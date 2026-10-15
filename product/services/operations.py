"""Operational services for the dispatch decision-support prototype.

These wrap the validated research environment for **product demonstration**. They run the
simulator and the trained policies, and they read the frozen forecaster, but they write nothing:
no service here calls `save()`, and none opens a research path for writing.

Everything these services produce is `PRODUCT DEMONSTRATION` evidence -- a single simulated draw,
not a research result. The distinction is carried in the return values, not left to the caller to
remember: `SimulationOutcome` and `DispatchRecommendation` both declare their evidence class.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any

from product.services.research_repository import PRODUCT, ResearchArtefactMissing

SCENARIOS: tuple[str, ...] = ("normal", "surge", "disruption", "combined")

SCENARIO_LABEL = {
    "normal": "Normal operation",
    "surge": "Demand surge",
    "disruption": "Train disruption",
    "combined": "Combined stress",
}

#: Uncertainty bands, expressed as sigma relative to the forecast mean. Thresholds are product
#: presentation choices, not research findings, and are labelled as such in the UI.
UNCERTAINTY_BANDS = ((0.10, "LOW"), (0.20, "MEDIUM"))


class ProductServiceError(RuntimeError):
    """A service could not produce a result. Surfaced to the operator, never substituted."""


@dataclass
class OperatingState:
    """What the operator sees about the line right now, in one simulated episode."""

    epoch: int
    total_epochs: int
    clock: str
    demand_now: float
    forecast_mean: float
    forecast_sigma: float
    carries_uncertainty: bool
    trains_in_service: int
    fleet_size: int
    trains_available: int
    total_queue: float
    max_station_queue: float
    capacity_pressure: str
    uncertainty_level: str
    scenario: str
    evidence: str = PRODUCT

    @property
    def forecast_low(self) -> float:
        return max(0.0, self.forecast_mean - 1.2816 * self.forecast_sigma)

    @property
    def forecast_high(self) -> float:
        return self.forecast_mean + 1.2816 * self.forecast_sigma


@dataclass
class DispatchRecommendation:
    """A recommended action, its reasons, and what the simulator expects it to do."""

    action: int
    label: str
    reasons: list[str]
    feasible_actions: list[int]
    source: str
    evidence: str = PRODUCT
    expected: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationOutcome:
    """Before/after metrics for a product demonstration run."""

    scenario: str
    method: str
    epochs_run: int
    before: dict[str, Any]
    after: dict[str, Any]
    evidence: str = PRODUCT

    def delta(self, key: str) -> float | None:
        a, b = self.before.get(key), self.after.get(key)
        if a is None or b is None:
            return None
        try:
            return float(b) - float(a)
        except (TypeError, ValueError):
            return None


def capacity_pressure(queue: float, trains_in_service: int, fleet_size: int) -> str:
    """A coarse operator-facing band. A presentation device, not a research quantity."""
    utilisation = trains_in_service / fleet_size if fleet_size else 0.0
    if queue > 1500 or (queue > 800 and utilisation > 0.8):
        return "HIGH"
    if queue > 500 or utilisation > 0.6:
        return "MODERATE"
    return "LOW"


def uncertainty_level(mean: float, sigma: float) -> str:
    if mean <= 0 or sigma <= 0:
        return "NOT AVAILABLE"
    ratio = sigma / mean
    for threshold, label in UNCERTAINTY_BANDS:
        if ratio < threshold:
            return label
    return "HIGH"


class SessionService:
    """Owns one live simulated episode. Never writes."""

    def __init__(self, method: str, scenario: str, eval_seed: int,
                 train_seed: int | None = None) -> None:
        from src.demo.session import DemoSession, MissingArtifactError

        try:
            self.session = DemoSession(method, scenario, eval_seed=eval_seed,
                                       train_seed=train_seed)
        except MissingArtifactError as error:
            raise ProductServiceError(str(error)) from error
        self.method, self.scenario = method, scenario

    # -- state ---------------------------------------------------------------------------

    def operating_state(self) -> OperatingState:
        network = self.session.network_state()
        forecast = self.session.forecast_state()
        queues = [s["queue"] for s in network["stations"]]
        total_queue = float(sum(queues))
        in_service = int(network["trains_in_service"])
        fleet = int(network["fleet_size"])
        mean = float(forecast.get("mu_line_h1") or 0.0)
        sigma = float(forecast.get("sigma_line_h1") or 0.0)
        return OperatingState(
            epoch=int(network["epoch"]), total_epochs=int(network["n_decision_epochs"]),
            clock=str(network["clock"]),
            demand_now=total_queue, forecast_mean=mean, forecast_sigma=sigma,
            carries_uncertainty=bool(forecast.get("carries_uncertainty")),
            trains_in_service=in_service, fleet_size=fleet,
            trains_available=max(0, fleet - in_service),
            total_queue=total_queue, max_station_queue=float(max(queues) if queues else 0.0),
            capacity_pressure=capacity_pressure(total_queue, in_service, fleet),
            uncertainty_level=(uncertainty_level(mean, sigma)
                               if forecast.get("carries_uncertainty") else "NOT AVAILABLE"),
            scenario=self.scenario,
        )

    def network_state(self) -> dict[str, Any]:
        return self.session.network_state()

    def forecast_state(self) -> dict[str, Any]:
        return self.session.forecast_state()

    def metrics(self) -> dict[str, Any]:
        return self.session.metrics()

    def traceability(self) -> dict[str, Any]:
        return self.session.traceability()

    @property
    def terminated(self) -> bool:
        return bool(self.session.terminated)

    # -- actions -------------------------------------------------------------------------

    def recommend(self) -> DispatchRecommendation:
        """What the selected policy would do at the current decision epoch.

        The action comes from the policy itself, so the recommendation is whatever the validated
        artefact actually produces. The *reasons* are an operator-facing reading of the observable
        state, not a claim about the network's internal computation.
        """
        state = self.operating_state()
        action, feasible = self._peek_via_policy()
        reasons = self._reasons(state, action)
        return DispatchRecommendation(
            action=int(action),
            label=self._action_label(int(action)),
            reasons=reasons,
            feasible_actions=[int(a) for a in feasible],
            source=f"policy `{self.method}`",
        )

    def _peek_via_policy(self) -> tuple[int, list[int]]:
        """The action the policy would take now, without advancing the episode.

        Follows `DemoSession.step` exactly -- same mask, same clip to `k_max`, same truncation to
        the largest feasible action. A recommendation that did not clip identically could show the
        operator an action the environment would then refuse to execute.
        """
        import numpy as np

        session = self.session
        env, policy = session.env, session.policy
        try:
            mask = env.action_masks()
        except Exception as error:  # noqa: BLE001
            raise ProductServiceError(f"action masking failed: {error!r}") from error
        feasible = [int(i) for i in np.flatnonzero(mask)]
        if not feasible:
            raise ProductServiceError("no action is feasible at this epoch")
        try:
            requested = int(policy.act(session.observation, env, mask))
        except Exception as error:  # noqa: BLE001
            raise ProductServiceError(
                f"the policy could not produce an action: {error!r}") from error
        requested = int(np.clip(requested, 0, env.cfg.k_max))
        return min(requested, feasible[-1]), feasible

    @staticmethod
    def _action_label(action: int) -> str:
        if action == 0:
            return "NO ADDITIONAL DISPATCH"
        if action == 1:
            return "DISPATCH 1 TRAINSET"
        return f"DISPATCH {action} TRAINSETS"

    @staticmethod
    def _reasons(state: OperatingState, action: int) -> list[str]:
        reasons: list[str] = []
        if state.total_queue > 800:
            reasons.append(f"Queue pressure is elevated ({state.total_queue:,.0f} waiting).")
        elif state.total_queue > 300:
            reasons.append(f"Queue is building ({state.total_queue:,.0f} waiting).")
        else:
            reasons.append(f"Queue is light ({state.total_queue:,.0f} waiting).")
        if state.forecast_mean > 0:
            reasons.append(
                f"Forecast demand for the next hour is {state.forecast_mean:,.0f}"
                + (f" (80% interval {state.forecast_low:,.0f}–{state.forecast_high:,.0f})."
                   if state.carries_uncertainty else ", with no uncertainty channel on this arm."))
        reasons.append(f"{state.trains_available} of {state.fleet_size} trainsets are available; "
                       f"{state.trains_in_service} are in service.")
        reasons.append(f"Capacity pressure is {state.capacity_pressure}.")
        if action == 0:
            reasons.append("The policy is withholding additional dispatch at this epoch.")
        else:
            reasons.append(f"The policy is releasing {action} trainset(s) at this epoch.")
        return reasons

    def _snapshot(self) -> dict[str, Any]:
        """The episode summary, which is where the operator-facing quantities live."""
        payload = self.session.metrics()
        snapshot = dict(payload.get("summary") or {})
        snapshot["epochs_stepped"] = payload.get("epochs_stepped")
        snapshot["cumulative_reward"] = payload.get("cumulative_reward")
        endpoint = payload.get("primary_endpoint") or {}
        if "value" in endpoint:
            snapshot.setdefault(str(endpoint.get("name")), endpoint["value"])
        return snapshot

    def step(self, epochs: int = 1) -> SimulationOutcome:
        before = self._snapshot()
        ran = 0
        for _ in range(max(1, epochs)):
            if self.session.terminated:
                break
            self.session.step()
            ran += 1
        return SimulationOutcome(scenario=self.scenario, method=self.method, epochs_run=ran,
                                 before=before, after=self._snapshot())

    def reset(self) -> None:
        self.session.reset()


@functools.lru_cache(maxsize=8)
def available_methods() -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    """(method_id, label, train_seeds) for every policy the product may offer.

    Pilot seeds are excluded upstream by `src.demo.session`, so a discarded policy can never be
    presented as an operational one.
    """
    from src.demo.session import available_policies

    return tuple((o.method_id, o.label, tuple(o.train_seeds))
                 for o in available_policies() if o.available)


def unavailable_methods() -> list[tuple[str, str]]:
    from src.demo.session import available_policies

    return [(o.method_id, o.reason) for o in available_policies() if not o.available]


__all__ = [
    "SCENARIOS", "SCENARIO_LABEL", "OperatingState", "DispatchRecommendation",
    "SimulationOutcome", "SessionService", "ProductServiceError", "available_methods",
    "unavailable_methods", "capacity_pressure", "uncertainty_level",
    "ResearchArtefactMissing",
]
