"""One interactive demonstration episode: build it, step it, read its state.

Deliberately free of any UI framework, so the behaviour the acceptance test cares about -- that
stepping advances the *actual* simulator, that the action comes from the *actual* policy, that a
missing artefact fails loudly -- can be tested without a browser.

Every quantity exposed here is read from the environment, the simulator or the policy. Nothing is
recomputed, and nothing is invented: a value the research code does not produce is reported as
absent rather than filled in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.common.logging import get_logger
from src.common.paths import MODELS_DIR, RESULTS_DIR, ensure_dir

logger = get_logger(__name__)

__all__ = ["DemoSession", "PolicyOption", "SCENARIOS", "available_policies", "DEMO_RESULTS_DIR"]

#: Demonstration output lives here and nowhere else. `results/runs/` holds confirmatory results
#: and is never opened for writing by this package -- `tests/test_demo.py` asserts it.
DEMO_RESULTS_DIR = RESULTS_DIR / "demo"

SCENARIOS: tuple[str, ...] = ("normal", "surge", "disruption", "combined")

_SCENARIO_LABELS = {
    "normal": "Normal",
    "surge": "Peak / Surge",
    "disruption": "Train Disruption",
    "combined": "Combined",
}

_POLICY_LABELS = {
    "fixed_timetable": "Fixed Timetable",
    "heuristic_threshold": "Tuned Heuristic",
    "milp_static": "MILP (static plan)",
    "milp_rolling": "MILP (rolling horizon)",
    "ppo_point": "Point-Forecast PPO",
    "ppo_uq": "Uncertainty-Aware PPO",
    "ppo_placebo": "Placebo-Sigma PPO",
}


def scenario_label(scenario: str) -> str:
    return _SCENARIO_LABELS.get(scenario, scenario)


@dataclass(frozen=True)
class PolicyOption:
    """A selectable policy, and whether its trained artefact actually exists.

    A policy whose artefact is missing is reported as unavailable with the reason, never silently
    replaced by a fallback. A dashboard that quietly substituted a different policy would be
    showing one method's behaviour under another's name.
    """

    method_id: str
    label: str
    available: bool
    reason: str = ""
    train_seeds: tuple[int, ...] = ()

    @property
    def needs_train_seed(self) -> bool:
        return self.method_id.startswith("ppo")


#: Training seeds at or above this are pilot seeds. The R-1 pilot's only job was to estimate the
#: between-seed SD before the sample size was fixed; protocol section 8.4 discards its policies
#: and admits no pilot observation to a reported result. Offering one here would let a viewer see
#: a discarded policy's behaviour presented as the study's -- so the demo refuses to list them.
PILOT_SEED_FLOOR = 900


def _trained_policy_seeds(method: str, weight_id: str = "w_balanced") -> tuple[int, ...]:
    directory = MODELS_DIR / "policies" / weight_id / method
    if not directory.is_dir():
        return ()
    seeds = []
    for child in directory.iterdir():
        if child.is_dir() and (child / "policy.zip").is_file():
            try:
                seed = int(child.name)
            except ValueError:
                continue
            if seed < PILOT_SEED_FLOOR:
                seeds.append(seed)
    return tuple(sorted(seeds))


def available_policies(weight_id: str = "w_balanced") -> list[PolicyOption]:
    """Every policy the dashboard could offer, each marked with whether it can actually run."""
    options: list[PolicyOption] = []

    # Artefact locations are imported from the modules that own them rather than restated here,
    # so a path change cannot leave the dashboard reporting a policy as missing that exists.
    from src.baselines.fixed_timetable import TIMETABLE_FILE

    options.append(PolicyOption(
        "fixed_timetable", _POLICY_LABELS["fixed_timetable"],
        available=TIMETABLE_FILE.is_file(),
        reason="" if TIMETABLE_FILE.is_file() else
               f"missing {TIMETABLE_FILE.name} (run the GTFS timetable extraction)"))

    from src.baselines.heuristic import HEURISTIC_FILE
    from src.baselines.milp import SURROGATE_FILE

    options.append(PolicyOption(
        "heuristic_threshold", _POLICY_LABELS["heuristic_threshold"],
        available=HEURISTIC_FILE.is_file(),
        reason="" if HEURISTIC_FILE.is_file() else
               f"missing {HEURISTIC_FILE.name} (run the tuning stage)"))

    for method in ("milp_static", "milp_rolling"):
        options.append(PolicyOption(
            method, _POLICY_LABELS[method],
            available=SURROGATE_FILE.is_file(),
            reason="" if SURROGATE_FILE.is_file() else
                   f"missing {SURROGATE_FILE.name} (run the prepare stage)"))

    for method in ("ppo_point", "ppo_uq", "ppo_placebo"):
        seeds = _trained_policy_seeds(method, weight_id)
        options.append(PolicyOption(
            method, _POLICY_LABELS[method],
            available=bool(seeds),
            reason="" if seeds else "no trained policy yet (run the training stage)",
            train_seeds=seeds))

    return options


@dataclass
class StepRecord:
    """What happened in one decision epoch, as the simulator and policy reported it."""

    epoch: int
    clock: str
    requested_action: int
    executed_action: int
    feasible_actions: list[int]
    action_mask: list[bool]
    reward: float
    reward_components: dict[str, float]
    metrics: dict[str, float]
    terminated: bool


class MissingArtifactError(RuntimeError):
    """Raised when a selected policy has no trained artefact. Never silently substituted."""


class DemoSession:
    """One episode, advanced under interactive control.

    Args:
        method: Registered method id, e.g. ``ppo_uq``.
        scenario: One of :data:`SCENARIOS`.
        eval_seed: Evaluation seed. The same value reproduces the same episode exactly.
        train_seed: Required for PPO methods; ignored otherwise.
        weight_id: Reward weighting; the frozen default is ``w_balanced``.
    """

    def __init__(self, method: str, scenario: str, eval_seed: int,
                 *, train_seed: int | None = None, weight_id: str = "w_balanced") -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")

        option = next((o for o in available_policies(weight_id) if o.method_id == method), None)
        if option is None:
            raise ValueError(f"unknown method {method!r}")
        if not option.available:
            raise MissingArtifactError(
                f"{option.label} cannot be demonstrated: {option.reason}. The dashboard does not "
                "substitute a different policy, because that would show one method's behaviour "
                "under another's name."
            )
        if option.needs_train_seed:
            if train_seed is None:
                train_seed = option.train_seeds[0]
            elif train_seed not in option.train_seeds:
                raise MissingArtifactError(
                    f"{option.label} has no trained policy at seed {train_seed}; "
                    f"available seeds are {list(option.train_seeds)}")

        self.method = method
        self.scenario = scenario
        self.eval_seed = int(eval_seed)
        self.train_seed = train_seed
        self.weight_id = weight_id
        self.option = option

        self.history: list[StepRecord] = []
        self._build()

    # ------------------------------------------------------------------ lifecycle

    def _build(self) -> None:
        from src.evaluation.runner import Cell, build_env, build_policy

        self._cell = Cell(method=self.method, scenario=self.scenario, eval_seed=self.eval_seed,
                          train_seed=self.train_seed, weight_id=self.weight_id,
                          variant="demo", experiment_id="M17-DEMO")
        self.env = build_env(self._cell, trained_forecaster=True)
        self.policy = build_policy(self._cell, self.env)
        self.reset()

    def reset(self) -> None:
        """Start the episode over. The same seed reproduces the same episode exactly."""
        self.observation, _ = self.env.reset(seed=self.eval_seed)
        if hasattr(self.policy, "reset"):
            self.policy.reset(self.env)
        self.history = []
        self.terminated = False

    # ------------------------------------------------------------------ stepping

    def step(self) -> StepRecord:
        """Advance one decision epoch under the actual policy.

        The action is whatever the policy returns. It is clipped to the feasible set exactly as
        `src.evaluation.policies.run_episode` clips it, so a demonstration step and a confirmatory
        step take the same path through the environment.
        """
        if self.terminated:
            raise RuntimeError("the episode has ended; call reset() to start another")

        mask = self.env.action_masks()
        requested = int(self.policy.act(self.observation, self.env, mask))
        requested = int(np.clip(requested, 0, self.env.cfg.k_max))
        feasible = int(np.flatnonzero(mask)[-1])
        executed = min(requested, feasible)

        epoch = int(self.env.simulator.epoch)
        self.observation, reward, terminated, truncated, info = self.env.step(executed)
        self.terminated = bool(terminated or truncated)

        record = self.env.episode_records()[-1]
        components = {key: float(record[key]) for key in
                      ("W_minutes", "U_passengers", "E_units", "C_units") if key in record}

        step_record = StepRecord(
            epoch=epoch,
            clock=self._clock(epoch),
            requested_action=requested,
            executed_action=executed,
            feasible_actions=list(range(feasible + 1)),
            action_mask=[bool(m) for m in mask],
            reward=float(reward),
            reward_components=components,
            metrics={k: float(v) for k, v in record.items()
                     if isinstance(v, (int, float)) and not isinstance(v, bool)},
            terminated=self.terminated,
        )
        self.history.append(step_record)
        return step_record

    def run(self, steps: int) -> list[StepRecord]:
        """Advance up to ``steps`` epochs, stopping early if the episode ends."""
        produced = []
        for _ in range(max(0, steps)):
            if self.terminated:
                break
            produced.append(self.step())
        return produced

    def _clock(self, epoch: int) -> str:
        config = self.env.sim_config
        seconds = config.day_start_hour * 3600 + epoch * config.decision_interval_seconds
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours % 24:02d}:{minutes:02d}:{secs:02d}"

    # ------------------------------------------------------------------ state readout

    def network_state(self) -> dict[str, Any]:
        """Stations, queues and train positions, read from the simulator."""
        from src.simulator.entities import TrainStatus

        simulator = self.env.simulator
        network = self.env.network

        stations = []
        for index in range(network.n_stations):
            queues = [simulator.queues[index][direction] for direction in (0, 1)]
            stations.append({
                "index": index,
                "station_id": network.station_ids[index],
                "name": network.station_names[index],
                "queue": float(sum(q.size for q in queues)),
                "queue_by_direction": [float(q.size) for q in queues],
                "oldest_wait_s": max(float(q.oldest_wait_s(simulator.time_s)) for q in queues),
            })

        trains = []
        for train in simulator.trains:
            trains.append({
                "train_id": f"T{train.train_id:02d}",
                "status": TrainStatus(train.status).name,
                "in_service": bool(train.in_service),
                "station": int(train.station),
                "station_name": network.station_names[int(train.station)],
                "direction": int(train.direction),
                "occupancy": float(train.occupancy),
                "capacity": float(self.env.sim_config.capacity_passengers),
                "occupancy_fraction": float(
                    train.occupancy / max(1e-9, self.env.sim_config.capacity_passengers)),
                "timer_s": float(train.timer_s),
            })

        return {
            "stations": stations,
            "trains": trains,
            "trains_in_service": sum(1 for t in simulator.trains if t.in_service),
            "fleet_size": int(self.env.sim_config.fleet_size),
            "clock": self._clock(int(simulator.epoch)),
            "epoch": int(simulator.epoch),
            "n_decision_epochs": int(self.env.sim_config.n_decision_epochs),
        }

    def forecast_state(self) -> dict[str, Any]:
        """The forecast the policy is actually looking at, in physical units.

        Read from the raw (unscaled) observation, so the numbers shown are passengers rather than
        standardised values. `sigma` is present only when the method's arm carries it: the
        point-forecast arm zeroes that channel, and showing a non-zero dispersion for it would
        misrepresent what the policy sees.
        """
        raw = self.env.raw_observation()
        spec = self.env.obs_spec

        def block(name: str) -> np.ndarray | None:
            try:
                return raw[spec.slice_of(name)]
            except (KeyError, ValueError):
                return None

        mu = block("forecast_mu_h1")
        sigma = block("forecast_sigma_h1")
        mu_line = block("forecast_mu_line_h2h3")
        sigma_line = block("forecast_sigma_line_h2h3")

        carries_sigma = sigma is not None and float(np.abs(sigma).sum()) > 0.0
        return {
            "arm": self.env.cfg.arm,
            "carries_uncertainty": carries_sigma,
            "horizon_hours": [1, 2, 3],
            "mu_per_station": mu.tolist() if mu is not None else [],
            "sigma_per_station": sigma.tolist() if sigma is not None else [],
            "mu_line_h1": float(mu.sum()) if mu is not None else float("nan"),
            "sigma_line_h1": (float(np.sqrt((sigma ** 2).sum()))
                              if sigma is not None else float("nan")),
            "mu_line_h2h3": mu_line.tolist() if mu_line is not None else [],
            "sigma_line_h2h3": sigma_line.tolist() if sigma_line is not None else [],
            "interval_note": (
                "sigma is derived from the calibrated 80% interval as (q90 - q10) / (2 x 1.2816); "
                "the interval shown is mu +/- 1.2816 sigma"
                if carries_sigma else
                "this arm's sigma channel is zeroed, so no interval is shown"
            ),
        }

    def realised_demand(self) -> list[dict[str, Any]]:
        """Demand that has already arrived, per stepped epoch.

        Displayed as an outcome *after* the fact. It is read from the episode record, which is
        written when an epoch is stepped, so it is not available to the policy at decision time --
        the policy has already acted by then.
        """
        return [
            {"epoch": record.epoch, "clock": record.clock,
             "arrived": float(record.metrics.get("arrived", float("nan"))),
             "boarded": float(record.metrics.get("boarded", float("nan"))),
             "left_behind": float(record.metrics.get("left_behind", float("nan")))}
            for record in self.history
        ]

    def metrics(self) -> dict[str, Any]:
        """Cumulative episode metrics, from the simulator's own accounting."""
        summary = self.env.episode_summary()
        cumulative_reward = float(sum(r.reward for r in self.history))
        latest = self.history[-1] if self.history else None
        return {
            "epochs_stepped": len(self.history),
            "cumulative_reward": cumulative_reward,
            "current_reward": latest.reward if latest else float("nan"),
            "current_components": latest.reward_components if latest else {},
            "summary": {k: v for k, v in summary.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)},
            "primary_endpoint": {
                "name": "left_behind_fraction",
                "value": float(summary.get("left_behind_fraction", float("nan"))),
                "note": "distinct left-behind passengers as a fraction of arrivals (amendment A-01)",
            },
        }

    # ------------------------------------------------------------------ provenance

    def traceability(self) -> dict[str, Any]:
        """Everything needed to reproduce this demonstration run."""
        from src.common.manifest import git_info, package_versions

        git = git_info()
        forecast_metadata: dict[str, Any] = {}
        path = MODELS_DIR / "forecasting" / "metadata.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            forecast_metadata = {
                "source": payload.get("source"),
                "calibration_factor": payload.get("calibration_factor"),
                "refit_agreement": payload.get("refit_agreement"),
                "source_run_git_commit": payload.get("source_run_git_commit"),
            }

        return {
            "kind": "INTERACTIVE DEMONSTRATION -- NOT A CONFIRMATORY RESULT",
            "method": self.method,
            "method_label": self.option.label,
            "scenario": self.scenario,
            "scenario_label": scenario_label(self.scenario),
            "eval_seed": self.eval_seed,
            "train_seed": self.train_seed,
            "weight_id": self.weight_id,
            "git_commit": git.commit,
            "git_dirty": git.dirty,
            "packages": package_versions(),
            "network": dict(self.env.network.provenance),
            "n_stations": int(self.env.network.n_stations),
            "horizon_epochs": int(self.env.sim_config.n_decision_epochs),
            "decision_interval_s": float(self.env.sim_config.decision_interval_seconds),
            "fleet_size": int(self.env.sim_config.fleet_size),
            "forecast_model": forecast_metadata,
            "reward_scales": dict(self.env.scales.provenance),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }

    def save(self, name: str | None = None) -> Path:
        """Write this run to ``results/demo/``. Never to the confirmatory tree."""
        ensure_dir(DEMO_RESULTS_DIR)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        filename = name or (f"{self.method}_{self.scenario}_seed{self.eval_seed}_{stamp}.json")
        path = DEMO_RESULTS_DIR / filename
        path.write_text(json.dumps({
            "traceability": self.traceability(),
            "metrics": self.metrics(),
            "steps": [
                {"epoch": r.epoch, "clock": r.clock, "requested": r.requested_action,
                 "executed": r.executed_action, "reward": r.reward,
                 "components": r.reward_components}
                for r in self.history
            ],
        }, indent=2, default=float), encoding="utf-8")
        logger.info("wrote demonstration run to %s", path)
        return path
