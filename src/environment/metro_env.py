"""Gymnasium environment for metro train induction (M6, protocol section 6).

One episode is one operating day. At each of the 228 decision epochs the policy chooses how many
trainsets to induct over the next five minutes; the simulator then advances that interval and
returns the outcomes from which the multi-objective reward is computed.

Three properties are structural rather than matters of discipline, because each is a place where
a scheduling study can quietly go wrong:

**No future information.** Every observation channel is built from simulator state at or before
the decision instant, and the forecast is recomputed in-episode from realised history through
:class:`~src.environment.forecast_provider.HistoryBuffer`, whose accessors refuse to return any
hour at or after the issue time. `tests/test_leakage.py` verifies it empirically by changing the
future and asserting the observation does not move.

**Common random numbers.** The environment's random streams come from
``environment_streams(scenario, eval_seed)``, which takes no method argument -- the demand and
failure realisations *cannot* depend on which policy is being evaluated. The placebo shuffle uses
an agent-side stream instead, so it cannot perturb them either.

**Constraints are masked, not penalised.** An infeasible induction is unrepresentable rather than
expensive, so a reported zero headway violations is a guarantee, not a tuning outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.common.logging import get_logger
from src.common.seeding import agent_streams, environment_streams
from src.environment.action_mask import InfeasibleActionError, action_mask
from src.environment.arms import METHOD_TO_ARM, SigmaArm, make_sigma_arm
from src.environment.forecast_provider import (
    REQUIRED_HISTORY_HOURS,
    ForecastProvider,
    HistoryBuffer,
)
from src.environment.observation import (
    ObservationScaler,
    ObservationSpec,
    build_observation_spec,
)
from src.environment.reward import (
    WEIGHT_GRID,
    ReferenceScales,
    RewardComponents,
    RewardWeights,
    compute_reward,
)
from src.simulator.demand import DemandProcess
from src.simulator.disruptions import ScenarioSpec, generate_scenario
from src.simulator.engine import MetroSimulator, SimulatorConfig
from src.simulator.entities import TrainStatus
from src.simulator.network import DIRECTIONS, Network

logger = get_logger(__name__)

__all__ = ["MetroEnv", "EnvConfig", "HistorySampler"]

#: Callable supplying pre-episode observable history, shape ``(hours, n_stations)``.
HistorySampler = Callable[[np.random.Generator, str], np.ndarray]


def _sanitise(payload: dict[str, Any]) -> dict[str, Any]:
    """Replace non-finite floats with ``None`` for the per-step ``info`` dict.

    Two quantities are legitimately undefined in an epoch where nobody boarded: the mean waiting
    time over boarders, and the reward diagnostic derived from it. NaN is the honest value, and
    it is what :meth:`MetroEnv.episode_records` stores. But ``NaN != NaN``, so leaving it in
    ``info`` makes Gymnasium's environment checker report this deterministic environment as
    non-deterministic -- and would make any downstream equality check on ``info`` silently
    wrong. ``None`` says "undefined" without lying and compares equal to itself.
    """
    return {
        key: (None if isinstance(value, float) and not np.isfinite(value) else value)
        for key, value in payload.items()
    }


@dataclass
class EnvConfig:
    """Frozen environment parameters (``configs/env.yaml``).

    Attributes:
        k_max: Largest induction action. ``Discrete(k_max + 1)``.
        n_segments: ``K`` frozen line segments for the position summary.
        include_availability_channel: Protocol section 6.1 item 8; off for ``ppo_uq_noavail``.
        availability_noise_sd: Standard deviation of the noisy readout of latent train health.
        frailty_log_sd: Log-normal spread of the per-train hazard multiplier. Zero reproduces
            the homogeneous M5 failure process exactly.
        weight_id: Which row of the frozen weight grid this run uses.
        arm: ``point`` | ``placebo`` | ``uq``.
        scenario: One of the four frozen scenario families.
        day_type: ``weekday`` or ``weekend``.
        strict_scaler: Refuse to run on an unfitted observation scaler. True for reportable
            runs; False for smoke tests.
    """

    k_max: int = 2
    n_segments: int = 4
    include_availability_channel: bool = True
    availability_noise_sd: float = 0.10
    frailty_log_sd: float = 0.35
    weight_id: str = "w_balanced"
    arm: str = "uq"
    scenario: str = "normal"
    day_type: str = "weekday"
    strict_scaler: bool = False
    availability_mask_threshold: float = 0.0

    @classmethod
    def from_yaml(
        cls,
        path: Path | None = None,
        *,
        allow_unresolved: bool = False,
        sweep_tag: str | None = None,
        **overrides,
    ) -> "EnvConfig":
        """Load from ``configs/env.yaml`` through the provenance-enforcing loader.

        Two of this file's parameters are ``unresolved`` (assumption A14: no public per-train
        availability record exists for the modelled fleet), so the loader refuses to resolve it
        unless the run explicitly opts in. That refusal is the point -- an ungrounded value must
        be impossible to use by accident, and a run that uses one is self-identifying in its own
        manifest.

        Args:
            path: Config file; defaults to ``configs/env.yaml``.
            allow_unresolved: Passed through; must be paired with
                ``sweep_tag='sensitivity_sweep'``.
            sweep_tag: Passed through.
            **overrides: Applied after loading, for per-experiment fields such as ``arm``,
                ``scenario`` and ``weight_id`` that the registry supplies rather than the file.
        """
        from src.common.config import load_config
        from src.common.paths import REPO_ROOT

        config = load_config(
            path or REPO_ROOT / "configs" / "env.yaml",
            allow_unresolved=allow_unresolved,
            sweep_tag=sweep_tag,
        )
        values = dict(
            k_max=int(config.get("action.k_max")),
            n_segments=int(config.get("observation.n_segments")),
            include_availability_channel=bool(
                config.get("observation.include_availability_channel")
            ),
            availability_noise_sd=float(config.get("availability_signal.noise_sd")),
            frailty_log_sd=float(config.get("availability_signal.frailty_log_sd")),
            availability_mask_threshold=float(config.get("availability_signal.mask_threshold")),
            weight_id=str(config.get("reward.weight_id")),
            scenario=str(config.get("episode.scenario")),
            day_type=str(config.get("episode.day_type")),
            strict_scaler=bool(config.get("episode.strict_scaler")),
        )
        values.update(overrides)
        return cls(**values)


class MetroEnv(gym.Env):
    """Metro train induction as a Gymnasium environment.

    Args:
        network: The modelled line.
        demand_process: Ground-truth demand ``D_true``, fitted on `train` days. The forecaster
            never sees the path this samples -- that separation is what makes H1 testable
            (protocol section 5.1).
        simulator_config: Frozen simulator parameters.
        env_config: Frozen environment parameters.
        forecast_provider: Produces (mu, sigma-hat) in-episode from realised history.
        history_sampler: Supplies the observable history preceding the simulated day.
        scales: Frozen reward reference scales.
        scaler: Frozen observation normalisation.
        train_seed: Policy-side seed. Drives the placebo shuffle and nothing in the
            environment's own stochasticity.
        eval_seed: Environment-side seed. With ``scenario``, fully determines the demand and
            failure realisations -- identically for every method.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        network: Network,
        demand_process: DemandProcess,
        simulator_config: SimulatorConfig,
        env_config: EnvConfig,
        forecast_provider: ForecastProvider,
        history_sampler: HistorySampler,
        scales: ReferenceScales,
        scaler: ObservationScaler | None = None,
        *,
        train_seed: int = 101,
        eval_seed: int = 1,
    ) -> None:
        super().__init__()

        if env_config.weight_id not in WEIGHT_GRID:
            raise ValueError(
                f"unknown weight_id {env_config.weight_id!r}; the grid is frozen in protocol "
                f"section 6.5 as {sorted(WEIGHT_GRID)}"
            )

        self.network = network
        self.demand_process = demand_process
        self.sim_config = simulator_config
        self.cfg = env_config
        self.forecast_provider = forecast_provider
        self.history_sampler = history_sampler
        self.scales = scales
        self.weights: RewardWeights = WEIGHT_GRID[env_config.weight_id]
        self.train_seed = int(train_seed)
        self.eval_seed = int(eval_seed)

        self.simulator = MetroSimulator(network, demand_process, simulator_config)
        self.obs_spec: ObservationSpec = build_observation_spec(
            station_ids=tuple(network.station_ids),
            n_segments=env_config.n_segments,
            include_availability_channel=env_config.include_availability_channel,
        )
        self.scaler = scaler or ObservationScaler.identity(self.obs_spec.size)
        if self.scaler.fitted and self.scaler.centre.size != self.obs_spec.size:
            # The frozen scaler is fitted on the full layout. Arms that drop a block -- currently
            # only `ppo_uq_noavail`, which has no availability channel -- see a shorter
            # observation, and the constants must be restricted to the blocks they actually have.
            # Projecting rather than refitting keeps every surviving feature scaled exactly as in
            # the full-observation arms, so the H7 ablation differs only by the missing channel.
            full_spec = build_observation_spec(
                station_ids=tuple(network.station_ids),
                n_segments=env_config.n_segments,
                include_availability_channel=True,
            )
            self.scaler = self.scaler.project(full_spec, self.obs_spec)
        if env_config.strict_scaler:
            self.scaler.require_fitted()

        self.sigma_arm: SigmaArm = make_sigma_arm(
            env_config.arm,
            agent_streams("ppo_placebo", self.train_seed)["placebo_shuffle"]
            if env_config.arm == "placebo"
            else None,
        )

        self.action_space = spaces.Discrete(env_config.k_max + 1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_spec.size,), dtype=np.float32
        )

        self._segment_bounds = self._compute_segment_bounds(env_config.n_segments)
        self._history = HistoryBuffer(network.n_stations)
        self._availability: np.ndarray = np.ones(simulator_config.fleet_size)
        self._prev_boarded_cumulative = np.zeros(network.n_stations)
        self._boardings_prev_epoch = np.zeros(network.n_stations)
        self._hour_boardings_accumulator = np.zeros(network.n_stations)
        self._epoch_records: list[dict[str, Any]] = []
        self._scenario_spec: ScenarioSpec | None = None
        self._streams: dict[str, np.random.Generator] = {}
        self._trains_lost_today = 0
        self._raw_cache: np.ndarray | None = None
        self._raw_cache_epoch: int = -1

    # ------------------------------------------------------------------ setup helpers

    def _compute_segment_bounds(self, k: int) -> list[tuple[int, int]]:
        """Split the line into K contiguous station ranges of near-equal length."""
        n = self.network.n_stations
        edges = np.linspace(0, n, k + 1).round().astype(int)
        return [(int(edges[i]), int(edges[i + 1])) for i in range(k)]

    @property
    def epochs_per_hour(self) -> int:
        return int(round(3600.0 / self.sim_config.decision_interval_seconds))

    # ------------------------------------------------------------------ gymnasium API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a fresh operating day.

        ``seed`` is accepted for Gymnasium compliance and, when given, is used as the
        **evaluation** seed. It never seeds a global generator: all stochasticity flows through
        the named streams so that a change anywhere else in the codebase cannot shift the demand
        realisation.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.eval_seed = int(seed)
        if options:
            self.eval_seed = int(options.get("eval_seed", self.eval_seed))
            self.cfg.scenario = str(options.get("scenario", self.cfg.scenario))

        self._streams = environment_streams(self.cfg.scenario, self.eval_seed)
        self._scenario_spec = generate_scenario(
            self.cfg.scenario,
            self._streams,
            day_seconds=self.sim_config.day_seconds,
            fleet_size=self.sim_config.fleet_size,
            day_type=self.cfg.day_type,
        )
        self.simulator.reset(self._scenario_spec, self._streams)

        self._draw_train_health()

        # Drawn from a dedicated stream, not from `demand`: sampling history off the demand
        # stream would advance it and make the realised arrivals depend on how much history the
        # forecaster happens to need.
        history = self.history_sampler(self._streams["history"], self.cfg.day_type)
        self._history = HistoryBuffer(self.network.n_stations)
        self._history.set_history(history)
        self.forecast_provider.reset(
            self._history,
            day_start_hour=self.sim_config.day_start_hour,
            is_weekend=self.cfg.day_type == "weekend",
        )
        self.sigma_arm.reset()

        self._prev_boarded_cumulative = np.zeros(self.network.n_stations)
        self._boardings_prev_epoch = np.zeros(self.network.n_stations)
        self._hour_boardings_accumulator = np.zeros(self.network.n_stations)
        self._epoch_records = []
        self._trains_lost_today = 0
        self._raw_cache = None
        self._raw_cache_epoch = -1

        observation = self._build_observation()
        return observation, self._reset_info()

    def step(
        self, action: int | np.integer
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one decision epoch.

        Raises:
            InfeasibleActionError: if ``action`` is masked. Protocol section 6.3 requires a
                raise rather than a silent no-op, and requires simulator state to be unchanged
                afterwards -- which holds here because the check precedes every mutation.
        """
        action = int(action)
        if not 0 <= action <= self.cfg.k_max:
            raise InfeasibleActionError(
                f"action {action} outside Discrete({self.cfg.k_max + 1})"
            )

        mask = self.action_masks()
        if not mask[action]:
            raise InfeasibleActionError(
                f"action {action} is masked at epoch {self.simulator.epoch}: "
                f"feasible actions are {np.flatnonzero(mask).tolist()}. "
                "Constraints are enforced structurally (protocol section 6.3), so an infeasible "
                "induction is rejected rather than penalised."
            )

        failures_before = self.simulator._totals["failures"]
        metrics = self.simulator.step_epoch(action)
        self._trains_lost_today += int(self.simulator._totals["failures"] - failures_before)

        self._update_boardings(metrics)

        components = compute_reward(metrics, self.weights, self.scales)
        record = {
            "epoch": self.simulator.epoch - 1,
            "action": action,
            **components.to_dict(),
            **metrics.to_dict(),
        }
        self._epoch_records.append(record)

        terminated = bool(self.simulator.done)
        truncated = False
        observation = self._build_observation()

        info: dict[str, Any] = {
            "components": _sanitise(components.to_dict()),
            "epoch_metrics": _sanitise(metrics.to_dict()),
            "action_mask": mask,
            "scenario": self.cfg.scenario,
            "eval_seed": self.eval_seed,
        }
        if terminated:
            info["episode_summary"] = self.episode_summary()

        return observation, float(components.reward), terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Feasible-action mask. The name is what `sb3-contrib`'s MaskablePPO looks for."""
        return action_mask(
            self.simulator,
            self.cfg.k_max,
            availability_probability=(
                self._availability if self.cfg.include_availability_channel else None
            ),
            availability_threshold=self.cfg.availability_mask_threshold,
        )

    # ------------------------------------------------------------------ internals

    def _draw_train_health(self) -> None:
        """Latent per-train health and its noisy readout (assumption A14, decision D-04).

        The signal is a noisy observation of a latent state drawn **at episode start** that
        *causes* the hazard. It is therefore causal, not anticipatory: the agent sees a health
        indicator, not a preview of which train will fail. No claim of train-health prediction
        performance is made anywhere -- D-04 demoted the module to an information channel whose
        value to the policy is what gets measured (hypothesis H7).

        The frailty multiplier is normalised to mean exactly 1, so the fleet-aggregate failure
        rate stays equal to the value calibrated against published MDBF in M5.
        """
        fleet = self.sim_config.fleet_size
        if not self.cfg.include_availability_channel or self.cfg.frailty_log_sd <= 0:
            self._availability = np.ones(fleet)
            return

        rng = self._streams["maintenance"]
        sd = self.cfg.frailty_log_sd
        # E[lognormal(mu, sd)] = exp(mu + sd^2/2); choosing mu = -sd^2/2 gives mean exactly 1.
        frailty = rng.lognormal(mean=-0.5 * sd * sd, sigma=sd, size=fleet)
        for train in self.simulator.trains:
            train.frailty = float(frailty[train.train_id])

        latent_health = 1.0 / (1.0 + frailty)          # in (0, 1), decreasing in frailty
        noise = rng.normal(0.0, self.cfg.availability_noise_sd, size=fleet)
        self._availability = np.clip(latent_health + noise, 0.0, 1.0)

    def _update_boardings(self, metrics) -> None:
        """Track per-station boardings for the observation and for the forecaster's history."""
        cumulative = np.array(
            [
                sum(self.simulator.queues[s][d].boarded for d in DIRECTIONS)
                for s in range(self.network.n_stations)
            ]
        )
        self._boardings_prev_epoch = cumulative - self._prev_boarded_cumulative
        self._prev_boarded_cumulative = cumulative
        self._hour_boardings_accumulator += self._boardings_prev_epoch

        # A completed clock hour is appended to the forecaster's observable history. This is the
        # only channel by which in-episode realisations reach the forecaster, and it carries
        # exclusively hours that have already finished.
        if self.simulator.epoch % self.epochs_per_hour == 0:
            self._history.append_hour(self._hour_boardings_accumulator)
            self._hour_boardings_accumulator = np.zeros(self.network.n_stations)

    def _current_issue_hour(self) -> int:
        """Absolute hour index at which a forecast issued *now* would be issued.

        Equal to the number of hours realised so far, so the forecaster's window ends at the
        last completed hour and the current, still-incomplete hour is invisible to it.
        """
        return self._history.n_hours

    def _build_observation(self) -> np.ndarray:
        sim = self.simulator
        n = self.network.n_stations
        forecast = self.forecast_provider.forecast(self._current_issue_hour())

        mu = forecast.mu
        sigma_observed = self.sigma_arm.apply(forecast.sigma)

        # Line aggregate. Means add; dispersions are combined in quadrature, i.e. under an
        # independence assumption across stations. Station demands are certainly correlated, so
        # this understates the true line-level spread -- but it is a monotone function of the
        # per-station sigmas either way, so the information available to the policy is
        # unchanged. Recorded as assumption A-ENV-01 in docs/rl_environment.md.
        mu_line = mu.sum(axis=0)
        sigma_line = np.sqrt((sigma_observed**2).sum(axis=0))

        queue = np.array(
            [sum(sim.queues[s][d].size for d in DIRECTIONS) for s in range(n)]
        )

        n_idle = sum(1 for t in sim.trains if t.status == TrainStatus.DEPOT)
        n_service = sum(1 for t in sim.trains if t.in_service)
        n_unavailable = sum(
            1 for t in sim.trains if t.status in (TrainStatus.FAILED, TrainStatus.MAINTENANCE)
        )

        segment_counts = np.zeros(2 * self.cfg.n_segments)
        for train in sim.trains:
            if not train.in_service:
                continue
            for index, (low, high) in enumerate(self._segment_bounds):
                if low <= train.station < high:
                    segment_counts[index + self.cfg.n_segments * train.direction] += 1
                    break

        in_service = [t for t in sim.trains if t.in_service]
        if in_service:
            ratios = np.array(
                [t.occupancy / self.sim_config.capacity_passengers for t in in_service]
            )
            occupancy = np.array([ratios.mean(), ratios.max()])
        else:
            occupancy = np.zeros(2)

        elapsed = sim.time_s
        time_of_day = (self.sim_config.day_start_hour * 3600.0 + elapsed) % 86400.0
        epochs_remaining = 1.0 - sim.epoch / max(1, self.sim_config.n_decision_epochs)
        time_index = np.array(
            [
                np.sin(2 * np.pi * time_of_day / 86400.0),
                np.cos(2 * np.pi * time_of_day / 86400.0),
                epochs_remaining,
            ]
        )

        spec = self._scenario_spec
        active_surge = 1.0 if (spec is not None and spec.scale_at(elapsed) > 1.0) else 0.0
        disruption = np.array(
            [1.0 if self._trains_lost_today > 0 else 0.0, float(self._trains_lost_today), active_surge]
        )

        last_induction = sim.last_induction_s()
        since_induction = (
            self.sim_config.day_seconds if last_induction == float("-inf") else elapsed - last_induction
        )
        headway_state = np.array([since_induction, sim.min_observed_headway_s()])

        parts = [
            mu[:, 0],
            sigma_observed[:, 0],
            mu_line[1:3],
            sigma_line[1:3],
            queue,
            self._boardings_prev_epoch,
            np.array([n_idle, n_service, n_unavailable], dtype=float),
        ]
        if self.cfg.include_availability_channel:
            idle = [t.train_id for t in sim.trains if t.status == TrainStatus.DEPOT]
            if idle:
                signal = self._availability[idle]
                parts.append(np.array([signal.mean(), signal.min()]))
            else:
                parts.append(np.zeros(2))
        parts += [segment_counts, occupancy, time_index, disruption, headway_state]

        raw = np.concatenate([np.asarray(p, dtype=float).ravel() for p in parts])
        if raw.size != self.obs_spec.size:
            raise RuntimeError(
                f"observation assembled {raw.size} values but the frozen layout declares "
                f"{self.obs_spec.size}; the two must not drift apart"
            )
        if not np.all(np.isfinite(raw)):
            bad = [
                self.obs_spec.feature_names[i]
                for i in np.flatnonzero(~np.isfinite(raw))
            ]
            raise RuntimeError(f"non-finite observation values in channels {bad}")

        self._raw_cache = raw
        self._raw_cache_epoch = sim.epoch
        return self.scaler.transform(raw).astype(np.float32)

    def raw_observation(self, *, refresh: bool = False) -> np.ndarray:
        """The observation before normalisation, in physical units.

        Returns the vector cached during the most recent :meth:`_build_observation` when it is
        still current. Rebuilding it would re-issue the forecast, the single most expensive
        operation in a step -- and the heuristic and MILP policies call this on every epoch, so
        an uncached implementation made a rolling-horizon MILP episode 11 times slower.

        Args:
            refresh: Force a rebuild. **Any test that mutates simulator state mid-epoch and then
                asks whether the observation moved must pass this**, or the cache will return the
                pre-mutation vector and the test will pass without having tested anything. The
                leakage tests depend on it.
        """
        if not refresh and self._raw_cache is not None and self._raw_cache_epoch == self.simulator.epoch:
            return self._raw_cache
        scaler, self.scaler = self.scaler, ObservationScaler.identity(self.obs_spec.size)
        try:
            return self._build_observation().astype(float)
        finally:
            self.scaler = scaler

    # ------------------------------------------------------------------ reporting

    def _reset_info(self) -> dict[str, Any]:
        return {
            "scenario": self.cfg.scenario,
            "eval_seed": self.eval_seed,
            "train_seed": self.train_seed,
            "arm": self.cfg.arm,
            "weight_id": self.cfg.weight_id,
            "scenario_spec": self._scenario_spec.to_dict() if self._scenario_spec else None,
            "action_mask": self.action_masks(),
        }

    def episode_records(self) -> list[dict[str, Any]]:
        """Per-epoch records: action, every reward component in physical units, and outcomes."""
        return list(self._epoch_records)

    def episode_summary(self) -> dict[str, Any]:
        """Simulator summary plus the environment-level accounting."""
        summary = dict(self.simulator.episode_summary())
        summary["total_reward"] = float(sum(r["reward"] for r in self._epoch_records))
        summary["arm"] = self.cfg.arm
        summary["weight_id"] = self.cfg.weight_id
        summary["scenario"] = self.cfg.scenario
        summary["eval_seed"] = self.eval_seed
        summary["train_seed"] = self.train_seed
        summary["n_epochs"] = len(self._epoch_records)
        for key in ("W_minutes", "U_passengers", "E_units", "C_units"):
            summary[f"sum_{key}"] = float(sum(r[key] for r in self._epoch_records))
        return summary


def constant_history_sampler(
    demand_process: DemandProcess,
    day_start_hour: int,
    hours: int = 4 * REQUIRED_HISTORY_HOURS + 24,
) -> HistorySampler:
    """History sampler that draws pre-episode days from the ground-truth process.

    Used for smoke runs and unit tests, and as the fallback when the real observed history for
    the sampled date is unavailable. The draws come from the caller's ``demand`` stream, so the
    history is reproducible under ``(scenario, eval_seed)`` like everything else.

    For reportable experiments the history should be the **real observed series** preceding the
    simulated date; that sampler is supplied at M7 alongside the fixed-timetable baseline, when
    episode dates are drawn from the `test` split.
    """

    def sampler(rng: np.random.Generator, day_type: str) -> np.ndarray:
        n_days = int(np.ceil(hours / 24.0)) + 1
        blocks = [demand_process.hourly_intensity(day_type, rng) for _ in range(n_days)]
        stacked = np.vstack(blocks)  # (24 * n_days, n_stations), hour 0 of each day first
        # Cut so the **last history row is the hour immediately before the operating day
        # starts**. Without this the buffer's hour indexing is offset from the episode's, and
        # the 168-hour seasonal lag -- an exact multiple of 24 -- would silently fetch the wrong
        # hour of day, quietly degrading every forecast the agent sees.
        end = stacked.shape[0] - 24 + day_start_hour
        return stacked[end - hours : end]

    return sampler
