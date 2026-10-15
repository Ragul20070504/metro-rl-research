"""Tuned threshold heuristic (M8).

Protocol section 7.1: *"Induct when forecast demand or queue exceeds a threshold; thresholds tuned
by grid search on `train` days only, selected on `val`. Information available: the same mu as
`PPO-Point`."*

Two rules that matter for a fair comparison:

**It sees exactly what `ppo_point` sees, and no more.** It reads the forecast mean and the
observed queue; it never reads sigma. If it did, H4 would no longer be "RL versus a heuristic" but
"two methods with uncertainty versus one without".

**Its thresholds are tuned on days disjoint from the evaluation seeds.** A heuristic tuned on the
evaluation days would be a strictly stronger baseline than the RL arms, which never see them --
the asymmetry would run *against* the proposed method, but it would still be an asymmetry, and
protocol section 7.1 forbids it in either direction.

The heuristic is deliberately simple: two thresholds and a coarse grid. A baseline elaborate
enough to need its own tuning budget stops being a baseline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.common.logging import get_logger
from src.common.paths import MODELS_DIR, ensure_dir
from src.environment.metro_env import MetroEnv

logger = get_logger(__name__)

__all__ = ["HeuristicParameters", "HeuristicPolicy", "tune_heuristic", "HEURISTIC_FILE"]

HEURISTIC_FILE = MODELS_DIR / "heuristic_threshold.json"

#: Coarse, frozen grid. Chosen before any tuning run, and small on purpose.
#:
#: The first tuning run selected the largest queue threshold in this grid, which is evidence the
#: optimum lies outside it. Amendment A-10 extends the grid upward by doubling until the selection
#: is strictly interior, under a stopping rule fixed before the re-tune. The original values stay
#: here as the starting grid; the extension is generated, not hand-picked.
QUEUE_THRESHOLDS: tuple[float, ...] = (0.0, 50.0, 100.0, 200.0, 400.0, 800.0)
FORECAST_THRESHOLDS: tuple[float, ...] = (0.0, 2000.0, 4000.0, 6000.0, 8000.0)

#: Hard stop on the extension, so a pathological surface cannot spin forever. Reaching it is
#: itself a finding and is recorded rather than silently accepted.
MAX_EXTENSION_ROUNDS = 6


@dataclass
class HeuristicParameters:
    """Frozen thresholds and the provenance of their selection."""

    queue_threshold: float
    forecast_threshold: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path = HEURISTIC_FILE) -> Path:
        ensure_dir(path.parent)
        path.write_text(json.dumps({
            "queue_threshold": self.queue_threshold,
            "forecast_threshold": self.forecast_threshold,
            "provenance": self.provenance,
        }, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path = HEURISTIC_FILE) -> "HeuristicParameters":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            queue_threshold=float(payload["queue_threshold"]),
            forecast_threshold=float(payload["forecast_threshold"]),
            provenance=payload.get("provenance", {}),
        )


class HeuristicPolicy:
    """Induct one trainset when either pressure signal exceeds its threshold, two when both do."""

    method_id = "heuristic_threshold"

    def __init__(self, parameters: HeuristicParameters) -> None:
        self.parameters = parameters
        self._queue_index: slice | None = None
        self._mu_index: slice | None = None

    @classmethod
    def load(cls, path: Path = HEURISTIC_FILE) -> "HeuristicPolicy":
        return cls(HeuristicParameters.load(path))

    def reset(self, env: MetroEnv) -> None:
        # Resolved once per episode from the frozen layout rather than hard-coded, so a change to
        # the observation spec cannot silently point this at the wrong channel.
        self._queue_index = env.obs_spec.slice_of("queue_length")
        self._mu_index = env.obs_spec.slice_of("forecast_mu_h1")

    def act(self, observation: np.ndarray, env: MetroEnv, mask: np.ndarray) -> int:
        # Read physical units, not the normalised vector: the thresholds are in passengers and
        # must mean the same thing whether or not a scaler happens to be fitted.
        raw = env.raw_observation()
        queue = float(raw[self._queue_index].sum())
        forecast = float(raw[self._mu_index].sum())

        over_queue = queue > self.parameters.queue_threshold
        over_forecast = forecast > self.parameters.forecast_threshold
        return int(over_queue) + int(over_forecast)


class _QueueRecordingPolicy(HeuristicPolicy):
    """A heuristic that also remembers the largest queue it was ever asked to judge.

    A-10's stopping rule needs the largest value the rule `queue > threshold` is ever compared
    against, because a threshold above that can never fire. Taking it from the environment after
    the episode would give the *final* queue, not the peak; taking it from the per-epoch records
    would give a similar but not identical quantity. Recording it at the comparison itself is the
    only version that is exactly the number the rule uses.
    """

    def __init__(self, parameters: HeuristicParameters) -> None:
        super().__init__(parameters)
        self.peak_queue = 0.0

    def act(self, observation: np.ndarray, env: MetroEnv, mask: np.ndarray) -> int:
        raw = env.raw_observation()
        self.peak_queue = max(self.peak_queue, float(raw[self._queue_index].sum()))
        return super().act(observation, env, mask)


def _score_point(payload: tuple[float, float, tuple[int, ...], tuple[str, ...], bool]) -> dict:
    """Evaluate one grid point. Runs in a worker process; must be importable at module level.

    Each grid point builds its own environments from frozen artefacts and its own seeded streams,
    so evaluating points concurrently cannot change any score -- only the wall clock.
    """
    from src.common.compute import single_threaded

    single_threaded()

    queue_threshold, forecast_threshold, tuning_seeds, scenarios, trained = payload

    from src.evaluation.policies import run_episode
    from src.evaluation.runner import Cell, build_env

    parameters = HeuristicParameters(queue_threshold, forecast_threshold)
    rewards: list[float] = []
    peak_queue = 0.0
    for scenario in scenarios:
        cell = Cell(method="heuristic_threshold", scenario=scenario, eval_seed=0)
        env = build_env(cell, trained_forecaster=trained)
        for seed in tuning_seeds:
            policy = _QueueRecordingPolicy(parameters)
            result = run_episode(env, policy, eval_seed=seed)
            rewards.append(float(result.summary["total_reward"]))
            peak_queue = max(peak_queue, policy.peak_queue)

    return {
        "queue_threshold": queue_threshold,
        "forecast_threshold": forecast_threshold,
        "mean_total_reward": float(np.mean(rewards)),
        "peak_observed_queue": peak_queue,
    }


def _evaluate_grid(
    queue_values: tuple[float, ...],
    forecast_values: tuple[float, ...],
    tuning_seeds: tuple[int, ...],
    scenarios: tuple[str, ...],
    trained_forecaster: bool,
    workers: int | None,
) -> list[dict]:
    """Score every (queue, forecast) pair, in parallel where the machine allows."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    payloads = [(q, f, tuning_seeds, scenarios, trained_forecaster)
                for q in queue_values for f in forecast_values]
    workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)

    if workers <= 1 or len(payloads) == 1:
        return [_score_point(p) for p in payloads]

    try:
        with ProcessPoolExecutor(max_workers=min(workers, len(payloads))) as pool:
            return list(pool.map(_score_point, payloads))
    except Exception as error:  # noqa: BLE001 - Windows pools break; the result must not depend on it
        logger.warning("grid pool failed (%r); scoring sequentially", error)
        return [_score_point(p) for p in payloads]


def tune_heuristic(
    *,
    tuning_seeds: tuple[int, ...] = tuple(range(2001, 2007)),
    scenarios: tuple[str, ...] = ("normal", "surge"),
    trained_forecaster: bool = True,
    workers: int | None = None,
) -> HeuristicParameters:
    """Grid-search the thresholds, selecting on mean reward over tuning days.

    The grid is **extended upward until the selected queue threshold is strictly interior**
    (amendment A-10). An optimum on the edge of a search region is evidence the true optimum lies
    outside it, and an under-tuned heuristic would be exactly the strawman baseline protocol
    section 7.1 forbids -- biased in favour of the proposed method.

    Args:
        tuning_seeds: **Disjoint from the frozen evaluation pool (1-30) and training pool
            (101-110).** Protocol section 7.1 requires baseline tuning to use `train`/`val` only;
            `tests/test_leakage.py` asserts no tuning routine reads a `test`-split artefact.
        scenarios: Tuned across both an undisrupted and a surge day so the thresholds are not
            specialised to the scenario the primary endpoint is measured under.
        trained_forecaster: Use the M4 model, so the heuristic sees the same mu as `ppo_point`.
        workers: Parallelism for the grid. Scheduling only; scores are unaffected.

    Returns:
        The selected parameters, with every trial and the extension history in their provenance.
    """
    queue_values = list(QUEUE_THRESHOLDS)
    trials: list[dict[str, Any]] = []
    extensions: list[dict[str, Any]] = []
    scored: set[float] = set()

    for round_index in range(MAX_EXTENSION_ROUNDS + 1):
        pending = tuple(q for q in queue_values if q not in scored)
        if pending:
            logger.info("heuristic grid round %d: scoring %d new queue threshold(s) %s",
                        round_index, len(pending), list(pending))
            trials.extend(_evaluate_grid(pending, FORECAST_THRESHOLDS, tuning_seeds,
                                         scenarios, trained_forecaster, workers))
            scored.update(pending)

        best_trial = max(trials, key=lambda t: t["mean_total_reward"])
        selected_queue = float(best_trial["queue_threshold"])
        largest = max(queue_values)
        interior = selected_queue < largest

        peak_queue = max(float(t.get("peak_observed_queue", 0.0)) for t in trials)
        next_value = largest * 2 if largest > 0 else 50.0
        # A threshold above the largest queue ever observed can never fire, so every larger value
        # produces an identical policy. At that point the boundary is not a real boundary.
        inert = next_value > peak_queue

        extensions.append({
            "round": round_index,
            "grid_max": largest,
            "selected_queue_threshold": selected_queue,
            "selected_is_interior": interior,
            "peak_observed_queue": peak_queue,
            "next_threshold_would_be": next_value,
            "next_threshold_can_never_fire": inert,
        })

        if interior:
            logger.info("selection %.0f is interior to a grid topping out at %.0f; stopping",
                        selected_queue, largest)
            break
        if inert:
            logger.info(
                "selection %.0f is at the grid maximum, but the next value %.0f exceeds the "
                "largest observed queue %.0f, so every larger threshold is behaviourally "
                "identical; stopping", selected_queue, next_value, peak_queue)
            break
        if round_index == MAX_EXTENSION_ROUNDS:
            logger.warning("extension hit its round cap with the selection still at the boundary")
            break

        logger.info("selection %.0f is at the grid maximum; extending to %.0f",
                    selected_queue, next_value)
        queue_values.append(next_value)

    best_trial = max(trials, key=lambda t: t["mean_total_reward"])
    score = float(best_trial["mean_total_reward"])
    parameters = HeuristicParameters(float(best_trial["queue_threshold"]),
                                     float(best_trial["forecast_threshold"]))

    final = extensions[-1]
    forecast_values = list(FORECAST_THRESHOLDS)
    parameters.provenance = {
        "selection_criterion": "mean total reward over tuning episodes",
        "selected_mean_total_reward": score,
        "tuning_seeds": list(tuning_seeds),
        "tuning_scenarios": list(scenarios),
        "seed_pool_note": (
            "seeds 2001+ are disjoint from the frozen evaluation pool (1-30) and training pool "
            "(101-110), so no threshold is tuned on a day the comparison is later scored on"
        ),
        "grid": {"queue": queue_values, "forecast": forecast_values},
        "starting_grid": {"queue": list(QUEUE_THRESHOLDS), "forecast": forecast_values},
        "amendment": "A-10: extend the queue grid until the selection is strictly interior",
        "extension_history": extensions,
        "selection_is_interior": {
            "queue": bool(final["selected_is_interior"]),
            "forecast": bool(parameters.forecast_threshold < max(forecast_values)),
            "note": (
                "a selection at a grid boundary means the optimum probably lies outside the grid, "
                "which would make this baseline weaker than it should be and bias H4-H6 in favour "
                "of the proposed method"
            ),
        },
        "stopping_rule": (
            "extend by doubling until the selected queue threshold is strictly interior, or until "
            "the next threshold exceeds the largest observed queue and therefore can never fire"
        ),
        "trials": trials,
        "information_available": "forecast mean mu and observed queue; never sigma",
        "tuned_utc": datetime.now(timezone.utc).isoformat(),
    }
    logger.info("selected heuristic thresholds: queue %.0f, forecast %.0f (reward %.1f, "
                "interior=%s)", parameters.queue_threshold, parameters.forecast_threshold,
                score, final["selected_is_interior"])
    return parameters
