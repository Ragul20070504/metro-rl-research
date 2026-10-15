"""Scenario and disruption generation (M5, protocol section 7.2).

Exactly four scenario families, generated **reproducibly from the evaluation seed** rather
than hard-coded, so every method faces byte-identical conditions in a given
``(scenario, eval_seed)`` cell. The generator takes no method argument, which is what makes
the common-random-numbers guarantee structural rather than a matter of care.

The failure hazard is **synthetic in shape but calibrated in rate**: the per-train-km hazard is
scaled so simulated failures match the published Mean Distance Between Failures for the car
class operating the modelled line. MDBF constrains the aggregate rate only -- the within-day
timing structure (constant hazard per km) remains fabricated, and is recorded as assumption
A13.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.common.paths import RAW_DIR

logger = get_logger(__name__)

SCENARIOS = ("normal", "surge", "disruption", "combined")


@dataclass(frozen=True)
class ScenarioSpec:
    """A fully realised scenario instance for one episode.

    Attributes:
        name: Scenario family.
        demand_scale: Multiplier applied to the ground-truth demand intensity.
        surge_start_s: Start of the surge window, seconds after the operating-day start.
        surge_end_s: End of the surge window.
        forced_failures: ``(time_seconds, n_trains)`` events removing in-service trains.
        day_type: ``"weekday"`` or ``"weekend"``.
    """

    name: str
    demand_scale: float = 1.0
    surge_start_s: float | None = None
    surge_end_s: float | None = None
    forced_failures: tuple[tuple[float, int], ...] = ()
    day_type: str = "weekday"

    def scale_at(self, elapsed_s: float) -> float:
        """Demand multiplier at a point in the operating day."""
        if self.surge_start_s is None or self.surge_end_s is None:
            return 1.0
        return self.demand_scale if self.surge_start_s <= elapsed_s < self.surge_end_s else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "demand_scale": self.demand_scale,
            "surge_start_s": self.surge_start_s,
            "surge_end_s": self.surge_end_s,
            "forced_failures": [list(f) for f in self.forced_failures],
            "day_type": self.day_type,
        }


def estimate_failure_rate_per_km(
    car_class: str = "R143", cars_per_train: int = 8, lookback_months: int = 24
) -> dict[str, float]:
    """Estimate a per-train-km failure hazard from published MDBF.

    MDBF is published per *car*: miles a rail car travels before a mechanical failure delays
    the train. A trainset of ``cars_per_train`` cars therefore fails roughly ``cars_per_train``
    times as often per train-mile, since any car's failure delays the whole train.

    Returns the rate together with the inputs used, so the calibration is auditable and the
    assumption about ``cars_per_train`` is visible rather than buried.
    """
    path = RAW_DIR / "mta_mdbf" / "data.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the M3 download first")

    frame = pd.read_parquet(path)
    frame = frame[frame["car_class"].astype(str).str.strip() == car_class].copy()
    if frame.empty:
        raise ValueError(f"car class {car_class!r} not present in the MDBF dataset")

    for column in ("total_miles", "number_of_failures", "number_of_cars"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["month"] = pd.to_datetime(frame["month"])
    recent = frame.sort_values("month").tail(lookback_months)

    total_miles = float(recent["total_miles"].sum())
    total_failures = float(recent["number_of_failures"].sum())
    if total_miles <= 0:
        raise ValueError("MDBF rows contain no positive mileage")

    car_failures_per_mile = total_failures / total_miles
    train_failures_per_mile = car_failures_per_mile * cars_per_train
    train_failures_per_km = train_failures_per_mile / 1.609344

    result = {
        "car_class": car_class,
        "cars_per_train": float(cars_per_train),
        "months_used": float(len(recent)),
        "first_month": str(recent["month"].min().date()),
        "last_month": str(recent["month"].max().date()),
        "total_car_miles": total_miles,
        "total_failures": total_failures,
        "mdbf_car_miles": total_miles / total_failures if total_failures else float("inf"),
        "train_failures_per_km": train_failures_per_km,
    }
    logger.info(
        "failure calibration from MDBF: %s, %d months, MDBF %.0f car-miles -> "
        "%.3e train failures per km",
        car_class, len(recent), result["mdbf_car_miles"], train_failures_per_km,
    )
    return result


def generate_scenario(
    scenario: str,
    rng_streams: dict[str, np.random.Generator],
    *,
    day_seconds: float,
    fleet_size: int,
    day_type: str = "weekday",
    surge_scale_range: tuple[float, float] = (1.30, 1.50),
    surge_duration_range_s: tuple[float, float] = (5400.0, 10800.0),
    disruption_trains_range: tuple[int, int] = (1, 3),
) -> ScenarioSpec:
    """Realise one scenario instance from the environment's random streams.

    Args:
        scenario: One of :data:`SCENARIOS`.
        rng_streams: Streams from :func:`src.common.seeding.environment_streams`. These depend
            only on ``(scenario, eval_seed)``, never on the method being evaluated.
        day_seconds: Length of the operating day.
        fleet_size: Used to bound how many trains a disruption may remove.
        surge_scale_range: Manuscript section 6 specifies 30-50% above baseline.

    Notes:
        Surge and disruption draws use *separate* streams, and since M6 the scenario name is
        excluded from the stream entropy (decision D-14), so the ``combined`` scenario carries
        the same surge and the same failures as the individual families at that seed, on the
        same base demand. That is what makes the interaction test in ``EXP-ROBUST-003`` a
        genuine factorial contrast rather than a comparison across unrelated realisations.

        Before D-14 this claim was false: the scenario name was part of the entropy, so every
        family had its own demand, surge and failure draws. The comment asserting otherwise had
        been in place since M2 and was only caught when the M6 responsiveness report showed
        ``surge`` scoring better than ``normal``.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")

    surge_rng = rng_streams["surge"]
    failure_rng = rng_streams["failures"]

    demand_scale, surge_start, surge_end = 1.0, None, None
    if scenario in ("surge", "combined"):
        demand_scale = float(surge_rng.uniform(*surge_scale_range))
        duration = float(surge_rng.uniform(*surge_duration_range_s))
        latest_start = max(0.0, day_seconds - duration)
        surge_start = float(surge_rng.uniform(0.0, latest_start))
        surge_end = surge_start + duration

    failures: list[tuple[float, int]] = []
    if scenario in ("disruption", "combined"):
        n_events = int(failure_rng.integers(disruption_trains_range[0], disruption_trains_range[1] + 1))
        # Mid-day losses, per manuscript section 6: sampled from the middle of the operating
        # day rather than the edges, where there is little service to disrupt.
        times = np.sort(failure_rng.uniform(0.25 * day_seconds, 0.75 * day_seconds, size=n_events))
        failures = [(float(t), 1) for t in times]

    return ScenarioSpec(
        name=scenario,
        demand_scale=demand_scale,
        surge_start_s=surge_start,
        surge_end_s=surge_end,
        forced_failures=tuple(failures),
        day_type=day_type,
    )
