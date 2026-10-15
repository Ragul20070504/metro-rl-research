"""Ground-truth passenger demand process for the simulator (M5).

**This module implements manuscript correction MC-06 / decision D-01, and it is the reason the
central experiment is testable at all.**

The manuscript (section 5) specifies passenger arrivals "generated as Poisson processes whose
rate is set by the demand-forecast model plus noise". That design is circular: if the
simulator's ground truth comes from the forecaster, the forecaster is correct by construction,
forecast error is whatever noise term we chose, and the predictive dispersion sigma is an
artefact of our own parameter rather than a property of a real forecasting problem. H1 would
then measure our noise setting, not uncertainty-awareness.

Here the ground truth is instead fitted to **observed historical ridership on training days
only**, and the forecaster never sees the sampled path -- it receives only observable history.
Forecast error therefore arises from the genuine difficulty of the task.
``tests/test_no_circular_demand.py`` asserts that this module holds no reference to any
forecasting artefact and that perturbing the forecaster leaves the realised demand stream
bit-identical under a fixed seed.

Process. A doubly-stochastic (Cox) process, which separates the two sources of variation that
matter operationally:

1. **Day-level intensity.** For each (station, day type, hour) an hourly volume is drawn from a
   negative binomial fitted to the across-day spread in the training split. This captures the
   real day-to-day variation an operator faces, which a plain Poisson badly understates.
2. **Within-hour arrivals.** The drawn hourly volume is spread across simulation steps by a
   fixed intra-hour profile and realised as Poisson counts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.common.paths import PROCESSED_DIR, ensure_dir

logger = get_logger(__name__)

DEMAND_MODEL_FILE = PROCESSED_DIR / "network" / "demand_process.json"

#: Day types the process distinguishes. Weekday/weekend is the dominant split in transit
#: demand and is the only one the data supports reliably at station-hour resolution.
DAY_TYPES = ("weekday", "weekend")


def day_type_of(timestamp: pd.Timestamp) -> str:
    return "weekend" if timestamp.dayofweek >= 5 else "weekday"


@dataclass
class DemandProcess:
    """Fitted ground-truth demand process.

    Attributes:
        station_ids: Stations in network order.
        mean: ``mean[day_type][hour, station]`` -- fitted hourly mean boardings.
        dispersion: ``dispersion[day_type][hour, station]`` -- negative-binomial ``r``.
            ``inf`` means the fitted variance did not exceed the mean, so the cell is Poisson.
        fitted_on: Provenance record (split, date range, day counts).
    """

    station_ids: tuple[str, ...]
    mean: dict[str, np.ndarray]
    dispersion: dict[str, np.ndarray]
    fitted_on: dict[str, Any]

    @property
    def n_stations(self) -> int:
        return len(self.station_ids)

    def hourly_intensity(
        self, day_type: str, rng: np.random.Generator, *, scale: float = 1.0
    ) -> np.ndarray:
        """Draw one day's hourly volumes, shape ``(24, n_stations)``.

        Args:
            day_type: ``"weekday"`` or ``"weekend"``.
            rng: The environment's ``demand`` stream. Passing it explicitly is what keeps the
                realisation identical across methods under common random numbers.
            scale: Multiplicative demand scaling, used by the surge scenario.
        """
        mean = self.mean[day_type] * float(scale)
        dispersion = self.dispersion[day_type]

        drawn = np.empty_like(mean)
        poisson_cells = ~np.isfinite(dispersion)
        # Poisson where the data showed no over-dispersion.
        drawn[poisson_cells] = rng.poisson(np.clip(mean[poisson_cells], 0, None))

        nb_cells = ~poisson_cells
        if nb_cells.any():
            mu = np.clip(mean[nb_cells], 1e-9, None)
            r = np.clip(dispersion[nb_cells], 1e-6, None)
            # NB as a gamma-Poisson mixture: gamma shape r, scale mu/r.
            lam = rng.gamma(shape=r, scale=mu / r)
            drawn[nb_cells] = rng.poisson(lam)
        return np.clip(drawn, 0.0, None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "station_ids": list(self.station_ids),
            "day_types": list(DAY_TYPES),
            "mean": {k: v.tolist() for k, v in self.mean.items()},
            "dispersion": {
                k: [[None if not np.isfinite(x) else float(x) for x in row] for row in v]
                for k, v in self.dispersion.items()
            },
            "fitted_on": self.fitted_on,
        }

    def save(self, path: Path = DEMAND_MODEL_FILE) -> Path:
        ensure_dir(path.parent)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path = DEMAND_MODEL_FILE) -> "DemandProcess":
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: python -m src.simulator.fit_demand"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        dispersion = {
            k: np.array([[np.inf if x is None else float(x) for x in row] for row in v], dtype=float)
            for k, v in payload["dispersion"].items()
        }
        return cls(
            station_ids=tuple(payload["station_ids"]),
            mean={k: np.array(v, dtype=float) for k, v in payload["mean"].items()},
            dispersion=dispersion,
            fitted_on=payload["fitted_on"],
        )


def fit_demand_process(
    demand: pd.DataFrame, station_ids: tuple[str, ...], *, split: str = "train"
) -> DemandProcess:
    """Fit the ground-truth process to observed demand on one split.

    Fitted on the **training split only**. Using validation or test days here would leak
    evaluation-period information into the environment the policy is trained in, which is a
    subtler leak than feeding the agent future demand but just as disqualifying.
    """
    if "split" in demand.columns:
        block = demand[demand["split"] == split]
        if block.empty:
            raise ValueError(f"no rows in split {split!r}")
    else:
        block = demand
        logger.warning("demand frame carries no split column; fitting on all supplied rows")

    working = block.copy()
    working["timestamp"] = pd.to_datetime(working["transit_timestamp"])
    working["hour"] = working["timestamp"].dt.hour
    working["date"] = working["timestamp"].dt.normalize()
    working["day_type"] = np.where(working["timestamp"].dt.dayofweek >= 5, "weekend", "weekday")

    index = {station: position for position, station in enumerate(station_ids)}
    working = working[working["station_complex_id"].isin(index)]
    working["station_index"] = working["station_complex_id"].map(index)

    means: dict[str, np.ndarray] = {}
    dispersions: dict[str, np.ndarray] = {}
    day_counts: dict[str, int] = {}

    for day_type in DAY_TYPES:
        subset = working[working["day_type"] == day_type]
        mean = np.zeros((24, len(station_ids)), dtype=float)
        dispersion = np.full((24, len(station_ids)), np.inf, dtype=float)
        day_counts[day_type] = int(subset["date"].nunique())

        if subset.empty:
            logger.warning("no %s rows in split %s; that cell will generate zero demand", day_type, split)
            means[day_type], dispersions[day_type] = mean, dispersion
            continue

        grouped = subset.groupby(["hour", "station_index"], observed=True)["ridership"]
        cell_mean = grouped.mean()
        cell_var = grouped.var(ddof=1)

        for (hour, station_index), mu in cell_mean.items():
            mean[int(hour), int(station_index)] = float(mu)
            variance = float(cell_var.get((hour, station_index), np.nan))
            # Negative binomial only where the data actually show over-dispersion; otherwise
            # Poisson. Forcing NB on an under-dispersed cell would invent variance.
            if np.isfinite(variance) and variance > mu > 0:
                dispersion[int(hour), int(station_index)] = float(mu ** 2 / (variance - mu))

        means[day_type], dispersions[day_type] = mean, dispersion

    over_dispersed = {
        day_type: float(np.mean(np.isfinite(dispersions[day_type]))) for day_type in DAY_TYPES
    }
    logger.info("demand process fitted on split=%s: %s days, over-dispersed cell fraction %s",
                split, day_counts, {k: round(v, 3) for k, v in over_dispersed.items()})

    return DemandProcess(
        station_ids=station_ids,
        mean=means,
        dispersion=dispersions,
        fitted_on={
            "split": split,
            "days": day_counts,
            "first_date": str(working["date"].min().date()) if not working.empty else None,
            "last_date": str(working["date"].max().date()) if not working.empty else None,
            "over_dispersed_cell_fraction": over_dispersed,
            "note": "Fitted to OBSERVED ridership. Independent of any forecasting artefact "
                    "(manuscript correction MC-06 / decision D-01).",
        },
    )


def intra_hour_weights(steps_per_hour: int, profile: str = "uniform") -> np.ndarray:
    """Weights spreading an hourly volume across simulation steps (assumption A02).

    The source data is hourly, so the within-hour shape is unobserved. ``uniform`` is the
    honest default; ``smooth`` provides the swept alternative and is deliberately mild, since
    inventing sharp within-hour structure would fabricate detail the data cannot support.
    """
    if profile == "uniform":
        return np.full(steps_per_hour, 1.0 / steps_per_hour)
    if profile == "smooth":
        position = (np.arange(steps_per_hour) + 0.5) / steps_per_hour
        weights = 1.0 + 0.20 * np.sin(2 * np.pi * position)
        return weights / weights.sum()
    raise ValueError(f"unknown intra-hour profile {profile!r}")


def build_synthetic_process(
    station_ids: tuple[str, ...], peak_mean: float = 400.0, seed: int = 0
) -> DemandProcess:
    """A deterministic stand-in process for unit tests and toy environments.

    Never used for a reported result; it exists so simulator tests do not require the full
    ingested dataset.
    """
    rng = np.random.default_rng(seed)
    n = len(station_ids)
    hours = np.arange(24)
    shape = np.exp(-0.5 * ((hours - 8) / 2.0) ** 2) + 0.9 * np.exp(-0.5 * ((hours - 18) / 2.2) ** 2)
    shape[:5] = 0.0

    means: dict[str, np.ndarray] = {}
    dispersions: dict[str, np.ndarray] = {}
    for day_type, factor in (("weekday", 1.0), ("weekend", 0.55)):
        station_weight = 0.5 + rng.uniform(0, 1, size=n)
        means[day_type] = peak_mean * factor * np.outer(shape, station_weight)
        dispersions[day_type] = np.full((24, n), 8.0)
    return DemandProcess(
        station_ids=station_ids, mean=means, dispersion=dispersions,
        fitted_on={"split": "synthetic", "note": "tests only; never a reported result"},
    )
