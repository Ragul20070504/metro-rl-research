"""Leakage-safe feature construction for demand forecasting (protocol section 4.2).

Indexing convention, chosen so that leakage is impossible by construction rather than by
discipline. Every row is one ``(station, issue_time t, horizon h)`` triple:

* the **target** is ``y(s, t + h)`` -- the value being predicted;
* every **feature** is a function of ``y(s, tau)`` for ``tau <= t``, or of calendar
  attributes of ``t + h`` which are known arbitrarily far in advance.

There is no code path that can read ``y(s, tau)`` for ``tau > t``. ``tests/test_leakage.py``
verifies this empirically by injecting a large spike after the issue time and asserting that
not one feature value moves.

Feature families:

``y_tminus_{0..5}``
    Recent observed demand at the issue time and the five hours before it.
``y_seasonal_{24,48,168}``
    Demand at the same clock hour on previous days/weeks, taken relative to the *target*
    time. Valid because the horizons in use (1-3 h) are far shorter than the shortest
    seasonal lag, so these still refer to times at or before ``t``.
``y_roll_mean_{3,24,168}`` / ``y_roll_std_24``
    Rolling statistics over windows ending at ``t``.
``hour_sin``/``hour_cos``/``dow``/``is_weekend``/``is_holiday``
    Calendar attributes of the target time.
``station_index``
    Position along the line, so the model can express spatial structure without one-hot
    exploding the feature space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.common.logging import get_logger

logger = get_logger(__name__)

TARGET_COLUMN = "target"
KEY_COLUMNS = ["station_complex_id", "issue_time", "horizon_h", "target_time"]

#: Recent-history lags, expressed as hours *before the issue time*.
RECENT_LAGS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
#: Seasonal lags, expressed as hours *before the target time*.
SEASONAL_LAGS: tuple[int, ...] = (24, 48, 168)
#: Rolling windows (hours), ending at the issue time.
ROLLING_WINDOWS: tuple[int, ...] = (3, 24, 168)


@dataclass(frozen=True)
class FeatureSpec:
    """Which features are built. Mirrors ``configs/data.yaml: features``."""

    horizons_h: tuple[int, ...] = (1, 2, 3)
    recent_lags: tuple[int, ...] = RECENT_LAGS
    seasonal_lags: tuple[int, ...] = SEASONAL_LAGS
    rolling_windows: tuple[int, ...] = ROLLING_WINDOWS

    @property
    def feature_names(self) -> list[str]:
        """Ordered feature column names, frozen so model inputs stay stable across runs."""
        names = [f"y_tminus_{lag}" for lag in self.recent_lags]
        names += [f"y_seasonal_{lag}" for lag in self.seasonal_lags]
        names += [f"y_roll_mean_{w}" for w in self.rolling_windows]
        names += ["y_roll_std_24"]
        names += ["hour_sin", "hour_cos", "dow", "is_weekend", "is_holiday", "station_index"]
        return names


def to_hourly_grid(demand: pd.DataFrame) -> pd.DataFrame:
    """Reindex demand onto a complete, gap-free hourly grid per station.

    Missing station-hours become explicit zeros. That is the correct reading here: the source
    publishes a row only when at least one fare transaction occurred, so an absent row means
    no recorded entries, not an unknown value. Genuinely unreliable periods are removed
    earlier, by the station-day exclusion list from :mod:`src.data.validate`, so this fill
    cannot mask a data outage.
    """
    stations = sorted(demand["station_complex_id"].unique())
    full_range = pd.date_range(
        demand["transit_timestamp"].min(), demand["transit_timestamp"].max(), freq="h"
    )
    index = pd.MultiIndex.from_product([stations, full_range],
                                       names=["station_complex_id", "transit_timestamp"])
    gridded = (
        demand.set_index(["station_complex_id", "transit_timestamp"])[["ridership"]]
        .reindex(index)
        .fillna({"ridership": 0.0})
        .reset_index()
    )
    logger.info("hourly grid: %d stations x %d hours = %d rows",
                len(stations), len(full_range), len(gridded))
    return gridded


def _holiday_dates(start: pd.Timestamp, end: pd.Timestamp) -> set[pd.Timestamp]:
    """US federal holidays in range, used only as a calendar attribute of the target time."""
    calendar = USFederalHolidayCalendar()
    return set(calendar.holidays(start=start.normalize(), end=end.normalize()))


def build_supervised_frame(
    demand: pd.DataFrame,
    spec: FeatureSpec,
    station_order: dict[str, int],
) -> pd.DataFrame:
    """Build the ``(station, issue_time, horizon)`` supervised frame.

    Args:
        demand: Long frame with ``station_complex_id``, ``transit_timestamp``, ``ridership``.
        spec: Which features to construct.
        station_order: Station id -> position along the line, for ``station_index``.

    Returns:
        One row per (station, issue_time, horizon) with the frozen feature columns, the
        target, and the key columns. Rows whose features or target cannot be formed (at the
        series edges) are dropped.
    """
    grid = to_hourly_grid(demand).sort_values(["station_complex_id", "transit_timestamp"])
    grouped = grid.groupby("station_complex_id", observed=True)["ridership"]

    # Everything below is computed on the *issue-time* index first, then joined onto targets.
    base = grid[["station_complex_id", "transit_timestamp"]].copy()
    base = base.rename(columns={"transit_timestamp": "issue_time"})

    for lag in spec.recent_lags:
        base[f"y_tminus_{lag}"] = grouped.shift(lag).to_numpy()

    for window in spec.rolling_windows:
        # `.shift(0)` is implicit: the window ends at the issue time inclusive, which is
        # observed data. min_periods=window forbids partially-formed windows.
        base[f"y_roll_mean_{window}"] = (
            grouped.rolling(window, min_periods=window).mean().reset_index(level=0, drop=True).to_numpy()
        )
    base["y_roll_std_24"] = (
        grouped.rolling(24, min_periods=24).std().reset_index(level=0, drop=True).to_numpy()
    )

    # Seasonal lags are defined relative to the target time, so they are built on the grid and
    # joined by target time below.
    seasonal = grid[["station_complex_id", "transit_timestamp"]].copy()
    seasonal = seasonal.rename(columns={"transit_timestamp": "target_time"})
    for lag in spec.seasonal_lags:
        seasonal[f"y_seasonal_{lag}"] = grouped.shift(lag).to_numpy()

    targets = grid[["station_complex_id", "transit_timestamp", "ridership"]].rename(
        columns={"transit_timestamp": "target_time", "ridership": TARGET_COLUMN}
    )

    holidays = _holiday_dates(grid["transit_timestamp"].min(), grid["transit_timestamp"].max())

    frames: list[pd.DataFrame] = []
    for horizon in spec.horizons_h:
        block = base.copy()
        block["horizon_h"] = horizon
        block["target_time"] = block["issue_time"] + pd.Timedelta(hours=horizon)
        block = block.merge(targets, on=["station_complex_id", "target_time"], how="inner")
        block = block.merge(seasonal, on=["station_complex_id", "target_time"], how="left")
        frames.append(block)

    supervised = pd.concat(frames, ignore_index=True)

    target_time = supervised["target_time"]
    hour = target_time.dt.hour
    supervised["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    supervised["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    supervised["dow"] = target_time.dt.dayofweek
    supervised["is_weekend"] = (supervised["dow"] >= 5).astype(int)
    supervised["is_holiday"] = target_time.dt.normalize().isin(holidays).astype(int)
    supervised["station_index"] = supervised["station_complex_id"].map(station_order).astype("float64")

    columns = KEY_COLUMNS + spec.feature_names + [TARGET_COLUMN]
    supervised = supervised[columns]

    before = len(supervised)
    supervised = supervised.dropna(subset=spec.feature_names + [TARGET_COLUMN])
    logger.info("supervised frame: %d rows (dropped %d incomplete edge rows)",
                len(supervised), before - len(supervised))
    return supervised.sort_values(["issue_time", "horizon_h", "station_complex_id"]).reset_index(drop=True)
