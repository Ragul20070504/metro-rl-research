"""Tests for the M3 pipeline: regime screening, gridding, GTFS parsing, quality checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features import to_hourly_grid
from src.data.gtfs import gtfs_time_to_seconds
from src.data.splits import (
    daily_line_demand,
    make_chronological_split,
    population_stability_index,
    screen_demand_regime,
)
from src.data.validate import (
    check_duplicates,
    check_non_negative,
    check_required_fields,
    check_station_ids,
    check_timestamps,
)

SCREEN_KWARGS = dict(
    short_window_days=28, yoy_lag_days=365,
    max_abs_yoy_growth=0.10, min_stable_run_days=200,
    fallback_start="2024-01-01",
)


def _daily_series(values: np.ndarray, start: str = "2020-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=index)


class TestRegimeScreen:
    """The screen must find a genuinely stable run, not the first tolerance crossing."""

    def test_detects_stability_after_a_shock_and_recovery(self) -> None:
        """Flat, then a crash, then recovery, then flat again.

        The screen must land in the final flat regime, not during the recovery ramp where
        year-on-year growth is large.
        """
        flat_before = np.full(400, 1000.0)
        crash = np.full(200, 300.0)
        recovery = np.linspace(300, 1000, 400)
        flat_after = np.full(800, 1000.0)
        daily = _daily_series(np.concatenate([flat_before, crash, recovery, flat_after]))

        result = screen_demand_regime(daily, **SCREEN_KWARGS)

        assert not result.used_fallback
        assert result.max_abs_yoy_in_run <= 0.10
        assert result.stable_run_days >= 200
        # The stable run must begin after the recovery ramp has finished.
        ramp_end = daily.index[0] + pd.Timedelta(days=400 + 200 + 400)
        assert result.start_date >= ramp_end - pd.Timedelta(days=30), (
            f"screen chose {result.start_date.date()}, which is inside the recovery ramp"
        )

    def test_falls_back_and_says_so_when_nothing_is_stable(self) -> None:
        """Perpetual growth has no stable regime; the fallback must be reported, not hidden."""
        daily = _daily_series(np.exp(np.linspace(0, 3, 1200)) * 100)
        result = screen_demand_regime(daily, **SCREEN_KWARGS)
        assert result.used_fallback
        assert result.start_date == pd.Timestamp("2024-01-01")
        assert "finding" in result.detail.lower()

    def test_transient_dip_through_tolerance_does_not_qualify(self) -> None:
        """A brief excursion into tolerance must not be mistaken for a stable regime."""
        # Growth that only briefly flattens, far shorter than min_stable_run_days.
        segment = np.concatenate([np.linspace(100, 400, 300), np.full(50, 400.0), np.linspace(400, 1600, 300)])
        daily = _daily_series(np.concatenate([np.full(400, 100.0), segment]))
        result = screen_demand_regime(daily, **SCREEN_KWARGS)
        if not result.used_fallback:
            assert result.stable_run_days >= SCREEN_KWARGS["min_stable_run_days"]

    def test_daily_aggregation_sums_across_stations(self) -> None:
        demand = pd.DataFrame({
            "transit_timestamp": pd.to_datetime(
                ["2024-01-01 05:00", "2024-01-01 06:00", "2024-01-02 05:00"]
            ),
            "station_complex_id": ["A", "B", "A"],
            "ridership": [10.0, 5.0, 7.0],
        })
        daily = daily_line_demand(demand)
        assert daily.loc[pd.Timestamp("2024-01-01")] == 15.0
        assert daily.loc[pd.Timestamp("2024-01-02")] == 7.0


class TestChronologicalSplit:
    def test_fractions_are_respected_approximately(self) -> None:
        boundaries = make_chronological_split(
            pd.Timestamp("2024-01-01"), pd.Timestamp("2026-12-31"),
            train_fraction=0.70, val_fraction=0.15, embargo_days=7, align_to_weekday=None,
        )
        total = (boundaries.test_end - boundaries.train_start).days + 1
        train_days = (boundaries.train_end - boundaries.train_start).days + 1
        assert 0.63 < train_days / total < 0.75

    def test_boundaries_snap_to_the_requested_weekday(self) -> None:
        boundaries = make_chronological_split(
            pd.Timestamp("2024-01-01"), pd.Timestamp("2026-12-31"),
            train_fraction=0.70, val_fraction=0.15, embargo_days=7, align_to_weekday=0,
        )
        assert boundaries.val_start.dayofweek == 0
        assert boundaries.test_start.dayofweek == 0

    def test_window_too_short_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            make_chronological_split(
                pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-10"),
                train_fraction=0.70, val_fraction=0.15, embargo_days=7,
            )

    def test_invalid_fractions_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid split fractions"):
            make_chronological_split(
                pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31"),
                train_fraction=0.9, val_fraction=0.2, embargo_days=7,
            )


class TestPSI:
    def test_identical_samples_give_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        sample = rng.normal(size=5000)
        assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-9)

    def test_shifted_samples_give_a_large_value(self) -> None:
        rng = np.random.default_rng(0)
        assert population_stability_index(rng.normal(size=5000), rng.normal(loc=3.0, size=5000)) > 0.25


class TestHourlyGrid:
    def test_missing_hours_become_explicit_zeros(self) -> None:
        demand = pd.DataFrame({
            "station_complex_id": ["A", "A"],
            "transit_timestamp": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 03:00"]),
            "ridership": [5.0, 9.0],
        })
        grid = to_hourly_grid(demand)
        assert len(grid) == 4
        assert grid["ridership"].tolist() == [5.0, 0.0, 0.0, 9.0]

    def test_grid_is_complete_for_every_station(self) -> None:
        demand = pd.DataFrame({
            "station_complex_id": ["A", "B"],
            "transit_timestamp": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 02:00"]),
            "ridership": [1.0, 2.0],
        })
        grid = to_hourly_grid(demand)
        assert set(grid["station_complex_id"]) == {"A", "B"}
        assert grid.groupby("station_complex_id").size().unique().tolist() == [3]


class TestGtfsTimeParsing:
    def test_ordinary_times(self) -> None:
        parsed = gtfs_time_to_seconds(pd.Series(["00:00:00", "01:30:00", "23:59:59"]))
        assert parsed.tolist() == [0, 5400, 86399]

    def test_times_past_midnight_are_not_wrapped(self) -> None:
        """GTFS encodes post-midnight service as hour >= 24; wrapping would corrupt runtimes."""
        parsed = gtfs_time_to_seconds(pd.Series(["24:00:00", "25:14:00", "27:05:30"]))
        assert parsed.tolist() == [86400, 90840, 97530]


class TestQualityChecks:
    def _frame(self, **overrides) -> pd.DataFrame:
        frame = pd.DataFrame({
            "transit_timestamp": pd.to_datetime(["2024-01-01 05:00", "2024-01-01 06:00"]),
            "station_complex_id": ["118", "119"],
            "ridership": [10.0, 20.0],
            "transfers": [1.0, 2.0],
            "dataset_id": ["d", "d"],
        })
        for key, value in overrides.items():
            frame[key] = value
        return frame

    def test_required_fields_detects_missing(self) -> None:
        frame = self._frame()
        frame.loc[0, "ridership"] = np.nan
        result = check_required_fields(frame)
        assert not result.passed and result.severity == "blocking"

    def test_duplicate_station_hours_are_blocking(self) -> None:
        frame = pd.concat([self._frame(), self._frame()], ignore_index=True)
        result = check_duplicates(frame)
        assert not result.passed
        assert result.metrics["duplicate_rows"] == 4

    def test_negative_demand_is_blocking(self) -> None:
        frame = self._frame()
        frame.loc[0, "ridership"] = -1.0
        assert not check_non_negative(frame).passed

    def test_unaligned_timestamps_are_detected(self) -> None:
        frame = self._frame()
        frame.loc[0, "transit_timestamp"] = pd.Timestamp("2024-01-01 05:30")
        result = check_timestamps(frame)
        assert not result.passed
        assert result.metrics["not_hour_aligned"] == 1

    def test_station_ids_must_match_the_expected_set(self) -> None:
        frame = self._frame()
        result = check_station_ids(frame, ["118", "119"])
        assert result.passed
        assert not check_station_ids(frame, ["118", "119", "120"]).passed
