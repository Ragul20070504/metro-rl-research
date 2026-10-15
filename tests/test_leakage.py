"""Temporal-leakage tests (protocol section 9.2).

The central requirement of the whole project: a forecast issued at time ``t`` may use nothing
from after ``t``, and the RL agent may never observe realised future demand. These are the
tests that make that requirement checkable rather than merely asserted in a document.

The strongest test here is empirical rather than structural: inject a huge spike *after* the
issue time and assert that not one feature value moves. A structural review of the code can
miss an off-by-one in a shift; this cannot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features import FeatureSpec, build_supervised_frame
from src.data.splits import SplitBoundaries, make_chronological_split


def synthetic_demand(n_days: int = 40, stations: tuple[str, ...] = ("A", "B")) -> pd.DataFrame:
    """A deterministic hourly demand series with daily and weekly structure."""
    timestamps = pd.date_range("2024-01-01", periods=n_days * 24, freq="h")
    rows = []
    for index, station in enumerate(stations):
        hour = timestamps.hour.to_numpy()
        dow = timestamps.dayofweek.to_numpy()
        base = 100 + 40 * np.sin(2 * np.pi * hour / 24) + 10 * index - 15 * (dow >= 5)
        rows.append(pd.DataFrame({
            "station_complex_id": station,
            "transit_timestamp": timestamps,
            "ridership": np.clip(base, 0, None),
        }))
    return pd.concat(rows, ignore_index=True)


STATION_ORDER = {"A": 0, "B": 1}
SPEC = FeatureSpec(horizons_h=(1, 2, 3))


class TestFeatureLeakage:
    """Features must be a pure function of data at or before the issue time."""

    def test_future_spike_does_not_move_any_feature(self) -> None:
        """Inject demand *after* the issue time; every feature must be bit-identical.

        This is the empirical leakage test. If any feature secretly reads forward, the huge
        injected value will move it and the comparison fails.
        """
        demand = synthetic_demand()
        baseline = build_supervised_frame(demand, SPEC, STATION_ORDER)

        # Pick an issue time in the middle, and corrupt everything strictly after it.
        issue_time = pd.Timestamp("2024-01-20 12:00")
        corrupted = demand.copy()
        future = corrupted["transit_timestamp"] > issue_time
        corrupted.loc[future, "ridership"] = 1_000_000.0

        after = build_supervised_frame(corrupted, SPEC, STATION_ORDER)

        left = baseline[baseline["issue_time"] == issue_time].sort_values(
            ["station_complex_id", "horizon_h"]).reset_index(drop=True)
        right = after[after["issue_time"] == issue_time].sort_values(
            ["station_complex_id", "horizon_h"]).reset_index(drop=True)

        assert not left.empty, "test setup produced no rows at the chosen issue time"
        for column in SPEC.feature_names:
            np.testing.assert_array_equal(
                left[column].to_numpy(), right[column].to_numpy(),
                err_msg=f"feature {column!r} changed when only post-issue-time data was altered",
            )

    def test_target_does_move_when_the_future_changes(self) -> None:
        """Control for the test above: the *target* must react, or the test proves nothing."""
        demand = synthetic_demand()
        baseline = build_supervised_frame(demand, SPEC, STATION_ORDER)

        issue_time = pd.Timestamp("2024-01-20 12:00")
        corrupted = demand.copy()
        corrupted.loc[corrupted["transit_timestamp"] > issue_time, "ridership"] = 1_000_000.0
        after = build_supervised_frame(corrupted, SPEC, STATION_ORDER)

        left = baseline[baseline["issue_time"] == issue_time].sort_values(
            ["station_complex_id", "horizon_h"])["target"].to_numpy()
        right = after[after["issue_time"] == issue_time].sort_values(
            ["station_complex_id", "horizon_h"])["target"].to_numpy()
        assert not np.array_equal(left, right), (
            "targets did not change when future demand was altered -- the leakage test above "
            "would pass vacuously"
        )

    def test_every_feature_timestamp_is_at_or_before_issue_time(self) -> None:
        """Structural check complementing the empirical one.

        Reconstructs each lag feature's source timestamp from its name and asserts it never
        exceeds the issue time.
        """
        for lag in SPEC.recent_lags:
            # y_tminus_k is observed at issue_time - k hours.
            assert lag >= 0, f"recent lag {lag} would read after the issue time"
        for lag in SPEC.seasonal_lags:
            # y_seasonal_k is observed at target_time - k = issue_time + h - k.
            for horizon in SPEC.horizons_h:
                offset_from_issue = horizon - lag
                assert offset_from_issue <= 0, (
                    f"seasonal lag {lag} at horizon {horizon} reads {offset_from_issue}h "
                    "after the issue time"
                )

    def test_rolling_windows_end_at_the_issue_time(self) -> None:
        """A rolling mean over a window ending at t must equal the manual computation."""
        demand = synthetic_demand()
        frame = build_supervised_frame(demand, SPEC, STATION_ORDER)

        issue_time = pd.Timestamp("2024-01-20 12:00")
        station = "A"
        series = demand[demand["station_complex_id"] == station].set_index("transit_timestamp")["ridership"]

        row = frame[(frame["issue_time"] == issue_time)
                    & (frame["station_complex_id"] == station)].iloc[0]
        for window in SPEC.rolling_windows:
            expected = series.loc[issue_time - pd.Timedelta(hours=window - 1): issue_time].mean()
            assert row[f"y_roll_mean_{window}"] == pytest.approx(expected), (
                f"y_roll_mean_{window} does not match a window ending at the issue time"
            )


class TestSplitLeakage:
    """Splits must be chronological, ordered, and separated by a real embargo."""

    def test_splits_are_disjoint_and_ordered(self) -> None:
        boundaries = make_chronological_split(
            pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31"),
            train_fraction=0.70, val_fraction=0.15, embargo_days=7,
        )
        boundaries.validate()
        assert boundaries.train_end < boundaries.val_start
        assert boundaries.val_end < boundaries.test_start

    def test_embargo_gap_is_enforced(self) -> None:
        boundaries = make_chronological_split(
            pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31"),
            train_fraction=0.70, val_fraction=0.15, embargo_days=7,
        )
        assert (boundaries.val_start - boundaries.train_end).days - 1 >= 7
        assert (boundaries.test_start - boundaries.val_end).days - 1 >= 7

    def test_embargo_days_are_labelled_and_therefore_dropped(self) -> None:
        """Days inside an embargo must be labelled `embargo`, not silently absorbed."""
        boundaries = make_chronological_split(
            pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31"),
            train_fraction=0.70, val_fraction=0.15, embargo_days=7,
        )
        gap_day = boundaries.train_end + pd.Timedelta(days=1)
        labels = boundaries.label_for(pd.Series([gap_day]))
        assert labels.iloc[0] == "embargo"

    def test_misordered_boundaries_are_rejected(self) -> None:
        bad = SplitBoundaries(
            train_start=pd.Timestamp("2024-01-01"), train_end=pd.Timestamp("2024-06-01"),
            val_start=pd.Timestamp("2024-05-01"), val_end=pd.Timestamp("2024-07-01"),
            test_start=pd.Timestamp("2024-08-01"), test_end=pd.Timestamp("2024-09-01"),
            embargo_days=7,
        )
        with pytest.raises(ValueError, match="not strictly ordered"):
            bad.validate()

    def test_insufficient_embargo_is_rejected(self) -> None:
        bad = SplitBoundaries(
            train_start=pd.Timestamp("2024-01-01"), train_end=pd.Timestamp("2024-06-01"),
            val_start=pd.Timestamp("2024-06-02"), val_end=pd.Timestamp("2024-07-01"),
            test_start=pd.Timestamp("2024-07-02"), test_end=pd.Timestamp("2024-09-01"),
            embargo_days=7,
        )
        with pytest.raises(ValueError, match="embargo gaps too small"):
            bad.validate()

    def test_no_random_splitting_is_possible(self) -> None:
        """The split API accepts only dates, so a shuffled split cannot be expressed."""
        import inspect

        parameters = set(inspect.signature(make_chronological_split).parameters)
        assert "shuffle" not in parameters and "random_state" not in parameters, (
            "the split API must not expose randomisation for a time series"
        )


# ---------------------------------------------------------------------------------------------
# M6 additions: the RL observation must be a pure function of state at or before the decision
# instant (protocol section 9.2 item 2).
# ---------------------------------------------------------------------------------------------

from src.environment.forecast_provider import (  # noqa: E402
    REQUIRED_HISTORY_HOURS,
    HistoryBuffer,
    SeasonalNaiveProvider,
)
from tests.test_environment import make_env  # noqa: E402


class TestHistoryBufferRefusesTheFuture:
    """The buffer is the only path by which realised demand reaches the forecaster, so the
    guarantee is enforced there once rather than in each provider's discipline."""

    @staticmethod
    def _buffer(hours: int = REQUIRED_HISTORY_HOURS + 48) -> HistoryBuffer:
        buffer = HistoryBuffer(3)
        rng = np.random.default_rng(0)
        buffer.set_history(rng.gamma(4.0, 25.0, size=(hours, 3)))
        return buffer

    def test_reading_at_the_issue_hour_is_an_error(self) -> None:
        buffer = self._buffer()
        with pytest.raises(IndexError, match="future information"):
            buffer.value_at(buffer.origin_index, buffer.origin_index)

    def test_reading_after_the_issue_hour_is_an_error(self) -> None:
        buffer = self._buffer()
        with pytest.raises(IndexError, match="future information"):
            buffer.value_at(buffer.origin_index + 5, buffer.origin_index)

    def test_a_window_ending_beyond_realised_hours_is_an_error(self) -> None:
        buffer = self._buffer()
        with pytest.raises(IndexError, match="beyond"):
            buffer.window_ending_at(buffer.n_hours + 1, 24)

    def test_a_window_ends_strictly_before_the_issue_hour(self) -> None:
        buffer = self._buffer()
        issue = buffer.origin_index
        window = buffer.window_ending_at(issue, 10)
        assert window.shape == (10, 3)
        assert np.array_equal(window[-1], buffer.value_at(issue - 1, issue))

    def test_short_history_is_refused(self) -> None:
        buffer = HistoryBuffer(3)
        with pytest.raises(ValueError, match="reaches back"):
            buffer.set_history(np.zeros((10, 3)))

    def test_appended_hours_become_readable_only_after_the_issue_moves_on(self) -> None:
        buffer = self._buffer()
        issue = buffer.n_hours
        buffer.append_hour(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(IndexError):
            buffer.value_at(issue, issue)
        assert np.array_equal(buffer.value_at(issue, issue + 1), np.array([1.0, 2.0, 3.0]))


class TestForecastUsesOnlyThePast:
    def test_changing_a_future_hour_does_not_move_the_forecast(self) -> None:
        """The empirical test. A structural review can miss an off-by-one; this cannot."""
        rng = np.random.default_rng(1)
        history = rng.gamma(4.0, 25.0, size=(4 * REQUIRED_HISTORY_HOURS + 48, 3))

        clean = HistoryBuffer(3)
        clean.set_history(history.copy())
        provider_a = SeasonalNaiveProvider(3)
        provider_a.reset(clean, day_start_hour=5, is_weekend=False)
        issue = clean.origin_index - 24
        before = provider_a.forecast(issue)

        spiked = history.copy()
        spiked[issue:] *= 1000.0          # everything from the issue hour onwards
        buffer_b = HistoryBuffer(3)
        buffer_b.set_history(spiked)
        provider_b = SeasonalNaiveProvider(3)
        provider_b.reset(buffer_b, day_start_hour=5, is_weekend=False)
        after = provider_b.forecast(issue)

        assert np.array_equal(before.mu, after.mu)
        assert np.array_equal(before.sigma, after.sigma)

    def test_changing_a_past_hour_does_move_the_forecast(self) -> None:
        """The positive control. Without it, a forecaster that ignores its input would pass."""
        rng = np.random.default_rng(2)
        history = rng.gamma(4.0, 25.0, size=(4 * REQUIRED_HISTORY_HOURS + 48, 3))

        clean = HistoryBuffer(3)
        clean.set_history(history.copy())
        provider_a = SeasonalNaiveProvider(3)
        provider_a.reset(clean, day_start_hour=5, is_weekend=False)
        issue = clean.origin_index - 24
        before = provider_a.forecast(issue)

        perturbed = history.copy()
        perturbed[issue - 168 + 1] *= 50.0     # the seasonal lag the h=1 forecast reads
        buffer_b = HistoryBuffer(3)
        buffer_b.set_history(perturbed)
        provider_b = SeasonalNaiveProvider(3)
        provider_b.reset(buffer_b, day_start_hour=5, is_weekend=False)
        after = provider_b.forecast(issue)

        assert not np.array_equal(before.mu, after.mu)


class TestObservationHasNoFutureInformation:
    """Protocol section 9.2 item 2, asserted on the assembled observation vector."""

    def test_observation_is_unchanged_when_future_demand_is_altered(self) -> None:
        """Inject a large change into every *later* hour of the day and require the observation
        at the current epoch to be bit-identical."""
        env_a = make_env(eval_seed=21)
        env_a.reset()
        for _ in range(40):
            env_a.step(1)
        before = env_a.raw_observation(refresh=True)

        env_b = make_env(eval_seed=21)
        env_b.reset()
        for _ in range(40):
            env_b.step(1)
        # The simulator reads its hourly intensity for the *current* hour on each step, so
        # raising every later hour changes only the future.
        current_hour = int(
            env_b.sim_config.day_start_hour + env_b.simulator.time_s // 3600
        ) % 24
        env_b.simulator._hourly_intensity[current_hour + 1 :] *= 100.0
        after = env_b.raw_observation(refresh=True)

        assert np.array_equal(before, after)

    def test_observation_does_respond_to_the_realised_past(self) -> None:
        """Positive control: a change that has already happened must be visible."""
        env = make_env(eval_seed=22)
        env.reset()
        for _ in range(40):
            env.step(1)
        before = env.raw_observation(refresh=True)
        for row in env.simulator.queues:
            row[0].add(5000.0, env.simulator.time_s)
        after = env.raw_observation(refresh=True)
        assert not np.array_equal(before, after)

    def test_no_observation_channel_is_labelled_as_future_information(self) -> None:
        """Every block must declare when it becomes available, and none may say 'future'."""
        spec = make_env().obs_spec
        for block in spec.blocks:
            assert block.time_availability, block.name
            assert "future" not in block.time_availability.lower(), block.name

    def test_the_availability_signal_does_not_predict_the_realised_failure(self) -> None:
        """The signal is a noisy readout of a latent health state that *causes* the hazard
        (assumption A14, decision D-04). Causal, not anticipatory: it is drawn before any
        failure occurs and never updated when one happens."""
        env = make_env(scenario="disruption", eval_seed=23, failure_rate_per_km=4e-5)
        env.reset()
        signal_at_start = env._availability.copy()
        while True:
            _, _, terminated, truncated, _ = env.step(1)
            assert np.array_equal(env._availability, signal_at_start)
            if terminated or truncated:
                break

    def test_disruption_flag_is_zero_before_any_failure_occurs(self) -> None:
        env = make_env(scenario="disruption", eval_seed=24)
        env.reset()
        index = env.obs_spec.slice_of("disruption_flags")
        flags = env.raw_observation(refresh=True)[index]
        assert flags[0] == 0.0 and flags[1] == 0.0
        assert env.simulator._totals["failures"] == 0


class TestNormalisationConstantsComeFromTrainOnly:
    """Protocol section 9.2 item 5."""

    def test_reward_scales_are_fitted_off_the_evaluation_seeds(self) -> None:
        from src.common.paths import REPO_ROOT
        import json

        path = REPO_ROOT / "configs" / "reward_scales.json"
        if not path.exists():
            pytest.skip("reward scales not fitted in this checkout")
        provenance = json.loads(path.read_text(encoding="utf-8"))["provenance"]
        assert "1001" in provenance["seeds"]
        # Protocol section 8.1 freezes evaluation seeds at 1..30 and training seeds at 101..110.
        assert "disjoint" in provenance["seeds"]
