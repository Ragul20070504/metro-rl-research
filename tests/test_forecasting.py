"""Tests for forecasting metrics, calibration and models (M4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.calibrate import apply_factor, fit_interval_scale
from src.forecasting.metrics import (
    coverage,
    crps_from_quantiles,
    evaluate_probabilistic,
    mae,
    mape,
    mase,
    mean_interval_width,
    pinball_loss,
    quantile_calibration_error,
    rmse,
    winkler_score,
)
from src.forecasting.models import (
    DEFAULT_QUANTILES,
    Z90,
    SeasonalNaive,
    XGBPointForecaster,
    XGBQuantileForecaster,
    sigma_from_interval,
)


class TestPointMetrics:
    def test_perfect_forecast_scores_zero(self) -> None:
        actual = np.array([1.0, 2.0, 3.0])
        assert mae(actual, actual) == 0.0
        assert rmse(actual, actual) == 0.0

    def test_rmse_penalises_large_errors_more_than_mae(self) -> None:
        actual = np.array([0.0, 0.0, 0.0, 0.0])
        predicted = np.array([0.0, 0.0, 0.0, 4.0])
        assert rmse(actual, predicted) > mae(actual, predicted)

    def test_mase_below_one_means_better_than_naive(self) -> None:
        actual = np.array([10.0, 12.0, 11.0, 13.0])
        model = np.array([10.5, 11.5, 11.2, 12.8])
        naive_errors = np.array([2.0, 2.0, 2.0, 2.0])
        assert mase(actual, model, naive_errors) < 1.0

    def test_mase_is_nan_when_the_naive_scale_is_zero(self) -> None:
        """A degenerate reference must yield NaN, not a division blow-up."""
        assert np.isnan(mase(np.array([1.0]), np.array([1.0]), np.array([0.0])))

    def test_mape_floor_excludes_near_zero_actuals(self) -> None:
        """Overnight zeros must not be allowed to dominate a percentage error."""
        actual = np.array([0.0, 0.0, 100.0])
        predicted = np.array([5.0, 5.0, 110.0])
        assert mape(actual, predicted, floor=1.0) == pytest.approx(10.0)

    def test_non_finite_values_are_dropped_pairwise(self) -> None:
        actual = np.array([1.0, np.nan, 3.0])
        predicted = np.array([1.0, 2.0, 3.0])
        assert mae(actual, predicted) == 0.0


class TestPinballAndIntervals:
    def test_pinball_is_asymmetric_in_the_expected_direction(self) -> None:
        """At q=0.9, under-prediction must be penalised more than over-prediction."""
        actual = np.array([10.0])
        under = pinball_loss(actual, np.array([8.0]), 0.9)
        over = pinball_loss(actual, np.array([12.0]), 0.9)
        assert under > over

    def test_pinball_at_median_is_half_the_absolute_error(self) -> None:
        assert pinball_loss(np.array([10.0]), np.array([8.0]), 0.5) == pytest.approx(1.0)

    def test_coverage_counts_points_inside_the_interval(self) -> None:
        actual = np.array([1.0, 5.0, 9.0])
        assert coverage(actual, np.array([0.0, 0.0, 0.0]), np.array([6.0, 6.0, 6.0])) == pytest.approx(2 / 3)

    def test_interval_width(self) -> None:
        assert mean_interval_width(np.array([0.0, 1.0]), np.array([2.0, 5.0])) == pytest.approx(3.0)

    def test_winkler_penalises_a_miss(self) -> None:
        """A miss must score worse than the same interval containing the point."""
        contained = winkler_score(np.array([5.0]), np.array([0.0]), np.array([10.0]), alpha=0.2)
        missed = winkler_score(np.array([50.0]), np.array([0.0]), np.array([10.0]), alpha=0.2)
        assert missed > contained

    def test_winkler_penalises_needless_width(self) -> None:
        narrow = winkler_score(np.array([5.0]), np.array([4.0]), np.array([6.0]), alpha=0.2)
        wide = winkler_score(np.array([5.0]), np.array([0.0]), np.array([100.0]), alpha=0.2)
        assert wide > narrow

    def test_crps_is_lower_for_a_sharper_correct_forecast(self) -> None:
        actual = np.full(200, 10.0)
        sharp = {q: np.full(200, 10.0 + (q - 0.5)) for q in DEFAULT_QUANTILES}
        diffuse = {q: np.full(200, 10.0 + (q - 0.5) * 40) for q in DEFAULT_QUANTILES}
        assert crps_from_quantiles(actual, sharp) < crps_from_quantiles(actual, diffuse)


class TestCalibrationError:
    def test_well_calibrated_quantiles_score_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        actual = rng.normal(size=20_000)
        from scipy.stats import norm
        predictions = {q: np.full(20_000, norm.ppf(q)) for q in DEFAULT_QUANTILES}
        assert quantile_calibration_error(actual, predictions) < 0.02

    def test_miscalibrated_quantiles_score_high(self) -> None:
        rng = np.random.default_rng(0)
        actual = rng.normal(size=20_000)
        predictions = {q: np.full(20_000, 5.0) for q in DEFAULT_QUANTILES}
        assert quantile_calibration_error(actual, predictions) > 0.3


class TestSigmaDefinition:
    def test_sigma_matches_the_frozen_formula(self) -> None:
        """sigma = (q90 - q10) / (2 * 1.2816); for a standard normal this recovers ~1."""
        q10 = np.array([-1.2816])
        q90 = np.array([1.2816])
        assert sigma_from_interval(q10, q90)[0] == pytest.approx(1.0, rel=1e-6)

    def test_sigma_is_clipped_at_zero_on_crossing(self) -> None:
        """Quantile crossing must not produce a negative dispersion."""
        assert sigma_from_interval(np.array([5.0]), np.array([1.0]))[0] == 0.0

    def test_z90_constant_is_the_protocol_value(self) -> None:
        assert Z90 == pytest.approx(1.2816)


class TestIntervalCalibration:
    # Demand is non-negative, and `apply_factor` clips intervals at zero accordingly. These
    # fixtures are therefore centred well away from zero, matching the domain: a fixture
    # centred at 0 would have its lower bound clipped and could never be narrowed, which is a
    # property of the domain constraint rather than of the calibration search.

    def test_widening_is_selected_when_coverage_is_too_low(self) -> None:
        rng = np.random.default_rng(0)
        actual = rng.normal(loc=50.0, scale=9.0, size=4000)
        median = np.full(4000, 50.0)
        lower, upper = np.full(4000, 49.0), np.full(4000, 51.0)  # far too narrow
        result = fit_interval_scale(actual, median, lower, upper)
        assert result.factor > 1.0
        assert result.improved
        assert abs(result.coverage_after - 0.80) < abs(result.coverage_before - 0.80)

    def test_narrowing_is_selected_when_coverage_is_too_high(self) -> None:
        rng = np.random.default_rng(0)
        actual = rng.normal(loc=50.0, scale=1.0, size=4000)
        median = np.full(4000, 50.0)
        lower, upper = np.full(4000, 20.0), np.full(4000, 80.0)  # far too wide
        result = fit_interval_scale(actual, median, lower, upper)
        assert result.factor < 1.0
        assert result.improved
        assert 0.75 < result.coverage_after < 0.85

    def test_never_returns_a_factor_worse_than_no_correction(self) -> None:
        """Guard against a search-range limitation silently degrading calibration.

        Here the interval straddles zero, so its lower bound is clipped and narrowing cannot
        reduce coverage. The correct behaviour is to fall back to the identity, not to return
        a boundary factor that makes calibration worse and then feed a corrupted sigma into H1.
        """
        rng = np.random.default_rng(0)
        actual = np.abs(rng.normal(scale=1.0, size=2000))
        median = np.zeros(2000)
        lower, upper = np.full(2000, -20.0), np.full(2000, 20.0)
        result = fit_interval_scale(actual, median, lower, upper)
        assert abs(result.coverage_after - 0.80) <= abs(result.coverage_before - 0.80) + 1e-9

    def test_search_boundary_hit_is_flagged_not_silently_accepted(self) -> None:
        """A factor pinned to the range endpoint means the range binds; it must be visible.

        Silently accepting a clamped factor would feed a knowingly miscalibrated sigma into
        the uncertainty channel that H1 depends on.
        """
        rng = np.random.default_rng(0)
        actual = rng.normal(loc=50.0, scale=0.05, size=2000)
        median = np.full(2000, 50.0)
        lower, upper = np.full(2000, 0.0), np.full(2000, 100.0)  # absurdly wide
        result = fit_interval_scale(actual, median, lower, upper, search_min=0.5, search_max=2.0)
        assert result.at_boundary
        assert result.to_dict()["at_search_boundary"] is True

    def test_no_boundary_flag_when_the_optimum_is_interior(self) -> None:
        rng = np.random.default_rng(0)
        actual = rng.normal(loc=50.0, scale=5.0, size=4000)
        median = np.full(4000, 50.0)
        lower, upper = np.full(4000, 40.0), np.full(4000, 60.0)
        result = fit_interval_scale(actual, median, lower, upper)
        assert not result.at_boundary

    def test_identity_is_always_a_reachable_candidate(self) -> None:
        rng = np.random.default_rng(1)
        actual = rng.normal(loc=100.0, scale=10.0, size=3000)
        median = np.full(3000, 100.0)
        lower, upper = np.full(3000, 100 - 12.816), np.full(3000, 100 + 12.816)  # already ~80%
        result = fit_interval_scale(actual, median, lower, upper)
        assert result.factor == pytest.approx(1.0, abs=0.15)

    def test_calibration_never_moves_the_median(self) -> None:
        """The point forecast must be untouched, or calibration would confound the ablation."""
        median = np.array([10.0, 20.0])
        lower, upper = np.array([8.0, 15.0]), np.array([12.0, 25.0])
        lo, hi = apply_factor(median, lower, upper, 2.5)
        assert np.allclose((lo + hi) / 2.0, median)

    def test_calibrated_output_is_clipped_at_zero(self) -> None:
        median = np.array([1.0])
        lo, _ = apply_factor(median, np.array([0.5]), np.array([1.5]), 10.0)
        assert lo[0] >= 0.0


def _toy_frame(n: int = 600) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    features = pd.DataFrame({
        "y_tminus_0": rng.uniform(0, 100, n),
        "y_seasonal_168": rng.uniform(0, 100, n),
        "hour_sin": rng.uniform(-1, 1, n),
        "station_index": rng.integers(0, 24, n).astype(float),
    })
    target = 0.8 * features["y_tminus_0"] + 0.2 * features["y_seasonal_168"] + rng.normal(0, 5, n)
    return features, np.clip(target.to_numpy(), 0, None)


class TestModels:
    def test_seasonal_naive_returns_the_seasonal_lag(self) -> None:
        features, target = _toy_frame()
        model = SeasonalNaive().fit(features, target)
        output = model.predict(features)
        np.testing.assert_array_equal(output.mean, features["y_seasonal_168"].to_numpy())

    def test_seasonal_naive_requires_its_lag_column(self) -> None:
        with pytest.raises(KeyError, match="y_seasonal_168"):
            SeasonalNaive().fit(pd.DataFrame({"other": [1.0]}), np.array([1.0]))

    def test_point_model_learns_and_never_predicts_negative_demand(self) -> None:
        features, target = _toy_frame()
        model = XGBPointForecaster(n_estimators=40, max_depth=3).fit(features, target)
        predictions = model.predict(features).mean
        assert (predictions >= 0).all()
        assert rmse(target, predictions) < rmse(target, np.full_like(target, target.mean()))

    def test_quantile_model_emits_monotone_quantiles_and_a_sigma(self) -> None:
        features, target = _toy_frame()
        model = XGBQuantileForecaster(n_estimators=40, max_depth=3).fit(features, target)
        output = model.predict(features)

        levels = sorted(output.quantiles)
        stacked = np.column_stack([output.quantiles[q] for q in levels])
        assert (np.diff(stacked, axis=1) >= -1e-9).all(), "quantiles are not monotone after repair"
        assert (output.sigma >= 0).all()
        assert output.sigma.shape == output.mean.shape

    def test_quantile_model_mean_is_the_median(self) -> None:
        features, target = _toy_frame()
        model = XGBQuantileForecaster(n_estimators=40, max_depth=3).fit(features, target)
        output = model.predict(features)
        np.testing.assert_allclose(output.mean, output.quantiles[0.50])

    def test_calibration_factor_widens_sigma_without_moving_the_mean(self) -> None:
        features, target = _toy_frame()
        model = XGBQuantileForecaster(n_estimators=40, max_depth=3).fit(features, target)
        before = model.predict(features)
        model.calibration_factor = 2.0
        after = model.predict(features)

        np.testing.assert_allclose(before.mean, after.mean, atol=1e-9)
        assert after.sigma.mean() > before.sigma.mean()

    def test_point_output_carries_zero_sigma(self) -> None:
        """The point arm must receive an informationally empty uncertainty channel."""
        features, target = _toy_frame()
        output = XGBPointForecaster(n_estimators=20, max_depth=3).fit(features, target).predict(features)
        assert output.quantiles == {}
        assert np.all(output.sigma == 0.0)


class TestProbabilisticEvaluation:
    def test_evaluate_probabilistic_populates_the_full_block(self) -> None:
        rng = np.random.default_rng(0)
        actual = rng.normal(loc=50, scale=10, size=3000)
        from scipy.stats import norm
        predictions = {q: np.full(3000, 50 + 10 * norm.ppf(q)) for q in DEFAULT_QUANTILES}
        metrics = evaluate_probabilistic(actual, predictions)

        assert 0.75 < metrics.picp_80 < 0.85
        assert 0.85 < metrics.picp_90 < 0.95
        assert np.isfinite(metrics.crps)
        assert np.isfinite(metrics.winkler_80)
        assert len(metrics.reliability) == len(DEFAULT_QUANTILES)


class TestStoredForecastProvenance:
    """Protocol section 4.6 requires every stored forecast to name the model that produced it.

    Without these columns a forecast file is anonymous: nothing in it distinguishes a calibrated
    quantile model's output from a point model's, or one fitted configuration from another.
    """

    @staticmethod
    def _frames_and_outputs():
        from types import SimpleNamespace

        import pandas as pd

        frame = pd.DataFrame({
            "station_complex_id": ["A", "A", "B"],
            "issue_time": pd.to_datetime(["2026-01-01 08:00"] * 3),
            "horizon_h": [1, 2, 1],
            "target_time": pd.to_datetime(
                ["2026-01-01 09:00", "2026-01-01 10:00", "2026-01-01 09:00"]),
            "target": [10.0, 12.0, 8.0],
        })
        output = SimpleNamespace(
            mean=np.array([10.0, 11.0, 9.0]),
            quantiles={0.10: np.array([5.0, 6.0, 4.0]),
                       0.50: np.array([10.0, 11.0, 9.0]),
                       0.90: np.array([15.0, 16.0, 14.0])},
            sigma=np.array([3.9, 3.9, 3.9]),
        )
        return {"val": frame}, {"val": output}

    def test_the_three_protocol_columns_are_written(self, tmp_path) -> None:
        import pandas as pd

        from src.forecasting.train import write_forecast_parquet

        frames, outputs = self._frames_and_outputs()
        path = write_forecast_parquet(
            frames, outputs, tmp_path / "forecast.parquet",
            model_id="xgb_quantile", model_version="abc123", calibrated=True)

        stored = pd.read_parquet(path)
        assert list(stored["model_id"].unique()) == ["xgb_quantile"]
        assert list(stored["model_version"].unique()) == ["abc123"]
        assert list(stored["calibrated"].unique()) == [True]

    def test_the_provenance_travels_with_every_row(self, tmp_path) -> None:
        """A per-file sidecar would be lost the moment two forecast files were concatenated."""
        import pandas as pd

        from src.forecasting.train import write_forecast_parquet

        frames, outputs = self._frames_and_outputs()
        a = pd.read_parquet(write_forecast_parquet(
            frames, outputs, tmp_path / "q.parquet",
            model_id="xgb_quantile", model_version="v1", calibrated=True))
        b = pd.read_parquet(write_forecast_parquet(
            frames, outputs, tmp_path / "p.parquet",
            model_id="xgb_point", model_version="v1", calibrated=False))

        merged = pd.concat([a, b], ignore_index=True)
        assert set(merged["model_id"]) == {"xgb_quantile", "xgb_point"}
        assert merged[merged["model_id"] == "xgb_point"]["calibrated"].eq(False).all()


class TestCodePathDigest:
    """The digest is what lets a completed run prove which bytes it executed."""

    def test_it_is_stable_across_calls(self) -> None:
        from src.forecasting.train import code_path_digest

        assert code_path_digest() == code_path_digest()

    def test_it_covers_the_files_that_determine_the_result(self) -> None:
        from src.forecasting.train import CODE_PATH

        for required in ("src/forecasting/train.py", "src/forecasting/models.py",
                         "src/data/features.py", "src/data/splits.py",
                         "configs/forecasting.yaml"):
            assert required in CODE_PATH

    def test_it_changes_when_a_covered_file_changes(self, tmp_path, monkeypatch) -> None:
        """A digest that ignored content would pass every check while proving nothing."""
        import src.forecasting.train as train_module

        (tmp_path / "configs").mkdir()
        target = tmp_path / "configs" / "forecasting.yaml"
        target.write_text("random_state: 0\n", encoding="utf-8")

        monkeypatch.setattr(train_module, "CODE_PATH", ("configs/forecasting.yaml",))
        monkeypatch.setattr("src.common.paths.REPO_ROOT", tmp_path)

        before = train_module.code_path_digest()
        target.write_text("random_state: 1\n", encoding="utf-8")
        assert train_module.code_path_digest() != before
