"""The forecasting models the RL arms load must be the models the M4 run evaluated.

`src/forecasting/train.py` evaluates its models and discards them, so the artefacts under
`models/forecasting/` are reconstructed. That reconstruction is the seam where a *different* model
could reach the experiment without anything downstream noticing: the environment would load it,
every arm would consume it, and the only symptom would be results that quietly do not correspond
to the reported forecasting metrics.

These tests hold the reconstruction to the two properties that close that seam: nothing unverified
is ever written, and the verification covers sigma as well as the median.
"""

from __future__ import annotations

import json

import pytest

from scripts import persist_forecast_models as persist


class TestStationIndex:
    """The manifest is a manifest, not the mapping. Reading it flat crashed on a timestamp."""

    def test_it_reads_the_nested_mapping(self, tmp_path, monkeypatch):
        network = tmp_path / "network"
        network.mkdir()
        (network / "station_order.json").write_text(json.dumps({
            "derived_utc": "2026-08-26T16:11:58.027016+00:00",
            "source": "gtfs",
            "n_stations": 2,
            "station_complex_to_index": {"A": 0, "B": 1},
        }), encoding="utf-8")
        monkeypatch.setattr(persist, "PROCESSED_DIR", tmp_path)

        assert persist._station_index() == {"A": 0, "B": 1}

    def test_it_still_accepts_a_flat_mapping(self, tmp_path, monkeypatch):
        network = tmp_path / "network"
        network.mkdir()
        (network / "station_order.json").write_text(
            json.dumps({"A": 0, "B": 1}), encoding="utf-8")
        monkeypatch.setattr(persist, "PROCESSED_DIR", tmp_path)

        assert persist._station_index() == {"A": 0, "B": 1}

    def test_a_missing_manifest_is_empty_not_fatal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(persist, "PROCESSED_DIR", tmp_path)
        assert persist._station_index() == {}

    def test_a_malformed_mapping_stops_rather_than_guesses(self, tmp_path, monkeypatch):
        network = tmp_path / "network"
        network.mkdir()
        (network / "station_order.json").write_text(
            json.dumps({"station_complex_to_index": {"A": "not-an-index"}}), encoding="utf-8")
        monkeypatch.setattr(persist, "PROCESSED_DIR", tmp_path)

        with pytest.raises(SystemExit):
            persist._station_index()


class TestVerificationGate:
    """A model that disagrees with the stored forecasts must not reach the experiment."""

    @staticmethod
    def _setup(tmp_path, monkeypatch, *, p50_shift: float = 0.0, sigma_shift: float = 0.0):
        import numpy as np
        import pandas as pd
        from types import SimpleNamespace

        keys = {
            "station_complex_id": ["A", "B", "C"],
            "issue_time": pd.to_datetime(["2026-01-01 08:00"] * 3),
            "horizon_h": [1, 1, 1],
            "target_time": pd.to_datetime(["2026-01-01 09:00"] * 3),
        }
        forecasts = pd.DataFrame({**keys, "split": "val",
                                  "p50": [10.0, 20.0, 30.0], "sigma": [3.0, 4.0, 5.0]})
        features = pd.DataFrame({**keys, "x": [1.0, 2.0, 3.0]})

        (tmp_path / "forecasts").mkdir()
        forecasts.to_parquet(tmp_path / "forecasts" / "forecast.parquet", index=False)
        features.to_parquet(tmp_path / "supervised_val.parquet", index=False)
        monkeypatch.setattr(persist, "PROCESSED_DIR", tmp_path)

        model = SimpleNamespace(predict=lambda frame: SimpleNamespace(
            mean=np.array([10.0, 20.0, 30.0]) + p50_shift,
            sigma=np.array([3.0, 4.0, 5.0]) + sigma_shift,
        ))
        return model

    def test_an_exact_reproduction_passes(self, tmp_path, monkeypatch):
        model = self._setup(tmp_path, monkeypatch)
        agreement = persist._verify_against_stored_forecasts(model, ["x"])
        assert agreement["max_abs_p50_difference"] == 0.0
        assert agreement["max_abs_sigma_difference"] == 0.0
        assert agreement["n_compared"] == 3

    def test_a_median_disagreement_stops_the_run(self, tmp_path, monkeypatch):
        model = self._setup(tmp_path, monkeypatch, p50_shift=0.5)
        with pytest.raises(SystemExit, match="max_abs_p50_difference"):
            persist._verify_against_stored_forecasts(model, ["x"])

    def test_a_sigma_disagreement_stops_the_run(self, tmp_path, monkeypatch):
        """The regression this test exists for: only the median used to be gated.

        Sigma is the channel H1 is about. A model reproducing the median while disagreeing on
        dispersion would be wrong exactly where the experiment is sensitive.
        """
        model = self._setup(tmp_path, monkeypatch, sigma_shift=0.5)
        with pytest.raises(SystemExit, match="max_abs_sigma_difference"):
            persist._verify_against_stored_forecasts(model, ["x"])

    def test_no_overlapping_rows_is_a_failure_not_a_pass(self, tmp_path, monkeypatch):
        """An empty comparison used to skip verification silently."""
        import pandas as pd

        model = self._setup(tmp_path, monkeypatch)
        # Replace the features with rows that share no key with the stored forecasts.
        pd.DataFrame({
            "station_complex_id": ["Z"],
            "issue_time": pd.to_datetime(["2020-01-01 08:00"]),
            "horizon_h": [1],
            "target_time": pd.to_datetime(["2020-01-01 09:00"]),
            "x": [1.0],
        }).to_parquet(tmp_path / "supervised_val.parquet", index=False)

        with pytest.raises(SystemExit, match="share no rows"):
            persist._verify_against_stored_forecasts(model, ["x"])


class TestProviderIntegration:
    """The environment must load the persisted model, not silently fall back to a naive one.

    A silent fallback is the dangerous case: every RL arm would still run, every metric would
    still be produced, and the uncertainty channel H1 is about would be seasonal-naive dispersion
    rather than the calibrated model the M4 gate passed.
    """

    def test_the_trained_provider_loads_the_persisted_quantile_model(self):
        from src.common.paths import REPO_ROOT

        if not (REPO_ROOT / "models" / "forecasting" / "xgb_quantile.json").exists():
            pytest.skip("no persisted forecasting models yet")

        from src.environment.forecast_provider import (
            TrainedForecastProvider,
            load_trained_provider,
        )
        from src.evaluation.runner import _network

        provider = load_trained_provider(_network())
        assert isinstance(provider, TrainedForecastProvider)
        assert provider.model_id == "xgb_quantile"

    def test_loading_fails_loudly_when_the_model_is_absent(self, tmp_path):
        """Missing artefacts must raise, never degrade to a different forecaster."""
        from src.environment.forecast_provider import load_trained_provider
        from src.evaluation.runner import _network

        with pytest.raises((FileNotFoundError, SystemExit, OSError, ValueError)):
            load_trained_provider(_network(), model_dir=tmp_path / "absent")

    def test_the_calibration_factor_travels_with_the_model(self):
        """Sigma is derived from the calibrated interval; an uncalibrated load would misstate it."""
        from src.common.paths import REPO_ROOT

        path = REPO_ROOT / "models" / "forecasting" / "metadata.json"
        if not path.exists():
            pytest.skip("no persisted forecasting models yet")

        metadata = json.loads(path.read_text(encoding="utf-8"))
        recorded = float(metadata["calibration_factor"])

        manifest = json.loads(
            (REPO_ROOT / "results" / "forecasting_metrics.json").read_text(encoding="utf-8"))
        assert recorded == pytest.approx(float(manifest["calibration"]["factor"]))


def test_the_persisted_metadata_records_a_verified_agreement():
    """Whatever is on disk now must carry evidence that it reproduces the run's forecasts."""
    from src.common.paths import REPO_ROOT

    path = REPO_ROOT / "models" / "forecasting" / "metadata.json"
    if not path.exists():
        pytest.skip("no persisted forecasting models yet")

    metadata = json.loads(path.read_text(encoding="utf-8"))
    agreement = metadata.get("refit_agreement") or {}
    assert agreement, "the persisted models carry no verification record"
    assert agreement["max_abs_p50_difference"] <= persist.AGREEMENT_TOLERANCE
    assert agreement["max_abs_sigma_difference"] <= persist.AGREEMENT_TOLERANCE
