"""Persist the fitted M4 forecasters (deterministic refit at the selected hyperparameters).

DECISION: the forecasting models are recovered by refitting at the hyperparameters the full fit
selected, rather than by re-running the search.
REASON: `src/forecasting/train.py` evaluates its models and discards them -- it writes forecasts
and metrics but no model artefact -- and the RL arms need the fitted quantile model to produce
in-episode forecasts. XGBoost at a fixed ``random_state`` on the frozen training split is
deterministic, so refitting at the selected configuration reproduces exactly the model the run
selected rather than approximating it. The expensive part of M4 was the 12-candidate search; a
single refit costs roughly one candidate.
SCIENTIFIC CONSEQUENCE: none. Hyperparameter selection, splits, calibration factor and all
reported metrics come from the completed full fit and are unchanged. Only the model *object* is
reconstructed.

The refit is verified rather than assumed: predictions from the reconstructed model are compared
against the forecasts the original run stored, and the script fails if they disagree.

    python scripts/persist_forecast_models.py
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.common.paths import PROCESSED_DIR, REPO_ROOT, RESULTS_DIR, ensure_dir

logger = get_logger(__name__)

MODEL_DIR = REPO_ROOT / "models" / "forecasting"
METRICS = RESULTS_DIR / "forecasting_metrics.json"

#: Largest tolerated disagreement between the refit and the run's stored forecasts, in passengers.
AGREEMENT_TOLERANCE = 1e-4


def git_commit() -> str:
    from src.common.manifest import _git_executable

    executable = _git_executable()
    if executable is None:
        return "unknown"
    try:
        return subprocess.check_output([executable, "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       text=True).strip()
    except Exception as error:  # noqa: BLE001 - an unknown commit is recorded, not fatal
        logger.warning("could not read HEAD: %r", error)
        return "unknown"


def _load_into(forecaster, path, columns: list[str]):
    """Attach a saved XGBoost booster to a forecaster wrapper, as the environment does."""
    import xgboost as xgb

    model = xgb.XGBRegressor()
    model.load_model(str(path))
    forecaster._model = model
    forecaster._columns = list(columns)
    return forecaster


def _station_index() -> dict[str, int]:
    """The station-to-observation-index mapping, from the network manifest.

    The file is a manifest -- `derived_utc`, `source`, `n_stations`, and the mapping nested under
    `station_complex_to_index` -- not the mapping itself. Reading it as a flat mapping crashes on
    the timestamp, which is what it did.
    """
    path = PROCESSED_DIR / "network" / "station_order.json"
    if not path.exists():
        logger.warning("no station order manifest at %s; station_index will be empty", path)
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload.get("station_complex_to_index", payload)
    if not isinstance(mapping, dict):
        raise SystemExit(f"{path} does not contain a station-to-index mapping")
    try:
        return {str(k): int(v) for k, v in mapping.items()}
    except (TypeError, ValueError) as error:
        raise SystemExit(f"{path} holds a malformed station-to-index mapping: {error}") from error


def _verify_against_stored_forecasts(quantile, columns: list[str]) -> dict[str, float]:
    """Compare the model's predictions against the forecasts the M4 run stored.

    Both the median and sigma are gated. Sigma is the channel H1 is about, so a model that
    reproduced the median while disagreeing on dispersion would be the wrong model to ship
    precisely where it matters.
    """
    stored = pd.read_parquet(PROCESSED_DIR / "forecasts" / "forecast.parquet")
    sample = stored[stored["split"] == "val"].head(2000)
    if not len(sample):
        raise SystemExit("no stored validation forecasts to verify the model against")

    merged = sample.merge(
        pd.read_parquet(PROCESSED_DIR / "supervised_val.parquet"),
        on=["station_complex_id", "issue_time", "horizon_h", "target_time"],
        how="inner", suffixes=("", "_features"),
    )
    if not len(merged):
        raise SystemExit(
            "stored forecasts and validation features share no rows, so the refit cannot be "
            "verified; shipping an unverified model is not an option"
        )

    output = quantile.predict(merged[columns])
    agreement = {
        "n_compared": int(len(merged)),
        "max_abs_p50_difference": float(np.max(np.abs(output.mean - merged["p50"]))),
        "max_abs_sigma_difference": float(np.max(np.abs(output.sigma - merged["sigma"]))),
    }
    logger.info("refit agreement: %s", agreement)

    for key in ("max_abs_p50_difference", "max_abs_sigma_difference"):
        if agreement[key] > AGREEMENT_TOLERANCE:
            raise SystemExit(
                f"the model does not reproduce the stored forecasts ({key} = "
                f"{agreement[key]:.6g}, tolerance {AGREEMENT_TOLERANCE:g}). It is not the model "
                "the run evaluated, so it must not be used downstream."
            )
    return agreement


def main() -> int:
    from src.data.features import FeatureSpec
    from src.forecasting.models import XGBPointForecaster, XGBQuantileForecaster

    if not METRICS.exists():
        raise SystemExit(f"{METRICS} not found -- has the M4 full fit finished?")
    manifest = json.loads(METRICS.read_text(encoding="utf-8"))
    if manifest.get("quick_mode"):
        raise SystemExit("refusing to persist models from a quick-mode run")

    selected = manifest["selected_hyperparameters"]
    columns = list(manifest["feature_columns"])
    calibration = manifest["calibration"]
    factor = float(calibration.get("factor", calibration.get("calibration_factor", 1.0)))

    train = pd.read_parquet(PROCESSED_DIR / "supervised_train.parquet")
    quantile_path = MODEL_DIR / "xgb_quantile.json"
    point_path = MODEL_DIR / "xgb_point.json"

    # If models are already on disk, verify *those* rather than refitting and assuming. This is
    # both cheaper and stricter: it holds whatever the environment will actually load to the same
    # standard, instead of certifying a fresh object and trusting that the file matches it.
    source = "existing artefacts on disk, verified against the run's stored forecasts"
    quantile = XGBQuantileForecaster(**selected["xgb_quantile"])
    point = XGBPointForecaster(**selected["xgb_point"])
    reused = False

    if quantile_path.exists() and point_path.exists():
        try:
            _load_into(quantile, quantile_path, columns)
            _load_into(point, point_path, columns)
            quantile.calibration_factor = factor
            agreement = _verify_against_stored_forecasts(quantile, columns)
            reused = True
            logger.info("reused the persisted models; they reproduce the stored forecasts")
        except SystemExit:
            raise
        except Exception as error:  # noqa: BLE001 - unloadable artefacts are replaced, not trusted
            logger.warning("could not load the persisted models (%r); refitting", error)

    if not reused:
        logger.info("refitting on %d training rows at the selected hyperparameters", len(train))
        features = train[columns]
        target = train["target"].to_numpy(dtype=float)

        quantile = XGBQuantileForecaster(**selected["xgb_quantile"]).fit(features, target)
        quantile.calibration_factor = factor
        point = XGBPointForecaster(**selected["xgb_point"]).fit(features, target)

        # Verified BEFORE saving. Saving first left unverified artefacts on disk whenever
        # verification failed, and `stage_prepare` skips persisting when the files exist -- so a
        # failed run would have been silently promoted to a trusted model by the next stage.
        agreement = _verify_against_stored_forecasts(quantile, columns)
        source = "deterministic refit at the hyperparameters selected by the M4 full fit"

        ensure_dir(MODEL_DIR)
        quantile._model.save_model(str(quantile_path))
        point._model.save_model(str(point_path))

    station_index = _station_index()

    metadata = {
        "source": source,
        "source_manifest": str(METRICS.relative_to(REPO_ROOT)),
        "source_run_git_commit": manifest.get("git_commit"),
        "hyperparameters": selected["xgb_quantile"],
        "point_hyperparameters": selected["xgb_point"],
        "quantiles": list(quantile.quantiles),
        "calibration_factor": factor,
        "calibration_record": calibration,
        "feature_columns": columns,
        "feature_spec": {
            "recent_lags": list(FeatureSpec().recent_lags),
            "seasonal_lags": list(FeatureSpec().seasonal_lags),
            "rolling_windows": list(FeatureSpec().rolling_windows),
        },
        "station_index": station_index,
        "training_rows": int(len(train)),
        "refit_agreement": agreement,
        "sigma_definition": "(q90 - q10) / (2 * 1.2816), protocol section 4.3",
        "git_commit": git_commit(),
        "persisted_utc": datetime.now(timezone.utc).isoformat(),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str),
                                             encoding="utf-8")
    logger.info("persisted forecasting models to %s", MODEL_DIR)
    print(f"persisted -> {MODEL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
