"""Fit, calibrate and evaluate the demand forecasters (M4).

Pipeline, in the order the protocol freezes:

1. Load the chronological supervised splits produced by :mod:`src.data.prepare`.
2. Fit the seasonal-naive reference (no parameters).
3. Select hyperparameters for the point and quantile models on **validation**, using grids of
   identical size so neither model receives a tuning advantage.
4. Fit the interval-width multiplier on **validation** only.
5. Evaluate every model on validation and test.
6. Apply the **blocking gate**: the quantile model must beat seasonal naive on pinball loss and
   its calibrated 80% coverage must land inside the pre-registered band. RL training may not
   consume forecasts otherwise.
7. Write ``forecast.parquet`` with the columns the RL state and the evaluation need.

Run::

    python -m src.forecasting.train --config configs/forecasting.yaml
    python -m src.forecasting.train --config configs/forecasting.yaml --quick   # smoke test
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.common.logging import get_logger, setup_logging
from src.common.manifest import git_info, package_versions
from src.common.paths import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, ensure_dir
from src.forecasting.calibrate import fit_interval_scale
from src.forecasting.metrics import evaluate_point, evaluate_probabilistic
from src.forecasting.models import (
    SeasonalNaive,
    XGBPointForecaster,
    XGBQuantileForecaster,
)

logger = get_logger(__name__)

FORECAST_DIR = PROCESSED_DIR / "forecasts"
KEY_COLUMNS = ["station_complex_id", "issue_time", "horizon_h", "target_time"]

#: The files whose contents determine what this run computes. A change to any of them while the
#: fit is in flight makes the stored result attributable to neither the old nor the new version.
CODE_PATH: tuple[str, ...] = (
    "src/forecasting/train.py", "src/forecasting/models.py", "src/forecasting/metrics.py",
    "src/forecasting/calibrate.py", "src/data/features.py", "src/data/splits.py",
    "src/common/config.py", "configs/forecasting.yaml",
)


def code_path_digest() -> str:
    """SHA-256 over the contents of every file this run executes, in a fixed order.

    Stronger than a git commit for this purpose: it captures uncommitted working-tree state, which
    is exactly the case a commit hash misses.
    """
    from src.common.paths import REPO_ROOT

    digest = hashlib.sha256()
    for relative in CODE_PATH:  # already in a fixed order; not sorted, so the order is auditable
        path = REPO_ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes() if path.is_file() else b"<absent>")
    return digest.hexdigest()


def load_split(name: str) -> pd.DataFrame:
    """Load one supervised split written by :mod:`src.data.prepare`."""
    path = PROCESSED_DIR / f"supervised_{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.data.prepare --config configs/data.yaml"
        )
    return pd.read_parquet(path)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Feature columns are everything that is neither a key, the target, nor the split label."""
    excluded = set(KEY_COLUMNS) | {"target", "split"}
    return [c for c in frame.columns if c not in excluded]


def naive_absolute_errors(frame: pd.DataFrame) -> np.ndarray:
    """In-sample absolute errors of the seasonal-naive reference, used to scale MASE."""
    return np.abs(frame["target"].to_numpy(dtype=float) - frame["y_seasonal_168"].to_numpy(dtype=float))


def _grid_candidates(search: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the frozen grid into an explicit, ordered candidate list."""
    grid = search["grid"]
    names = sorted(grid)
    return [dict(zip(names, values)) for values in itertools.product(*(grid[name] for name in names))]


def select_hyperparameters(
    model_class: type,
    train: pd.DataFrame,
    val: pd.DataFrame,
    columns: list[str],
    search: dict[str, Any],
    *,
    score: str,
    quantiles: tuple[float, ...] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit every grid candidate on train, score on validation, return the best.

    Both models traverse grids of identical size, so the point-forecast arm is never
    handicapped relative to the arm that supplies the uncertainty channel.
    """
    candidates = _grid_candidates(search)
    fixed = dict(search["fixed"])
    trials: list[dict[str, Any]] = []

    x_train, y_train = train[columns], train["target"].to_numpy(dtype=float)
    x_val, y_val = val[columns], val["target"].to_numpy(dtype=float)

    best_params: dict[str, Any] | None = None
    best_score = np.inf

    for index, candidate in enumerate(candidates, start=1):
        params = {**fixed, **candidate}
        model = model_class(quantiles=quantiles, **params) if quantiles else model_class(**params)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        output = model.predict(x_val)

        if score == "rmse":
            value = float(np.sqrt(np.mean((y_val - output.mean) ** 2)))
        elif score == "pinball_mean":
            value = evaluate_probabilistic(y_val, output.quantiles).pinball_mean
        else:
            raise ValueError(f"unknown selection score {score!r}")

        trials.append({
            "candidate": candidate, "score_metric": score, "score": value,
            "fit_seconds": round(time.perf_counter() - started, 2),
        })
        logger.info("  [%d/%d] %s -> %s=%.4f", index, len(candidates), candidate, score, value)

        if np.isfinite(value) and value < best_score:
            best_score, best_params = value, params

    if best_params is None:
        raise RuntimeError(f"no candidate produced a finite {score}")
    logger.info("selected %s (%s=%.4f)", {k: v for k, v in best_params.items() if k in candidates[0]},
                score, best_score)
    return best_params, trials


def evaluate_model(
    name: str, output: Any, frame: pd.DataFrame, naive_errors: np.ndarray, floor: float
) -> dict[str, Any]:
    """Assemble the full metric block for one model on one split."""
    actual = frame["target"].to_numpy(dtype=float)
    block: dict[str, Any] = {
        "model_id": name,
        "point": evaluate_point(actual, output.mean, naive_errors, floor).to_dict(),
    }
    if output.quantiles:
        block["probabilistic"] = evaluate_probabilistic(actual, output.quantiles).to_dict()
    return block


def write_forecast_parquet(
    frames: dict[str, pd.DataFrame],
    outputs: dict[str, Any],
    path: Path,
    *,
    model_id: str,
    model_version: str,
    calibrated: bool,
) -> Path:
    """Write the stored predictions the RL environment and the evaluation consume.

    Columns are those protocol section 4.6 requires: timestamp, station, horizon, actual, p10,
    p50, p90, mean, sigma, plus the split label and the three provenance columns -- `model_id`,
    `model_version` and `calibrated` -- so any downstream number can be traced back to the model
    that produced it and to whether its interval had been calibrated.

    `model_version` is the run's configuration hash, which identifies the fitted model far more
    precisely than a version string a human would have to remember to bump.
    """
    blocks: list[pd.DataFrame] = []
    for split_name, frame in frames.items():
        output = outputs[split_name]
        block = frame[KEY_COLUMNS].copy()
        block["split"] = split_name
        block["actual"] = frame["target"].to_numpy(dtype=float)
        block["mean"] = output.mean
        block["p10"] = output.quantiles.get(0.10, np.full(len(frame), np.nan))
        block["p50"] = output.quantiles.get(0.50, output.mean)
        block["p90"] = output.quantiles.get(0.90, np.full(len(frame), np.nan))
        block["sigma"] = output.sigma
        block["model_id"] = model_id
        block["model_version"] = model_version
        block["calibrated"] = bool(calibrated)
        blocks.append(block)

    combined = pd.concat(blocks, ignore_index=True)
    ensure_dir(path.parent)
    combined.to_parquet(path, index=False)
    logger.info("wrote %s (%d rows)", path, len(combined))
    return path


def run(config_path: str | Path, *, quick: bool = False) -> dict[str, Any]:
    """Execute M4 end to end and return the results manifest."""
    config = load_config(config_path)
    ensure_dir(FORECAST_DIR)
    ensure_dir(MODELS_DIR)

    # Provenance is captured at *launch* as well as completion. A multi-hour fit runs while work
    # continues elsewhere in the repository, so HEAD at completion says nothing about the code the
    # run actually executed. The digest is over the file contents the run is about to import, and
    # is recomputed at the end: if the two differ, the code changed underneath the run and the
    # result is not attributable to either version.
    launch_git = git_info()
    launch_digest = code_path_digest()

    train, val, test = load_split("train"), load_split("val"), load_split("test")
    if quick:
        # Smoke path: a recent slice of each split. Protocol section 10 stage 1 -- prove the
        # pipeline runs before spending the full fit.
        train, val, test = train.tail(60_000), val.tail(20_000), test.tail(20_000)
        logger.warning("quick mode: using truncated splits (%d/%d/%d rows) -- NOT a result",
                       len(train), len(val), len(test))

    columns = feature_columns(train)
    logger.info("train=%d val=%d test=%d rows, %d features", len(train), len(val), len(test), len(columns))

    naive_errors = {name: naive_absolute_errors(frame)
                    for name, frame in (("train", train), ("val", val), ("test", test))}
    floor = float(config.get("metrics.smape_mape_floor", 1.0))
    search = config.get("search")
    quantiles = tuple(float(q) for q in config.get("quantiles"))

    models: dict[str, Any] = {}
    trials: dict[str, Any] = {}

    # --- 1. seasonal naive -----------------------------------------------------------------
    logger.info("fitting %s", SeasonalNaive.model_id)
    models[SeasonalNaive.model_id] = SeasonalNaive().fit(train[columns], train["target"].to_numpy())

    # --- 2. point model --------------------------------------------------------------------
    logger.info("selecting hyperparameters for %s", XGBPointForecaster.model_id)
    point_params, trials["xgb_point"] = select_hyperparameters(
        XGBPointForecaster, train, val, columns, search, score="rmse")
    models[XGBPointForecaster.model_id] = XGBPointForecaster(**point_params).fit(
        train[columns], train["target"].to_numpy(dtype=float))

    # --- 3. quantile model -----------------------------------------------------------------
    logger.info("selecting hyperparameters for %s", XGBQuantileForecaster.model_id)
    quantile_params, trials["xgb_quantile"] = select_hyperparameters(
        XGBQuantileForecaster, train, val, columns, search, score="pinball_mean", quantiles=quantiles)
    quantile_model = XGBQuantileForecaster(quantiles=quantiles, **quantile_params).fit(
        train[columns], train["target"].to_numpy(dtype=float))
    models[XGBQuantileForecaster.model_id] = quantile_model

    # --- 4. calibrate on validation only ---------------------------------------------------
    calibration_config = config.get("calibration")
    val_output = quantile_model.predict(val[columns])
    calibration = fit_interval_scale(
        val["target"].to_numpy(dtype=float),
        val_output.quantiles[0.50], val_output.quantiles[0.10], val_output.quantiles[0.90],
        target_coverage=float(calibration_config["target_coverage"]),
        search_min=float(calibration_config["search_min"]),
        search_max=float(calibration_config["search_max"]),
        n_candidates=int(calibration_config["n_candidates"]),
    )
    quantile_model.calibration_factor = calibration.factor

    # --- 5. evaluate ------------------------------------------------------------------------
    frames = {"train": train, "val": val, "test": test}
    evaluation: dict[str, dict[str, Any]] = {}
    stored_outputs: dict[str, dict[str, Any]] = {}

    for model_id, model in models.items():
        evaluation[model_id] = {}
        stored_outputs[model_id] = {}
        for split_name, frame in frames.items():
            output = model.predict(frame[columns])
            stored_outputs[model_id][split_name] = output
            evaluation[model_id][split_name] = evaluate_model(
                model_id, output, frame, naive_errors[split_name], floor)

    # --- 6. blocking gate --------------------------------------------------------------------
    gate_config = config.get("gate")
    naive_pinball = evaluation["naive_seasonal"]["val"].get("probabilistic", {}).get("pinball_mean")
    if naive_pinball is None:
        # Seasonal naive emits no quantiles; score it as a degenerate predictive distribution
        # (all mass at the point forecast) so the comparison is defined and fair.
        naive_output = stored_outputs["naive_seasonal"]["val"]
        degenerate = {q: naive_output.mean for q in quantiles}
        naive_pinball = evaluate_probabilistic(val["target"].to_numpy(dtype=float), degenerate).pinball_mean

    quantile_pinball = evaluation["xgb_quantile"]["val"]["probabilistic"]["pinball_mean"]
    picp = evaluation["xgb_quantile"]["val"]["probabilistic"]["picp_80"]
    beats_naive = bool(np.isfinite(quantile_pinball) and quantile_pinball < naive_pinball)
    coverage_ok = bool(float(gate_config["coverage_min"]) <= picp <= float(gate_config["coverage_max"]))

    gate = {
        "beats_naive_on_pinball": beats_naive,
        "naive_pinball_val": float(naive_pinball),
        "quantile_pinball_val": float(quantile_pinball),
        "picp_80_val": float(picp),
        "coverage_band": [float(gate_config["coverage_min"]), float(gate_config["coverage_max"])],
        "coverage_within_band": coverage_ok,
        "passed": bool(beats_naive and coverage_ok),
    }
    if not gate["passed"]:
        logger.error("FORECASTING GATE FAILED: %s", gate)
    else:
        logger.info("forecasting gate passed: %s", gate)

    # --- 7. artefacts -------------------------------------------------------------------------
    forecast_path = write_forecast_parquet(
        frames, stored_outputs["xgb_quantile"], FORECAST_DIR / "forecast.parquet",
        model_id=XGBQuantileForecaster.model_id, model_version=config.hash,
        calibrated=calibration.factor != 1.0)
    write_forecast_parquet(
        frames, stored_outputs["xgb_point"], FORECAST_DIR / "forecast_point.parquet",
        # The point model emits no interval, so there is nothing to calibrate. Recording `False`
        # states that, rather than leaving a reader to infer it from a missing column.
        model_id=XGBPointForecaster.model_id, model_version=config.hash, calibrated=False)

    info = git_info()
    final_digest = code_path_digest()
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "quick_mode": quick,
        "git_commit": info.commit,
        "git_dirty": info.dirty,
        "git_commit_at_launch": launch_git.commit,
        "git_dirty_at_launch": launch_git.dirty,
        "code_path_digest_at_launch": launch_digest,
        "code_path_digest_at_completion": final_digest,
        "code_unchanged_during_run": launch_digest == final_digest,
        "packages": package_versions(),
        "config_path": str(config_path),
        "config_hash": config.hash,
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "feature_columns": columns,
        "selected_hyperparameters": {
            "xgb_point": point_params,
            "xgb_quantile": quantile_params,
        },
        "search_trials": trials,
        "calibration": calibration.to_dict(),
        "quantile_crossing_rate": quantile_model.crossing_rate,
        "gate": gate,
        "metrics": evaluation,
        "artifacts": {
            "forecast_quantile": str(forecast_path),
            "forecast_point": str(FORECAST_DIR / "forecast_point.parquet"),
        },
    }
    ensure_dir(RESULTS_DIR)
    (RESULTS_DIR / "forecasting_metrics.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    _write_report(manifest)
    return manifest


def _write_report(manifest: dict[str, Any]) -> None:
    """Human-readable forecasting report, generated entirely from the metrics manifest."""
    gate = manifest["gate"]
    lines = [
        "# Forecasting Results (M4)",
        "",
        "Generated by `python -m src.forecasting.train`. Every number computed, none typed.",
        "",
        f"- Generated (UTC): {manifest['generated_utc']}",
        f"- Git commit: `{manifest['git_commit']}`" + ("  **(dirty tree)**" if manifest["git_dirty"] else ""),
        f"- Rows: train {manifest['rows']['train']:,} / val {manifest['rows']['val']:,} / test {manifest['rows']['test']:,}",
        f"- Quick mode: **{manifest['quick_mode']}**" + ("  — NOT a reportable result" if manifest["quick_mode"] else ""),
        "",
        "## Blocking gate (protocol section 4.5)",
        "",
        f"**{'PASSED' if gate['passed'] else 'FAILED'}**",
        "",
        "| Condition | Value | Required | Met |",
        "|---|---|---|---|",
        f"| quantile beats seasonal naive on mean pinball (val) | {gate['quantile_pinball_val']:.4f} vs {gate['naive_pinball_val']:.4f} | lower | {gate['beats_naive_on_pinball']} |",
        f"| calibrated 80% coverage (val) | {gate['picp_80_val']:.4f} | {gate['coverage_band'][0]}–{gate['coverage_band'][1]} | {gate['coverage_within_band']} |",
        "",
        "RL training may not consume forecasts unless both conditions hold. If the gate fails, "
        "the forecaster is revised **before** any RL run, never after seeing RL results.",
        "",
        "## Calibration",
        "",
        f"- Interval-width multiplier fitted on **validation only**: **{manifest['calibration']['factor']:.3f}**",
        f"- 80% coverage before → after: {manifest['calibration']['coverage_before']:.4f} → {manifest['calibration']['coverage_after']:.4f}",
        f"- Quantile crossing rate before monotone repair: {manifest['quantile_crossing_rate']:.4%}",
        "",
        "## Point accuracy",
        "",
        "| Model | Split | MAE | RMSE | MASE | SMAPE % | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for model_id, splits in manifest["metrics"].items():
        for split_name in ("val", "test"):
            point = splits[split_name]["point"]
            lines.append(
                f"| `{model_id}` | {split_name} | {point['mae']:.3f} | {point['rmse']:.3f} | "
                f"{point['mase']:.3f} | {point['smape']:.2f} | {point['n']:,} |"
            )

    lines += [
        "",
        f"MASE < 1 means the model beats seasonal naive. SMAPE/MAPE are computed only over rows "
        f"whose actual value is at least {manifest['metrics']['xgb_point']['val']['point']['smape_mape_floor']}, "
        "because overnight station-hours are legitimately zero and would otherwise dominate a "
        "percentage error.",
        "",
        "## Uncertainty quality",
        "",
        "| Model | Split | pinball | PICP80 | MPIW80 | MPIW80/mean | Winkler80 | CRPS | ECE |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for model_id, splits in manifest["metrics"].items():
        for split_name in ("val", "test"):
            prob = splits[split_name].get("probabilistic")
            if not prob:
                continue
            lines.append(
                f"| `{model_id}` | {split_name} | {prob['pinball_mean']:.4f} | {prob['picp_80']:.4f} | "
                f"{prob['mpiw_80']:.3f} | {prob['mpiw_80_normalised']:.3f} | {prob['winkler_80']:.3f} | "
                f"{prob['crps']:.4f} | {prob['ece']:.4f} |"
            )
    lines += [
        "",
        "Coverage is always reported next to interval width: an interval can reach any coverage "
        "by widening, so coverage alone is not evidence of a good forecaster. Winkler penalises "
        "width and misses jointly.",
        "",
    ]
    (RESULTS_DIR / "forecasting_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/forecasting.yaml")
    parser.add_argument("--quick", action="store_true", help="smoke run on truncated splits")
    args = parser.parse_args()

    setup_logging(level=logging.INFO, log_file="forecasting_train.log")
    manifest = run(args.config, quick=args.quick)

    gate = manifest["gate"]
    print(f"\ngate passed: {gate['passed']}")
    print(f"  pinball  quantile={gate['quantile_pinball_val']:.4f} naive={gate['naive_pinball_val']:.4f}")
    print(f"  PICP80   {gate['picp_80_val']:.4f} (band {gate['coverage_band']})")
    for model_id, splits in manifest["metrics"].items():
        point = splits["test"]["point"]
        print(f"  {model_id:16s} test MAE={point['mae']:.3f} RMSE={point['rmse']:.3f} MASE={point['mase']:.3f}")
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
