"""End-to-end M3 pipeline: raw -> validated -> cleaned -> split -> model-ready.

Orchestrates :mod:`src.data.validate`, :mod:`src.data.splits` and :mod:`src.data.features`,
and writes the artefacts every later milestone consumes:

``data/processed/demand_hourly.parquet``
    Cleaned station-hour demand for the study window, with a ``split`` label.
``data/processed/supervised_<split>.parquet``
    Model-ready supervised frames for the forecaster.
``data/processed/network/station_order.json``
    Station complex -> position along the line, derived from GTFS.
``data/processed/split_manifest.json``
    The regime screen result, the frozen split boundaries, distribution-shift diagnostics,
    and row counts -- the record of *how* the split was chosen.

Run::

    python -m src.data.prepare --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.common.logging import get_logger, setup_logging
from src.common.paths import INTERIM_DIR, PROCESSED_DIR, RAW_DIR, RESULTS_DIR, ensure_dir
from src.data.features import FeatureSpec, build_supervised_frame
from src.data.gtfs import complex_station_order
from src.data.splits import (
    daily_line_demand,
    make_chronological_split,
    population_stability_index,
    regime_diagnostics,
    screen_demand_regime,
)
from src.data.validate import load_hourly_demand

logger = get_logger(__name__)

NETWORK_DIR = PROCESSED_DIR / "network"


def apply_exclusions(demand: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop station-days flagged by the validation stage.

    The exclusion list is a committed artefact (``data/interim/excluded_station_days.csv``),
    so which rows left the analysis is auditable rather than implicit.
    """
    path = INTERIM_DIR / "excluded_station_days.csv"
    if not path.exists():
        return demand, 0

    excluded = pd.read_csv(path)
    if excluded.empty:
        return demand, 0

    keys = set(
        zip(pd.to_datetime(excluded["date"]).dt.normalize(), excluded["station_complex_id"].astype(str))
    )
    dates = demand["transit_timestamp"].dt.normalize()
    mask = pd.Series(list(zip(dates, demand["station_complex_id"])), index=demand.index).isin(keys)
    removed = int(mask.sum())
    logger.info("excluded %d station-hour rows from %d flagged station-days", removed, len(excluded))
    return demand.loc[~mask].reset_index(drop=True), removed


def build_station_order() -> dict[str, int]:
    """Derive station ordering along the line from GTFS, and persist it."""
    stations = pd.read_parquet(RAW_DIR / "mta_subway_stations" / "data.parquet")
    order = complex_station_order(stations, "L")
    ensure_dir(NETWORK_DIR)
    (NETWORK_DIR / "station_order.json").write_text(
        json.dumps(
            {
                "derived_utc": datetime.now(timezone.utc).isoformat(),
                "source": "GTFS modal stopping pattern for route L, joined to "
                          "mta_subway_stations via gtfs_stop_id",
                "n_stations": len(order),
                "station_complex_to_index": dict(sorted(order.items(), key=lambda kv: kv[1])),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return order


def prepare(config_path: str | Path) -> dict[str, Any]:
    """Run the full M3 pipeline and return the split manifest."""
    config = load_config(config_path)
    ensure_dir(PROCESSED_DIR)

    demand = load_hourly_demand()
    demand, excluded_rows = apply_exclusions(demand)

    expected = list(config.get("line.station_complex_ids"))
    demand = demand[demand["station_complex_id"].isin(expected)].reset_index(drop=True)

    # --- regime screen (decision D-11) --------------------------------------------------
    screen_config = config.get("regime_screen")
    daily = daily_line_demand(demand)
    screen = screen_demand_regime(
        daily,
        short_window_days=int(screen_config["short_window_days"]),
        yoy_lag_days=int(screen_config["yoy_lag_days"]),
        max_abs_yoy_growth=float(screen_config["max_abs_yoy_growth"]),
        min_stable_run_days=int(screen_config["min_stable_run_days"]),
        fallback_start=str(screen_config["fallback_start"]),
    )

    study = demand[demand["transit_timestamp"] >= screen.start_date].reset_index(drop=True)
    if study.empty:
        raise RuntimeError(f"regime screen start {screen.start_date} leaves no data")

    first_date = study["transit_timestamp"].min().normalize()
    last_date = study["transit_timestamp"].max().normalize()

    # --- chronological split -------------------------------------------------------------
    split_config = config.get("split")
    boundaries = make_chronological_split(
        first_date, last_date,
        train_fraction=float(split_config["train_fraction"]),
        val_fraction=float(split_config["val_fraction"]),
        embargo_days=int(split_config["embargo_days"]),
        align_to_weekday=split_config.get("align_to_weekday"),
    )
    study["split"] = boundaries.label_for(study["transit_timestamp"])

    counts = study["split"].value_counts().to_dict()
    logger.info("split row counts: %s", counts)

    # --- features -------------------------------------------------------------------------
    station_order = build_station_order()
    feature_config = config.get("features")
    spec = FeatureSpec(horizons_h=tuple(int(h) for h in feature_config["horizons_h"]))

    # Features are built on the full study frame so that lag windows can reach across split
    # boundaries *backwards in time only*. That is legitimate -- a validation row may use its
    # own genuine past, which at prediction time would be observed. It is the target that must
    # never cross forward, and the supervised frame's split label comes from the TARGET time.
    supervised = build_supervised_frame(study, spec, station_order)
    supervised["split"] = boundaries.label_for(supervised["target_time"])

    modelling = supervised[supervised["split"].isin({"train", "val", "test"})].reset_index(drop=True)

    for split_name in ("train", "val", "test"):
        block = modelling[modelling["split"] == split_name]
        path = PROCESSED_DIR / f"supervised_{split_name}.parquet"
        block.to_parquet(path, index=False)
        logger.info("wrote %s (%d rows)", path.name, len(block))

    study.to_parquet(PROCESSED_DIR / "demand_hourly.parquet", index=False)

    # --- distribution-shift diagnostics (reported, not gated) -----------------------------
    train_target = modelling.loc[modelling["split"] == "train", "target"].to_numpy(dtype=float)
    shift = {
        f"psi_train_vs_{name}": population_stability_index(
            train_target, modelling.loc[modelling["split"] == name, "target"].to_numpy(dtype=float)
        )
        for name in ("val", "test")
    }

    manifest: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "regime_screen": screen.to_dict(),
        "regime_diagnostics": regime_diagnostics(
            daily,
            short_window_days=int(screen_config["short_window_days"]),
            yoy_lag_days=int(screen_config["yoy_lag_days"]),
            study_start=screen.start_date,
        ),
        "split": boundaries.to_dict(),
        "study_window": {"first": str(first_date.date()), "last": str(last_date.date()),
                         "days": int((last_date - first_date).days + 1)},
        "rows": {
            "raw_station_hours": int(len(demand)),
            "excluded_by_quality": excluded_rows,
            "study_station_hours": int(len(study)),
            "supervised_total": int(len(modelling)),
            **{f"supervised_{k}": int(v) for k, v in modelling["split"].value_counts().items()},
        },
        "hourly_split_counts": {str(k): int(v) for k, v in counts.items()},
        "distribution_shift": shift,
        "feature_names": spec.feature_names,
        "horizons_h": list(spec.horizons_h),
        "n_stations": len(station_order),
        "pre_regime_period_reserved": {
            "note": "Data before the regime start is NOT used for training or evaluation. It is "
                    "reserved for the out-of-regime robustness check EXP-ROBUST-004 (decision D-11).",
            "available_before": str(screen.start_date.date()),
        },
    }
    (PROCESSED_DIR / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _write_split_report(manifest, daily, screen)
    return manifest


def _write_split_report(manifest: dict[str, Any], daily: pd.Series, screen: Any) -> None:
    """Human-readable companion to the split manifest."""
    ensure_dir(RESULTS_DIR)
    split = manifest["split"]
    rows = manifest["rows"]
    lines = [
        "# Study Window and Split Report",
        "",
        "Generated by `python -m src.data.prepare`. Every number computed, none typed.",
        "",
        "## Demand-regime screen (decision D-11)",
        "",
        f"- Screen outcome: **{screen.detail}**",
        f"- Study window starts: **{manifest['regime_screen']['start_date']}**",
        f"- Used configured fallback: **{screen.used_fallback}**",
        f"- Full available daily series: {daily.index.min().date()} .. {daily.index.max().date()}",
        "",
        "Data before the regime start is deliberately **not** used for training or evaluation; "
        "it is reserved for the out-of-regime robustness check `EXP-ROBUST-004`.",
        "",
    ]

    diagnostics = manifest["regime_diagnostics"]
    window = diagnostics["study_window"]
    lines += [
        "### Residual regime instability (reported, not corrected)",
        "",
        "| Year | median YoY growth | days within ±10% | mean daily ridership |",
        "|---|---|---|---|",
    ]
    for year, stats in sorted(diagnostics["by_year"].items()):
        mean_daily = diagnostics["mean_daily_by_year"].get(year)
        mean_text = f"{mean_daily:,.0f}" if mean_daily else "—"
        lines.append(
            f"| {year} | {stats['median_yoy']:+.3f} | {stats['fraction_within_10pct']:.0%} | {mean_text} |"
        )
    lines += [
        "",
        f"Inside the study window the median year-on-year growth of the 28-day mean is "
        f"**{window['median_yoy']:+.1%}** (max |YoY| {window['max_abs_yoy']:.1%}); "
        f"{window['fraction_within_10pct']:.0%} of days fall within ±10% and "
        f"{window['fraction_within_15pct']:.0%} within ±15%.",
        "",
        "Longest run satisfying each tolerance, across the whole series:",
        "",
        "| tolerance | longest qualifying run (days) |",
        "|---|---|",
    ]
    for tolerance, run_length in diagnostics["longest_run_by_tolerance"].items():
        lines.append(f"| ±{float(tolerance):.0%} | {run_length} |")
    lines += [
        "",
        "**This table is a diagnostic, not a licence to retune.** The pre-registered tolerance "
        "is fixed; reading this after the fact and loosening the rule so the screen 'passes' "
        "would be exactly the post-hoc adjustment the protocol forbids. The screen's outcome "
        "stands as recorded above.",
        "",
        "**Consequence for the hypotheses.** Residual upward drift inflates forecast error for "
        "*every* model equally, so it does not differentially advantage the point-forecast or "
        "the uncertainty-aware arm — the H1 comparison stays internally valid. It does mean "
        "part of the fitted dispersion reflects slow trend rather than short-horizon demand "
        "uncertainty, which is carried into the limitations. Because the split is "
        "chronological, any drift makes the test period harder rather than easier, which is "
        "the conservative direction.",
        "",
        "## Chronological split",
        "",
        "| Split | Start | End |",
        "|---|---|---|",
        f"| train | {split['train_start']} | {split['train_end']} |",
        f"| val | {split['val_start']} | {split['val_end']} |",
        f"| test | {split['test_start']} | {split['test_end']} |",
        "",
        f"Embargo gap between adjacent splits: **{split['embargo_days']} days**, discarded.",
        "Random splitting is never used: with lag features reaching back 168 hours, a shuffled "
        "split would place a row's own recent history on the other side of the partition.",
        "",
        "## Row counts",
        "",
        "| Stage | Rows |",
        "|---|---|",
        f"| raw station-hours ingested | {rows['raw_station_hours']:,} |",
        f"| excluded by data quality | {rows['excluded_by_quality']:,} |",
        f"| study-window station-hours | {rows['study_station_hours']:,} |",
        f"| supervised rows (train) | {rows.get('supervised_train', 0):,} |",
        f"| supervised rows (val) | {rows.get('supervised_val', 0):,} |",
        f"| supervised rows (test) | {rows.get('supervised_test', 0):,} |",
        "",
        "## Distribution shift (reported, not gated)",
        "",
        "| Comparison | PSI |",
        "|---|---|",
    ]
    for key, value in manifest["distribution_shift"].items():
        lines.append(f"| {key} | {value:.4f} |")
    lines += [
        "",
        "PSI is reported rather than used as a gate. A chronological split of a real series is "
        "expected to show some shift, and gating on it would amount to selecting the test set "
        "for convenience. Conventional reading: < 0.1 negligible, 0.1-0.25 moderate, > 0.25 large.",
        "",
    ]
    (RESULTS_DIR / "split_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    setup_logging(level=logging.INFO, log_file="data_prepare.log")
    manifest = prepare(args.config)

    print("\nregime screen :", manifest["regime_screen"]["detail"])
    print("study window  :", manifest["study_window"])
    print("split         :", manifest["split"])
    print("rows          :", manifest["rows"])
    print("shift         :", {k: round(v, 4) for k, v in manifest["distribution_shift"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
