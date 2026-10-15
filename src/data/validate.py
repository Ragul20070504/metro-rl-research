"""Data-quality validation (protocol section 3.2).

Ten checks, each an explicit assertion with a recorded outcome. The guiding rule is that
**anomalies are flagged and counted, never silently deleted** -- a pipeline that quietly drops
inconvenient rows produces clean-looking numbers that nobody can audit.

Severities:

* ``blocking``  -- the pipeline must not continue; the data are not what we think they are.
* ``advisory``  -- recorded in the report and carried into the limitations; does not stop work.

Run::

    python -m src.data.validate --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.common.logging import get_logger, setup_logging
from src.common.paths import INTERIM_DIR, RAW_DIR, RESULTS_DIR, ensure_dir
from src.data.sources import HOURLY_RIDERSHIP

logger = get_logger(__name__)


@dataclass
class CheckResult:
    """Outcome of one data-quality check."""

    name: str
    passed: bool
    severity: str
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def blocking_failure(self) -> bool:
        return (not self.passed) and self.severity == "blocking"


#: Datasets belonging to the post-recovery demand regime, which is what the study window is
#: drawn from. The 2017-2019 dataset is deliberately EXCLUDED here: decision D-11 reserves it
#: for the out-of-regime robustness check (EXP-ROBUST-004), and merging it into the main
#: pipeline would reintroduce exactly the structural break the regime screen exists to avoid.
#: Loading it requires naming it explicitly, so it cannot drift into the study by accident.
IN_REGIME_DATASETS: tuple[str, ...] = (
    "mta_hourly_ridership_2020",
    "mta_hourly_ridership_2025",
)

PRE_REGIME_DATASETS: tuple[str, ...] = ("mta_hourly_ridership_2017",)


def load_hourly_demand(dataset_ids: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Concatenate the month partitions of the requested hourly-ridership datasets.

    Args:
        dataset_ids: Which datasets to load. Defaults to :data:`IN_REGIME_DATASETS`.
            Pass :data:`PRE_REGIME_DATASETS` only for the registered out-of-regime experiment.

    The datasets have identical schemas and contiguous coverage (``docs/data_sources.md``
    section 1). The originating ``dataset_id`` is retained so any seam is visible downstream.
    """
    selected = set(dataset_ids) if dataset_ids is not None else set(IN_REGIME_DATASETS)
    frames: list[pd.DataFrame] = []
    for source in HOURLY_RIDERSHIP:
        if source.dataset_id not in selected:
            continue
        months_dir = RAW_DIR / source.dataset_id / "months"
        if not months_dir.is_dir():
            continue
        for path in sorted(months_dir.glob("*.parquet")):
            frame = pd.read_parquet(path)
            if frame.empty:
                continue
            frame["dataset_id"] = source.dataset_id
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            "no hourly ridership partitions found under data/raw/. "
            "Run: python -m src.data.download --config configs/data.yaml"
        )
    demand = pd.concat(frames, ignore_index=True)
    demand["transit_timestamp"] = pd.to_datetime(demand["transit_timestamp"])
    demand["station_complex_id"] = demand["station_complex_id"].astype(str)
    return demand.sort_values(["transit_timestamp", "station_complex_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------------------

def check_required_fields(demand: pd.DataFrame) -> CheckResult:
    """1. No missing timestamp, station id, or ridership."""
    required = ["transit_timestamp", "station_complex_id", "ridership"]
    counts = {column: int(demand[column].isna().sum()) for column in required}
    total = sum(counts.values())
    return CheckResult(
        name="required_fields_present",
        passed=total == 0,
        severity="blocking",
        detail=f"{total} missing values across {required}",
        metrics=counts,
    )


def check_duplicates(demand: pd.DataFrame) -> CheckResult:
    """2. No duplicate (timestamp, station) rows.

    Server-side aggregation already collapses fare class and payment method, so a duplicate
    here would mean overlapping coverage between the three source datasets -- which would
    double-count demand.
    """
    duplicated = demand.duplicated(subset=["transit_timestamp", "station_complex_id"], keep=False)
    n = int(duplicated.sum())
    overlap = (
        demand.loc[duplicated]
        .groupby("dataset_id", observed=True)
        .size()
        .to_dict()
    ) if n else {}
    return CheckResult(
        name="no_duplicate_station_hours",
        passed=n == 0,
        severity="blocking",
        detail=f"{n} duplicated (timestamp, station) rows" + (f"; by dataset {overlap}" if overlap else ""),
        metrics={"duplicate_rows": n, "by_dataset": {str(k): int(v) for k, v in overlap.items()}},
    )


def check_timestamps(demand: pd.DataFrame) -> CheckResult:
    """3. Timestamps parse, are hour-aligned, and lie in a plausible range."""
    timestamps = demand["transit_timestamp"]
    unparsed = int(timestamps.isna().sum())
    not_hour_aligned = int(((timestamps.dt.minute != 0) | (timestamps.dt.second != 0)).sum())
    first, last = timestamps.min(), timestamps.max()
    implausible = int(((timestamps < pd.Timestamp("2010-01-01")) | (timestamps > pd.Timestamp.now() + pd.Timedelta(days=1))).sum())
    problems = unparsed + not_hour_aligned + implausible
    return CheckResult(
        name="timestamps_valid",
        passed=problems == 0,
        severity="blocking",
        detail=(
            f"coverage {first} .. {last}; unparsed={unparsed}, "
            f"not hour-aligned={not_hour_aligned}, implausible={implausible}"
        ),
        metrics={
            "first": str(first), "last": str(last),
            "unparsed": unparsed, "not_hour_aligned": not_hour_aligned, "implausible": implausible,
        },
    )


def check_non_negative(demand: pd.DataFrame) -> CheckResult:
    """4. The forecasting target must be non-negative.

    Severity is scoped to the column's actual role, and the scoping is deliberate rather than
    convenient:

    * ``ridership`` is the forecasting target and drives the simulator's demand process. A
      negative value there would invalidate both, so it stays **blocking**.
    * ``transfers`` is ingested for provenance completeness but is **not used downstream** --
      not as a target, not as a feature, not by the simulator. The published series carries a
      handful of small negative values (net-of-adjustment artefacts). They are counted and
      reported as an upstream defect, but they cannot affect any result, so failing the whole
      pipeline on them would be theatre rather than rigour.

    If ``transfers`` ever becomes an input, this must be promoted back to blocking.
    """
    ridership_negatives = int((demand["ridership"] < 0).sum())
    transfer_negatives = (
        int((demand["transfers"] < 0).sum()) if "transfers" in demand.columns else 0
    )
    detail = f"ridership: {ridership_negatives} negative (blocking)"
    if transfer_negatives:
        worst = float(demand.loc[demand["transfers"] < 0, "transfers"].min())
        detail += (
            f"; transfers: {transfer_negatives} negative (upstream defect, min {worst:.0f}) -- "
            "recorded, not blocking, because transfers is not used downstream"
        )
    return CheckResult(
        name="non_negative_demand",
        passed=ridership_negatives == 0,
        severity="blocking",
        detail=detail,
        metrics={
            "ridership_negative": ridership_negatives,
            "transfers_negative": transfer_negatives,
            "transfers_used_downstream": False,
        },
    )


def check_outliers(demand: pd.DataFrame, mad_k: float) -> CheckResult:
    """5. Robust outlier screen, per station. Flags and counts; never deletes.

    Uses median +/- k * MAD rather than a standard-deviation rule because hourly transit
    counts are strongly right-skewed and heavy-tailed, so an SD-based screen would flag most
    of the peak period.
    """
    flagged_by_station: dict[str, int] = {}
    for station, group in demand.groupby("station_complex_id", observed=True):
        values = group["ridership"].to_numpy(dtype=float)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        if mad <= 0:
            continue
        # 1.4826 rescales MAD to a standard-deviation equivalent for a normal distribution.
        threshold = median + mad_k * 1.4826 * mad
        count = int((values > threshold).sum())
        if count:
            flagged_by_station[str(station)] = count
    total = sum(flagged_by_station.values())
    return CheckResult(
        name="outlier_screen",
        passed=True,  # advisory by construction
        severity="advisory",
        detail=f"{total} station-hours above median + {mad_k}*1.4826*MAD; flagged, not removed",
        metrics={"flagged_total": total, "flagged_by_station": flagged_by_station},
    )


def check_station_ids(demand: pd.DataFrame, expected: list[str]) -> CheckResult:
    """6. Station identifiers resolve, and every expected complex is present."""
    observed = set(demand["station_complex_id"].unique())
    expected_set = set(expected)
    unexpected = sorted(observed - expected_set)
    missing = sorted(expected_set - observed)
    return CheckResult(
        name="station_ids_resolve",
        passed=not unexpected and not missing,
        severity="blocking",
        detail=f"{len(observed)} complexes observed; unexpected={unexpected}; missing={missing}",
        metrics={"observed": len(observed), "unexpected": unexpected, "missing": missing},
    )


def check_temporal_continuity(
    demand: pd.DataFrame, max_missing_hours: int, start_hour: int, end_hour: int
) -> tuple[CheckResult, pd.DataFrame]:
    """7. Count missing operating hours per station-day; list days to exclude.

    Only *operating* hours are counted. The line does run overnight, but the simulator's
    episode is the 05:00-24:00 operating day frozen in the protocol, so continuity outside
    that window is irrelevant to the study and would produce misleading gap counts.
    """
    working = demand.copy()
    working["date"] = working["transit_timestamp"].dt.date
    working["hour"] = working["transit_timestamp"].dt.hour
    operating = working[(working["hour"] >= start_hour) & (working["hour"] < end_hour)]

    expected_hours = end_hour - start_hour
    present = (
        operating.groupby(["date", "station_complex_id"], observed=True)["hour"]
        .nunique()
        .rename("hours_present")
        .reset_index()
    )
    present["hours_missing"] = expected_hours - present["hours_present"]
    excluded = present[present["hours_missing"] > max_missing_hours].copy()

    # A station-day that is entirely absent will not appear above at all; count those too.
    all_dates = pd.Index(sorted(operating["date"].unique()), name="date")
    all_stations = pd.Index(sorted(operating["station_complex_id"].unique()), name="station_complex_id")
    full_index = pd.MultiIndex.from_product([all_dates, all_stations])
    seen = pd.MultiIndex.from_frame(present[["date", "station_complex_id"]])
    absent = full_index.difference(seen)

    return (
        CheckResult(
            name="temporal_continuity",
            passed=True,  # advisory: gaps are handled by exclusion, not by failing the run
            severity="advisory",
            detail=(
                f"{len(excluded)} station-days exceed {max_missing_hours} missing operating hours; "
                f"{len(absent)} station-days entirely absent; both excluded downstream"
            ),
            metrics={
                "expected_operating_hours": expected_hours,
                "station_days_total": int(len(present)),
                "station_days_excluded": int(len(excluded)),
                "station_days_absent": int(len(absent)),
            },
        ),
        excluded,
    )


def check_service_delivered_scaling(config_max: float) -> CheckResult:
    """8. Guard the known upstream mis-scaling in ``32ch-sei3``.

    The published ``service_delivered`` field reads ~0.00913 where ``num_actual_trains /
    num_sched_trains`` is 0.9135 -- a factor of 100. The pipeline recomputes the ratio and
    must never use the published field. This check asserts the recomputed value is sane and
    confirms the published one is indeed mis-scaled, so the defect is documented rather than
    assumed.
    """
    path = RAW_DIR / "mta_service_delivered" / "data.parquet"
    if not path.exists():
        return CheckResult("service_delivered_scaling", True, "advisory", "dataset not ingested; skipped")

    frame = pd.read_parquet(path)
    for column in ("num_actual_trains", "num_sched_trains", "service_delivered"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    valid = frame[frame["num_sched_trains"] > 0].copy()
    valid["recomputed"] = valid["num_actual_trains"] / valid["num_sched_trains"]

    out_of_range = int((valid["recomputed"] > config_max).sum())
    ratio = (valid["recomputed"] / valid["service_delivered"]).replace([np.inf, -np.inf], np.nan).dropna()
    median_ratio = float(ratio.median()) if len(ratio) else float("nan")
    mis_scaled = bool(np.isfinite(median_ratio) and median_ratio > 10)

    return CheckResult(
        name="service_delivered_scaling",
        passed=out_of_range == 0,
        severity="blocking",
        detail=(
            f"recomputed ratio in range for {len(valid) - out_of_range}/{len(valid)} rows; "
            f"published field is mis-scaled by a median factor of {median_ratio:.1f} "
            f"({'confirmed' if mis_scaled else 'NOT confirmed -- upstream may have been fixed'})"
        ),
        metrics={
            "rows": int(len(valid)),
            "out_of_range": out_of_range,
            "median_scaling_factor": median_ratio,
            "mis_scaling_confirmed": mis_scaled,
        },
    )


def check_manifests_present(dataset_ids: tuple[str, ...]) -> CheckResult:
    """9. Every dataset **used by this pipeline** carries a manifest with file hashes.

    Scoped to the datasets actually loaded, plus the reference and validation tables. A
    dataset that exists on disk but is not consumed (for example the pre-regime series while
    it is still being ingested) cannot affect a result, so it is reported separately rather
    than blocking the run. Data that *is* consumed without provenance remains blocking --
    that is the condition this check exists for.
    """
    required = set(dataset_ids) | {
        d.name for d in RAW_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("mta_hourly_ridership_")
    }
    missing: list[str] = []
    unused_without_manifest: list[str] = []
    hashed = 0

    for directory in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        manifest_path = directory / "MANIFEST.json"
        if not manifest_path.exists():
            (missing if directory.name in required else unused_without_manifest).append(directory.name)
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        hashed += sum(1 for f in manifest.get("files", []) if f.get("sha256"))

    detail = f"{hashed} files hashed; consumed datasets missing a manifest: {missing or 'none'}"
    if unused_without_manifest:
        detail += f"; present but not consumed by this pipeline: {unused_without_manifest}"
    return CheckResult(
        name="manifests_present",
        passed=not missing,
        severity="blocking",
        detail=detail,
        metrics={
            "files_hashed": hashed,
            "missing_manifests_for_consumed_datasets": missing,
            "present_but_unused": unused_without_manifest,
        },
    )


def check_demand_magnitude(demand: pd.DataFrame, min_daily: float) -> CheckResult:
    """10. Implausibly low station-days, which usually indicate a station closure.

    Advisory rather than blocking: the modelled line genuinely had extended closures, and
    treating a real closure as a data error would be the wrong correction.
    """
    daily = (
        demand.assign(date=demand["transit_timestamp"].dt.date)
        .groupby(["date", "station_complex_id"], observed=True)["ridership"]
        .sum()
    )
    low = daily[daily < min_daily]
    by_station = low.groupby("station_complex_id", observed=True).size().to_dict()
    return CheckResult(
        name="demand_magnitude",
        passed=True,
        severity="advisory",
        detail=f"{len(low)} station-days below {min_daily} total riders (possible closures)",
        metrics={"low_station_days": int(len(low)), "by_station": {str(k): int(v) for k, v in by_station.items()}},
    )


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------

def run_all_checks(config_path: str | Path) -> tuple[list[CheckResult], pd.DataFrame, pd.DataFrame]:
    """Run every check; return the results, the demand frame, and the exclusion list."""
    config = load_config(config_path)
    quality = config.get("quality")
    operating = quality["operating_day"]

    demand = load_hourly_demand(IN_REGIME_DATASETS)
    logger.info("loaded %d station-hour rows spanning %s .. %s (datasets: %s)",
                len(demand), demand["transit_timestamp"].min(), demand["transit_timestamp"].max(),
                ", ".join(IN_REGIME_DATASETS))

    continuity, excluded = check_temporal_continuity(
        demand, int(quality["max_missing_hours_per_day"]),
        int(operating["start_hour"]), int(operating["end_hour"]),
    )

    results = [
        check_required_fields(demand),
        check_duplicates(demand),
        check_timestamps(demand),
        check_non_negative(demand),
        check_outliers(demand, float(quality["outlier_mad_k"])),
        check_station_ids(demand, list(config.get("line.station_complex_ids"))),
        continuity,
        check_service_delivered_scaling(float(quality["service_delivered_max"])),
        check_manifests_present(IN_REGIME_DATASETS),
        check_demand_magnitude(demand, float(quality["min_daily_ridership"])),
    ]
    return results, demand, excluded


def write_report(results: list[CheckResult], demand: pd.DataFrame, excluded: pd.DataFrame) -> Path:
    """Write the human-readable and machine-readable data-quality reports."""
    ensure_dir(RESULTS_DIR)
    ensure_dir(INTERIM_DIR)
    if not excluded.empty:
        excluded.to_csv(INTERIM_DIR / "excluded_station_days.csv", index=False)

    blocking = [r for r in results if r.blocking_failure]
    lines = [
        "# Data Quality Report",
        "",
        "Generated by `python -m src.data.validate`. Every number here is computed, none typed.",
        "",
        f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Station-hour rows: {len(demand):,}",
        f"- Coverage: {demand['transit_timestamp'].min()} .. {demand['transit_timestamp'].max()}",
        f"- Station complexes: {demand['station_complex_id'].nunique()}",
        f"- **Blocking failures: {len(blocking)}**",
        "",
        "## Checks",
        "",
        "| # | Check | Severity | Result | Detail |",
        "|---|---|---|---|---|",
    ]
    for index, result in enumerate(results, start=1):
        status = "PASS" if result.passed else ("**FAIL**" if result.severity == "blocking" else "flagged")
        lines.append(f"| {index} | `{result.name}` | {result.severity} | {status} | {result.detail} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "Advisory checks flag and count anomalies; they never delete rows. Station-days with "
        "excessive missing operating hours are listed in `data/interim/excluded_station_days.csv` "
        "and excluded downstream, so the exclusion is auditable rather than implicit.",
        "",
        "The `service_delivered_scaling` check exists because the upstream `32ch-sei3` dataset "
        "publishes a mis-scaled ratio. The pipeline recomputes it from the underlying counts and "
        "never uses the published field.",
        "",
    ]

    report_path = RESULTS_DIR / "data_quality_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (RESULTS_DIR / "data_quality_report.json").write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "rows": int(len(demand)),
                "coverage": {
                    "first": str(demand["transit_timestamp"].min()),
                    "last": str(demand["transit_timestamp"].max()),
                },
                "checks": [asdict(r) for r in results],
            },
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    setup_logging(level=logging.INFO, log_file="data_validate.log")
    results, demand, excluded = run_all_checks(args.config)
    path = write_report(results, demand, excluded)

    for result in results:
        status = "PASS " if result.passed else ("FAIL " if result.severity == "blocking" else "FLAG ")
        print(f"{status} {result.name:32s} {result.detail}")

    blocking = [r for r in results if r.blocking_failure]
    print(f"\nreport: {path}")
    print(f"blocking failures: {len(blocking)}")
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())

