"""Independent validation of the M4 full forecasting fit.

`src/forecasting/train.py` writes its own metrics manifest. This script does **not** trust it: it
re-derives every claim from the stored artefacts, so a bug in the training code cannot certify
itself. Where a check can only be answered by the manifest (hyperparameters, search trials), that
is stated in the output rather than presented as an independent confirmation.

Covers the twelve verification items the execution directive sets for M4, and preserves the
full-data result as an immutable reference artefact under ``results/reference/m4_full_fit/`` with
a SHA-256 manifest.

Writes ``results/forecasting_validation.json`` and ``docs/forecasting.md``.

Usage::

    python scripts/validate_forecasting_run.py
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import hashlib
import json
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.common.paths import PROCESSED_DIR, REPO_ROOT, RESULTS_DIR

logger = get_logger(__name__)

METRICS = RESULTS_DIR / "forecasting_metrics.json"
FORECAST = PROCESSED_DIR / "forecasts" / "forecast.parquet"
FORECAST_POINT = PROCESSED_DIR / "forecasts" / "forecast_point.parquet"
SPLIT_MANIFEST = PROCESSED_DIR / "split_manifest.json"
TRAIN_LOG = REPO_ROOT / "logs" / "forecasting_train.log"
REFERENCE_DIR = RESULTS_DIR / "reference" / "m4_full_fit"

#: Protocol section 4.6 freezes the stored forecast schema.
REQUIRED_FORECAST_COLUMNS = (
    "issue_time", "target_time", "horizon_h", "station_complex_id", "split",
    "actual", "p10", "p50", "p90", "mean", "sigma",
    "model_id", "model_version", "calibrated",
)

#: Files the forecasting run actually executes. Used to check that none of them changed while the
#: run was in flight, since the manifest records HEAD at *completion*, not at launch.
FORECASTING_CODE_PATHS = (
    "src/forecasting/", "src/data/features.py", "src/data/splits.py",
    "src/common/config.py", "src/common/paths.py", "src/common/manifest.py",
    "configs/forecasting.yaml",
)


@dataclass
class Check:
    """One verification item: what was asked, what was found, and whether it passes."""

    item: str
    passed: bool
    detail: str
    blocking: bool = True
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item, "passed": self.passed, "blocking": self.blocking,
            "detail": self.detail, "evidence": self.evidence,
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    """Run a git command, or return "" if git is unavailable.

    Uses the same executable lookup as `src.common.manifest`, which tolerates git being installed
    but off PATH -- the common Windows case. A bare `["git", ...]` silently returned "" here, and
    an empty string is indistinguishable from a legitimately empty answer, so every check built on
    it reported "unknown" while looking like it had asked.
    """
    from src.common.manifest import _git_executable

    executable = _git_executable()
    if executable is None:
        logger.warning("git is not available; provenance checks that need it cannot be performed")
        return ""
    try:
        return subprocess.check_output([executable, *args], cwd=REPO_ROOT, text=True).strip()
    except Exception as error:  # noqa: BLE001 - a failed git call is reported, not fatal
        logger.warning("git %s failed: %r", " ".join(args), error)
        return ""


# --------------------------------------------------------------------------- individual checks


def check_run_completed(manifest: dict[str, Any]) -> list[Check]:
    checks = [
        Check(
            "1. run completed successfully",
            bool(manifest) and "gate" in manifest and "metrics" in manifest,
            "metrics manifest present with a gate verdict and a metric block",
            evidence={"generated_utc": manifest.get("generated_utc")},
        ),
        Check(
            "1b. run was NOT in quick mode",
            manifest.get("quick_mode") is False,
            "quick mode truncates the splits; a quick-mode result is not reportable",
            evidence={"quick_mode": manifest.get("quick_mode")},
        ),
    ]
    rows = manifest.get("rows", {})
    manifest_rows = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8")).get("rows", {})
    # The split manifest prefixes its supervised counts; comparing on the wrong key names makes
    # `all(...)` vacuously true over an empty sequence, which is how this check first passed on a
    # quick-mode run whose splits were truncated to 60k/20k/20k.
    expected = {
        split: manifest_rows.get(f"supervised_{split}") for split in ("train", "val", "test")
    }
    resolvable = all(value is not None for value in expected.values())
    matches = resolvable and all(rows.get(split) == expected[split] for split in expected)
    checks.append(
        Check(
            "1c. full splits were used",
            bool(rows) and matches,
            (
                f"run used {rows}, split manifest declares {expected}"
                if resolvable
                else "could not resolve supervised_{train,val,test} from the split manifest"
            ),
            evidence={"run_rows": rows, "split_manifest_rows": expected},
        )
    )
    return checks


def check_no_nan_or_inf(forecasts: pd.DataFrame) -> list[Check]:
    numeric = ["actual", "mean", "p10", "p50", "p90", "sigma"]
    bad = {
        column: int((~np.isfinite(forecasts[column].to_numpy(dtype=float))).sum())
        for column in numeric
        if column in forecasts
    }
    total = sum(bad.values())
    return [
        Check(
            "2. no NaN or Inf silently produced",
            total == 0,
            f"{total} non-finite values across {len(numeric)} numeric columns of "
            f"{len(forecasts):,} rows",
            evidence={"non_finite_by_column": bad, "rows": int(len(forecasts))},
        )
    ]


def check_quantiles(forecasts: pd.DataFrame, manifest: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    present = [q for q in ("p10", "p50", "p90") if q in forecasts]
    checks.append(
        Check(
            "3. all expected forecast quantiles generated",
            len(present) == 3,
            f"stored quantile columns: {present}",
            evidence={"present": present},
        )
    )

    horizons = sorted(forecasts["horizon_h"].unique().tolist())
    checks.append(
        Check(
            "3b. all three horizons present",
            horizons == [1, 2, 3],
            f"horizons present: {horizons}",
            evidence={"horizons": horizons},
        )
    )

    if len(present) == 3:
        p10 = forecasts["p10"].to_numpy(dtype=float)
        p50 = forecasts["p50"].to_numpy(dtype=float)
        p90 = forecasts["p90"].to_numpy(dtype=float)
        crossings = int(((p10 > p50) | (p50 > p90)).sum())
        rate = crossings / max(1, len(forecasts))
        # Independent quantile regressions can cross; a small rate is a known property of the
        # method, not a bug. A large rate would make sigma meaningless.
        checks.append(
            Check(
                "3c. quantiles are monotone (p10 <= p50 <= p90)",
                rate < 0.01,
                f"{crossings:,} crossings in {len(forecasts):,} rows ({rate:.4%}); "
                "independent quantile regressions may cross, but a high rate would invalidate "
                "the sigma derived from the interval",
                blocking=False,
                evidence={"crossings": crossings, "rate": rate,
                          "run_reported_rate": manifest.get("quantile_crossing_rate")},
            )
        )
    return checks


def check_uncertainty_valid(forecasts: pd.DataFrame, manifest: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    sigma = forecasts["sigma"].to_numpy(dtype=float)
    negative = int((sigma < 0).sum())
    zero = int((sigma == 0).sum())
    checks.append(
        Check(
            "4. sigma is non-negative and finite",
            negative == 0 and np.isfinite(sigma).all(),
            f"{negative} negative, {zero:,} exactly zero, "
            f"median {np.median(sigma):.3f}, p99 {np.quantile(sigma, 0.99):.3f}",
            evidence={"negative": negative, "zero": zero,
                      "median": float(np.median(sigma)),
                      "p99": float(np.quantile(sigma, 0.99))},
        )
    )

    # sigma-hat = (q90 - q10) / (2 * z90), frozen in protocol section 4.3.
    z90 = 1.2816
    implied = (forecasts["p90"].to_numpy(dtype=float) - forecasts["p10"].to_numpy(dtype=float))
    implied = np.maximum(implied, 0.0) / (2.0 * z90)
    agreement = float(np.max(np.abs(implied - sigma))) if len(sigma) else 0.0
    checks.append(
        Check(
            "4b. sigma matches the frozen definition (q90 - q10) / (2 * 1.2816)",
            agreement < 1e-6,
            f"maximum absolute deviation {agreement:.3e}",
            evidence={"max_abs_deviation": agreement},
        )
    )

    val = forecasts[forecasts["split"] == "val"]
    if len(val):
        inside = (
            (val["actual"] >= val["p10"]) & (val["actual"] <= val["p90"])
        ).mean()
        gate = manifest.get("gate", {})
        low, high = gate.get("coverage_band", [0.76, 0.84])
        checks.append(
            Check(
                "4c. empirical 80% coverage on val lies in the pre-registered band",
                bool(low <= inside <= high),
                f"recomputed PICP80 = {inside:.4f}, band [{low}, {high}]; "
                f"run reported {gate.get('picp_80_val')}",
                evidence={"recomputed_picp_80": float(inside),
                          "run_reported": gate.get("picp_80_val"),
                          "band": [low, high]},
            )
        )
    return checks


def check_temporal_separation(forecasts: pd.DataFrame) -> list[Check]:
    checks: list[Check] = []
    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))

    ranges = {}
    for split in ("train", "val", "test"):
        subset = forecasts[forecasts["split"] == split]
        if len(subset):
            ranges[split] = (
                str(subset["target_time"].min()), str(subset["target_time"].max())
            )

    ordered = True
    if {"train", "val", "test"} <= set(ranges):
        ordered = ranges["train"][1] < ranges["val"][0] < ranges["val"][1] < ranges["test"][0]
    checks.append(
        Check(
            "5. temporal train/val/test separation holds in the stored forecasts",
            ordered,
            f"target-time ranges: {ranges}",
            evidence={"ranges": ranges},
        )
    )

    split = manifest.get("split", {})
    embargo_days = split.get("embargo_days")
    gaps = {}
    ok = True
    if {"train", "val", "test"} <= set(ranges):
        for earlier, later in (("train", "val"), ("val", "test")):
            gap = (pd.Timestamp(ranges[later][0]) - pd.Timestamp(ranges[earlier][1])).days
            gaps[f"{earlier}->{later}"] = gap
            if embargo_days is not None and gap < embargo_days:
                ok = False
    checks.append(
        Check(
            "6. the embargo gap is present between consecutive splits",
            ok,
            f"observed gaps in days: {gaps}; configured embargo: {embargo_days} days",
            evidence={"gaps_days": gaps, "embargo_days": embargo_days,
                      "split_manifest": split},
        )
    )
    return checks


def check_no_future_information() -> list[Check]:
    """Re-run the blocking leakage assertions rather than citing that they exist."""
    import pytest

    code = pytest.main([
        str(REPO_ROOT / "tests" / "test_leakage.py"),
        "-q", "--no-header", "-p", "no:cacheprovider",
    ])
    return [
        Check(
            "7. no future information entered the features",
            code == 0,
            "tests/test_leakage.py re-executed against this checkout; it injects a future spike "
            "and requires every feature to be unchanged, with a positive control that a past "
            "change does move them",
            evidence={"pytest_exit_code": int(code)},
        )
    ]


def check_provenance(manifest: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []

    commit = manifest.get("git_commit", "")
    checks.append(
        Check(
            "9. exact git commit recorded",
            bool(commit),
            f"commit {commit or 'MISSING'}, dirty={manifest.get('git_dirty')}",
            evidence={"git_commit": commit, "git_dirty": manifest.get("git_dirty")},
        )
    )

    # The manifest records HEAD at *completion*. Work continued on other milestones while the fit
    # was in flight, so HEAD moved. That is only a provenance problem if a file the run executes
    # changed underneath it -- check that directly rather than assuming.
    # Strongest evidence: the run hashed the contents of every file it executes, at launch and
    # again at completion. That covers uncommitted working-tree state, which a commit hash misses.
    launch_digest = manifest.get("code_path_digest_at_launch")
    final_digest = manifest.get("code_path_digest_at_completion")

    launch = _launch_commit(manifest)
    changed: list[str] = []
    if launch and commit and launch != commit:
        diff = git("diff", "--name-only", f"{launch}..{commit}").splitlines()
        changed = [
            path for path in diff
            if any(path.startswith(prefix) for prefix in FORECASTING_CODE_PATHS)
        ]

    if launch_digest and final_digest:
        passed = launch_digest == final_digest
        detail = (
            f"code-path digest {'unchanged' if passed else 'CHANGED'} during the run "
            f"({launch_digest[:12]} -> {final_digest[:12]})"
        )
        source = "digest"
    elif launch and commit:
        passed = not changed
        detail = (
            f"HEAD moved {launch[:12]} -> {commit[:12]} during the run; "
            f"{len(changed)} file(s) on the forecasting code path changed"
            + (f": {changed}. A commit-level difference is not by itself proof the run read "
               "different bytes -- an uncommitted change present at launch and committed later "
               "shows up here too. Check 9c settles the configuration case; anything else needs "
               "an explicit written verification." if changed else "")
        )
        source = "commit-diff"
    else:
        # The check cannot be performed. Reporting that as a pass would be the worst outcome:
        # a green line asserting something nobody established.
        passed = False
        detail = (
            "could not establish the launch state: the manifest records no code-path digest and "
            f"the launch commit could not be inferred (launch={launch or 'unknown'}, "
            f"completion={commit or 'unknown'}). This check asserts nothing."
        )
        source = "unavailable"

    checks.append(
        Check(
            "9b. no forecasting code changed while the run was in flight",
            passed,
            detail,
            blocking=False,
            evidence={"launch_commit": launch, "completion_commit": commit,
                      "changed_on_code_path": changed, "evidence_source": source,
                      "code_path_digest_at_launch": launch_digest,
                      "code_path_digest_at_completion": final_digest},
        )
    )

    # Would re-running today use the same configuration? The config hash answers that directly,
    # and is immune to the commit-versus-working-tree ambiguity above.
    recorded_hash = str(manifest.get("config_hash", ""))
    recomputed = _recompute_config_hash()
    checks.append(
        Check(
            "9c. the configuration on disk still hashes to what the run recorded",
            bool(recorded_hash) and recomputed == recorded_hash,
            (f"recorded {recorded_hash[:12]}, recomputed from configs/forecasting.yaml "
             f"{(recomputed or 'FAILED')[:12]}"
             + ("" if recomputed == recorded_hash else
                " -- the configuration has changed since the run, so re-running would not "
                "reproduce the stored result")),
            blocking=False,
            evidence={"recorded": recorded_hash, "recomputed": recomputed},
        )
    )

    seed = _configured_seed()
    checks.append(
        Check(
            "10. random seed recorded",
            seed is not None,
            f"random_state = {seed} (configs/forecasting.yaml); XGBoost is seeded from it and "
            "the run is otherwise deterministic given the frozen splits",
            evidence={"random_state": seed},
        )
    )

    runtime = _runtime_from_log()
    checks.append(
        Check(
            "11. runtime and hardware recorded",
            runtime is not None,
            f"wall clock {runtime['wall_clock_s'] / 3600:.2f} h ({runtime['start']} -> "
            f"{runtime['end']}) on {platform.platform()}" if runtime else
            "could not derive runtime from logs/forecasting_train.log",
            evidence={
                "runtime": runtime,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python": platform.python_version(),
                "packages": manifest.get("packages"),
            },
        )
    )

    checks.append(
        Check(
            "8. model configuration and metadata saved",
            bool(manifest.get("selected_hyperparameters")) and bool(manifest.get("config_hash")),
            f"selected hyperparameters, {_trial_count(manifest)} search trials across "
            f"{len(manifest.get('search_trials', {}))} searches, "
            f"config hash {manifest.get('config_hash', '')[:12]}, calibration record, "
            f"{len(manifest.get('feature_columns', []))} feature columns",
            evidence={
                "selected_hyperparameters": manifest.get("selected_hyperparameters"),
                "config_hash": manifest.get("config_hash"),
                "calibration": manifest.get("calibration"),
                "n_search_trials": _trial_count(manifest),
                "trials_per_search": {
                    name: len(trials)
                    for name, trials in (manifest.get("search_trials") or {}).items()
                },
            },
        )
    )
    return checks


def check_schema(forecasts: pd.DataFrame) -> list[Check]:
    missing = [c for c in REQUIRED_FORECAST_COLUMNS if c not in forecasts.columns]
    return [
        Check(
            "3d. stored forecast schema matches protocol section 4.6",
            not missing,
            f"missing columns: {missing}" if missing else "all required columns present",
            blocking=False,
            evidence={"missing": missing, "present": list(forecasts.columns)},
        )
    ]


def _trial_count(manifest: dict[str, Any]) -> int:
    """Total hyperparameter trials, summed over the per-model searches.

    `search_trials` is a mapping `{model_id: [trial, ...]}`, so `len()` on it counts *searches*,
    not trials -- which read as "2 search trials" for a 24-trial grid.
    """
    trials = manifest.get("search_trials") or {}
    if isinstance(trials, dict):
        return sum(len(v) for v in trials.values() if isinstance(v, list))
    return len(trials)


def _launch_commit(manifest: dict[str, Any] | None = None) -> str | None:
    """HEAD at the time the run started.

    Preferred source is the manifest itself: runs now record `git_commit_at_launch` directly.
    Older manifests predate that field, and the fallback infers it from the training log -- but
    the log is appended across runs, so the *last* start marker is the one that belongs to the
    completed run. Reading the first line instead attributes the run to whichever earlier
    invocation happened to write the top of the file.
    """
    if manifest and manifest.get("git_commit_at_launch"):
        return str(manifest["git_commit_at_launch"])
    if not TRAIN_LOG.exists():
        return None
    lines = [line for line in TRAIN_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
             if line.strip()]
    starts = [line for line in lines if "train=" in line or "selecting hyperparameters" in line]
    marker = starts[-1] if starts else (lines[0] if lines else None)
    stamp = marker.split()[0] if marker else None
    if not stamp:
        return None
    commit = git("rev-list", "-1", f"--before={stamp}", "HEAD")
    return commit or None


def _recompute_config_hash() -> str | None:
    """Hash the forecasting configuration as it stands on disk right now."""
    try:
        from src.common.config import config_hash, load_config

        config = load_config(REPO_ROOT / "configs" / "forecasting.yaml")
        return config_hash(config.values if hasattr(config, "values") else config)
    except Exception as error:  # noqa: BLE001 - an unhashable config is a finding, not a crash
        logger.warning("could not recompute the config hash: %r", error)
        return None


def _configured_seed() -> int | None:
    path = REPO_ROOT / "configs" / "forecasting.yaml"
    if not path.exists():
        return None
    match = re.search(r"random_state:\s*(\d+)", path.read_text(encoding="utf-8"))
    return int(match.group(1)) if match else None


def _runtime_from_log() -> dict[str, Any] | None:
    if not TRAIN_LOG.exists():
        return None
    lines = [line for line in TRAIN_LOG.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    try:
        start = datetime.fromisoformat(lines[0].split()[0])
        end = datetime.fromisoformat(lines[-1].split()[0])
    except ValueError:
        return None
    return {
        "start": start.isoformat(), "end": end.isoformat(),
        "wall_clock_s": (end - start).total_seconds(),
    }


def preserve_reference(manifest: dict[str, Any]) -> dict[str, Any]:
    """Item 12: copy the full-data result to an immutable reference location with hashes.

    Refuses to preserve a quick-mode run. A reference artefact built from truncated splits would
    be the worst possible outcome of this script: every later comparison would be against a
    result that was never reportable, and the file's existence would imply otherwise.
    """
    if manifest.get("quick_mode"):
        logger.error("refusing to preserve a quick-mode run as the reference artefact")
        return {"skipped": "quick_mode run; not a reportable reference", "files": {}}

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    sources = [METRICS, RESULTS_DIR / "forecasting_report.md", FORECAST, FORECAST_POINT,
               SPLIT_MANIFEST]
    entries = {}
    for source in sources:
        if not source.exists():
            continue
        target = REFERENCE_DIR / source.name
        shutil.copy2(source, target)
        entries[source.name] = {
            "sha256": sha256(target),
            "bytes": target.stat().st_size,
            "copied_from": str(source.relative_to(REPO_ROOT)),
        }
    record = {
        "purpose": (
            "Immutable reference copy of the M4 full-data fit. Any later forecasting run -- "
            "including one accelerated by subsampling the hyperparameter search -- is compared "
            "against this, and never replaces it."
        ),
        "preserved_utc": datetime.now(timezone.utc).isoformat(),
        "run_git_commit": manifest.get("git_commit"),
        "files": entries,
    }
    (REFERENCE_DIR / "REFERENCE_MANIFEST.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    return record


# --------------------------------------------------------------------------- report


def write_report(checks: list[Check], manifest: dict[str, Any], reference: dict[str, Any]) -> None:
    metrics = manifest.get("metrics", {})
    gate = manifest.get("gate", {})
    blocking_failures = [c for c in checks if c.blocking and not c.passed]

    lines = [
        "# Forecasting (M4)",
        "",
        f"Generated from `results/forecasting_metrics.json` and the stored forecast artefacts on "
        f"{datetime.now(timezone.utc).date().isoformat()}. Every claim below is re-derived from "
        "the artefacts by `scripts/validate_forecasting_run.py`; the training code does not "
        "certify itself.",
        "",
        f"**Validation: {len(checks) - len(blocking_failures)}/{len(checks)} checks pass"
        + (f", {len(blocking_failures)} BLOCKING FAILURE(S)**" if blocking_failures else "**"),
        "",
        "## Blocking gate (protocol section 4.5)",
        "",
        "RL training may not consume forecasts until `xgb_quantile` beats `naive_seasonal` on "
        "pinball loss on `val`, **and** post-calibration empirical 80 % coverage on `val` lies in "
        "the pre-registered band.",
        "",
        f"| criterion | value | verdict |",
        "|---|---:|---|",
        f"| pinball, `xgb_quantile` on val | {gate.get('quantile_pinball_val', float('nan')):.4f} | |",
        f"| pinball, `naive_seasonal` on val | {gate.get('naive_pinball_val', float('nan')):.4f} | |",
        f"| beats the reference floor | | {'**PASS**' if gate.get('beats_naive_on_pinball') else '**FAIL**'} |",
        f"| PICP80 on val (calibrated) | {gate.get('picp_80_val', float('nan')):.4f} | |",
        f"| pre-registered band | {gate.get('coverage_band')} | "
        f"{'**PASS**' if gate.get('coverage_within_band') else '**FAIL**'} |",
        f"| **gate** | | {'**PASSED**' if gate.get('passed') else '**FAILED**'} |",
        "",
    ]

    lines += ["## Verification checks", "", "| # | check | result | detail |", "|---|---|---|---|"]
    for check in checks:
        mark = "pass" if check.passed else ("**FAIL**" if check.blocking else "warn")
        lines.append(f"| {check.item} | | {mark} | {check.detail} |")
    lines.append("")

    if metrics:
        lines += ["## Metrics", ""]
        for split in ("val", "test"):
            rows = []
            for model_id, per_split in metrics.items():
                block = per_split.get(split, {})
                point = block.get("point", {})
                prob = block.get("probabilistic", {})
                if not point and not prob:
                    continue
                rows.append(
                    f"| `{model_id}` | {point.get('mae', float('nan')):.2f} | "
                    f"{point.get('rmse', float('nan')):.2f} | "
                    f"{point.get('mase', float('nan')):.4f} | "
                    f"{prob.get('pinball_mean', float('nan')):.4f} | "
                    f"{prob.get('picp_80', float('nan')):.4f} | "
                    f"{prob.get('picp_90', float('nan')):.4f} | "
                    f"{prob.get('mpiw', float('nan')):.2f} | "
                    f"{prob.get('crps', float('nan')):.4f} | "
                    f"{prob.get('ece', float('nan')):.4f} |"
                )
            if rows:
                lines += [
                    f"### `{split}`", "",
                    "| model | MAE | RMSE | MASE | pinball | PICP80 | PICP90 | MPIW | CRPS | ECE |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                    *rows, "",
                ]

    lines += [
        "## Selected hyperparameters",
        "",
        "```json",
        json.dumps(manifest.get("selected_hyperparameters", {}), indent=2),
        "```",
        "",
        "Selected on `val` by RMSE for `xgb_point` and mean pinball loss for `xgb_quantile`, over "
        f"the same frozen grid for both — {len(manifest.get('search_trials', []))} trials in total "
        "(protocol section 4.3: identical search budget).",
        "",
        "## Calibration",
        "",
        "```json",
        json.dumps(manifest.get("calibration", {}), indent=2),
        "```",
        "",
        "A single scalar interval-width multiplier, fitted on `val` by minimising the absolute "
        "deviation of empirical 80 % coverage from 0.80, applied unchanged to `test`.",
        "",
        "## Immutable reference artefact",
        "",
        "The full-data result is preserved under `results/reference/m4_full_fit/` with SHA-256 "
        "digests. Any later run — including one accelerated by subsampling the hyperparameter "
        "search — is compared against it and never replaces it.",
        "",
        "| file | bytes | sha256 |",
        "|---|---:|---|",
    ]
    for name, entry in reference.get("files", {}).items():
        lines.append(f"| `{name}` | {entry['bytes']:,} | `{entry['sha256'][:16]}…` |")
    lines.append("")

    lines += [
        "## What this does NOT establish",
        "",
        "- Forecasting quality and downstream scheduling usefulness are **separate questions**. "
        "Nothing here is evidence for or against H1.",
        "- The gate is a *precondition* for testing H1, not a partial result about it: it "
        "establishes only that the uncertainty channel is calibrated enough to be worth feeding "
        "to a policy.",
        "- Point accuracy alone is insufficient (protocol section 4.4). If `xgb_point` does not "
        "beat `naive_seasonal` on MASE, that is a **finding** and is reported as one.",
        "",
    ]

    (REPO_ROOT / "docs" / "forecasting.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not METRICS.exists():
        raise SystemExit(f"{METRICS} not found — has the full fit finished?")
    manifest = json.loads(METRICS.read_text(encoding="utf-8"))
    forecasts = pd.read_parquet(FORECAST)

    checks: list[Check] = []
    checks += check_run_completed(manifest)
    checks += check_no_nan_or_inf(forecasts)
    checks += check_quantiles(forecasts, manifest)
    checks += check_schema(forecasts)
    checks += check_uncertainty_valid(forecasts, manifest)
    checks += check_temporal_separation(forecasts)
    checks += check_no_future_information()
    checks += check_provenance(manifest)

    reference = preserve_reference(manifest)

    payload = {
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "checks": [c.to_dict() for c in checks],
        "n_passed": sum(1 for c in checks if c.passed),
        "n_checks": len(checks),
        "blocking_failures": [c.item for c in checks if c.blocking and not c.passed],
        "gate": manifest.get("gate"),
        "reference_artifact": reference,
    }
    (RESULTS_DIR / "forecasting_validation.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    write_report(checks, manifest, reference)

    for check in checks:
        status = "PASS" if check.passed else ("FAIL" if check.blocking else "WARN")
        print(f"{status:5} {check.item}: {check.detail}")
    print(f"\n{payload['n_passed']}/{payload['n_checks']} checks passed")
    if payload["blocking_failures"]:
        print(f"BLOCKING FAILURES: {payload['blocking_failures']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
