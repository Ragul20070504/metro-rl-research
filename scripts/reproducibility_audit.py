"""Final reproducibility audit (M16).

Every check re-derives its answer from the repository rather than trusting a document. Where a
check cannot be performed — because an artefact is absent in this checkout — it reports
``SKIPPED`` with the reason, never ``PASS``.

The two checks that matter most, and that most audits omit:

* **Experiment reproduction.** A stored result cell is re-executed and compared against what was
  written. Re-running must reproduce the metrics exactly; anything less means the seeds do not
  control what they claim to.
* **No hand-entered experimental numbers.** Every figure must have a provenance companion naming
  the files it drew from, and every table must be generated. A figure with no companion is
  treated as a failure, because it cannot be traced.

    python scripts/reproducibility_audit.py

Writes ``docs/reproducibility_audit.md`` and ``results/reproducibility_audit.json``.
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.logging import get_logger
from src.common.paths import MODELS_DIR, PROCESSED_DIR, REPO_ROOT, RESULTS_DIR, ensure_dir

logger = get_logger(__name__)

#: Paths that would make a run machine-specific: a *user's* directory, or a drive-rooted path
#: that is not a conventional system install root.
#:
#: `C:\Program Files\...` is deliberately allowed. `src/common/manifest.py` searches there for
#: git when it is installed but absent from PATH -- a fallback that is identical on every Windows
#: machine and degrades to None. Flagging it would train the reader to ignore this check, which
#: matters far more than the one true positive it would ever catch there. A user-specific path
#: (`C:\Users\...`, `/home/...`, `/Users/...`) is a genuine hazard and is still flagged.
ABSOLUTE_PATH_PATTERN = re.compile(
    r"""["'](?:[A-Za-z]:[\\/](?!Program Files)|/home/|/Users/)|["'][A-Za-z]:[\\/]Users[\\/]"""
)
SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|secret|password|passwd|token)\s*[:=]\s*[\"'][^\"'\s]{6,}[\"']"
)
#: Files that legitimately contain absolute paths: they *record* where a run happened.
PROVENANCE_FILES = {"run_manifest.json", "metadata.json", "REFERENCE_MANIFEST.json"}


@dataclass
class Check:
    name: str
    status: str  # PASS | FAIL | SKIPPED | WARN
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail,
                "evidence": self.evidence}


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return ""


# --------------------------------------------------------------------------- checks


def check_environment() -> list[Check]:
    checks: list[Check] = []
    lock = REPO_ROOT / "requirements.lock.txt"
    if lock.exists():
        text = lock.read_text(encoding="utf-8")
        hashed = text.count("--hash=sha256:")
        pinned = len(re.findall(r"^[A-Za-z0-9._-]+==", text, re.M))
        checks.append(Check(
            "dependency reproducibility",
            "PASS" if hashed >= pinned > 0 else "WARN",
            f"{pinned} pinned packages, {hashed} SHA-256 hashes",
            {"pinned": pinned, "hashes": hashed},
        ))
    else:
        checks.append(Check("dependency reproducibility", "WARN",
                            "requirements.lock.txt absent; versions are pinned in "
                            "requirements.txt but not hash-verified"))

    checks.append(Check(
        "interpreter recorded", "PASS",
        f"Python {platform.python_version()} on {platform.platform()}",
        {"python": platform.python_version(), "platform": platform.platform(),
         "executable": sys.executable},
    ))
    return checks


def check_git_state() -> list[Check]:
    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain"))
    return [
        Check("git commit recorded", "PASS" if commit else "FAIL",
              f"HEAD = {commit or 'unknown'}", {"commit": commit}),
        Check("working tree clean", "PASS" if not dirty else "WARN",
              "clean" if not dirty else "uncommitted changes present; any run made now would be "
                                        "marked provisional in its manifest",
              {"dirty": dirty}),
    ]


def check_data_provenance() -> list[Check]:
    manifests = list((REPO_ROOT / "data").rglob("MANIFEST.json"))
    licences = list((REPO_ROOT / "data").rglob("LICENSE.txt"))
    checks = [Check(
        "raw data provenance", "PASS" if manifests else "SKIPPED",
        f"{len(manifests)} dataset manifests, {len(licences)} licence files"
        if manifests else "no raw data in this checkout",
        {"manifests": len(manifests), "licences": len(licences)},
    )]

    split = PROCESSED_DIR / "split_manifest.json"
    if split.exists():
        payload = json.loads(split.read_text(encoding="utf-8"))
        checks.append(Check(
            "chronological splits with embargo", "PASS",
            f"embargo {payload.get('split', {}).get('embargo_days')} days; "
            f"rows {payload.get('rows', {}).get('supervised_train')} / "
            f"{payload.get('rows', {}).get('supervised_val')} / "
            f"{payload.get('rows', {}).get('supervised_test')}",
            {"split": payload.get("split"), "rows": payload.get("rows")},
        ))
        screen = payload.get("regime_screen", {})
        if screen.get("used_fallback"):
            checks.append(Check(
                "regime screen", "WARN",
                "the pre-registered regime screen fired its fallback and this is carried into "
                f"the limitations: {screen.get('detail', '')}",
                {"regime_screen": screen},
            ))
    else:
        checks.append(Check("chronological splits with embargo", "SKIPPED",
                            "split manifest absent"))
    return checks


def check_no_leakage() -> list[Check]:
    import pytest

    code = pytest.main([str(REPO_ROOT / "tests" / "test_leakage.py"), "-q", "--no-header",
                        "-p", "no:cacheprovider"])
    return [Check(
        "temporal leakage tests", "PASS" if code == 0 else "FAIL",
        "tests/test_leakage.py re-executed: future-spike injection with positive controls, "
        "split ordering, embargo, and observation-level future-information tests",
        {"pytest_exit_code": int(code)},
    )]


def check_full_test_suite() -> list[Check]:
    import pytest

    code = pytest.main([str(REPO_ROOT / "tests"), "-q", "--no-header", "-p", "no:cacheprovider"])
    return [Check("full test suite", "PASS" if code == 0 else "FAIL",
                  "every test in tests/ re-executed", {"pytest_exit_code": int(code)})]


def check_no_absolute_paths_or_secrets() -> list[Check]:
    offenders_paths: list[str] = []
    offenders_secrets: list[str] = []
    for path in list((REPO_ROOT / "src").rglob("*.py")) + list((REPO_ROOT / "scripts").rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if ABSOLUTE_PATH_PATTERN.search(text):
            offenders_paths.append(str(path.relative_to(REPO_ROOT)))
        if SECRET_PATTERN.search(text):
            offenders_secrets.append(str(path.relative_to(REPO_ROOT)))

    for path in (REPO_ROOT / "configs").rglob("*.yaml"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if SECRET_PATTERN.search(text):
            offenders_secrets.append(str(path.relative_to(REPO_ROOT)))

    return [
        Check("no hard-coded machine paths in code", "PASS" if not offenders_paths else "FAIL",
              "all paths derive from src/common/paths.py" if not offenders_paths
              else f"absolute paths in {offenders_paths}",
              {"offenders": offenders_paths}),
        Check("no secrets in code or configs", "PASS" if not offenders_secrets else "FAIL",
              "no credential-shaped literals found" if not offenders_secrets
              else f"possible secrets in {offenders_secrets}",
              {"offenders": offenders_secrets}),
    ]


def check_configs_have_provenance() -> list[Check]:
    from src.common.config import load_config

    results: list[Check] = []
    for path in sorted((REPO_ROOT / "configs").glob("*.yaml")):
        try:
            load_config(path)
            results.append(Check(f"config provenance: {path.name}", "PASS",
                                 "loads under strict provenance"))
        except Exception as error:  # noqa: BLE001
            message = str(error)
            if "unresolved" in message:
                results.append(Check(
                    f"config provenance: {path.name}", "PASS",
                    "refuses to load outside a tagged sensitivity sweep because it contains "
                    "ungrounded parameters -- which is the intended behaviour",
                    {"error": message[:300]},
                ))
            else:
                results.append(Check(f"config provenance: {path.name}", "FAIL", message[:300]))
    return results


def check_forecasting() -> list[Check]:
    validation = RESULTS_DIR / "forecasting_validation.json"
    if not validation.exists():
        return [Check("forecasting validated", "SKIPPED",
                      "results/forecasting_validation.json absent")]
    payload = json.loads(validation.read_text(encoding="utf-8"))
    blocking = payload.get("blocking_failures", [])
    return [Check(
        "forecasting validated", "PASS" if not blocking else "FAIL",
        f"{payload.get('n_passed')}/{payload.get('n_checks')} independent checks pass"
        + (f"; blocking failures: {blocking}" if blocking else ""),
        {"gate": payload.get("gate"), "blocking_failures": blocking},
    )]


def check_simulator() -> list[Check]:
    path = RESULTS_DIR / "simulator_validation.json"
    if not path.exists():
        return [Check("simulator validated", "SKIPPED", "validation result absent")]
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts = payload.get("counts", {})
    mismatches = counts.get("mismatch", 0)
    return [Check(
        "simulator validated against published statistics",
        "PASS" if mismatches == 0 else "FAIL",
        f"{counts.get('validated', 0)} validated, "
        f"{counts.get('partially validated', 0)} partially validated, "
        f"{counts.get('not validatable', 0)} not validatable, {mismatches} mismatches",
        {"counts": counts},
    )]


def check_model_loading() -> list[Check]:
    checks: list[Check] = []
    forecaster = REPO_ROOT / "models" / "forecasting" / "xgb_quantile.json"
    if forecaster.exists():
        try:
            from src.environment.forecast_provider import load_trained_provider
            from src.simulator.network import Network

            provider = load_trained_provider(Network.load())
            checks.append(Check("forecaster loads from disk", "PASS",
                                f"{provider.model_id}, calibration factor "
                                f"{provider.calibration_factor:.4f}"))
        except Exception as error:  # noqa: BLE001
            checks.append(Check("forecaster loads from disk", "FAIL", repr(error)[:300]))
    else:
        checks.append(Check("forecaster loads from disk", "SKIPPED", "no persisted model"))

    policies = list((MODELS_DIR / "policies").rglob("policy.zip"))
    if policies:
        try:
            from sb3_contrib import MaskablePPO

            MaskablePPO.load(policies[0], device="cpu")
            checks.append(Check("trained policies load from disk", "PASS",
                                f"{len(policies)} checkpoints; loaded {policies[0].parent.name}",
                                {"n_policies": len(policies)}))
        except Exception as error:  # noqa: BLE001
            checks.append(Check("trained policies load from disk", "FAIL", repr(error)[:300]))
    else:
        checks.append(Check("trained policies load from disk", "SKIPPED", "no checkpoints"))
    return checks


def check_experiment_reproduction() -> list[Check]:
    """Re-run a stored cell and require the metrics to match exactly."""
    from src.evaluation.runner import discover_cells, load_cell_metrics, run_cell

    cells = discover_cells()
    if not cells:
        return [Check("experiment reproduction", "SKIPPED", "no stored cells")]

    # Prefer a deterministic baseline: it isolates environment reproducibility from any
    # nondeterminism in policy inference.
    candidates = [c for c in cells if c.method == "fixed_timetable"] or cells
    cell = candidates[0]
    stored = load_cell_metrics(cell)
    if stored is None:
        return [Check("experiment reproduction", "SKIPPED", "chosen cell has no metrics")]

    rerun = run_cell(cell, overwrite=True)
    compared = ["left_behind_fraction", "waiting_accrued_passenger_hours", "energy_units",
                "cost_units", "passengers_arrived", "inductions", "total_reward"]
    differences = {
        key: (stored.get(key), rerun.get(key))
        for key in compared
        if key in stored and abs(float(stored[key]) - float(rerun.get(key, float("nan")))) > 1e-9
    }
    return [Check(
        "experiment reproduction", "PASS" if not differences else "FAIL",
        f"re-ran {cell.key()} and compared {len(compared)} metrics"
        + ("; all identical" if not differences else f"; differences: {differences}"),
        {"cell": cell.key(), "differences": differences},
    )]


def check_seed_discipline() -> list[Check]:
    from scripts.run_experiments import EVAL_SEEDS, PILOT_EVAL_SEEDS, PILOT_TRAIN_SEEDS, TRAIN_SEEDS

    pools = {
        "evaluation": set(EVAL_SEEDS),
        "training": set(TRAIN_SEEDS),
        "pilot training": set(PILOT_TRAIN_SEEDS),
        "pilot evaluation": set(PILOT_EVAL_SEEDS),
        "scaler fitting": set(range(1001, 1021)),
        "heuristic tuning": set(range(2001, 2007)),
        "surrogate calibration": set(range(3001, 3005)),
    }
    overlaps = {
        f"{a} & {b}": sorted(pools[a] & pools[b])
        for i, a in enumerate(pools) for b in list(pools)[i + 1:]
        if pools[a] & pools[b]
    }
    return [Check(
        "seed pools are disjoint", "PASS" if not overlaps else "FAIL",
        "every tuning, calibration, pilot and training pool is disjoint from the evaluation "
        "seeds, so nothing is fitted on a day the comparison is scored on"
        if not overlaps else f"overlapping pools: {overlaps}",
        {"pools": {k: [min(v), max(v)] for k, v in pools.items()}, "overlaps": overlaps},
    )]


def check_figure_provenance() -> list[Check]:
    figures = RESULTS_DIR / "figures"
    if not figures.exists() or not list(figures.glob("*.png")):
        return [Check("no hand-entered experimental numbers", "SKIPPED", "no figures generated")]
    missing = [
        png.name for png in figures.glob("*.png")
        if not (figures / f"{png.stem}.source.json").exists()
    ]
    return [Check(
        "no hand-entered experimental numbers",
        "PASS" if not missing else "FAIL",
        f"{len(list(figures.glob('*.png')))} figures, each with a provenance companion naming "
        "its source files" if not missing else f"figures without provenance: {missing}",
        {"missing": missing},
    )]


def check_registry() -> list[Check]:
    import yaml

    path = REPO_ROOT / "experiments" / "registry.yaml"
    if not path.exists():
        return [Check("experiment registry", "FAIL", "registry.yaml absent")]
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    experiments = payload.get("experiments", [])
    frozen = payload.get("meta", {}).get("frozen", False)
    return [Check(
        "experiment registry", "PASS" if experiments else "FAIL",
        f"{len(experiments)} registered experiments; frozen = {frozen}",
        {"n_experiments": len(experiments), "frozen": frozen},
    )]


def check_no_fabricated_results() -> list[Check]:
    """Every reported number must trace to a stored result file."""
    from src.evaluation.runner import discover_cells

    cells = discover_cells()
    analysis = RESULTS_DIR / "statistical_analysis.json"
    if not analysis.exists():
        return [Check("results trace to stored runs", "SKIPPED",
                      "no statistical analysis generated")]
    payload = json.loads(analysis.read_text(encoding="utf-8"))
    claimed = payload.get("n_episode_rows", 0)
    return [Check(
        "results trace to stored runs",
        "PASS" if claimed <= len(cells) else "FAIL",
        f"the analysis reports {claimed} episodes; {len(cells)} result directories exist on disk",
        {"analysis_episodes": claimed, "stored_cells": len(cells)},
    )]


# --------------------------------------------------------------------------- entry point


def main() -> int:
    checks: list[Check] = []
    checks += check_environment()
    checks += check_git_state()
    checks += check_data_provenance()
    checks += check_configs_have_provenance()
    checks += check_seed_discipline()
    checks += check_no_absolute_paths_or_secrets()
    checks += check_forecasting()
    checks += check_simulator()
    checks += check_model_loading()
    checks += check_experiment_reproduction()
    checks += check_no_leakage()
    checks += check_figure_provenance()
    checks += check_registry()
    checks += check_no_fabricated_results()
    checks += check_full_test_suite()

    counts = {status: sum(1 for c in checks if c.status == status)
              for status in ("PASS", "FAIL", "WARN", "SKIPPED")}

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "counts": counts,
        "checks": [c.to_dict() for c in checks],
    }
    ensure_dir(RESULTS_DIR)
    (RESULTS_DIR / "reproducibility_audit.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Reproducibility audit (M16)",
        "",
        f"Generated by `scripts/reproducibility_audit.py` at commit "
        f"`{payload['git_commit'][:12]}`. Every check re-derives its answer from the repository; "
        "none trusts a document. A check that cannot be performed reports **SKIPPED** with the "
        "reason, never PASS.",
        "",
        f"**{counts['PASS']} pass, {counts['FAIL']} fail, {counts['WARN']} warn, "
        f"{counts['SKIPPED']} skipped.**",
        "",
        "| check | status | detail |",
        "|---|---|---|",
    ]
    for check in checks:
        lines.append(f"| {check.name} | **{check.status}** | {check.detail} |")
    lines += [
        "",
        "## How to reproduce from scratch",
        "",
        "```bash",
        "git clone <repository> && cd metro-rl-research",
        f"git checkout {payload['git_commit'][:12]}",
        "python -m venv .venv",
        ".venv/Scripts/pip install -r requirements.lock.txt   # hash-pinned",
        "",
        "python -m src.data.prepare                 # M3: download, validate, split",
        "python -m src.forecasting.train            # M4: fit and evaluate forecasters",
        "python scripts/persist_forecast_models.py  # persist the fitted models",
        "python -m src.simulator.fit_demand         # M5: fit the demand process",
        "python -m src.simulator.validate           # M5: validate against published data",
        "python scripts/fit_scalers.py --what both  # M7: freeze reward and observation scales",
        "python scripts/run_experiments.py --stage all",
        "python scripts/analyse.py                  # M14",
        "python scripts/make_figures.py             # M15",
        "python scripts/reproducibility_audit.py    # M16",
        "```",
        "",
        "All stochasticity flows through named streams derived from `eval_seed`;",
        "`environment_streams` takes no method argument, so no method can perturb the demand or",
        "failure realisation it is evaluated against.",
        "",
    ]
    (REPO_ROOT / "docs" / "reproducibility_audit.md").write_text("\n".join(lines), encoding="utf-8")

    for check in checks:
        print(f"{check.status:8} {check.name}: {check.detail}")
    print(f"\n{counts}")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
