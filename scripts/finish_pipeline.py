"""Carry the pipeline from the end of M4 through M16, unattended.

Waits for the running forecasting fit to finish, then executes every remaining stage in
dependency order. Each step is resumable, so re-running after an interruption picks up where it
stopped rather than repeating work.

**The R-1 power gate is honoured, not bypassed.** After the pilot, this script reads the decision
`scripts/power_analysis.py` reached and uses the sample size that decision names. It does not
choose a sample size itself, and it does not proceed past a pilot that failed to produce one.

    python scripts/finish_pipeline.py                 # wait for M4, then run everything
    python scripts/finish_pipeline.py --skip-wait     # M4 is already done
    python scripts/finish_pipeline.py --from sweep    # resume at a named stage
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.common.logging import get_logger
from src.common.paths import MODELS_DIR, REPO_ROOT, RESULTS_DIR, ensure_dir

logger = get_logger(__name__)

METRICS = RESULTS_DIR / "forecasting_metrics.json"
PROGRESS = RESULTS_DIR / "pipeline_progress.json"
POLL_SECONDS = 120
MAX_WAIT_SECONDS = 6 * 3600


def record(step: str, status: str, detail: str = "") -> None:
    ensure_dir(RESULTS_DIR)
    history = []
    if PROGRESS.exists():
        try:
            history = json.loads(PROGRESS.read_text(encoding="utf-8")).get("history", [])
        except json.JSONDecodeError:
            history = []
    history.append({"step": step, "status": status, "detail": detail,
                    "utc": datetime.now(timezone.utc).isoformat()})
    PROGRESS.write_text(json.dumps({"history": history}, indent=2), encoding="utf-8")
    logger.info("[%s] %s %s", status, step, detail)


def run(script: str, *args: str, required: bool = True) -> bool:
    """Run a script in a fresh interpreter so its module-level caches start clean.

    Each stage's console output is captured to its own file under ``logs/pipeline/``. An
    unattended run that fails at 04:00 is only diagnosable if its output was written down.
    """
    command = [sys.executable, str(REPO_ROOT / "scripts" / script), *args]
    stem = "_".join([script.removesuffix(".py"), *(a.lstrip("-") for a in args)])
    log_path = ensure_dir(REPO_ROOT / "logs" / "pipeline") / f"{stem}.log"

    logger.info("running %s %s (output -> %s)", script, " ".join(args), log_path)
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8", errors="replace") as sink:
        sink.write(f"\n===== {datetime.now(timezone.utc).isoformat()} {' '.join(command)} =====\n")
        sink.flush()
        result = subprocess.run(command, cwd=REPO_ROOT, stdout=sink, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started

    ok = result.returncode == 0
    record(f"{script} {' '.join(args)}".strip(),
           "OK" if ok else "FAILED", f"exit {result.returncode}, {elapsed:.0f}s, log {log_path.name}")
    if not ok and required:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
        raise SystemExit(f"{script} failed with exit code {result.returncode}\n{tail}")
    return ok


def wait_for_forecasting() -> None:
    """Block until the full fit has written a non-quick-mode manifest."""
    record("wait_for_m4", "START")
    deadline = time.time() + MAX_WAIT_SECONDS
    while time.time() < deadline:
        if METRICS.exists():
            try:
                manifest = json.loads(METRICS.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                time.sleep(5)  # a partial write; try again shortly
                continue
            if manifest.get("quick_mode") is False:
                record("wait_for_m4", "OK", "full-fit manifest present")
                return
        time.sleep(POLL_SECONDS)
    raise SystemExit("timed out waiting for the M4 full fit")


def amended_train_seeds() -> int:
    """The sample size the power analysis decided on. Never chosen here."""
    path = RESULTS_DIR / "power_analysis.json"
    if not path.exists():
        raise SystemExit(
            "the R-1 power gate produced no decision; the sweep must not proceed without one"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    n = int(payload.get("amended_train_seeds", 0))
    if n < 3:
        raise SystemExit(f"power analysis returned an unusable sample size ({n})")
    logger.info("power gate: %s -> S = %d", payload.get("decision"), n)
    record("power_gate", "OK", f"{payload.get('decision')}; S = {n}")
    return n


STAGES = ("validate_m4", "persist", "scaler", "prepare", "baselines", "pilot", "freeze",
          "train", "sweep", "ablations", "shap", "analyse", "figures",
          "traceability", "audit", "review")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-wait", action="store_true")
    parser.add_argument("--from", dest="start", choices=STAGES, default="validate_m4")
    args = parser.parse_args()

    start = STAGES.index(args.start)

    def active(stage: str) -> bool:
        return STAGES.index(stage) >= start

    if not args.skip_wait and start == 0:
        wait_for_forecasting()

    # --- M4 completion -------------------------------------------------------------------
    if active("validate_m4"):
        run("validate_forecasting_run.py")
    if active("persist"):
        run("persist_forecast_models.py")
    if active("scaler"):
        run("fit_scalers.py", "--what", "obs", "--days", "10")

    # --- M7-M9 preparation ---------------------------------------------------------------
    if active("prepare"):
        run("run_experiments.py", "--stage", "prepare")
    if active("baselines"):
        run("run_experiments.py", "--stage", "baselines")

    # --- R-1 gate ------------------------------------------------------------------------
    if active("pilot"):
        run("run_experiments.py", "--stage", "pilot")
    seeds = amended_train_seeds()

    # The pre-registration is frozen here: after the gate that fixes the sample size, and before
    # the first confirmatory run. `stage_sweep` refuses to start against an unfrozen registry.
    if active("freeze"):
        run("freeze_registry.py")

    # --- M10-M12 -------------------------------------------------------------------------
    if active("train"):
        run("run_experiments.py", "--stage", "train", "--train-seeds", str(seeds))
    if active("sweep"):
        run("run_experiments.py", "--stage", "sweep", "--train-seeds", str(seeds))
    if active("ablations"):
        run("run_experiments.py", "--stage", "ablations", "--train-seeds", str(seeds))

    # --- M13-M16 -------------------------------------------------------------------------
    if active("shap"):
        run("run_shap.py", required=False)
    if active("analyse"):
        run("analyse.py")
    if active("figures"):
        run("make_figures.py")
    if active("traceability"):
        run("traceability.py")
    if active("audit"):
        run("reproducibility_audit.py", required=False)
    if active("review"):
        run("peer_review.py")

    record("pipeline", "COMPLETE")
    print("pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
