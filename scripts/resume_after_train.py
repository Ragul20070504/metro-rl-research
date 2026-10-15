"""Chain the remaining pipeline once the in-flight training stage exits.

The driver was stopped deliberately so that `ppo_uq_noavail` -- whose 10 policies failed under a
scaler bug now fixed -- is trained BEFORE the sweep runs. Letting the sweep start first would have
produced 1,200 cells that fail twice each and litter the confirmatory tree with FAILED markers for
a reason already understood and repaired.

Every step is resumable: completed policies and cells are skipped, never recomputed.

    .venv/Scripts/python.exe scripts/resume_after_train.py
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import json
import subprocess
import sys
import time
from datetime import datetime

from src.common.paths import MODELS_DIR, REPO_ROOT, RESULTS_DIR


def _train_marker() -> tuple[str, str] | None:
    """The stage marker `run_experiments.py` writes when its `main()` completes.

    Watching this rather than the process table: `wmic` has been removed from Windows 11 and
    returned silently empty, which made a process check report "finished" while training was still
    running. A completion artefact the stage itself writes cannot lie in that direction.
    """
    path = RESULTS_DIR / "last_run.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None  # mid-write; try again on the next poll
    return (",".join(payload.get("stages", [])), str(payload.get("finished_utc")))


def _wait_for_train(previous: tuple[str, str] | None) -> None:
    """Block until the marker changes to a completed `train` stage."""
    while True:
        marker = _train_marker()
        if marker is not None and marker != previous and marker[0] == "train":
            return
        time.sleep(30)


def _confirmatory_policies() -> int:
    root = MODELS_DIR / "policies"
    if not root.is_dir():
        return 0
    return sum(1 for p in root.rglob("policy.zip") if int(p.parent.name) < 900)


def _run(*args: str) -> int:
    command = [sys.executable, str(REPO_ROOT / "scripts" / args[0]), *args[1:]]
    print(f"[{datetime.now():%H:%M}] running {' '.join(args)}", flush=True)
    return subprocess.run(command, cwd=REPO_ROOT).returncode


def main() -> int:
    baseline = _train_marker()
    print(f"[{datetime.now():%H:%M}] waiting for the in-flight training stage "
          f"(marker now: {baseline})", flush=True)
    _wait_for_train(baseline)
    print(f"[{datetime.now():%H:%M}] training stage finished", flush=True)

    # Fresh interpreter: picks up the scaler projection fix, trains the 10 ppo_uq_noavail
    # policies, and skips the 40 that already exist.
    _run("run_experiments.py", "--stage", "train", "--train-seeds", "10")
    print(f"[{datetime.now():%H:%M}] confirmatory policies: "
          f"{_confirmatory_policies()}/50", flush=True)

    # Hand back to the driver for everything downstream.
    return _run("finish_pipeline.py", "--from", "sweep", "--skip-wait")


if __name__ == "__main__":
    raise SystemExit(main())
