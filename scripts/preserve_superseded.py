"""Preserve results produced under a superseded configuration, as inspectable artefacts.

Amendments A-10 and A-11 corrected two baselines that protocol section 7.1 would have rejected as
strawmen. The cells produced under the uncorrected models were deleted from `results/runs/` so
they could never be mixed with the corrected ones -- but deleting is not the same as discarding,
and a reader checking that the correction mattered should not have to reconstruct the old numbers
from git plumbing.

This recovers them from the commit that last held them and writes them to
`results/superseded/<label>/`, with a manifest naming the amendment, the commit and the reason.
Nothing under `results/runs/` is touched, and the superseded tree is never read by the analysis.

    python scripts/preserve_superseded.py --commit add7f26 --label pre_A10_A11
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.common.logging import get_logger
from src.common.paths import REPO_ROOT, RESULTS_DIR, ensure_dir

logger = get_logger(__name__)

SUPERSEDED_DIR = RESULTS_DIR / "superseded"


def _git(*args: str) -> str:
    from src.common.manifest import _git_executable

    executable = _git_executable()
    if executable is None:
        raise SystemExit("git is unavailable, so superseded results cannot be recovered")
    return subprocess.check_output([executable, *args], cwd=REPO_ROOT, text=True)


#: What was measured under the superseded configuration, recorded when the defects were found.
#: Mean `left_behind_fraction` over 30 evaluation seeds per (method, scenario).
SUPERSEDED_MEASUREMENTS = {
    "note": (
        "Per-cell result files are NOT available: they were deleted when the defects were found, "
        "before results/runs was first committed, so git holds no copy. These aggregates are the "
        "measurements that were recorded at the time and that justified amendments A-10 and A-11."
    ),
    "metric": "left_behind_fraction, mean over 30 evaluation seeds",
    "by_method_and_scenario": {
        "fixed_timetable": {"normal": 0.0188, "surge": 0.0255,
                            "disruption": 0.0221, "combined": 0.0286},
        "heuristic_threshold": {"normal": 0.0053, "surge": 0.0121,
                                "disruption": 0.0103, "combined": 0.0179},
        "milp_static": {"normal": 0.0567, "surge": 0.0737,
                        "disruption": 0.0609, "combined": 0.0785},
        "milp_rolling": {"normal": 0.0224, "surge": 0.0282, "disruption": 0.0258,
                         "combined": "incomplete when the defect was found"},
    },
    "heuristic_configuration": {"queue_threshold": 800.0, "forecast_threshold": 4000.0,
                                "defect": "selected value was the maximum of its search grid"},
    "milp_diagnostic": {
        "surrogate_mean_predicted_queue": 0.5,
        "simulator_mean_realised_queue": 1017.9,
        "surrogate_preferred": "MILP plan",
        "simulator_preferred": "fixed timetable",
        "defect": "surrogate and simulator ranked the same two schedules oppositely",
    },
    "fixed_timetable_note": (
        "fixed_timetable was unaffected by both amendments; its cells were neither deleted nor "
        "re-run, and its numbers above are the same ones in results/runs today"
    ),
}


def _write_unrecoverable_manifest(target: Path, args) -> None:
    (target / "SUPERSEDED_MANIFEST.json").write_text(json.dumps({
        "label": args.label,
        "status": "SUPERSEDED -- not evidence for any claim",
        "preserved_utc": datetime.now(timezone.utc).isoformat(),
        "per_cell_artefacts": "NOT RECOVERABLE",
        "why_not_recoverable": (
            f"the affected cells were deleted before results/runs was first committed, so "
            f"commit {args.commit.strip()} and every earlier commit hold no copy of them"
        ),
        "amendments": args.amendments,
        "reason": args.reason,
        "recorded_measurements": SUPERSEDED_MEASUREMENTS,
        "warning": (
            "These results were produced under a configuration later found to violate protocol "
            "section 7.1's prohibition on strawman baselines. They are recorded so the effect of "
            "the correction can be inspected, NOT so they can be reported. Nothing under "
            "results/runs/ derives from them, and scripts/analyse.py never reads this tree."
        ),
    }, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True, help="commit that still holds the old results")
    parser.add_argument("--label", required=True, help="directory name under results/superseded/")
    parser.add_argument("--methods", nargs="+",
                        default=["heuristic_threshold", "milp_static", "milp_rolling"])
    parser.add_argument("--amendments", nargs="+", default=["A-10", "A-11"])
    parser.add_argument("--reason", default=(
        "produced under baseline configurations that amendments A-10 and A-11 corrected: the "
        "heuristic's tuned threshold sat at its grid boundary, and the MILP surrogate lacked a "
        "standing-queue term and therefore ranked schedules opposite to the simulator"))
    args = parser.parse_args()

    target = ensure_dir(SUPERSEDED_DIR / args.label)
    if any(target.iterdir()):
        logger.info("%s already populated; nothing to do", target)
        print(f"{target} already exists; leaving it untouched")
        return 0

    listing = _git("ls-tree", "-r", "--name-only", args.commit, "results/runs/").splitlines()
    wanted = [p for p in listing
              if any(f"/{method}/" in p for method in args.methods) and p.endswith(".json")]
    if not wanted:
        # The per-cell files were deleted before they were ever committed, so there is nothing to
        # recover. Recording that plainly is the honest outcome: a manifest that quietly held
        # nothing would read as though the artefacts had been preserved.
        _write_unrecoverable_manifest(target, args)
        print(f"no per-cell artefacts recoverable from {args.commit}; "
              f"wrote an explicit record to {target.relative_to(REPO_ROOT)}")
        return 0

    recovered = 0
    for path in wanted:
        content = _git("show", f"{args.commit}:{path}")
        destination = target / Path(path).relative_to("results/runs")
        ensure_dir(destination.parent)
        destination.write_text(content, encoding="utf-8")
        recovered += 1

    counts: dict[str, int] = {}
    for path in wanted:
        if path.endswith("metrics.json"):
            for method in args.methods:
                if f"/{method}/" in path:
                    counts[method] = counts.get(method, 0) + 1

    (target / "SUPERSEDED_MANIFEST.json").write_text(json.dumps({
        "label": args.label,
        "status": "SUPERSEDED -- not evidence for any claim",
        "preserved_utc": datetime.now(timezone.utc).isoformat(),
        "recovered_from_commit": args.commit.strip(),
        "amendments": args.amendments,
        "reason": args.reason,
        "cells_by_method": counts,
        "files_recovered": recovered,
        "warning": (
            "These results were produced under a configuration later found to violate protocol "
            "section 7.1's prohibition on strawman baselines. They are preserved so the effect of "
            "the correction can be inspected, NOT so they can be reported. Nothing under "
            "results/runs/ was taken from here, and scripts/analyse.py never reads this tree."
        ),
    }, indent=2), encoding="utf-8")

    logger.info("preserved %d file(s) from %s into %s", recovered, args.commit, target)
    print(f"preserved {recovered} file(s) -> {target.relative_to(REPO_ROOT)}")
    for method, count in sorted(counts.items()):
        print(f"  {method}: {count} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
