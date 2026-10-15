"""Re-tune the threshold heuristic under amendment A-10, and record whether it is interior.

The first tuning run selected the largest queue threshold in its grid, which is evidence the
optimum lies outside it -- an under-tuned heuristic, and exactly the strawman baseline protocol
section 7.1 forbids. `tune_heuristic` now extends the grid upward by doubling until the selection
is strictly interior, under a stopping rule fixed in advance (see A-10).

This script runs that, refuses to save a result that is still at a boundary without saying so, and
prints the extension history so the decision is visible rather than buried in a JSON blob.

    python scripts/tune_heuristic.py
    python scripts/tune_heuristic.py --workers 6
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import argparse

from src.common.logging import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=0, help="0 = one per core, less one")
    args = parser.parse_args()

    setup_logging(log_file="tune_heuristic.log")

    from src.baselines.heuristic import tune_heuristic

    parameters = tune_heuristic(workers=args.workers or None)
    provenance = parameters.provenance
    interior = provenance["selection_is_interior"]

    print("\n=== extension history (amendment A-10) ===")
    for row in provenance["extension_history"]:
        print(f"  round {row['round']}: grid max {row['grid_max']:.0f}, "
              f"selected {row['selected_queue_threshold']:.0f}, "
              f"interior={row['selected_is_interior']}, "
              f"peak observed queue {row['peak_observed_queue']:.0f}, "
              f"next would be {row['next_threshold_would_be']:.0f} "
              f"(can never fire: {row['next_threshold_can_never_fire']})")

    print(f"\nselected: queue {parameters.queue_threshold:.0f}, "
          f"forecast {parameters.forecast_threshold:.0f}, "
          f"mean reward {provenance['selected_mean_total_reward']:.1f}")
    print(f"queue selection interior:    {interior['queue']}")
    print(f"forecast selection interior: {interior['forecast']}")
    print(f"grid searched: queue {provenance['grid']['queue']}, "
          f"forecast {provenance['grid']['forecast']} "
          f"({len(provenance['trials'])} points)")

    if not interior["queue"]:
        final = provenance["extension_history"][-1]
        if final["next_threshold_can_never_fire"]:
            print("\nThe selection sits at the grid maximum, but the next threshold exceeds the "
                  "largest queue ever observed, so every larger value produces an identical "
                  "policy. The boundary is not a real one; A-10's stopping rule is satisfied.")
        else:
            raise SystemExit(
                "the selection is still at a live grid boundary after the extension cap. Saving "
                "this would ship a baseline that is probably under-tuned, which biases H4-H6 in "
                "favour of the proposed method. Investigate before proceeding."
            )

    path = parameters.save()
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
