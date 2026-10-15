"""Feasible-action masking (protocol section 6.3).

The manuscript (section 3.4) requires that operational constraints be enforced by **masking**,
not by reward penalties. The distinction is not cosmetic. A penalty leaves an infeasible action
reachable and lets the policy trade a safety violation against passenger benefit; with masking
the violation is unrepresentable, so a reported headway-violation count of zero is a structural
guarantee rather than evidence the penalty was tuned high enough.

A candidate action `k` (induct `k` trainsets during the next interval) is feasible only if all of:

* `k <= ` idle trainsets available now;
* inducting respects the minimum headway on the departure segment;
* each inducted trainset has completed turnaround (in this model a trainset returns to the depot
  only after a full round trip, so depot residency already implies it);
* platform / terminal capacity is not exceeded;
* the trainset is predicted-available, when the availability channel is on.

Action 0 -- induct nothing -- is always feasible, so the mask is never empty and the policy can
never be trapped.
"""

from __future__ import annotations

import numpy as np

from src.common.logging import get_logger
from src.simulator.engine import MetroSimulator

logger = get_logger(__name__)

__all__ = ["action_mask", "InfeasibleActionError"]


class InfeasibleActionError(ValueError):
    """Raised when a masked action is submitted to :meth:`MetroEnv.step`.

    Protocol section 6.3 requires that an attempted masked action **raise rather than silently
    no-op**, and that simulator state be unchanged afterwards. A silent no-op would let a buggy
    or unmasked policy appear to respect the constraints while actually ignoring them, and the
    resulting metrics would look clean.
    """


def action_mask(
    simulator: MetroSimulator,
    k_max: int,
    *,
    availability_probability: np.ndarray | None = None,
    availability_threshold: float = 0.0,
) -> np.ndarray:
    """Boolean mask of length ``k_max + 1`` over the discrete induction action.

    Args:
        simulator: The simulator at the decision instant.
        k_max: Largest number of trainsets inductable in one decision interval.
        availability_probability: Per-train noisy availability signal, or ``None`` when the
            availability channel is off. Trains whose signal falls at or below
            ``availability_threshold`` are treated as not offerable, per section 6.3
            ("predicted-available ... or actually available").
        availability_threshold: Signal level below which an idle train is not offered. Default
            0.0 offers every idle train, which is the frozen configuration: a positive threshold
            would let the *mask* encode availability information and confound the H7 ablation,
            whose whole point is to vary what the **observation** carries.

    Returns:
        ``mask[k]`` is True when inducting ``k`` trainsets is feasible. ``mask[0]`` is always
        True.
    """
    feasible = simulator.feasible_inductions()

    if availability_probability is not None and availability_threshold > 0.0:
        offerable = 0
        for train in simulator.trains:
            if train.available_for_induction:
                if availability_probability[train.train_id] > availability_threshold:
                    offerable += 1
        feasible = min(feasible, offerable)

    feasible = int(np.clip(feasible, 0, k_max))
    mask = np.zeros(k_max + 1, dtype=bool)
    mask[: feasible + 1] = True
    mask[0] = True
    return mask
