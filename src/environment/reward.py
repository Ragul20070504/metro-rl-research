"""Multi-objective reward (protocol section 6.4).

The manuscript's form is kept exactly::

    r_t = -( alpha * W_t + beta * U_t + gamma * E_t + delta * C_t )

**The unit problem the manuscript does not address.** The four terms are minutes, passengers,
energy units and cost units. Adding them with weights alpha..delta only means something once
they are dimensionless, and until then the weights cannot be read as priorities: changing the
energy *unit* would silently reweight the objective. The frozen fix is a fixed **reference
scale** per term, computed once from a reference policy over `train`-period days, stored in
``configs/reward_scales.json`` and never recomputed. alpha..delta then act on dimensionless
quantities, and -- usefully -- the unresolved energy and cost constants (assumptions A15/A16)
largely cancel, because they appear in both the term and its own reference scale.

**What the terms are** (protocol section 6.4 as amended):

===== ============================================================ =====================
term   quantity                                                     amendment
===== ============================================================ =====================
W_t    passenger-minutes of platform waiting accrued in the epoch   A-02 (decision D-13)
U_t    distinct passengers left behind by a departing full train    A-01 (decision D-12)
E_t    energy, in train-km-equivalent units (A15 unresolved)        --
C_t    dispatch and fleet cost, induction-equivalent units (A16)    --
===== ============================================================ =====================

**Component logging is mandatory.** Every step records ``W_t, U_t, E_t, C_t`` in physical units,
the four normalised terms, and ``r_t``. `tests/test_reward.py` asserts the stored components
reconstruct ``r_t`` exactly. No experiment may report the scalar reward alone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.common.logging import get_logger
from src.simulator.engine import EpochMetrics

logger = get_logger(__name__)

__all__ = [
    "RewardWeights",
    "ReferenceScales",
    "RewardComponents",
    "compute_reward",
    "WEIGHT_GRID",
]


@dataclass(frozen=True)
class RewardWeights:
    """One row of the frozen weight grid (protocol section 6.5)."""

    weight_id: str
    alpha: float  # waiting time
    beta: float   # unmet demand / left-behind passengers
    gamma: float  # energy
    delta: float  # dispatch and fleet cost

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Protocol section 6.5. Weights are never chosen and left unjustified (brief section 15).
WEIGHT_GRID: dict[str, RewardWeights] = {
    "w_balanced": RewardWeights("w_balanced", 1.0, 1.0, 1.0, 1.0),
    "w_service": RewardWeights("w_service", 2.0, 2.0, 0.5, 0.5),
    "w_cost": RewardWeights("w_cost", 0.5, 0.5, 2.0, 2.0),
    "w_wait_only": RewardWeights("w_wait_only", 1.0, 0.0, 0.0, 0.0),
    "w_unmet_heavy": RewardWeights("w_unmet_heavy", 1.0, 3.0, 1.0, 1.0),
}

#: The primary configuration for the confirmatory H1 test.
PRIMARY_WEIGHT_ID = "w_balanced"


@dataclass
class ReferenceScales:
    """Fixed per-term divisors that make alpha..delta interpretable.

    Attributes:
        ref_W: Reference passenger-minutes of waiting accrued per epoch (the queue integral,
            protocol amendment A-02).
        ref_U: Reference left-behind passengers per epoch.
        ref_E: Reference energy per epoch, in the config's energy units.
        ref_C: Reference cost per epoch, in the config's cost units.
        provenance: Which policy, which day set, which commit produced these. Required: a
            reference scale with no recorded source is indistinguishable from a tuned constant.
    """

    ref_W: float
    ref_U: float
    ref_E: float
    ref_C: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("ref_W", "ref_U", "ref_E", "ref_C"):
            value = getattr(self, name)
            if not (value > 0):
                raise ValueError(
                    f"{name} must be strictly positive; got {value!r}. A zero reference scale "
                    "would make its term infinite or silently drop it from the objective."
                )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "ReferenceScales":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            ref_W=float(payload["ref_W"]),
            ref_U=float(payload["ref_U"]),
            ref_E=float(payload["ref_E"]),
            ref_C=float(payload["ref_C"]),
            provenance=payload.get("provenance", {}),
        )

    @classmethod
    def unit(cls) -> "ReferenceScales":
        """All-ones scales, for unit tests where hand-checkable arithmetic matters more than
        interpretable weights."""
        return cls(1.0, 1.0, 1.0, 1.0, provenance={"status": "unit", "note": "tests only"})


@dataclass(frozen=True)
class RewardComponents:
    """Every quantity needed to reconstruct ``r_t``, in physical and normalised form."""

    # physical units
    W_minutes: float          # passenger-minutes accrued in the epoch (the queue integral)
    U_passengers: float
    E_units: float
    C_units: float
    # logged diagnostic, NOT a reward term: mean wait of the passengers who boarded this epoch.
    # NaN when nobody boarded, which is exactly why it is unusable as an objective (A-02).
    W_mean_over_boarders_minutes: float
    # normalised (dimensionless)
    W_norm: float
    U_norm: float
    E_norm: float
    C_norm: float
    # weighted contributions (each already negated in the total)
    reward: float
    weight_id: str

    def reconstruct(self, weights: RewardWeights) -> float:
        """Recompute ``r_t`` from the stored components. Used by `tests/test_reward.py`."""
        return -(
            weights.alpha * self.W_norm
            + weights.beta * self.U_norm
            + weights.gamma * self.E_norm
            + weights.delta * self.C_norm
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_reward(
    metrics: EpochMetrics,
    weights: RewardWeights,
    scales: ReferenceScales,
) -> RewardComponents:
    """Evaluate the reward for one decision epoch.

    Args:
        metrics: Outcomes of the epoch just simulated.
        weights: One row of :data:`WEIGHT_GRID`.
        scales: Frozen reference scales.

    Notes:
        ``W_t`` is the **queue integral**: passenger-minutes of platform waiting accrued during
        the epoch by everyone on a platform, served or not (protocol amendment A-02, decision
        D-13). The earlier reading -- mean wait of the passengers who *boarded* in the epoch --
        was measured to make the reward degenerate: under a policy that runs no service at all
        nobody boards, so the mean is undefined; no train departs, so nothing is left behind;
        nothing moves, so energy and cost are zero. That policy scored ``r = 0``, the global
        maximum, while stranding 100 % of demand. The queue integral separates the two by a
        factor of 249 and is defined whether or not anybody boards. The mean over boarders is
        still logged, as a diagnostic.

        ``U_t`` is **left-behind passengers**, not the end-of-day residual queue, following
        protocol amendment A-01 / decision D-12. The residual queue was measured to be
        degenerate (approximately 0.05 % under every policy), which would have made beta a
        weight on a constant. Residual queue is still recorded, as a secondary metric.
    """
    diagnostic_mean = metrics.waiting_time_mean_s / 60.0

    physical = (
        float(metrics.waiting_accrued_passenger_s / 60.0),
        float(metrics.left_behind),
        float(metrics.energy_units),
        float(metrics.cost_units),
    )
    normalised = (
        physical[0] / scales.ref_W,
        physical[1] / scales.ref_U,
        physical[2] / scales.ref_E,
        physical[3] / scales.ref_C,
    )

    reward = -(
        weights.alpha * normalised[0]
        + weights.beta * normalised[1]
        + weights.gamma * normalised[2]
        + weights.delta * normalised[3]
    )

    return RewardComponents(
        W_minutes=physical[0],
        W_mean_over_boarders_minutes=float(diagnostic_mean),
        U_passengers=physical[1],
        E_units=physical[2],
        C_units=physical[3],
        W_norm=normalised[0],
        U_norm=normalised[1],
        E_norm=normalised[2],
        C_norm=normalised[3],
        reward=float(reward),
        weight_id=weights.weight_id,
    )
