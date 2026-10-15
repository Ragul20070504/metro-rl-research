"""Origin-destination model (M5, assumption A03).

**This is the most consequential synthetic assumption in the project and it is labelled as
such everywhere it appears.** The demand source publishes station *entries* only -- no exits,
no destinations - so no public data constrains where passengers go. Occupancy, crowding and
unmet demand all inherit whatever is assumed here, and unmet demand is the pre-registered
primary endpoint.

Because it cannot be validated, it is *swept*: three structurally different variants are
implemented and ``EXP-SENS-002`` runs the primary comparison under all three. The primary
conclusion must hold under every variant or be reported as OD-dependent.

Variants:

``uniform_downstream``
    Every other station equally likely. Deliberately naive; the null model.
``distance_decay``
    Probability falls exponentially with distance. Reflects the usual finding that short
    trips dominate urban rail, without encoding any city-specific structure.
``cbd_attracted``
    Distance decay multiplied by an attraction that rises toward the Manhattan end of the
    line. Encodes commuter directionality, which is the most obvious real feature the other
    two miss -- and, being the most "informed" variant, the one most at risk of smuggling in
    an assumption that flatters the results. Hence all three are always reported.
"""

from __future__ import annotations

import numpy as np

from src.common.logging import get_logger

logger = get_logger(__name__)

OD_MODELS = ("uniform_downstream", "distance_decay", "cbd_attracted")


def cumulative_km(segment_km: tuple[float, ...]) -> np.ndarray:
    """Distance of each station from the line's first station."""
    return np.concatenate([[0.0], np.cumsum(np.asarray(segment_km, dtype=float))])


def build_od_matrix(
    segment_km: tuple[float, ...],
    model: str = "distance_decay",
    *,
    decay_km: float = 4.0,
    cbd_strength: float = 1.5,
) -> np.ndarray:
    """Return ``od[i, j]``: probability a passenger boarding at ``i`` alights at ``j``.

    Rows sum to 1 and the diagonal is zero. The matrix is deterministic given its parameters,
    so it contributes no Monte-Carlo noise -- the stochasticity in the simulator comes from
    passenger *arrivals*, which is the part the data actually supports.
    """
    if model not in OD_MODELS:
        raise ValueError(f"unknown OD model {model!r}; expected one of {OD_MODELS}")

    positions = cumulative_km(segment_km)
    n = positions.size
    distance = np.abs(positions[:, None] - positions[None, :])

    if model == "uniform_downstream":
        weights = np.ones((n, n), dtype=float)
    else:
        weights = np.exp(-distance / max(decay_km, 1e-6))
        if model == "cbd_attracted":
            # Station index n-1 is the Manhattan terminus of the modelled line.
            attraction = np.exp(cbd_strength * np.linspace(0.0, 1.0, n))
            weights = weights * attraction[None, :]

    np.fill_diagonal(weights, 0.0)
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    return weights / row_sums


def direction_split(od: np.ndarray) -> np.ndarray:
    """Fraction of boarders at each station travelling in direction 0 (increasing index)."""
    n = od.shape[0]
    upper = np.triu(od, k=1).sum(axis=1)
    total = od.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(total > 0, upper / total, 0.0)
    # The terminus in each direction has nowhere further to go.
    share[n - 1] = 0.0
    share[0] = 1.0
    return share


def destination_weights(od: np.ndarray, station: int, direction: int) -> np.ndarray:
    """Normalised destination distribution for boarders at ``station`` in ``direction``.

    Returns a full-length vector so it can be added straight onto a train's load array; entries
    for stations not reachable in this direction are zero.
    """
    n = od.shape[0]
    weights = np.zeros(n, dtype=float)
    if direction == 0:
        if station >= n - 1:
            return weights
        weights[station + 1:] = od[station, station + 1:]
    else:
        if station <= 0:
            return weights
        weights[:station] = od[station, :station]
    total = weights.sum()
    if total > 0:
        weights /= total
    return weights
