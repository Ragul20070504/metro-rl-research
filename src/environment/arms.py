"""The three uncertainty arms (protocol section 6.2) -- the placebo-controlled design.

The three arms have the same observation width. The placebo resamples dispersion scalars
with replacement from a rolling pool of observed values across stations and epochs. This
approximates a past empirical marginal; it is not a permutation and does not preserve a finite
sample's marginal exactly. Pool composition can convey demand-regime information, so a
placebo comparison distinguishes channel constructions without perfectly separating useful
information from temporal/distributional effects. These limitations are disclosed in the
revision manuscript and independent replication protocol.

All three arms share identical observation dimensionality, architecture, parameter count,
training budget, seeds and environment. Only the *content* of the sigma channels differs.

**Common random numbers are unaffected.** The placebo shuffle draws from an agent-side stream
keyed on ``(method, train_seed)``, never from the environment streams keyed on
``(scenario, eval_seed)``. The realised demand and failure streams are therefore byte-identical
across arms, which `tests/test_environment.py` asserts.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from src.common.logging import get_logger

logger = get_logger(__name__)

__all__ = ["SigmaArm", "make_sigma_arm", "ARM_NAMES"]

ARM_NAMES = ("point", "placebo", "uq")

#: How many past sigma scalars the placebo pool remembers. Large enough that the empirical
#: marginal is well represented across a full day (228 epochs x N stations), bounded so a long
#: training run cannot grow it without limit.
PLACEBO_POOL_SIZE = 50_000


class SigmaArm:
    """Transforms the true sigma channels into what a given arm actually observes."""

    def __init__(self, name: str, rng: np.random.Generator | None = None) -> None:
        if name not in ARM_NAMES:
            raise ValueError(f"unknown arm {name!r}; expected one of {ARM_NAMES}")
        self.name = name
        self._rng = rng
        self._pool: deque[float] = deque(maxlen=PLACEBO_POOL_SIZE)
        if name == "placebo" and rng is None:
            raise ValueError("the placebo arm needs an agent-side rng; see agent_streams()")

    def reset(self) -> None:
        """Called at episode start.

        The placebo pool deliberately **persists across episodes**: protocol section 6.2
        specifies shuffling across stations *and epochs* within a rollout batch, and a rollout
        batch spans episode boundaries. Clearing it every episode would confine the shuffle to
        one day and let the agent recover the within-day sigma trajectory.
        """

    def apply(self, sigma: np.ndarray) -> np.ndarray:
        """Return the sigma values this arm observes, given the true ones.

        Args:
            sigma: True calibrated sigma-hat, any shape.

        Returns:
            An array of the **same shape** -- the shape invariance is what keeps parameter
            counts identical across arms.
        """
        if self.name == "point":
            # Informationally empty, but present: the observation vector keeps its width so the
            # policy network keeps its input layer size.
            return np.zeros_like(sigma)

        if self.name == "uq":
            return sigma

        # Placebo: bootstrap the changing empirical pool; association is disrupted, not proven absent.
        #
        # This causal online bootstrap is the inherited implementation, not a literal
        # permutation of a rollout buffer. Sampling with replacement does not preserve the
        # finite marginal exactly. Current values are included, so regime information may
        # remain; the pool also changes temporal dependence. Do not describe it as a proven
        # information-free control. The algorithm is retained for replication, without retuning.
        flat = np.asarray(sigma, dtype=float).ravel()
        self._pool.extend(float(v) for v in flat)
        assert self._rng is not None
        pool = np.asarray(self._pool, dtype=float)
        picks = self._rng.integers(pool.size, size=flat.size)
        return pool[picks].reshape(np.shape(sigma))


def make_sigma_arm(name: str, agent_rng: np.random.Generator | None = None) -> SigmaArm:
    """Build the sigma transform for an arm.

    Args:
        name: One of :data:`ARM_NAMES`.
        agent_rng: Required for ``placebo``. Must come from
            :func:`src.common.seeding.agent_streams`, i.e. keyed on ``(method, train_seed)``,
            so that the shuffle cannot perturb the environment's ``(scenario, eval_seed)``
            streams and break common random numbers.
    """
    return SigmaArm(name, agent_rng)


#: Maps a protocol `method_id` to its sigma arm. Methods that do not consume sigma at all still
#: map to an arm, because the observation always has the channels.
METHOD_TO_ARM = {
    "ppo_point": "point",
    "ppo_placebo": "placebo",
    "ppo_uq": "uq",
    "ppo_uq_single_obj": "uq",
    "ppo_uq_noavail": "uq",
}
