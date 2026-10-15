"""The three uncertainty arms (protocol section 6.2) -- the placebo-controlled design.

`PPO-Point` versus `PPO-UQ` alone cannot distinguish *"uncertainty information helps"* from
*"a wider observation helps"*: the second arm simply has more numbers in it. `PPO-Placebo`
is the control that separates them. It receives sigma values with the **correct marginal
distribution** but the association to station and epoch destroyed, so it has the same input
dimensionality, the same input statistics, and no usable information.

A result in which `PPO-UQ` beats `PPO-Point` but only matches `PPO-Placebo` **refutes** the
manuscript's mechanism, even though it would look like a win in a two-arm study. That outcome is
a legitimate finding and the protocol requires it be reported as such.

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

        # placebo: correct marginal, destroyed association.
        #
        # Implemented as a bootstrap draw from the pool of sigma values seen so far rather than
        # as a literal permutation of one rollout buffer. Both preserve the marginal exactly and
        # both destroy the (station, epoch) association; the bootstrap is O(1) per element,
        # whereas permuting a 50k pool on each of 228 epochs per episode would dominate rollout
        # time. The deviation is recorded in docs/rl_environment.md.
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
