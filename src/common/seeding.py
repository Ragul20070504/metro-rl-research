"""Reproducible random-number streams, and the common-random-numbers guarantee.

The fair-comparison principle (research brief section 4, protocol section 7.1) requires that
every method face an *identical* demand realisation, an *identical* disruption realisation and
identical initial conditions within a given ``(scenario, eval_seed)`` cell. That guarantee is
easy to state and easy to break: if the environment draws its demand from the same generator
that the policy draws its action samples from, then two methods that happen to sample a
different number of times diverge, and the comparison silently stops being paired.

This module makes the guarantee structural rather than a matter of discipline:

* Environment streams are derived from ``(eval_seed, stream_name)`` **only**. The method name is
  not an input, so it *cannot* influence the demand or disruption realisation.
* Agent streams are derived from ``(method, train_seed, stream_name)``, a disjoint namespace.
* Every stream is an independent ``numpy.random.Generator``; consuming one never advances
  another.

**The scenario name is deliberately absent from the entropy** (fixed at M6; see
`docs/decisions.md` D-14). It used to be part of it, which meant ``normal`` seed 1 and ``surge``
seed 1 were entirely unrelated demand realisations -- matched in name only. Protocol section 7.4
defines the degradation ratio as the metric under a stressed scenario divided by the metric under
``normal``, *matched on* ``eval_seed``, and section 7.2 requires the ``combined`` family to carry
the same surge and the same failures as the individual families at that seed. Neither held.
Measured on the real network, the confound was large enough to invert the sign: ``surge`` scored
*better* than ``normal`` in three of five seeds.

With the scenario name removed, all four families share one base world per ``eval_seed`` and
differ only by the perturbation they draw -- and because surge draws and failure draws come from
separate streams, ``combined`` reuses exactly the surge of ``surge`` and the failures of
``disruption``. That is the factorial contrast the protocol describes.

``tests/test_common_random_numbers.py`` asserts the property end to end; this module is what
makes it true.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

__all__ = [
    "SCENARIO_NAMES",
    "ENVIRONMENT_STREAMS",
    "AGENT_STREAMS",
    "stable_entropy",
    "environment_streams",
    "agent_streams",
    "SeedPair",
]

# Fixed stream names. Adding a name is safe; reordering or renaming changes every draw, so
# these are frozen alongside the protocol.
ENVIRONMENT_STREAMS: tuple[str, ...] = (
    "demand",           # passenger arrival realisation
    "od",               # origin-destination assignment
    "failures",         # train failure hazard
    "maintenance",      # start-of-day maintenance holds
    "surge",            # surge window placement and magnitude
    "running_time",     # inter-station running-time noise
    "dwell",            # dwell-time noise
    "tiebreak",         # deterministic-looking tie-breaks inside the simulator
    "history",          # pre-episode observable history supplied to the in-episode forecaster
)

#: The four frozen scenario families (protocol section 7.2). Duplicated here rather than imported
#: from :mod:`src.simulator.disruptions` to keep this module free of simulator dependencies;
#: `tests/test_seeding.py` asserts the two lists agree.
SCENARIO_NAMES: tuple[str, ...] = ("normal", "surge", "disruption", "combined")

AGENT_STREAMS: tuple[str, ...] = (
    "policy_init",      # network weight initialisation
    "action_sample",    # stochastic action selection
    "rollout",          # rollout ordering / vectorised env seeding
    "placebo_shuffle",  # the shuffle applied to the placebo arm's sigma channel
)


def stable_entropy(*parts: object) -> int:
    """Map arbitrary labels to a reproducible 128-bit integer seed.

    ``hash()`` is deliberately not used: CPython randomises string hashing per process unless
    ``PYTHONHASHSEED`` is fixed, which would make seeding depend on an environment variable
    rather than on the experiment definition. SHA-256 of the joined repr is stable across
    processes, machines and Python versions.
    """
    payload = "\x1f".join(repr(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def _named_streams(entropy_parts: tuple[object, ...], names: tuple[str, ...]) -> dict[str, np.random.Generator]:
    """Build one independent generator per name, reproducibly.

    Each stream gets its own ``SeedSequence`` derived from the shared entropy *and* its own
    name, rather than from ``spawn()`` ordering. That way adding a new stream later does not
    renumber and therefore does not change the draws of existing ones.
    """
    return {
        name: np.random.default_rng(np.random.SeedSequence(stable_entropy(*entropy_parts, name)))
        for name in names
    }


def environment_streams(scenario: str, eval_seed: int) -> dict[str, np.random.Generator]:
    """Return the environment's random streams for a ``(scenario, eval_seed)`` cell.

    Args:
        scenario: The scenario family. **Validated but deliberately excluded from the entropy**,
            so that every family shares one base world at a given ``eval_seed``. See the module
            docstring and decision D-14 for why: including it made cross-scenario comparisons
            unmatched, and the resulting realisation noise was large enough to make ``surge``
            score better than ``normal``. The argument is kept because the scenario is part of
            the cell's identity and because dropping it would silently change every call site.
        eval_seed: The evaluation seed. The sole source of environment entropy.

    Returns:
        One independent generator per name in :data:`ENVIRONMENT_STREAMS`.

    Notes:
        The method under evaluation is **not** an argument. This is the common-random-numbers
        guarantee: every method evaluated in this cell sees the same demand, the same failures
        and the same surge, because there is no way to tell this function which method is
        running.
    """
    if scenario not in SCENARIO_NAMES:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIO_NAMES}")
    return _named_streams(("env", int(eval_seed)), ENVIRONMENT_STREAMS)


def agent_streams(method: str, train_seed: int) -> dict[str, np.random.Generator]:
    """Return an agent's random streams, in a namespace disjoint from the environment's."""
    return _named_streams(("agent", method, int(train_seed)), AGENT_STREAMS)


@dataclass(frozen=True)
class SeedPair:
    """The (training, evaluation) seed pair identifying one run.

    Protocol section 8.1 requires the two seed sets to be disjoint. ``validate`` enforces it
    at construction sites rather than leaving it to a downstream test.
    """

    train_seed: int | None
    eval_seed: int

    def validate(self, train_pool: set[int], eval_pool: set[int]) -> None:
        """Raise ``ValueError`` if the seed pools overlap or this pair is out of pool."""
        overlap = train_pool & eval_pool
        if overlap:
            raise ValueError(
                f"training and evaluation seed pools must be disjoint; shared seeds: {sorted(overlap)}"
            )
        if self.train_seed is not None and self.train_seed not in train_pool:
            raise ValueError(f"train_seed {self.train_seed} is not in the registered training pool")
        if self.eval_seed not in eval_pool:
            raise ValueError(f"eval_seed {self.eval_seed} is not in the registered evaluation pool")
