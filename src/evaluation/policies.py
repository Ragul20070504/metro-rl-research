"""The common policy interface every method is evaluated through (protocol section 7.1).

Fair comparison requires that a fixed timetable, a tuned heuristic, a MILP schedule and a trained
PPO policy all face **the same environment, the same episode loop and the same mask**. If each
method had its own evaluation loop, a difference in how one of them handled an infeasible action
or an end-of-episode boundary would show up as a difference in performance.

So there is exactly one loop, in :func:`run_episode`, and methods differ only in
:meth:`Policy.act`. The loop applies the mask to every method identically: a method that requests
an infeasible action has its request reduced to the largest feasible one, and the shortfall is
recorded rather than silently absorbed. That is a property of the infrastructure, not of the
method, so it must be measured the same way for all of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from src.common.logging import get_logger
from src.environment.metro_env import MetroEnv

logger = get_logger(__name__)

__all__ = ["Policy", "ConstantPolicy", "run_episode", "EpisodeResult"]


class Policy(Protocol):
    """What the evaluation loop needs from a method."""

    method_id: str

    def reset(self, env: MetroEnv) -> None:
        """Called once per episode, after ``env.reset()``."""

    def act(self, observation: np.ndarray, env: MetroEnv, mask: np.ndarray) -> int:
        """Return the desired induction count. May be infeasible; the loop clips it."""


@dataclass
class EpisodeResult:
    """One evaluated ``(method, scenario, train_seed, eval_seed)`` cell."""

    summary: dict[str, Any]
    epochs: list[dict[str, Any]]
    requested_actions: list[int]
    executed_actions: list[int]
    inference_latency_ms: list[float] = field(default_factory=list)

    @property
    def mask_clip_rate(self) -> float:
        """Fraction of epochs where the method asked for more than the mask allowed.

        Protocol section 7.4 records induction success rate. A method whose requests are clipped
        often is being limited by the infrastructure, not by its own logic, and reporting its
        outcome without this number would misattribute the cause.
        """
        if not self.requested_actions:
            return 0.0
        clipped = sum(1 for r, e in zip(self.requested_actions, self.executed_actions) if r != e)
        return clipped / len(self.requested_actions)


class ConstantPolicy:
    """Induct a fixed number every epoch. Used for reference scaling and sanity probes."""

    def __init__(self, k: int, method_id: str | None = None) -> None:
        self.k = int(k)
        self.method_id = method_id or f"constant_{k}"

    def reset(self, env: MetroEnv) -> None:
        return None

    def act(self, observation: np.ndarray, env: MetroEnv, mask: np.ndarray) -> int:
        return self.k


def run_episode(env: MetroEnv, policy: Policy, *, eval_seed: int | None = None) -> EpisodeResult:
    """Run one operating day. The single evaluation loop shared by every method.

    Args:
        env: A configured environment. Its ``scenario`` and reward weights are already set.
        policy: Any object satisfying :class:`Policy`.
        eval_seed: Passed to ``env.reset``; omit to use the environment's own.

    Returns:
        The episode summary, per-epoch records, and both the requested and executed action
        sequences so that mask clipping is auditable after the fact.
    """
    import time

    observation, _ = env.reset(seed=eval_seed)
    policy.reset(env)

    requested: list[int] = []
    executed: list[int] = []
    latency: list[float] = []

    while True:
        mask = env.action_masks()
        start = time.perf_counter()
        want = int(policy.act(observation, env, mask))
        latency.append((time.perf_counter() - start) * 1000.0)

        # Clip to the mask identically for every method. Raising here instead would make the
        # comparison depend on which methods happen to be mask-aware.
        want = int(np.clip(want, 0, env.cfg.k_max))
        feasible = int(np.flatnonzero(mask)[-1])
        action = min(want, feasible)

        requested.append(want)
        executed.append(action)

        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return EpisodeResult(
                summary=info["episode_summary"],
                epochs=env.episode_records(),
                requested_actions=requested,
                executed_actions=executed,
                inference_latency_ms=latency,
            )
