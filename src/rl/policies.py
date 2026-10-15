"""Trained-policy wrapper for evaluation (M10).

Wraps a `MaskablePPO` checkpoint in the same :class:`~src.evaluation.policies.Policy` interface
every other method uses, so the evaluation loop cannot treat RL differently from a heuristic.

Actions are taken **deterministically** at evaluation. Protocol section 8.1 gives evaluation
seeds a single job -- driving the environment -- and a stochastic policy would add a second,
uncontrolled source of variation that differs between arms and breaks the paired comparison under
common random numbers.
"""

from __future__ import annotations

import numpy as np

from src.common.logging import get_logger
from src.environment.metro_env import MetroEnv

logger = get_logger(__name__)

__all__ = ["TrainedPPOPolicy"]


class TrainedPPOPolicy:
    """A trained MaskablePPO checkpoint, evaluated greedily."""

    def __init__(self, method_id: str, model) -> None:
        self.method_id = method_id
        self.model = model

    def reset(self, env: MetroEnv) -> None:
        return None

    def act(self, observation: np.ndarray, env: MetroEnv, mask: np.ndarray) -> int:
        action, _ = self.model.predict(
            observation, action_masks=mask, deterministic=True
        )
        return int(action)
