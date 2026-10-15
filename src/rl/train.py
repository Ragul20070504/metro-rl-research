"""PPO training for every RL arm (M10).

**One frozen hyperparameter configuration, applied unchanged to all arms.**

`docs/decisions.md` carries a standing constraint: *"The uncertainty-aware arm is not tuned more
aggressively than its ablations. One hyperparameter configuration, selected on `val` using the
point-forecast arm, frozen and applied unchanged to all RL arms."*

DECISION: no PPO hyperparameter search is run; a single documented configuration is used for
every arm.
REASON: the constraint's purpose is symmetry, and one untuned configuration satisfies it exactly.
A search conducted on the point arm would spend the compute budget without improving the validity
of the comparison, and would risk selecting a configuration that happens to suit the point arm's
narrower input — an asymmetry running *against* the proposed method, but an asymmetry all the
same.
SCIENTIFIC CONSEQUENCE: all arms may be jointly suboptimal. This bounds any claim about the
*absolute* quality of the learned policies. It does not bias the *relative* comparison, which is
what H1 and H1c test. Recorded as a limitation in `docs/m10_training.md`.

Every arm receives identical architecture, parameter count, training budget, seeds and
environment. The only difference is the content of the two sigma channels, which is the whole
point of the placebo design (protocol section 6.2).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.common.logging import get_logger
from src.common.paths import MODELS_DIR, ensure_dir

logger = get_logger(__name__)

__all__ = ["PPOConfig", "train_policy", "policy_path", "RL_METHODS"]

#: The five registered RL arms (protocol section 7.1).
RL_METHODS: tuple[str, ...] = (
    "ppo_point", "ppo_placebo", "ppo_uq", "ppo_uq_single_obj", "ppo_uq_noavail",
)

#: Arms whose reward weights differ from the cell's default (protocol section 7.1).
METHOD_WEIGHT_OVERRIDE: dict[str, str] = {"ppo_uq_single_obj": "w_wait_only"}


@dataclass(frozen=True)
class PPOConfig:
    """Frozen PPO hyperparameters, identical for every arm.

    Values are Stable-Baselines3 defaults except where the episode structure requires otherwise:
    ``n_steps`` is set to a whole number of decision epochs so a rollout never splits an
    operating day at an arbitrary point, and the network is small because the observation is a
    123-value summary rather than raw sensor data.
    """

    total_timesteps: int = 60_000
    n_steps: int = 456          # exactly two operating days of 228 epochs
    batch_size: int = 152       # divides n_steps evenly
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.995        # a 228-step episode; 0.99 would discount the evening away
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01      # the action space is tiny; a little exploration pressure helps
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: tuple[int, ...] = (64, 64)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "net_arch": list(self.net_arch)}


def policy_path(method: str, train_seed: int, weight_id: str) -> Path:
    return MODELS_DIR / "policies" / weight_id / method / str(train_seed) / "policy.zip"


def build_training_env(method: str, train_seed: int, weight_id: str, *,
                       trained_forecaster: bool = True):
    """The environment a policy trains in.

    Training days are drawn from the **training seed pool**, disjoint from the evaluation pool
    (protocol section 8.1), so no policy is ever optimised on a day it is later scored on.
    """
    from src.evaluation.runner import Cell, build_env

    cell = Cell(
        method=method,
        scenario="normal",
        eval_seed=train_seed,
        train_seed=train_seed,
        weight_id=METHOD_WEIGHT_OVERRIDE.get(method, weight_id),
    )
    return build_env(cell, trained_forecaster=trained_forecaster)


class _TrainingSeedCycler:
    """Cycle episodes over the training seed pool and the scenario families.

    Training on ``normal`` alone would produce policies that have never seen a surge or a
    failure, and the primary endpoint is measured under ``surge``. Training on the evaluation
    seeds would be test-set leakage. So training days come from a disjoint pool and cover all
    four families in a fixed rotation — identical for every arm, so it cannot advantage one.
    """

    SCENARIOS = ("normal", "surge", "disruption", "combined")

    def __init__(self, env, train_seed: int) -> None:
        self.env = env
        self.rng = np.random.default_rng(train_seed)
        self.counter = 0
        # Training days: 5000-block, disjoint from evaluation (1-30), scaler fitting (1001+),
        # heuristic tuning (2001+) and surrogate calibration (3001+).
        self.pool = tuple(range(5000 + train_seed * 100, 5000 + train_seed * 100 + 100))

    def next(self) -> tuple[str, int]:
        scenario = self.SCENARIOS[self.counter % len(self.SCENARIOS)]
        seed = self.pool[self.counter % len(self.pool)]
        self.counter += 1
        return scenario, seed


def train_policy(
    method: str,
    train_seed: int,
    *,
    weight_id: str = "w_balanced",
    config: PPOConfig | None = None,
    trained_forecaster: bool = True,
    overwrite: bool = False,
    store_as: str | None = None,
) -> Path:
    """Train one arm at one seed and write the checkpoint. Skips completed work.

    ``store_as`` redirects **only the checkpoint location**, leaving the reward weighting and
    every other aspect of training untouched. It exists so the FOLLOWUP-UQ-TRAINING study can
    keep one checkpoint per training budget without colliding with the registered policies, which
    share a method and seed with it. Leaving it ``None`` reproduces the original behaviour
    exactly, so no registered artefact can move.
    """
    from sb3_contrib import MaskablePPO

    effective_weight = METHOD_WEIGHT_OVERRIDE.get(method, weight_id)
    target = policy_path(method, train_seed, store_as or weight_id)
    if target.exists() and not overwrite:
        logger.info("policy %s/%s already trained; skipping", method, train_seed)
        return target

    config = config or PPOConfig()
    env = build_training_env(method, train_seed, weight_id,
                             trained_forecaster=trained_forecaster)
    cycler = _TrainingSeedCycler(env, train_seed)

    # Rotate the training day at every reset, through the environment's own options channel, so
    # the rotation cannot perturb the environment's random streams.
    original_reset = env.reset

    def rotating_reset(*, seed=None, options=None):
        scenario, day_seed = cycler.next()
        merged = dict(options or {})
        merged.update({"scenario": scenario, "eval_seed": day_seed})
        return original_reset(seed=None, options=merged)

    env.reset = rotating_reset  # type: ignore[method-assign]

    model = MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs={"net_arch": list(config.net_arch)},
        seed=train_seed,
        verbose=0,
        device="cpu",
    )

    started = time.perf_counter()
    model.learn(total_timesteps=config.total_timesteps, progress_bar=False)
    elapsed = time.perf_counter() - started

    ensure_dir(target.parent)
    model.save(target)

    metadata = {
        "method": method,
        "arm": env.cfg.arm,
        "train_seed": train_seed,
        "weight_id": effective_weight,
        "requested_weight_id": weight_id,
        "config": config.to_dict(),
        "parameter_count": int(sum(p.numel() for p in model.policy.parameters())),
        "observation_size": int(env.obs_spec.size),
        "include_availability_channel": env.cfg.include_availability_channel,
        "training_scenarios": list(_TrainingSeedCycler.SCENARIOS),
        "training_day_seed_pool": [cycler.pool[0], cycler.pool[-1]],
        "seed_pool_note": (
            "training days are drawn from a 5000-block disjoint from the frozen evaluation pool "
            "(1-30), so no policy is optimised on a day it is later scored on"
        ),
        "forecaster": getattr(env.forecast_provider, "model_id", "unknown"),
        "obs_scaler": env.scaler.provenance,
        "reward_scales": env.scales.provenance,
        "hyperparameter_search": (
            "NONE. One frozen configuration is applied unchanged to every arm, per the standing "
            "constraint in docs/decisions.md. All arms may be jointly suboptimal; the relative "
            "comparison is unaffected."
        ),
        "wall_clock_s": elapsed,
        "trained_utc": datetime.now(timezone.utc).isoformat(),
    }
    (target.parent / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    logger.info("trained %s seed %d in %.1f s -> %s", method, train_seed, elapsed, target)
    return target
