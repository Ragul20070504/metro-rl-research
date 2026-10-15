"""Tiny PPO smoke test (protocol section 10, stage 1) -- software validation, not science.

This proves only that the plumbing works: MaskablePPO can initialise against this observation
and action space, collect rollouts, take gradient steps, save and reload a checkpoint, and be
evaluated. It uses a training budget far too small to learn anything.

**Nothing here is a research result.** No number produced by this file may appear in the paper,
and the test asserts nothing about policy quality -- deliberately, because an assertion that a
128-step run improves the return would be a test that passes on noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.environment.forecast_provider import SeasonalNaiveProvider
from src.environment.metro_env import EnvConfig, MetroEnv, constant_history_sampler
from src.environment.reward import ReferenceScales
from src.simulator.demand import build_synthetic_process
from src.simulator.engine import SimulatorConfig
from src.simulator.network import build_toy_network

sb3_contrib = pytest.importorskip("sb3_contrib")

TINY_STATIONS = 5
TINY_BUDGET = 128


def tiny_env(arm: str = "uq", eval_seed: int = 1) -> MetroEnv:
    """A deliberately small environment: 5 stations, a short day, a handful of trainsets."""
    network = build_toy_network(TINY_STATIONS, 120.0)
    process = build_synthetic_process(network.station_ids, peak_mean=150.0)
    config = SimulatorConfig(
        fleet_size=6,
        capacity_passengers=400.0,
        failure_rate_per_km=0.0,
        day_start_hour=8,
        day_end_hour=11,          # 3-hour day = 36 decision epochs
    )
    return MetroEnv(
        network=network,
        demand_process=process,
        simulator_config=config,
        env_config=EnvConfig(arm=arm, scenario="normal", n_segments=2),
        forecast_provider=SeasonalNaiveProvider(network.n_stations),
        history_sampler=constant_history_sampler(process, config.day_start_hour),
        scales=ReferenceScales.unit(),
        eval_seed=eval_seed,
    )


def tiny_model(env):
    from sb3_contrib import MaskablePPO

    return MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        policy_kwargs={"net_arch": [16, 16]},
        seed=101,
        verbose=0,
        device="cpu",
    )


class TestPPOPlumbing:
    def test_maskable_ppo_initialises_against_this_environment(self) -> None:
        model = tiny_model(tiny_env())
        assert model.policy is not None
        assert model.observation_space.shape == (tiny_env().obs_spec.size,)
        assert model.action_space.n == 3

    def test_rollout_collection_and_gradient_steps_run(self) -> None:
        model = tiny_model(tiny_env())
        model.learn(total_timesteps=TINY_BUDGET)
        assert model.num_timesteps >= TINY_BUDGET

    def test_training_changes_the_policy_parameters(self) -> None:
        """Not a quality claim -- only that gradients actually reach the weights."""
        import torch

        model = tiny_model(tiny_env())
        before = [p.detach().clone() for p in model.policy.parameters()]
        model.learn(total_timesteps=TINY_BUDGET)
        after = list(model.policy.parameters())
        assert any(not torch.equal(a, b) for a, b in zip(before, after))

    def test_checkpoint_saves_and_reloads(self, tmp_path) -> None:
        from sb3_contrib import MaskablePPO

        model = tiny_model(tiny_env())
        model.learn(total_timesteps=TINY_BUDGET)
        path = tmp_path / "smoke_policy.zip"
        model.save(path)
        assert path.exists()
        reloaded = MaskablePPO.load(path, device="cpu")
        assert reloaded.action_space.n == model.action_space.n

    def test_a_trained_policy_can_be_evaluated_without_violating_the_mask(self) -> None:
        from sb3_contrib.common.maskable.utils import get_action_masks

        env = tiny_env()
        model = tiny_model(env)
        model.learn(total_timesteps=TINY_BUDGET)

        observation, _ = env.reset()
        total = 0.0
        while True:
            masks = get_action_masks(env)
            action, _ = model.predict(observation, action_masks=masks, deterministic=True)
            assert masks[int(action)], "policy selected a masked action"
            observation, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        assert np.isfinite(total)
        assert info["episode_summary"]["headway_violations"] == 0

    def test_all_three_arms_train_with_the_same_parameter_count(self) -> None:
        """Protocol section 6.2: identical architecture and parameter count across arms."""
        counts = {}
        for arm in ("point", "placebo", "uq"):
            model = tiny_model(tiny_env(arm=arm))
            counts[arm] = sum(p.numel() for p in model.policy.parameters())
            model.learn(total_timesteps=32)
        assert len(set(counts.values())) == 1, counts
