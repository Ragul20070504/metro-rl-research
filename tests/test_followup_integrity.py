"""Follow-up inference must never turn missing episodes into a smaller passing study."""
from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
import pandas as pd
import pytest

from scripts import analyse_followup as analysis
from scripts import run_followup as runner
from src.evaluation.statistics import bca_bootstrap_ci


@pytest.fixture
def grid():
    return pd.DataFrame([
        {"method": arm, "train_seed": seed, "scenario": scenario, "eval_seed": evaluation,
         "variant": "main" if budget == 60_000 else f"budget_{budget}",
         "weight_id": "w_balanced", "left_behind_fraction": 0.01}
        for budget in runner.BUDGETS for arm in runner.ARMS for seed in runner.TRAIN_SEEDS
        for scenario in analysis.SCENARIOS for evaluation in range(1, 31)
    ])


def test_completeness_checks_each_episode_and_rejects_duplicates_and_nonfinite(grid):
    assert all(row["complete"] for row in analysis.budget_coverage(grid).values())
    missing = analysis.budget_coverage(grid.iloc[1:])[15_000]
    assert not missing["complete"] and missing["missing"] == 1
    duplicate = analysis.budget_coverage(pd.concat([grid, grid.iloc[:1]]))[15_000]
    assert not duplicate["complete"] and duplicate["duplicates"] == 1
    grid.loc[0, "left_behind_fraction"] = np.nan
    assert not analysis.budget_coverage(grid)[15_000]["complete"]


def test_incomplete_grid_writes_no_interim_inference(tmp_path, monkeypatch, grid):
    monkeypatch.setattr(analysis, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(analysis, "discover_cells", lambda: [])
    monkeypatch.setattr(analysis, "load_frame", lambda _: grid[grid.variant == "main"])
    # A stale aggregate claiming a complete grid must not be consulted.
    grid.to_parquet(tmp_path / "seed_level.parquet")
    assert analysis.main() == 1
    result = json.loads((tmp_path / "followup_analysis.json").read_text())
    assert result["primary_interaction"]["status"] == "NOT RUN"
    assert result["per_budget"] == [] and result["budgets_analysed"] == []
    assert result["coverage"]["60000"]["complete"]


def test_followup_interval_estimates_protocol_hodges_lehmann():
    differences = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 5.0, 9.0, 20.0])
    actual = analysis._bca_ci(differences, n_boot=500, seed=42)
    assert actual == bca_bootstrap_ci(differences, resamples=500, seed=42)
    assert actual != bca_bootstrap_ci(differences, statistic=np.mean, resamples=500, seed=42)


def test_training_stops_after_one_recorded_retry(tmp_path, monkeypatch):
    import src.rl.train as training

    monkeypatch.setattr(runner, "BUDGETS", (15_000,))
    monkeypatch.setattr(runner, "ARMS", ("ppo_point",))
    monkeypatch.setattr(runner, "TRAIN_SEEDS", (101,))
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(training, "policy_path", lambda *args: tmp_path / "missing.zip")
    calls = []

    def fail(job):
        calls.append(job)
        raise RuntimeError("deliberate failure")

    monkeypatch.setattr(runner, "_train_one", fail)
    with pytest.raises(RuntimeError, match="failed after retry"):
        runner.stage_train()
    assert len(calls) == 2
    assert json.loads((tmp_path / "followup_training_failures.json").read_text())


def test_budget_training_retains_episode_learning_curve(tmp_path, monkeypatch):
    import src.rl.train as training
    from src.environment.forecast_provider import SeasonalNaiveProvider
    from src.environment.metro_env import EnvConfig, MetroEnv, constant_history_sampler
    from src.environment.reward import ReferenceScales
    from src.simulator.demand import build_synthetic_process
    from src.simulator.engine import SimulatorConfig
    from src.simulator.network import build_toy_network

    network = build_toy_network(3, 120.0)
    process = build_synthetic_process(network.station_ids, peak_mean=50.0)
    config = SimulatorConfig(fleet_size=6, capacity_passengers=400.0,
                             failure_rate_per_km=0.0, day_start_hour=8, day_end_hour=9)
    env = MetroEnv(network=network, demand_process=process, simulator_config=config,
                   env_config=EnvConfig(arm="point", scenario="normal", n_segments=2),
                   forecast_provider=SeasonalNaiveProvider(network.n_stations),
                   history_sampler=constant_history_sampler(process, 8),
                   scales=ReferenceScales.unit(), eval_seed=101)
    monkeypatch.setattr(training, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(training, "build_training_env", lambda *args, **kwargs: env)
    target = training.train_policy("ppo_point", 101, store_as="budget_32",
                                   config=training.PPOConfig(total_timesteps=32, n_steps=16,
                                                            batch_size=16, n_epochs=1, net_arch=(8,)))
    metadata = json.loads(target.with_name("metadata.json").read_text())
    assert metadata["actual_timesteps"] == 32
    curve = pd.read_csv(target.with_name(metadata["learning_curve"]), comment="#")
    assert len(curve) >= 2 and np.isfinite(curve["r"]).all()
