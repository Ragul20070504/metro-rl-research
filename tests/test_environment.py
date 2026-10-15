"""RL environment tests (M6): Gymnasium conformance, invariants, arms, and sanity environments.

These cover the checklist the execution directive sets for M6 and the staged compute gates of
protocol section 10. The invariants that matter scientifically are the ones asserting the agent
cannot see the future, that the arms differ only in sigma content, and that common random numbers
survive the arm transform -- without those three, no comparison between methods means anything.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.environment.action_mask import InfeasibleActionError
from src.environment.arms import make_sigma_arm
from src.environment.forecast_provider import SeasonalNaiveProvider
from src.environment.metro_env import EnvConfig, MetroEnv, constant_history_sampler
from src.environment.observation import ObservationScaler, build_observation_spec
from src.environment.reward import WEIGHT_GRID, ReferenceScales
from src.simulator.demand import build_synthetic_process
from src.simulator.engine import SimulatorConfig
from src.simulator.entities import TrainStatus
from src.simulator.network import build_toy_network

TOY_STATIONS = 10


def make_env(
    *,
    arm: str = "uq",
    scenario: str = "normal",
    eval_seed: int = 1,
    train_seed: int = 101,
    weight_id: str = "w_balanced",
    availability: bool = True,
    peak_mean: float = 200.0,
    scaler: ObservationScaler | None = None,
    strict_scaler: bool = False,
    **sim_overrides,
) -> MetroEnv:
    network = build_toy_network(TOY_STATIONS, 120.0)
    process = build_synthetic_process(network.station_ids, peak_mean=peak_mean)
    defaults = dict(fleet_size=20, capacity_passengers=800.0, failure_rate_per_km=0.0)
    defaults.update(sim_overrides)
    config = SimulatorConfig(**defaults)
    return MetroEnv(
        network=network,
        demand_process=process,
        simulator_config=config,
        env_config=EnvConfig(
            arm=arm,
            scenario=scenario,
            weight_id=weight_id,
            include_availability_channel=availability,
            strict_scaler=strict_scaler,
        ),
        forecast_provider=SeasonalNaiveProvider(network.n_stations),
        history_sampler=constant_history_sampler(process, config.day_start_hour),
        scales=ReferenceScales.unit(),
        scaler=scaler,
        train_seed=train_seed,
        eval_seed=eval_seed,
    )


def rollout(env: MetroEnv, policy=lambda mask: int(np.flatnonzero(mask)[-1])):
    """Run one full episode under a mask-respecting policy. Returns (summary, records)."""
    env.reset()
    while True:
        mask = env.action_masks()
        _, _, terminated, truncated, info = env.step(policy(mask))
        if terminated or truncated:
            return info["episode_summary"], env.episode_records()


class TestScalerProjection:
    """The H7 ablation drops a block, so the frozen scaler must be restricted, not refitted.

    `ppo_uq_noavail` has no availability channel and therefore sees 121 values where every other
    arm sees 123. Refitting a separate scaler for it would also change the centre and scale of the
    121 surviving features, so any outcome difference could be attributed to the scaling rather
    than to the missing information -- which is precisely what H7 is trying to measure.
    """

    @staticmethod
    def _specs():
        from src.environment.observation import build_observation_spec

        stations = tuple(f"S{i}" for i in range(6))
        full = build_observation_spec(station_ids=stations, n_segments=3,
                                      include_availability_channel=True)
        reduced = build_observation_spec(station_ids=stations, n_segments=3,
                                         include_availability_channel=False)
        return full, reduced

    def _fitted(self, spec):
        from src.environment.observation import ObservationScaler

        rng = np.random.default_rng(0)
        samples = rng.normal(loc=5.0, scale=2.0, size=(400, spec.size))
        return ObservationScaler.fit(samples, spec, {"source": "test"})

    def test_projection_matches_the_reduced_layout(self):
        full, reduced = self._specs()
        projected = self._fitted(full).project(full, reduced)
        assert projected.centre.size == reduced.size
        assert projected.fitted

    def test_every_retained_feature_keeps_its_original_constants(self):
        full, reduced = self._specs()
        scaler = self._fitted(full)
        projected = scaler.project(full, reduced)

        for block in reduced.blocks:
            np.testing.assert_allclose(
                projected.centre[reduced.slice_of(block.name)],
                scaler.centre[full.slice_of(block.name)],
                err_msg=f"{block.name} was rescaled by the projection")
            np.testing.assert_allclose(
                projected.scale[reduced.slice_of(block.name)],
                scaler.scale[full.slice_of(block.name)])

    def test_the_dropped_block_is_recorded(self):
        full, reduced = self._specs()
        projected = self._fitted(full).project(full, reduced)
        assert projected.provenance["dropped_blocks"] == ["availability_signal"]

    def test_a_non_nested_layout_is_refused(self):
        """Slicing layouts that are not nested would silently misalign every later block."""
        full, reduced = self._specs()
        with pytest.raises(ValueError, match="absent from the fitted layout"):
            self._fitted(reduced).project(reduced, full)

    def test_an_unfitted_scaler_projects_to_identity(self):
        from src.environment.observation import ObservationScaler

        full, reduced = self._specs()
        projected = ObservationScaler.identity(full.size).project(full, reduced)
        assert not projected.fitted
        assert projected.centre.size == reduced.size


class TestGymnasiumConformance:
    def test_passes_the_gymnasium_env_checker(self) -> None:
        check_env(make_env(), skip_render_check=True)

    def test_observation_matches_the_declared_space(self) -> None:
        env = make_env()
        observation, _ = env.reset()
        assert env.observation_space.contains(observation)

    def test_action_space_is_k_max_plus_one(self) -> None:
        env = make_env()
        assert env.action_space.n == env.cfg.k_max + 1

    def test_reset_returns_observation_and_info(self) -> None:
        observation, info = make_env().reset()
        assert observation.shape == (make_env().obs_spec.size,)
        assert {"scenario", "eval_seed", "arm", "weight_id"} <= set(info)

    def test_step_returns_the_five_tuple_with_finite_reward(self) -> None:
        env = make_env()
        env.reset()
        observation, reward, terminated, truncated, info = env.step(1)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert "components" in info and "epoch_metrics" in info

    def test_episode_terminates_after_exactly_the_configured_epochs(self) -> None:
        env = make_env()
        env.reset()
        steps = 0
        while True:
            _, _, terminated, truncated, _ = env.step(0)
            steps += 1
            if terminated or truncated:
                break
        assert steps == env.sim_config.n_decision_epochs
        assert terminated and not truncated


class TestObservationLayout:
    def test_size_matches_the_frozen_block_list(self) -> None:
        env = make_env()
        observation, _ = env.reset()
        assert observation.size == env.obs_spec.size
        assert observation.size == sum(b.size for b in env.obs_spec.blocks)

    def test_every_feature_has_a_name(self) -> None:
        env = make_env()
        assert len(env.obs_spec.feature_names) == env.obs_spec.size
        assert len(set(env.obs_spec.feature_names)) == env.obs_spec.size

    def test_observations_stay_finite_across_a_full_episode(self) -> None:
        env = make_env()
        observation, _ = env.reset()
        assert np.all(np.isfinite(observation))
        while True:
            observation, _, terminated, truncated, _ = env.step(1)
            assert np.all(np.isfinite(observation))
            if terminated or truncated:
                break

    def test_disabling_the_availability_channel_removes_exactly_two_values(self) -> None:
        with_channel = make_env(availability=True).obs_spec.size
        without = make_env(availability=False).obs_spec.size
        assert with_channel - without == 2

    def test_queue_channel_tracks_the_simulator(self) -> None:
        env = make_env()
        env.reset()
        env.step(0)
        raw = env.raw_observation(refresh=True)
        index = env.obs_spec.slice_of("queue_length")
        expected = [
            sum(q.size for q in env.simulator.queues[s])
            for s in range(env.network.n_stations)
        ]
        assert raw[index] == pytest.approx(expected)

    def test_fleet_counts_sum_to_the_fleet(self) -> None:
        env = make_env()
        env.reset()
        for _ in range(20):
            env.step(1)
        raw = env.raw_observation(refresh=True)
        counts = raw[env.obs_spec.slice_of("fleet_counts")]
        assert counts.sum() == pytest.approx(env.sim_config.fleet_size)


class TestObservationScaler:
    def test_identity_scaler_is_a_no_op(self) -> None:
        env = make_env()
        observation, _ = env.reset()
        assert observation == pytest.approx(env.raw_observation(refresh=True), rel=1e-5)

    def test_fitted_scaler_centres_and_scales(self) -> None:
        env = make_env()
        samples = []
        env.reset()
        for _ in range(30):
            env.step(1)
            samples.append(env.raw_observation(refresh=True))
        scaler = ObservationScaler.fit(np.array(samples), env.obs_spec, {"source": "test"})
        transformed = scaler.transform(np.array(samples))
        normalised = [b for b in env.obs_spec.blocks if b.normalised]
        index = env.obs_spec.slice_of(normalised[0].name)
        assert transformed[:, index].mean() == pytest.approx(0.0, abs=1e-9)

    def test_unnormalised_blocks_are_left_alone(self) -> None:
        env = make_env()
        samples = []
        env.reset()
        for _ in range(30):
            env.step(1)
            samples.append(env.raw_observation(refresh=True))
        scaler = ObservationScaler.fit(np.array(samples), env.obs_spec, {"source": "test"})
        index = env.obs_spec.slice_of("time_index")
        assert scaler.centre[index] == pytest.approx(0.0)
        assert scaler.scale[index] == pytest.approx(1.0)

    def test_strict_mode_refuses_an_unfitted_scaler(self) -> None:
        """A reportable run must not silently use identity normalisation."""
        with pytest.raises(RuntimeError, match="unfitted"):
            make_env(strict_scaler=True)

    def test_roundtrip_through_disk(self, tmp_path) -> None:
        env = make_env()
        scaler = ObservationScaler.fit(
            np.random.default_rng(0).normal(size=(50, env.obs_spec.size)),
            env.obs_spec,
            {"source": "test"},
        )
        path = scaler.save(tmp_path / "obs_scaler.json")
        loaded = ObservationScaler.load(path)
        assert loaded.centre == pytest.approx(scaler.centre)
        assert loaded.scale == pytest.approx(scaler.scale)
        assert loaded.fitted


class TestReproducibility:
    def test_same_seed_gives_an_identical_trajectory(self) -> None:
        first, records_a = rollout(make_env(eval_seed=7))
        second, records_b = rollout(make_env(eval_seed=7))
        assert first["passengers_arrived"] == second["passengers_arrived"]
        assert first["total_reward"] == pytest.approx(second["total_reward"])
        assert [r["reward"] for r in records_a] == pytest.approx([r["reward"] for r in records_b])

    def test_different_seeds_give_different_realisations(self) -> None:
        a, _ = rollout(make_env(eval_seed=7))
        b, _ = rollout(make_env(eval_seed=8))
        assert a["passengers_arrived"] != b["passengers_arrived"]

    def test_reset_is_reproducible_within_one_instance(self) -> None:
        env = make_env(eval_seed=11)
        first, _ = env.reset()
        second, _ = env.reset()
        assert first == pytest.approx(second)

    def test_seed_argument_to_reset_selects_the_evaluation_seed(self) -> None:
        env = make_env(eval_seed=1)
        env.reset(seed=42)
        assert env.eval_seed == 42


class TestCommonRandomNumbers:
    """The fairness guarantee of protocol section 7.1.

    Every method must see the same demand and the same failures in a given
    ``(scenario, eval_seed)`` cell. The arm transform touches only the observation, so it must
    not perturb the environment's streams -- which is why the placebo shuffle draws from an
    agent-side stream instead.
    """

    def test_all_three_arms_see_identical_demand(self) -> None:
        arrivals = {
            arm: rollout(make_env(arm=arm, eval_seed=3))[0]["passengers_arrived"]
            for arm in ("point", "placebo", "uq")
        }
        assert len(set(arrivals.values())) == 1

    def test_all_three_arms_see_identical_failures_and_surge(self) -> None:
        summaries = {
            arm: rollout(make_env(arm=arm, scenario="combined", eval_seed=5,
                                  failure_rate_per_km=4e-5))[0]
            for arm in ("point", "placebo", "uq")
        }
        assert len({s["failures"] for s in summaries.values()}) == 1
        assert len({s["passengers_arrived"] for s in summaries.values()}) == 1

    def test_changing_the_train_seed_does_not_change_the_environment(self) -> None:
        a, _ = rollout(make_env(arm="placebo", train_seed=101, eval_seed=4))
        b, _ = rollout(make_env(arm="placebo", train_seed=999, eval_seed=4))
        assert a["passengers_arrived"] == b["passengers_arrived"]


class TestUncertaintyArms:
    """Protocol section 6.2: identical shape, different content."""

    def test_all_arms_have_identical_observation_dimensionality(self) -> None:
        sizes = {make_env(arm=arm).obs_spec.size for arm in ("point", "placebo", "uq")}
        assert len(sizes) == 1

    def test_point_arm_sigma_channels_are_exactly_zero(self) -> None:
        env = make_env(arm="point")
        env.reset()
        env.step(1)
        raw = env.raw_observation(refresh=True)
        assert np.all(raw[env.obs_spec.slice_of("forecast_sigma_h1")] == 0.0)
        assert np.all(raw[env.obs_spec.slice_of("forecast_sigma_line_h2h3")] == 0.0)

    def test_uq_arm_sigma_channels_are_positive(self) -> None:
        env = make_env(arm="uq")
        env.reset()
        env.step(1)
        raw = env.raw_observation(refresh=True)
        assert np.all(raw[env.obs_spec.slice_of("forecast_sigma_h1")] > 0.0)

    def test_placebo_preserves_the_marginal_but_destroys_the_association(self) -> None:
        """The whole point of the control arm, asserted directly."""
        rng = np.random.default_rng(0)
        arm = make_sigma_arm("placebo", rng)
        truth = rng.gamma(2.0, 5.0, size=(500, 20))
        shuffled = np.array([arm.apply(row) for row in truth])

        # Same marginal: the pooled distributions agree closely.
        assert shuffled.mean() == pytest.approx(truth.mean(), rel=0.05)
        assert shuffled.std() == pytest.approx(truth.std(), rel=0.10)
        # Destroyed association: element-wise correlation collapses.
        correlation = np.corrcoef(truth.ravel(), shuffled.ravel())[0, 1]
        assert abs(correlation) < 0.10

    def test_placebo_output_keeps_the_input_shape(self) -> None:
        arm = make_sigma_arm("placebo", np.random.default_rng(1))
        assert arm.apply(np.ones((7, 3))).shape == (7, 3)

    def test_placebo_requires_an_agent_side_rng(self) -> None:
        with pytest.raises(ValueError, match="agent-side rng"):
            make_sigma_arm("placebo", None)

    def test_unknown_arm_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown arm"):
            make_sigma_arm("magic", None)

    def test_point_and_uq_differ_in_the_observation(self) -> None:
        point = make_env(arm="point")
        uq = make_env(arm="uq")
        point.reset()
        uq.reset()
        point.step(1)
        uq.step(1)
        index = point.obs_spec.slice_of("forecast_sigma_h1")
        assert not np.allclose(point.raw_observation()[index], uq.raw_observation()[index])


class TestInvariants:
    """Physical conservation, carried through the environment layer."""

    def test_passenger_conservation(self) -> None:
        summary, _ = rollout(make_env())
        residual = summary["passengers_arrived"] - (
            summary["passengers_alighted"]
            + summary["passengers_onboard_at_end"]
            + summary["unmet_demand"]
        )
        assert residual == pytest.approx(0.0, abs=1e-6)

    def test_train_conservation(self) -> None:
        env = make_env()
        env.reset()
        for _ in range(50):
            env.step(1)
        assert len(env.simulator.trains) == env.sim_config.fleet_size
        ids = {t.train_id for t in env.simulator.trains}
        assert len(ids) == env.sim_config.fleet_size

    def test_occupancy_is_never_negative_and_never_exceeds_capacity(self) -> None:
        env = make_env(peak_mean=2000.0, capacity_passengers=200.0)
        env.reset()
        while True:
            mask = env.action_masks()
            _, _, terminated, truncated, _ = env.step(int(np.flatnonzero(mask)[-1]))
            for train in env.simulator.trains:
                assert train.occupancy >= -1e-9
                assert train.occupancy <= env.sim_config.capacity_passengers + 1e-6
                assert np.all(train.load >= -1e-9)
            if terminated or truncated:
                break

    def test_queue_sizes_are_never_negative(self) -> None:
        env = make_env(peak_mean=1500.0, capacity_passengers=150.0)
        env.reset()
        while True:
            _, _, terminated, truncated, _ = env.step(1)
            for row in env.simulator.queues:
                for queue in row:
                    assert queue.size >= -1e-9
            if terminated or truncated:
                break

    def test_headway_is_never_violated(self) -> None:
        """Masking makes this structural; a non-zero count would be a bug, not a trade-off."""
        summary, _ = rollout(make_env())
        assert summary["headway_violations"] == 0

    def test_left_behind_fraction_stays_in_the_unit_interval(self) -> None:
        """The A-01 endpoint is a fraction of distinct passengers, so it is bounded."""
        for capacity in (100.0, 400.0, 800.0):
            summary, _ = rollout(make_env(capacity_passengers=capacity, peak_mean=1200.0))
            assert 0.0 <= summary["left_behind_fraction"] <= 1.0

    def test_rewards_are_finite_throughout(self) -> None:
        _, records = rollout(make_env())
        assert all(np.isfinite(r["reward"]) for r in records)

    def test_records_cover_every_epoch(self) -> None:
        env = make_env()
        _, records = rollout(env)
        assert len(records) == env.sim_config.n_decision_epochs
        assert [r["epoch"] for r in records] == list(range(len(records)))


class TestSanityEnvironments:
    """Protocol section 10, stage 2. These are gates, not suggestions."""

    def test_all_trains_unavailable_means_dispatch_is_impossible(self) -> None:
        env = make_env()
        env.reset()
        for train in env.simulator.trains:
            train.status = TrainStatus.MAINTENANCE
        mask = env.action_masks()
        assert mask.tolist() == [True] + [False] * env.cfg.k_max
        _, _, _, _, info = env.step(0)
        assert info["epoch_metrics"]["inductions"] == 0

    def test_zero_demand_makes_running_service_strictly_worse(self) -> None:
        """With nobody to carry, energy and cost are pure loss and the reward must say so."""
        idle, _ = rollout(make_env(peak_mean=0.0), policy=lambda m: 0)
        busy, _ = rollout(make_env(peak_mean=0.0), policy=lambda m: int(np.flatnonzero(m)[-1]))
        assert idle["total_reward"] > busy["total_reward"]
        assert idle["total_reward"] == pytest.approx(0.0)

    def test_extreme_demand_makes_running_service_strictly_better(self) -> None:
        """The converse. If this fails the objective is not measuring passenger service."""
        idle, _ = rollout(make_env(peak_mean=2000.0), policy=lambda m: 0)
        busy, _ = rollout(make_env(peak_mean=2000.0), policy=lambda m: int(np.flatnonzero(m)[-1]))
        assert busy["total_reward"] > idle["total_reward"]

    def test_doing_nothing_is_not_reward_optimal(self) -> None:
        """The regression guard for amendment A-02.

        Under the superseded reading of ``W_t`` -- the mean wait of passengers who *boarded* --
        a policy that ran no service scored exactly 0, the global maximum, while stranding all
        demand. Nobody boards, so there is no mean; no train departs, so nobody is left behind;
        nothing moves, so energy and cost vanish. The queue integral removes the degeneracy.
        """
        idle, _ = rollout(make_env(), policy=lambda m: 0)
        served, _ = rollout(make_env(), policy=lambda m: 1)
        assert served["total_reward"] > idle["total_reward"]
        assert idle["service_level"] == pytest.approx(0.0)
        assert served["service_level"] > 0.9


class TestScenarioFamilies:
    def test_all_four_scenarios_run(self) -> None:
        for scenario in ("normal", "surge", "disruption", "combined"):
            summary, _ = rollout(make_env(scenario=scenario, failure_rate_per_km=4e-5))
            assert summary["scenario"] == scenario
            assert summary["passengers_arrived"] > 0

    def test_surge_raises_demand_relative_to_normal(self) -> None:
        normal, _ = rollout(make_env(scenario="normal", eval_seed=2))
        surge, _ = rollout(make_env(scenario="surge", eval_seed=2))
        assert surge["passengers_arrived"] > normal["passengers_arrived"]

    def test_disruption_removes_trains(self) -> None:
        summary, _ = rollout(make_env(scenario="disruption", eval_seed=2))
        assert summary["failures"] >= 1

    def test_disruption_flag_reflects_a_loss_that_already_happened(self) -> None:
        env = make_env(scenario="disruption", eval_seed=2)
        env.reset()
        index = env.obs_spec.slice_of("disruption_flags")
        assert env.raw_observation(refresh=True)[index][0] == 0.0
        seen_flag = False
        while True:
            _, _, terminated, truncated, _ = env.step(1)
            flags = env.raw_observation(refresh=True)[index]
            if flags[0] == 1.0:
                seen_flag = True
                assert flags[1] >= 1.0
            if terminated or truncated:
                break
        assert seen_flag


class TestWeightGrid:
    def test_the_grid_is_exactly_the_frozen_five(self) -> None:
        assert set(WEIGHT_GRID) == {
            "w_balanced", "w_service", "w_cost", "w_wait_only", "w_unmet_heavy"
        }

    def test_an_unregistered_weight_id_is_refused(self) -> None:
        with pytest.raises(ValueError, match="frozen"):
            make_env(weight_id="w_whatever_works")

    def test_wait_only_ignores_energy_and_cost(self) -> None:
        _, records = rollout(make_env(weight_id="w_wait_only"))
        for record in records:
            assert record["reward"] == pytest.approx(-record["W_norm"])


class TestAvailabilityChannel:
    def test_signal_is_a_probability(self) -> None:
        env = make_env(availability=True)
        env.reset()
        raw = env.raw_observation(refresh=True)[env.obs_spec.slice_of("availability_signal")]
        assert np.all((raw >= 0.0) & (raw <= 1.0))

    def test_frailty_averages_to_one_so_the_calibrated_rate_is_preserved(self) -> None:
        """A heterogeneous hazard must not change the fleet-aggregate failure rate that M5
        calibrated against published MDBF."""
        env = make_env(availability=True)
        multipliers = []
        for seed in range(60):
            env.reset(seed=seed)
            multipliers.extend(t.frailty for t in env.simulator.trains)
        assert np.mean(multipliers) == pytest.approx(1.0, abs=0.05)

    def test_disabling_the_channel_leaves_the_homogeneous_hazard(self) -> None:
        env = make_env(availability=False)
        env.reset()
        assert all(t.frailty == 1.0 for t in env.simulator.trains)


class TestEpisodeAccounting:
    def test_summary_reports_every_reward_component_separately(self) -> None:
        summary, _ = rollout(make_env())
        for key in ("sum_W_minutes", "sum_U_passengers", "sum_E_units", "sum_C_units"):
            assert key in summary and np.isfinite(summary[key])

    def test_total_reward_equals_the_sum_of_step_rewards(self) -> None:
        summary, records = rollout(make_env())
        assert summary["total_reward"] == pytest.approx(sum(r["reward"] for r in records))

    def test_summary_carries_the_run_identity(self) -> None:
        summary, _ = rollout(make_env(arm="placebo", eval_seed=9, train_seed=103))
        assert summary["arm"] == "placebo"
        assert summary["eval_seed"] == 9
        assert summary["train_seed"] == 103

class TestEnvConfigFromYaml:
    """`configs/env.yaml` must be wired to the code, not decorative."""

    def test_unresolved_parameters_block_an_untagged_run(self) -> None:
        """Three availability parameters are `unresolved` (assumption A14). The provenance
        loader must refuse them outside a sensitivity sweep."""
        from src.common.config import ProvenanceError
        from src.environment.metro_env import EnvConfig

        with pytest.raises(ProvenanceError):
            EnvConfig.from_yaml()

    def test_a_tagged_run_resolves_and_matches_the_dataclass_defaults(self) -> None:
        from src.environment.metro_env import EnvConfig

        loaded = EnvConfig.from_yaml(allow_unresolved=True, sweep_tag="sensitivity_sweep")
        default = EnvConfig()
        assert loaded.k_max == default.k_max
        assert loaded.n_segments == default.n_segments
        assert loaded.include_availability_channel == default.include_availability_channel
        assert loaded.availability_noise_sd == default.availability_noise_sd
        assert loaded.frailty_log_sd == default.frailty_log_sd
        assert loaded.weight_id == default.weight_id

    def test_overrides_win(self) -> None:
        from src.environment.metro_env import EnvConfig

        loaded = EnvConfig.from_yaml(allow_unresolved=True, sweep_tag="sensitivity_sweep",
                                     arm="placebo", scenario="surge")
        assert loaded.arm == "placebo" and loaded.scenario == "surge"

    def test_the_mask_threshold_is_frozen_at_zero(self) -> None:
        """A positive threshold would put availability information into the action mask as well
        as the observation, and H7 would stop measuring what it claims to."""
        from src.environment.metro_env import EnvConfig

        assert EnvConfig.from_yaml(allow_unresolved=True, sweep_tag="sensitivity_sweep").availability_mask_threshold == 0.0
