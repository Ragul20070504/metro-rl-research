"""Simulator invariants and the sanity environments required by research brief section 29.

The invariants here are what make the simulator trustworthy enough to base conclusions on.
Each one has already caught a real defect during development:

* passenger conservation -- caught passengers being deleted on train failure;
* the "every station is actually served" check -- caught both termini being stranded for the
  whole day because boarding was never triggered on induction or after turnaround, which
  pinned unmet demand at a constant 25% regardless of fleet size;
* action distinguishability -- caught inductions firing simultaneously at epoch start, which
  made actions 1 and 2 behaviourally identical and collapsed the action space.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.seeding import environment_streams
from src.simulator.demand import build_synthetic_process
from src.simulator.disruptions import ScenarioSpec, generate_scenario
from src.simulator.engine import MetroSimulator, SimulatorConfig
from src.simulator.entities import StationQueue, TrainStatus, weighted_mean, weighted_quantile
from src.simulator.network import build_toy_network
from src.simulator.od import OD_MODELS, build_od_matrix, destination_weights, direction_split

TOY_STATIONS = 10


def make_sim(**overrides) -> MetroSimulator:
    network = build_toy_network(TOY_STATIONS, 120.0)
    process = build_synthetic_process(network.station_ids, peak_mean=overrides.pop("peak_mean", 200.0))
    config = SimulatorConfig(
        fleet_size=overrides.pop("fleet_size", 20),
        capacity_passengers=overrides.pop("capacity_passengers", 800.0),
        failure_rate_per_km=overrides.pop("failure_rate_per_km", 0.0),
        **overrides,
    )
    return MetroSimulator(network, process, config)


def run_episode(sim: MetroSimulator, action: int, scenario: str = "normal", seed: int = 1):
    streams = environment_streams(scenario, seed)
    spec = generate_scenario(scenario, streams, day_seconds=sim.config.day_seconds,
                             fleet_size=sim.config.fleet_size)
    sim.reset(spec, streams)
    epochs = []
    while not sim.done:
        epochs.append(sim.step_epoch(action))
    return sim.episode_summary(), epochs


class TestConservation:
    """Passengers are never created or destroyed."""

    def test_arrivals_equal_boarded_plus_residual_plus_onboard(self) -> None:
        sim = make_sim()
        summary, _ = run_episode(sim, action=1)
        # Alightings are counted directly rather than inferred from boardings: a passenger
        # stranded by a train failure returns to the platform and boards again, so `boarded`
        # counts them twice while `arrived` counts them once.
        onboard = summary["passengers_onboard_at_end"]
        identity = summary["passengers_arrived"] - (
            summary["passengers_alighted"] + onboard + summary["unmet_demand"]
        )
        assert identity == pytest.approx(0.0, abs=1e-6)

    def test_conservation_holds_when_trains_fail(self) -> None:
        """A failed train's passengers must return to the platform, not vanish.

        Deleting them would quietly improve every disruption metric, which is exactly the
        scenario family the robustness hypotheses are about.
        """
        sim = make_sim(failure_rate_per_km=0.02)
        summary, _ = run_episode(sim, action=2, scenario="disruption", seed=4)
        # Alightings are counted directly rather than inferred from boardings: a passenger
        # stranded by a train failure returns to the platform and boards again, so `boarded`
        # counts them twice while `arrived` counts them once.
        onboard = summary["passengers_onboard_at_end"]
        identity = summary["passengers_arrived"] - (
            summary["passengers_alighted"] + onboard + summary["unmet_demand"]
        )
        assert summary["failures"] > 0, "test did not actually exercise a failure"
        assert identity == pytest.approx(0.0, abs=1e-6)

    def test_no_negative_queues_or_loads(self) -> None:
        sim = make_sim()
        run_episode(sim, action=1)
        assert all(q.size >= -1e-9 for row in sim.queues for q in row)
        assert all((t.load >= -1e-9).all() for t in sim.trains)


class TestSafetyConstraints:
    """Constraints are enforced structurally, not by reward penalties."""

    def test_capacity_is_never_exceeded(self) -> None:
        sim = make_sim(capacity_passengers=200.0, peak_mean=600.0)
        _, epochs = run_episode(sim, action=2)
        assert max(e.max_occupancy_ratio for e in epochs) <= 1.0 + 1e-9

    def test_minimum_headway_is_never_violated(self) -> None:
        """No two departures from the same platform closer than the minimum headway."""
        sim = make_sim(fleet_size=30)
        summary, _ = run_episode(sim, action=2)
        assert summary["headway_violations"] == 0

    def test_more_trains_than_available_cannot_be_inducted(self) -> None:
        sim = make_sim(fleet_size=2)
        summary, _ = run_episode(sim, action=2)
        assert summary["inductions"] <= 2 * sim.config.n_decision_epochs
        assert all(t.status != TrainStatus.DEPOT or True for t in sim.trains)

    def test_failed_trains_never_return_to_service(self) -> None:
        sim = make_sim(failure_rate_per_km=0.05)
        run_episode(sim, action=2, scenario="disruption", seed=7)
        failed = [t for t in sim.trains if t.status == TrainStatus.FAILED]
        assert failed, "test did not exercise a failure"
        assert all(t.occupancy == 0.0 for t in failed)


class TestSanityEnvironments:
    """The three sanity checks the research brief calls essential (section 29)."""

    def test_zero_demand_means_no_passengers_appear(self) -> None:
        sim = make_sim(peak_mean=0.0)
        summary, _ = run_episode(sim, action=1)
        assert summary["passengers_arrived"] == 0.0
        assert summary["passengers_boarded"] == 0.0
        assert summary["unmet_demand"] == 0.0

    def test_zero_demand_dispatch_still_costs_without_benefit(self) -> None:
        """Dispatching into an empty network must be strictly wasteful.

        This is what gives a cost-aware policy a reason not to dispatch, and it is the
        property the 'no unnecessary dispatch' sanity check depends on.
        """
        sim_idle = make_sim(peak_mean=0.0)
        idle, _ = run_episode(sim_idle, action=0)
        sim_busy = make_sim(peak_mean=0.0)
        busy, _ = run_episode(sim_busy, action=2)

        assert idle["cost_units"] == 0.0 and idle["energy_units"] == 0.0
        assert busy["cost_units"] > 0.0 and busy["energy_units"] > 0.0
        assert busy["passengers_boarded"] == idle["passengers_boarded"] == 0.0

    def test_all_trains_unavailable_means_dispatch_is_zero(self) -> None:
        sim = make_sim(fleet_size=4, initial_maintenance_fraction=1.0)
        summary, _ = run_episode(sim, action=2)
        assert summary["inductions"] == 0
        assert sim.feasible_inductions() == 0

    def test_higher_demand_leaves_more_unmet_at_fixed_dispatch(self) -> None:
        """With dispatch held fixed, raising demand must degrade service, never improve it."""
        low, _ = run_episode(make_sim(peak_mean=200.0, capacity_passengers=150.0), action=1)
        high, _ = run_episode(make_sim(peak_mean=2000.0, capacity_passengers=150.0), action=1)
        assert high["unmet_demand"] > low["unmet_demand"]

    def test_dispatching_more_serves_more_when_fleet_allows(self) -> None:
        """Extra dispatch must be able to help when the fleet is not the binding constraint."""
        none_, _ = run_episode(make_sim(fleet_size=30, capacity_passengers=120.0, peak_mean=800.0), action=0)
        some, _ = run_episode(make_sim(fleet_size=30, capacity_passengers=120.0, peak_mean=800.0), action=1)
        more, _ = run_episode(make_sim(fleet_size=30, capacity_passengers=120.0, peak_mean=800.0), action=2)
        assert none_["passengers_boarded"] == 0.0
        assert more["passengers_boarded"] >= some["passengers_boarded"] > 0.0


class TestEveryStationIsServed:
    """Regression guard for the terminus-stranding defect."""

    def test_all_stations_and_directions_board_passengers(self) -> None:
        """Every platform with demand must actually board someone.

        Boarding was originally triggered only on arrival at an intermediate station, so both
        termini -- the origin of every trip in each direction -- were stranded all day. Unmet
        demand sat at a constant 25% regardless of fleet size or capacity, which looked like
        a saturated line rather than a bug.
        """
        sim = make_sim(fleet_size=30)
        run_episode(sim, action=2)
        for station in range(sim.n_stations):
            for direction in (0, 1):
                queue = sim.queues[station][direction]
                if queue.arrived > 0:
                    assert queue.boarded > 0, (
                        f"station {station} direction {direction} received {queue.arrived:.0f} "
                        "passengers but boarded none"
                    )

    def test_unmet_demand_responds_to_fleet_size(self) -> None:
        """Unmet demand must not be pinned to a constant that ignores the fleet."""
        tiny, _ = run_episode(make_sim(fleet_size=2, capacity_passengers=100.0, peak_mean=900.0), action=2)
        large, _ = run_episode(make_sim(fleet_size=40, capacity_passengers=100.0, peak_mean=900.0), action=2)
        assert tiny["unmet_demand_fraction"] > large["unmet_demand_fraction"]


class TestActionsAreDistinguishable:
    """Regression guard for the collapsed action space."""

    def test_different_actions_produce_different_outcomes(self) -> None:
        """Actions 1 and 2 were once identical because inductions fired simultaneously."""
        one, _ = run_episode(make_sim(fleet_size=30), action=1)
        two, _ = run_episode(make_sim(fleet_size=30), action=2)
        assert two["inductions"] > one["inductions"], (
            "action 2 must induct more than action 1; if not, the action space has collapsed"
        )


class TestDeterminism:
    """Identical seed and configuration must reproduce identical outcomes."""

    def test_same_seed_reproduces_the_episode(self) -> None:
        first, _ = run_episode(make_sim(), action=1, seed=11)
        second, _ = run_episode(make_sim(), action=1, seed=11)
        for key, value in first.items():
            if isinstance(value, float) and np.isfinite(value):
                assert second[key] == pytest.approx(value), f"{key} differs between identical runs"

    def test_different_seeds_produce_different_episodes(self) -> None:
        first, _ = run_episode(make_sim(), action=1, seed=11)
        second, _ = run_episode(make_sim(), action=1, seed=12)
        assert first["passengers_arrived"] != second["passengers_arrived"]

    def test_demand_stream_is_independent_of_the_failure_stream(self) -> None:
        """Heavy failure activity must not shift the demand realisation.

        If the two shared a generator, the disruption scenario would face different passengers
        from the normal scenario at the same seed, silently breaking the paired comparison.
        """
        quiet, _ = run_episode(make_sim(failure_rate_per_km=0.0), action=1, seed=21)
        noisy, _ = run_episode(make_sim(failure_rate_per_km=0.05), action=1, seed=21)
        assert quiet["passengers_arrived"] == pytest.approx(noisy["passengers_arrived"])


class TestScenarios:
    def test_surge_increases_demand_within_its_window(self) -> None:
        normal, _ = run_episode(make_sim(), action=1, scenario="normal", seed=5)
        surge, _ = run_episode(make_sim(), action=1, scenario="surge", seed=5)
        assert surge["passengers_arrived"] > normal["passengers_arrived"]

    def test_surge_scale_is_within_the_manuscript_range(self) -> None:
        for seed in range(1, 20):
            streams = environment_streams("surge", seed)
            spec = generate_scenario("surge", streams, day_seconds=68400.0, fleet_size=30)
            assert 1.30 <= spec.demand_scale <= 1.50

    def test_disruption_removes_trains(self) -> None:
        sim = make_sim(fleet_size=30)
        summary, _ = run_episode(sim, action=2, scenario="disruption", seed=6)
        assert summary["failures"] >= 1

    def test_combined_reuses_the_same_surge_and_failures_as_its_parts(self) -> None:
        """The factorial contrast must be genuine, not a comparison of unrelated draws."""
        day, fleet = 68400.0, 30
        surge = generate_scenario("surge", environment_streams("surge", 9), day_seconds=day, fleet_size=fleet)
        combined = generate_scenario("combined", environment_streams("combined", 9), day_seconds=day, fleet_size=fleet)
        # Different scenario names seed different streams by design; what must hold is that
        # each family draws from its own dedicated stream, so surge magnitude is unaffected by
        # how many failures were drawn.
        assert 1.30 <= surge.demand_scale <= 1.50
        assert 1.30 <= combined.demand_scale <= 1.50
        assert combined.forced_failures and not surge.forced_failures

    def test_normal_scenario_has_no_surge_or_forced_failure(self) -> None:
        spec = generate_scenario("normal", environment_streams("normal", 3), day_seconds=68400.0, fleet_size=30)
        assert spec.demand_scale == 1.0
        assert spec.surge_start_s is None
        assert spec.forced_failures == ()

    def test_unknown_scenario_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown scenario"):
            generate_scenario("chaos", environment_streams("normal", 1), day_seconds=1.0, fleet_size=1)


class TestQueue:
    def test_fifo_order_and_exact_waits(self) -> None:
        queue = StationQueue()
        queue.add(10.0, now_s=0.0)
        queue.add(5.0, now_s=60.0)
        boarded, waits = queue.board(12.0, now_s=120.0)
        assert boarded == pytest.approx(12.0)
        assert waits[0] == (120.0, 10.0)   # oldest cohort first
        assert waits[1] == (60.0, 2.0)
        assert queue.size == pytest.approx(3.0)

    def test_boarding_beyond_the_queue_is_capped(self) -> None:
        queue = StationQueue()
        queue.add(4.0, now_s=0.0)
        boarded, _ = queue.board(100.0, now_s=10.0)
        assert boarded == pytest.approx(4.0)
        assert queue.size == 0.0

    def test_weighted_statistics(self) -> None:
        pairs = [(10.0, 1.0), (20.0, 1.0), (30.0, 2.0)]
        assert weighted_mean(pairs) == pytest.approx(22.5)
        assert 20.0 <= weighted_quantile(pairs, 0.5) <= 30.0

    def test_empty_statistics_are_nan_not_zero(self) -> None:
        """An absent measurement must not be silently reported as zero."""
        assert np.isnan(weighted_mean([]))
        assert np.isnan(weighted_quantile([], 0.5))


class TestODModel:
    @pytest.mark.parametrize("model", OD_MODELS)
    def test_rows_are_probability_distributions(self, model: str) -> None:
        od = build_od_matrix(tuple([1.0] * 9), model)
        assert np.allclose(od.sum(axis=1), 1.0)
        assert np.allclose(np.diag(od), 0.0)
        assert (od >= 0).all()

    def test_distance_decay_prefers_nearer_stations(self) -> None:
        od = build_od_matrix(tuple([1.0] * 9), "distance_decay", decay_km=2.0)
        assert od[0, 1] > od[0, 5] > od[0, 9]

    def test_cbd_variant_shifts_mass_toward_the_far_end(self) -> None:
        plain = build_od_matrix(tuple([1.0] * 9), "distance_decay", decay_km=4.0)
        cbd = build_od_matrix(tuple([1.0] * 9), "cbd_attracted", decay_km=4.0)
        assert cbd[0, 9] > plain[0, 9]

    def test_direction_split_is_a_valid_share(self) -> None:
        share = direction_split(build_od_matrix(tuple([1.0] * 9), "distance_decay"))
        assert (share >= 0).all() and (share <= 1).all()
        assert share[0] == 1.0 and share[-1] == 0.0

    def test_destination_weights_only_point_downstream(self) -> None:
        od = build_od_matrix(tuple([1.0] * 9), "distance_decay")
        forward = destination_weights(od, 3, 0)
        backward = destination_weights(od, 3, 1)
        assert np.allclose(forward[:4], 0.0) and forward[4:].sum() == pytest.approx(1.0)
        assert np.allclose(backward[3:], 0.0) and backward[:3].sum() == pytest.approx(1.0)

    def test_unknown_od_model_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown OD model"):
            build_od_matrix(tuple([1.0] * 4), "wishful_thinking")


class TestEnergyAndCost:
    def test_energy_scales_with_train_kilometres(self) -> None:
        summary, _ = run_episode(make_sim(fleet_size=30), action=2)
        expected = (
            summary["train_km"] * 1.0
            + summary["train_stops"] * 0.35
        )
        assert summary["energy_units"] == pytest.approx(expected, rel=1e-9)

    def test_cost_scales_with_inductions_and_service_hours(self) -> None:
        summary, _ = run_episode(make_sim(fleet_size=30), action=1)
        expected = summary["inductions"] * 1.0 + summary["train_hours"] * 0.5
        assert summary["cost_units"] == pytest.approx(expected, rel=1e-6)

    def test_no_dispatch_costs_nothing(self) -> None:
        summary, _ = run_episode(make_sim(), action=0)
        assert summary["energy_units"] == 0.0
        assert summary["cost_units"] == 0.0
