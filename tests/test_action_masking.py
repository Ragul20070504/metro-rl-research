"""Action masking and action distinguishability (protocol section 6.3).

Two separate scientific risks are covered here.

**Masking must be real.** The manuscript (section 3.4) enforces operational constraints by
masking rather than by reward penalties. That is only meaningful if a masked action is genuinely
unreachable: if it silently no-ops, an unmasked policy looks compliant while ignoring the
constraint, and the reported headway-violation count of zero means nothing.

**The action space must not be degenerate.** M5 found a defect in which every induction fired at
the epoch boundary, so the terminal platform refused all but the first and actions 1 and 2 became
behaviourally identical -- a three-way action space collapsed to a binary one, and nothing in the
results would have shown it. These tests exist so that cannot recur silently.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.environment.action_mask import InfeasibleActionError, action_mask
from src.simulator.entities import TrainStatus
from tests.test_environment import make_env, rollout


class TestMaskShape:
    def test_mask_length_matches_the_action_space(self) -> None:
        env = make_env()
        env.reset()
        assert env.action_masks().shape == (env.action_space.n,)

    def test_mask_is_boolean(self) -> None:
        env = make_env()
        env.reset()
        assert env.action_masks().dtype == np.bool_

    def test_action_zero_is_always_feasible(self) -> None:
        """The mask can never be empty, so a policy can never be trapped."""
        env = make_env(peak_mean=1500.0, capacity_passengers=150.0)
        env.reset()
        while True:
            mask = env.action_masks()
            assert mask[0]
            assert mask.any()
            _, _, terminated, truncated, _ = env.step(int(np.flatnonzero(mask)[-1]))
            if terminated or truncated:
                break

    def test_mask_is_contiguous_from_zero(self) -> None:
        """Feasibility in k is monotone: if k trainsets can be inducted, so can k-1."""
        env = make_env()
        env.reset()
        for _ in range(60):
            mask = env.action_masks()
            feasible = np.flatnonzero(mask)
            assert feasible.tolist() == list(range(len(feasible)))
            env.step(int(feasible[-1]))


class TestMaskedActionsAreRejected:
    """Protocol section 6.3: raise rather than silently no-op, and leave state untouched."""

    def _fully_masked_env(self):
        env = make_env()
        env.reset()
        for train in env.simulator.trains:
            train.status = TrainStatus.MAINTENANCE
        return env

    def test_attempting_a_masked_action_raises(self) -> None:
        env = self._fully_masked_env()
        assert not env.action_masks()[1]
        with pytest.raises(InfeasibleActionError, match="masked"):
            env.step(1)

    def test_simulator_state_is_unchanged_after_a_rejected_action(self) -> None:
        env = self._fully_masked_env()
        before = {
            "time_s": env.simulator.time_s,
            "epoch": env.simulator.epoch,
            "totals": copy.deepcopy(env.simulator._totals),
            "queues": [[q.size for q in row] for row in env.simulator.queues],
            "statuses": [t.status for t in env.simulator.trains],
        }
        with pytest.raises(InfeasibleActionError):
            env.step(1)
        assert env.simulator.time_s == before["time_s"]
        assert env.simulator.epoch == before["epoch"]
        assert env.simulator._totals == before["totals"]
        assert [[q.size for q in row] for row in env.simulator.queues] == before["queues"]
        assert [t.status for t in env.simulator.trains] == before["statuses"]

    def test_no_epoch_record_is_written_for_a_rejected_action(self) -> None:
        env = self._fully_masked_env()
        with pytest.raises(InfeasibleActionError):
            env.step(1)
        assert env.episode_records() == []

    def test_out_of_range_actions_are_rejected(self) -> None:
        env = make_env()
        env.reset()
        for action in (-1, 99):
            with pytest.raises(InfeasibleActionError, match="outside"):
                env.step(action)

    def test_a_masked_action_never_executes_across_a_whole_episode(self) -> None:
        """The positive control: a mask-respecting policy runs a full day without raising."""
        env = make_env()
        env.reset()
        executed = []
        while True:
            mask = env.action_masks()
            action = int(np.flatnonzero(mask)[-1])
            assert mask[action]
            executed.append(action)
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        assert len(executed) == env.sim_config.n_decision_epochs


class TestConstraintsDriveTheMask:
    def test_no_idle_trainsets_means_only_action_zero(self) -> None:
        env = make_env()
        env.reset()
        for train in env.simulator.trains:
            train.status = TrainStatus.RUNNING
        assert env.action_masks().tolist() == [True, False, False]

    def test_an_occupied_terminal_platform_blocks_induction(self) -> None:
        env = make_env()
        env.reset()
        terminal = env.network.terminal_for(0)
        env.simulator._platform_occupant[terminal][0] = 0
        assert env.action_masks().tolist() == [True, False, False]

    def test_minimum_headway_blocks_induction(self) -> None:
        env = make_env()
        env.reset()
        terminal = env.network.terminal_for(0)
        env.simulator._last_departure_s[terminal][0] = env.simulator.time_s
        assert env.action_masks().tolist() == [True, False, False]

    def test_headway_block_clears_once_the_interval_has_passed(self) -> None:
        env = make_env()
        env.reset()
        terminal = env.network.terminal_for(0)
        env.simulator._last_departure_s[terminal][0] = (
            env.simulator.time_s - env.network.min_headway_seconds - 1.0
        )
        assert env.action_masks()[1]

    def test_mask_never_exceeds_the_idle_fleet(self) -> None:
        env = make_env()
        env.reset()
        for train in env.simulator.trains[1:]:
            train.status = TrainStatus.RUNNING
        assert env.action_masks().tolist() == [True, True, False]

    def test_availability_threshold_can_withhold_trains(self) -> None:
        """The frozen configuration keeps the threshold at 0 so that availability information
        reaches the policy through the observation only; the mechanism is still tested."""
        env = make_env()
        env.reset()
        for train in env.simulator.trains:
            train.status = TrainStatus.DEPOT
        low = np.zeros(env.sim_config.fleet_size)
        mask = action_mask(
            env.simulator, env.cfg.k_max,
            availability_probability=low, availability_threshold=0.5,
        )
        assert mask.tolist() == [True, False, False]

    def test_headway_violations_stay_at_zero_under_every_scenario(self) -> None:
        for scenario in ("normal", "surge", "disruption", "combined"):
            summary, _ = rollout(make_env(scenario=scenario, failure_rate_per_km=4e-5))
            assert summary["headway_violations"] == 0, scenario


class TestActionDistinguishability:
    """The M5 defect class: different valid actions must have different consequences."""

    @staticmethod
    def _by_action(**kwargs) -> dict[int, dict]:
        out = {}
        for action in (0, 1, 2):
            summary, _ = rollout(
                make_env(**kwargs),
                policy=lambda mask, a=action: a if mask[a] else int(np.flatnonzero(mask)[-1]),
            )
            out[action] = summary
        return out

    def test_action_zero_differs_sharply_from_running_service(self) -> None:
        results = self._by_action()
        assert results[0]["inductions"] == 0
        assert results[1]["inductions"] > 100
        assert results[0]["service_level"] == pytest.approx(0.0)
        assert results[1]["service_level"] > 0.9

    def test_action_one_and_two_produce_different_dispatch(self) -> None:
        results = self._by_action()
        assert results[2]["inductions"] > results[1]["inductions"]

    def test_action_one_and_two_produce_different_passenger_outcomes(self) -> None:
        """Under pressure the two actions must separate on the primary endpoint, not merely on
        the number of trains dispatched."""
        results = self._by_action(peak_mean=1200.0, capacity_passengers=200.0)
        assert results[1]["left_behind_fraction"] != results[2]["left_behind_fraction"]
        assert results[1]["waiting_accrued_passenger_hours"] != pytest.approx(
            results[2]["waiting_accrued_passenger_hours"]
        )

    def test_action_one_and_two_produce_different_energy_and_cost(self) -> None:
        results = self._by_action()
        assert results[2]["energy_units"] > results[1]["energy_units"]
        assert results[2]["cost_units"] > results[1]["cost_units"]

    def test_action_one_and_two_produce_different_fleet_state(self) -> None:
        results = self._by_action()
        assert results[2]["train_km"] != pytest.approx(results[1]["train_km"])

    def test_actions_produce_different_rewards(self) -> None:
        results = self._by_action()
        rewards = {a: results[a]["total_reward"] for a in results}
        assert len(set(round(v, 6) for v in rewards.values())) == 3

    def test_running_service_shortens_waiting(self) -> None:
        """The direction of the effect, not merely its existence. Wrong-signed responsiveness
        would look like distinguishability while making the objective meaningless."""
        results = self._by_action(peak_mean=1200.0, capacity_passengers=200.0)
        assert (
            results[1]["waiting_accrued_passenger_hours"]
            < results[0]["waiting_accrued_passenger_hours"]
        )

    def test_inducting_two_at_a_time_bunches_and_lengthens_the_tail(self) -> None:
        """Measured, and counterintuitive enough to pin down.

        Action 2 does **not** improve service. Because the terminal can only clear a train every
        ``min_headway`` seconds, asking for two inductions inside one 300 s interval produces a
        closely spaced pair followed by a long gap, rather than two evenly spaced departures.
        The same number of round trips is completed, but headways are less regular, so the
        95th-percentile wait rises (measured 210 s -> 330 s on the toy line) and more passengers
        are passed over by a full train.

        This is the bunching limitation recorded in protocol amendment A-01, showing up in the
        simulator exactly as that amendment predicted. If this test ever inverts, either the
        turnaround/headway model or the terminal dispatch logic has changed, and the primary
        endpoint's interpretation changes with it.
        """
        results = self._by_action(peak_mean=1200.0, capacity_passengers=200.0)
        assert results[2]["waiting_time_p95_s"] > results[1]["waiting_time_p95_s"]
        assert results[2]["left_behind_fraction"] > results[1]["left_behind_fraction"]
        assert results[2]["trips_completed"] <= results[1]["trips_completed"] + 1


class TestInductionSuccessRate:
    """Protocol section 7.4 records induction success rate; a persistent shortfall is a fact
    about the infrastructure, not a bug, and must be visible rather than inferred."""

    def test_requested_inductions_are_not_always_delivered(self) -> None:
        env = make_env()
        requested = 0
        env.reset()
        while True:
            mask = env.action_masks()
            action = int(np.flatnonzero(mask)[-1])
            requested += action
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                delivered = info["episode_summary"]["inductions"]
                break
        assert delivered <= requested
        assert delivered > 0
