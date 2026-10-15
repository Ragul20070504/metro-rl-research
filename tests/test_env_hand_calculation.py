"""A deterministic scenario worked out by hand, then asserted against the implementation.

Every other test in this suite checks an *invariant* -- conservation, boundedness, monotonicity.
Invariants are necessary but they are not sufficient: a simulator that halves every waiting time
satisfies all of them. This test is the one that pins the arithmetic down, by fixing a scenario
small enough to compute on paper and requiring the code to reproduce the numbers.

**The scenario.**

* **2 stations** in a line; the segment is 120 s of running time and 1.0 km.
* Minimum headway 150 s, turnaround 300 s, base dwell 30 s, 0.08 s of dwell per boarder.
* 2 trainsets, no maintenance holds, failures disabled.
* Capacity 1000, so nobody is ever denied.
* Internal step 30 s, decision interval 300 s, therefore **10 internal steps per epoch**.
* Demand: hourly intensity fixed at 120 passengers/hour at every station and hour, and arrivals
  taken as the rounded expectation rather than a Poisson draw. With a uniform intra-hour profile
  over 120 steps per hour, that is exactly **1 arrival per station per step**.
* Energy 1.0 unit/train-km and 0.35 units/stop; cost 1.0 unit/induction and 0.5 units/train-hour.

Two stations rather than three, deliberately. With three, a trainset inducted at station 0
reaches station 1 mid-epoch and boards **that** platform too, which pulls an extra term into
every quantity and makes the paper arithmetic much easier to get wrong than the code. (The first
draft of this test asserted 12 boarders in epoch 2 and the simulator returned 21; the simulator
was right and the derivation had simply forgotten station 1.) With two stations, station 1 is the
direction-0 terminus, so its forward platform is always empty and the epoch-2 boarding is
unambiguously station 0's queue alone.

**Epoch 1, action 0 (induct nothing).**

Each step k = 1..10 adds 2 passengers (one per station), so the line queue after step k is 2k.
The queue integral is taken after arrivals, so::

    accrued = sum_{k=1..10} 2k * 30 s = 60 * (1+2+...+10) = 60 * 55 = 3300 passenger-seconds
            = 55 passenger-minutes

    arrivals = 20      queue at epoch end = 20      left behind = 0
    energy = 0 (nothing moves)      cost = 0 (no inductions, nothing in service)

With unit reference scales and ``w_balanced`` (all weights 1), the reward is exactly::

    r = -(55 + 0 + 0 + 0) = -55.0

**Epoch 2, action 1 (induct one trainset).**

Station 0 is the direction-0 terminus, so every passenger arriving there travels forward and its
forward platform holds all of them; station 1 is the direction-1 terminus, so its arrivals all
queue backward and no forward-running train can pick them up.

*Step 1* (t = 300 s). Arrivals bring the line queue to 22, so the integral gains 22*30 = 660.
The induction then fires: the trainset enters at station 0 and boards the forward platform's
**11** waiting passengers -- the cohorts that arrived at t = 0, 30, ..., 300. Their waits are
300, 270, ..., 0 seconds, summing to 1650 passenger-seconds. Dwell is 30 + 0.08*11 = 30.88 s.
The queue drops to 11 (all at station 1, backward).

*Step 2* (t = 330 s). Arrivals take the queue to 13; one passenger at station 0 boards through
the still-open doors with a wait of 0 s, making **12** boarders whose mean wait is
1650/12 = **137.5 s**. The dwell timer expires, headway since the previous departure is
unbounded, so the train departs and begins a 120 s run.

*Steps 3-10.* Nothing else boards: the train is running, and station 1's forward platform is
empty by construction. The queue grows by 2 each step from 12, and the integral accumulates::

    step   1     2     3     4     5     6     7     8     9    10
    queue 22    13    14    16    18    20    22    24    26    28
    x30 s 660   390   420   480   540   600   660   720   780   840   -> 6090 passenger-seconds
                                                                      =  101.5 passenger-minutes

The train reaches station 1 at t = 450 s -- the epoch's only segment traversal::

    energy = 1.0 km * 1.0 unit/km + 0.35 units/stop = 1.35 units

It alights all 12 passengers, dwells 30 + 0.08*12 = 30.96 s, and at t = 510 s begins its 300 s
turnaround, which is still running when the epoch ends.

*Cost.* One induction, and one trainset in service for all 10 steps::

    cost = 1.0 + (10 * 30 s / 3600) * 0.5 = 1.0 + 0.0416667 = 1.0416667 units

*Reward.* ``r = -(101.5 + 0 + 1.35 + 1.0416667) = -103.8916667``
"""

from __future__ import annotations

import numpy as np
import pytest

from src.environment.forecast_provider import SeasonalNaiveProvider
from src.environment.metro_env import EnvConfig, MetroEnv, constant_history_sampler
from src.environment.reward import WEIGHT_GRID, ReferenceScales
from src.simulator.demand import build_synthetic_process
from src.simulator.engine import SimulatorConfig
from src.simulator.network import build_toy_network

N_STATIONS = 2
INTENSITY_PER_HOUR = 120.0


def hand_calc_env() -> MetroEnv:
    """The scenario described in the module docstring."""
    network = build_toy_network(N_STATIONS, 120.0)
    process = build_synthetic_process(network.station_ids, peak_mean=INTENSITY_PER_HOUR)
    config = SimulatorConfig(
        internal_step_seconds=30.0,
        decision_interval_seconds=300.0,
        dwell_base_seconds=30.0,
        dwell_per_boarder_seconds=0.08,
        capacity_passengers=1000.0,
        fleet_size=2,
        initial_maintenance_fraction=0.0,
        failures_enabled=False,
        failure_rate_per_km=0.0,
        deterministic_arrivals=True,
        energy_units_per_train_km=1.0,
        energy_units_per_stop=0.35,
        cost_units_per_induction=1.0,
        cost_units_per_train_hour=0.5,
    )
    env = MetroEnv(
        network=network,
        demand_process=process,
        simulator_config=config,
        env_config=EnvConfig(
            arm="point", scenario="normal", include_availability_channel=False, n_segments=2
        ),
        forecast_provider=SeasonalNaiveProvider(N_STATIONS),
        history_sampler=constant_history_sampler(process, config.day_start_hour),
        scales=ReferenceScales.unit(),
        eval_seed=1,
    )
    env.reset()
    # Pin the day's hourly intensity to the flat value the hand calculation assumes. The fitted
    # process draws a stochastic intensity path, which is correct for experiments and useless
    # for a paper-and-pencil check.
    env.simulator._hourly_intensity = np.full((24, N_STATIONS), INTENSITY_PER_HOUR)
    return env


class TestScenarioPreconditions:
    """If these do not hold, the arithmetic in the docstring does not describe this scenario."""

    def test_ten_internal_steps_per_epoch(self) -> None:
        assert hand_calc_env().sim_config.steps_per_decision == 10

    def test_one_arrival_per_station_per_step(self) -> None:
        env = hand_calc_env()
        weights = env.simulator._hour_weights
        rate = INTENSITY_PER_HOUR * weights[0]
        assert rate == pytest.approx(1.0)

    def test_station_zero_sends_everyone_forward(self) -> None:
        """Station 0 is the direction-0 terminus, so no backward trip can start there."""
        assert hand_calc_env().simulator.direction_share[0] == pytest.approx(1.0)

    def test_no_trainset_starts_on_a_maintenance_hold(self) -> None:
        env = hand_calc_env()
        assert all(t.available_for_induction for t in env.simulator.trains)


class TestEpochOneByHand:
    """Action 0: nothing runs, so every quantity is a closed-form sum."""

    def test_arrivals_are_exactly_twenty(self) -> None:
        env = hand_calc_env()
        _, _, _, _, info = env.step(0)
        assert info["epoch_metrics"]["arrived"] == pytest.approx(20.0)

    def test_queue_integral_is_3300_passenger_seconds(self) -> None:
        env = hand_calc_env()
        _, _, _, _, info = env.step(0)
        # sum_{k=1..10} 2k * 30 = 60 * 55
        assert info["epoch_metrics"]["waiting_accrued_passenger_s"] == pytest.approx(3300.0)

    def test_queue_at_epoch_end_is_twenty(self) -> None:
        env = hand_calc_env()
        _, _, _, _, info = env.step(0)
        assert info["epoch_metrics"]["queue_at_epoch_end"] == pytest.approx(20.0)

    def test_nothing_moves_so_energy_and_cost_are_zero(self) -> None:
        env = hand_calc_env()
        _, _, _, _, info = env.step(0)
        assert info["components"]["E_units"] == pytest.approx(0.0)
        assert info["components"]["C_units"] == pytest.approx(0.0)

    def test_nobody_is_left_behind_because_no_train_departs(self) -> None:
        env = hand_calc_env()
        _, _, _, _, info = env.step(0)
        assert info["components"]["U_passengers"] == pytest.approx(0.0)

    def test_reward_is_exactly_minus_55(self) -> None:
        """The whole chain: queue integral -> W_t -> normalisation -> weighted sum -> reward."""
        env = hand_calc_env()
        _, reward, _, _, _ = env.step(0)
        assert reward == pytest.approx(-55.0)


class TestEpochTwoByHand:
    """Action 1: one induction, and the trainset's first segment."""

    @staticmethod
    def _two_epochs() -> dict:
        env = hand_calc_env()
        env.step(0)
        _, reward, _, _, info = env.step(1)
        return {"reward": reward, **info}

    def test_twelve_passengers_board(self) -> None:
        """11 waiting at t=300 plus 1 arriving at t=330 into the still-open doors."""
        result = self._two_epochs()
        assert result["epoch_metrics"]["boarded"] == pytest.approx(12.0)

    def test_queue_integral_is_6090_passenger_seconds(self) -> None:
        """The step-by-step table in the module docstring, summed."""
        result = self._two_epochs()
        assert result["epoch_metrics"]["waiting_accrued_passenger_s"] == pytest.approx(6090.0)

    def test_mean_wait_of_boarders_is_137_point_5_seconds(self) -> None:
        """(300 + 270 + ... + 0) + 0 = 1650 passenger-seconds over 12 passengers."""
        result = self._two_epochs()
        assert result["epoch_metrics"]["waiting_time_mean_s"] == pytest.approx(1650.0 / 12.0)

    def test_energy_is_one_segment_traversal(self) -> None:
        """1.0 km * 1.0 unit/km + 0.35 units/stop. The second segment lands in epoch 3."""
        result = self._two_epochs()
        assert result["components"]["E_units"] == pytest.approx(1.35)

    def test_cost_is_one_induction_plus_ten_steps_of_service(self) -> None:
        result = self._two_epochs()
        expected = 1.0 + (10 * 30.0 / 3600.0) * 0.5
        assert result["components"]["C_units"] == pytest.approx(expected)

    def test_exactly_one_induction_happened(self) -> None:
        result = self._two_epochs()
        assert result["epoch_metrics"]["inductions"] == 1

    def test_nobody_is_left_behind_at_this_capacity(self) -> None:
        """Capacity is 1000 against a queue of 12, so a departure denies nobody."""
        result = self._two_epochs()
        assert result["components"]["U_passengers"] == pytest.approx(0.0)

    def test_reward_is_exactly_minus_103_point_89(self) -> None:
        """Every term derived independently on paper, then summed."""
        result = self._two_epochs()
        expected = -(6090.0 / 60.0 + 0.0 + 1.35 + (1.0 + (10 * 30.0 / 3600.0) * 0.5))
        assert expected == pytest.approx(-103.8916667)
        assert result["reward"] == pytest.approx(expected)


class TestHandCalculationDetectsRegression:
    """The point of the fixture is that it *fails* when the arithmetic changes."""

    def test_doubling_the_arrival_rate_doubles_the_queue_integral(self) -> None:
        env = hand_calc_env()
        env.simulator._hourly_intensity = np.full((24, N_STATIONS), INTENSITY_PER_HOUR * 2.0)
        _, _, _, _, info = env.step(0)
        assert info["epoch_metrics"]["waiting_accrued_passenger_s"] == pytest.approx(3300.0 * 2.0)

    def test_deterministic_arrivals_round_half_to_even(self) -> None:
        """A property of the test-only deterministic mode, asserted so it cannot surprise later.

        ``np.rint`` rounds half to even, so a per-step rate of exactly 0.5 yields **zero**
        arrivals, not one. Any hand-calculated scenario must therefore use a per-step rate that
        is an integer. Experiments never take this path -- they draw Poisson counts, where a
        rate of 0.5 behaves normally.
        """
        env = hand_calc_env()
        env.simulator._hourly_intensity = np.full((24, N_STATIONS), INTENSITY_PER_HOUR / 2.0)
        _, _, _, _, info = env.step(0)
        assert info["epoch_metrics"]["arrived"] == 0.0

    def test_a_wrong_energy_constant_would_be_caught(self) -> None:
        env = hand_calc_env()
        env.simulator.config.energy_units_per_stop = 0.70
        env.step(0)
        _, _, _, _, info = env.step(1)
        assert info["components"]["E_units"] == pytest.approx(1.70)
        assert info["components"]["E_units"] != pytest.approx(1.35)
