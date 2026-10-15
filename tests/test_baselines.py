"""Baseline tests (M7-M9): fixed timetable, heuristic, MILP, and the shared evaluation loop.

The properties asserted here are the ones that decide whether the comparison in EXP-BASE-001 and
EXP-BASE-005 means anything: that the fixed timetable really reproduces the published schedule and
really ignores the forecast, that the heuristic never reads sigma, that the MILP reports its
solver status honestly, and that every method goes through one identical episode loop.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines.fixed_timetable import (
    FixedTimetable,
    FixedTimetablePolicy,
    build_fixed_timetable,
)
from src.baselines.heuristic import HeuristicParameters, HeuristicPolicy
from src.baselines.milp import (
    SurrogateParameters,
    MILPPolicy,
    _fit_service_rate,
    _fit_wait_coefficients,
    solve_dispatch,
)
from src.environment.reward import WEIGHT_GRID, ReferenceScales
from src.evaluation.policies import ConstantPolicy, run_episode
from tests.test_environment import make_env


class TestFixedTimetable:
    def test_schedule_covers_every_decision_epoch(self) -> None:
        timetable = FixedTimetable.load()
        assert len(timetable.inductions["weekday"]) == 228
        assert len(timetable.inductions["weekend"]) == 228

    def test_weekday_service_is_substantial_and_bounded(self) -> None:
        """A schedule of zeros -- which an earlier dtype bug produced -- would make every other
        method look excellent by comparison."""
        timetable = FixedTimetable.load()
        weekday = timetable.inductions["weekday"]
        assert sum(weekday) > 150
        assert max(weekday) <= 2
        assert min(weekday) >= 0

    def test_weekend_runs_less_service_than_a_weekday(self) -> None:
        timetable = FixedTimetable.load()
        assert timetable.total("weekend") < timetable.total("weekday")

    def test_schedule_is_traceable_to_the_published_timetable(self) -> None:
        timetable = FixedTimetable.load()
        assert timetable.provenance["source"] == "real_data:mta_gtfs_static"
        assert timetable.provenance["by_day_type"]["weekday"]["service_ids"]

    def test_policy_ignores_the_observation_entirely(self) -> None:
        """Protocol section 7.1 gives this method nothing beyond the clock. If it read the
        forecast it would stop being the honest floor for H4."""
        env = make_env()
        env.reset()
        policy = FixedTimetablePolicy(FixedTimetable.load())
        policy.reset(env)
        mask = np.array([True, True, True])
        zeros = np.zeros(env.obs_spec.size)
        huge = np.full(env.obs_spec.size, 1e6)
        assert policy.act(zeros, env, mask) == policy.act(huge, env, mask)

    def test_policy_is_deterministic_across_episodes(self) -> None:
        env = make_env()
        policy = FixedTimetablePolicy(FixedTimetable.load())
        first = run_episode(env, policy, eval_seed=1).requested_actions
        second = run_episode(env, policy, eval_seed=2).requested_actions
        assert first == second

    def test_rebuilding_from_gtfs_reproduces_the_stored_schedule(self) -> None:
        rebuilt = build_fixed_timetable()
        stored = FixedTimetable.load()
        assert rebuilt.inductions["weekday"] == stored.inductions["weekday"]


class TestHeuristic:
    def test_thresholds_drive_the_action(self) -> None:
        env = make_env()
        env.reset()
        never = HeuristicPolicy(HeuristicParameters(1e12, 1e12))
        always = HeuristicPolicy(HeuristicParameters(0.0, 0.0))
        never.reset(env)
        always.reset(env)
        for _ in range(5):
            env.step(1)
        observation = env.raw_observation(refresh=True)
        assert never.act(observation, env, np.array([True, True, True])) == 0
        assert always.act(observation, env, np.array([True, True, True])) == 2

    def test_heuristic_never_reads_the_uncertainty_channel(self) -> None:
        """It is registered as having the same information as `ppo_point`: mu, never sigma.

        Asserted by construction: the policy resolves exactly two named blocks, and neither is a
        sigma channel.
        """
        env = make_env()
        env.reset()
        policy = HeuristicPolicy(HeuristicParameters(100.0, 100.0))
        policy.reset(env)
        assert policy._queue_index == env.obs_spec.slice_of("queue_length")
        assert policy._mu_index == env.obs_spec.slice_of("forecast_mu_h1")
        for name in ("forecast_sigma_h1", "forecast_sigma_line_h2h3"):
            assert policy._mu_index != env.obs_spec.slice_of(name)
            assert policy._queue_index != env.obs_spec.slice_of(name)

    def test_action_is_within_the_action_space(self) -> None:
        env = make_env()
        result = run_episode(env, HeuristicPolicy(HeuristicParameters(50.0, 1000.0)), eval_seed=3)
        assert all(0 <= a <= env.cfg.k_max for a in result.requested_actions)


class TestSurrogateCalibration:
    def test_service_rate_recovers_a_known_value(self) -> None:
        """Generate a trajectory from a known kappa and check the fit finds it."""
        rng = np.random.default_rng(0)
        kappa = 80.0
        rows = []
        queue = 0.0
        for _ in range(400):
            arrivals = float(rng.uniform(0, 4000))
            n = float(rng.integers(1, 30))
            end = max(0.0, queue + arrivals - kappa * n)
            rows.append((queue, arrivals, n, end))
            queue = end
        fitted, error = _fit_service_rate(rows)
        assert fitted == pytest.approx(kappa, abs=2.0)
        assert error < 5.0

    def test_wait_coefficients_recover_known_values(self) -> None:
        rng = np.random.default_rng(1)
        truth = [5.0, 2.0, 1.0]
        rows = []
        for _ in range(600):
            action = int(rng.integers(0, 3))
            arrivals = float(rng.uniform(100, 3000))
            queue = float(rng.uniform(0, 500))
            accrued = truth[action] * arrivals + 5.0 * queue
            rows.append((action, arrivals, queue, accrued))
        fitted, error = _fit_wait_coefficients(rows, k_max=2)
        assert fitted == pytest.approx(truth, abs=0.05)
        assert error < 1e-6

    def test_a_measured_increase_with_action_survives_the_fit(self) -> None:
        """The fit must report what the data says, including w2 > w1 (amendment A-12).

        This test previously asserted the opposite -- that the coefficients are monotone
        non-increasing in the action -- on the reasoning that "more service must never be modelled
        as more waiting, or the MILP would learn to withhold trains". That conflated more service
        *over time*, which does reduce waiting and is priced by the queue recursion and the
        standing-queue floor, with more service *at one instant*, which bunches departures and
        genuinely increases waiting. Measured on the simulator, the always-two probe strands five
        times as many passengers as the always-one probe for the same number of trains, and the
        unclamped fit reports w2 = 0.680 against w1 = 0.633. Clamping overwrote that measurement
        with an assumption and left the MILP believing a second trainset was free capacity.
        """
        rng = np.random.default_rng(2)
        truth = [4.0, 1.0, 1.6]  # deliberately NOT monotone: action 2 costs more than action 1
        rows = []
        for _ in range(900):
            action = int(rng.integers(0, 3))
            arrivals = float(rng.uniform(100, 3000))
            queue = float(rng.uniform(0, 500))
            rows.append((action, arrivals, queue, truth[action] * arrivals + 5.0 * queue))

        fitted, error = _fit_wait_coefficients(rows, k_max=2)
        assert fitted == pytest.approx(truth, abs=0.05), (
            "the fit must recover a non-monotone truth rather than flattening it")
        assert fitted[2] > fitted[1], "a measured increase with action must survive the fit"
        assert error < 1e-6

    def test_wait_coefficients_are_non_negative(self) -> None:
        """Removing the monotone clamp must not admit a negative waiting cost."""
        rng = np.random.default_rng(3)
        rows = [
            (int(rng.integers(0, 3)), float(rng.uniform(100, 3000)),
             float(rng.uniform(0, 500)), float(rng.uniform(0, 20000)))
            for _ in range(400)
        ]
        fitted, _ = _fit_wait_coefficients(rows, k_max=2)
        assert all(value >= 0.0 for value in fitted)

    def test_each_coefficient_uses_only_epochs_that_took_that_action(self) -> None:
        """Otherwise the three coefficients would be three views of one pooled average."""
        rows = [(0, 1000.0, 0.0, 5000.0)] * 50 + [(1, 1000.0, 0.0, 1000.0)] * 50
        fitted, _ = _fit_wait_coefficients(rows, k_max=2)
        assert fitted[0] == pytest.approx(5.0, abs=1e-6)
        assert fitted[1] == pytest.approx(1.0, abs=1e-6)

    def test_the_dispatch_envelope_never_binds_tighter_than_observed(self) -> None:
        """A-13's caps are observed maxima, so a plan the environment delivered stays feasible."""
        from src.baselines.milp import _fit_dispatch_capacity

        executed = [np.array([2, 0, 1, 1, 2, 0, 0, 2, 1, 0, 1, 1] * 4, dtype=int)]
        caps, provenance = _fit_dispatch_capacity(executed, k_max=2)
        for window_text, cap in caps.items():
            window = int(window_text)
            sums = np.convolve(executed[0], np.ones(window, dtype=int), mode="valid")
            assert cap >= int(sums.max()), (
                f"cap for window {window} is below a rate the environment actually delivered")
        assert provenance["source"].startswith("always-request-two probe")

    def test_stored_surrogate_is_traceable(self) -> None:
        surrogate = SurrogateParameters.load()
        assert surrogate.provenance["seeds"]
        assert "calibrated" in surrogate.provenance["method"]
        assert len(surrogate.wait_minutes_per_arrival) >= 3


class TestMILP:
    @staticmethod
    def _surrogate() -> SurrogateParameters:
        return SurrogateParameters(
            served_per_train_epoch=100.0, round_trip_epochs=20,
            energy_per_induction=46.0, cost_per_induction=1.0,
            cost_per_train_epoch=0.042, fleet_size=36,
            wait_minutes_per_arrival=[5.0, 2.0, 1.5],
        )

    def test_solver_returns_a_schedule_and_an_honest_report(self) -> None:
        demand = np.full(24, 1500.0)
        schedule, report = solve_dispatch(
            demand, self._surrogate(), WEIGHT_GRID["w_balanced"],
            ReferenceScales(3000.0, 20.0, 45.0, 1.6),
        )
        assert schedule.shape == (24,)
        assert set(np.unique(schedule)) <= {0, 1, 2}
        assert report.status in ("OPTIMAL", "FEASIBLE")
        assert np.isfinite(report.objective)

    def test_higher_demand_buys_more_service(self) -> None:
        surrogate, scales = self._surrogate(), ReferenceScales(3000.0, 20.0, 45.0, 1.6)
        low, _ = solve_dispatch(np.full(24, 100.0), surrogate,
                                WEIGHT_GRID["w_balanced"], scales)
        high, _ = solve_dispatch(np.full(24, 6000.0), surrogate,
                                 WEIGHT_GRID["w_balanced"], scales)
        assert high.sum() > low.sum()

    def test_a_cost_priority_weighting_buys_less_service(self) -> None:
        surrogate, scales = self._surrogate(), ReferenceScales(3000.0, 20.0, 45.0, 1.6)
        demand = np.full(24, 2000.0)
        service, _ = solve_dispatch(demand, surrogate, WEIGHT_GRID["w_service"], scales)
        cost, _ = solve_dispatch(demand, surrogate, WEIGHT_GRID["w_cost"], scales)
        assert cost.sum() <= service.sum()

    def test_fleet_size_bounds_concurrent_service(self) -> None:
        surrogate = self._surrogate()
        surrogate.fleet_size = 4
        schedule, _ = solve_dispatch(
            np.full(30, 20000.0), surrogate, WEIGHT_GRID["w_service"],
            ReferenceScales(3000.0, 20.0, 45.0, 1.6),
        )
        window = surrogate.round_trip_epochs
        for t in range(len(schedule)):
            concurrent = schedule[max(0, t - window + 1): t + 1].sum()
            assert concurrent <= surrogate.fleet_size

    def test_static_mode_plans_once_and_never_looks_at_the_episode(self) -> None:
        env = make_env()
        policy = MILPPolicy(mode="static", surrogate=self._surrogate())
        env.reset()
        policy.reset(env)
        assert policy._plan is not None
        assert len(policy.reports) == 1
        # The plan is open-loop: identical actions regardless of what the day does.
        first = list(policy._plan)
        env.reset(seed=99)
        policy.reset(env)
        assert list(policy._plan) == first

    def test_rolling_mode_records_one_instance_per_epoch(self) -> None:
        env = make_env()
        policy = MILPPolicy(mode="rolling", surrogate=self._surrogate())
        result = run_episode(env, policy, eval_seed=2)
        assert len(policy.reports) == len(result.requested_actions)
        summary = policy.solver_summary()
        assert summary["instances"] == len(policy.reports)
        assert summary["robustness_margin_z"] == 1.0

    def test_solver_summary_reports_status_and_gap(self) -> None:
        """Protocol section 7.1: instances that hit the time limit are reported as such, never
        silently treated as optimal."""
        env = make_env()
        policy = MILPPolicy(mode="rolling", surrogate=self._surrogate())
        run_episode(env, policy, eval_seed=2)
        summary = policy.solver_summary()
        assert set(summary) >= {
            "instances", "status_counts", "hit_time_limit",
            "solve_time_mean_s", "optimality_gap_mean", "time_limit_s",
        }
        assert sum(summary["status_counts"].values()) == summary["instances"]


class TestSharedEvaluationLoop:
    """Every method goes through one loop, so loop behaviour cannot differ between them."""

    def test_requested_and_executed_actions_are_both_recorded(self) -> None:
        env = make_env()
        result = run_episode(env, ConstantPolicy(2), eval_seed=1)
        assert len(result.requested_actions) == len(result.executed_actions)
        assert all(e <= r for r, e in zip(result.requested_actions, result.executed_actions))

    def test_mask_clipping_is_measured_not_hidden(self) -> None:
        env = make_env()
        greedy = run_episode(env, ConstantPolicy(2), eval_seed=1)
        idle = run_episode(env, ConstantPolicy(0), eval_seed=1)
        assert greedy.mask_clip_rate > 0.0
        assert idle.mask_clip_rate == 0.0

    def test_an_over_ambitious_request_is_clipped_rather_than_raising(self) -> None:
        """The environment raises on a masked action; the shared loop clips first, identically
        for every method, so mask-awareness is not a source of between-method difference."""
        env = make_env()
        result = run_episode(env, ConstantPolicy(99), eval_seed=1)
        assert all(a <= env.cfg.k_max for a in result.executed_actions)

    def test_inference_latency_is_recorded(self) -> None:
        env = make_env()
        result = run_episode(env, ConstantPolicy(1), eval_seed=1)
        assert len(result.inference_latency_ms) == len(result.executed_actions)
        assert all(t >= 0.0 for t in result.inference_latency_ms)
