"""Multi-objective reward tests (protocol section 6.4).

Protocol section 6.4 requires that the stored components reconstruct ``r_t`` **exactly**, and that
no experiment ever reports the scalar reward alone. Both are asserted here. So is the sign and
scaling of every term, because a sign error in a multi-objective reward does not crash -- it
quietly optimises for the wrong thing and produces a plausible-looking result.

The degeneracy guard at the end is the regression test for amendment A-02.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.environment.reward import (
    PRIMARY_WEIGHT_ID,
    WEIGHT_GRID,
    ReferenceScales,
    RewardWeights,
    compute_reward,
)
from src.simulator.engine import EpochMetrics
from tests.test_environment import make_env, rollout


def metrics(
    *, accrued_s: float = 0.0, left_behind: float = 0.0,
    energy: float = 0.0, cost: float = 0.0, mean_wait_s: float = float("nan"),
) -> EpochMetrics:
    return EpochMetrics(
        waiting_time_mean_s=mean_wait_s,
        waiting_accrued_passenger_s=accrued_s,
        left_behind=left_behind,
        energy_units=energy,
        cost_units=cost,
    )


class TestReconstruction:
    """Protocol section 6.4: `tests/test_reward.py` asserts the stored components reconstruct r_t."""

    def test_components_reconstruct_the_scalar_exactly(self) -> None:
        weights = WEIGHT_GRID["w_service"]
        component = compute_reward(
            metrics(accrued_s=600.0, left_behind=7.0, energy=13.0, cost=2.5),
            weights,
            ReferenceScales(2.0, 3.5, 5.0, 1.25),
        )
        assert component.reconstruct(weights) == pytest.approx(component.reward)

    def test_reconstruction_holds_for_every_weight_row(self) -> None:
        scales = ReferenceScales(2.0, 3.5, 5.0, 1.25)
        sample = metrics(accrued_s=1234.0, left_behind=9.0, energy=17.0, cost=3.0)
        for weights in WEIGHT_GRID.values():
            component = compute_reward(sample, weights, scales)
            assert component.reconstruct(weights) == pytest.approx(component.reward)

    def test_reconstruction_holds_across_a_whole_episode(self) -> None:
        env = make_env()
        _, records = rollout(env)
        for record in records:
            expected = -(
                env.weights.alpha * record["W_norm"]
                + env.weights.beta * record["U_norm"]
                + env.weights.gamma * record["E_norm"]
                + env.weights.delta * record["C_norm"]
            )
            assert record["reward"] == pytest.approx(expected)

    def test_physical_and_normalised_values_are_consistent(self) -> None:
        scales = ReferenceScales(2.0, 4.0, 8.0, 16.0)
        component = compute_reward(
            metrics(accrued_s=120.0, left_behind=8.0, energy=24.0, cost=32.0),
            WEIGHT_GRID["w_balanced"], scales,
        )
        assert component.W_norm == pytest.approx(component.W_minutes / scales.ref_W)
        assert component.U_norm == pytest.approx(component.U_passengers / scales.ref_U)
        assert component.E_norm == pytest.approx(component.E_units / scales.ref_E)
        assert component.C_norm == pytest.approx(component.C_units / scales.ref_C)


class TestComponentLogging:
    """No experiment reports the scalar reward alone (protocol section 6.4)."""

    def test_every_component_is_recorded_in_physical_units(self) -> None:
        _, records = rollout(make_env())
        for key in ("W_minutes", "U_passengers", "E_units", "C_units"):
            assert all(key in r for r in records)

    def test_physical_units_are_recoverable_without_the_scales(self) -> None:
        """A reader with only the results file must be able to see minutes and passengers."""
        _, records = rollout(make_env())
        assert sum(r["W_minutes"] for r in records) > 0
        assert all(np.isfinite(r["E_units"]) for r in records)

    def test_the_boarder_mean_diagnostic_is_kept_but_is_not_a_reward_term(self) -> None:
        _, records = rollout(make_env())
        assert "W_mean_over_boarders_minutes" in records[0]
        weights = WEIGHT_GRID["w_balanced"]
        for record in records[:20]:
            expected = -(record["W_norm"] + record["U_norm"] + record["E_norm"] + record["C_norm"])
            assert record["reward"] == pytest.approx(expected)


class TestSignAndScaling:
    def test_reward_is_never_positive(self) -> None:
        """r = -(non-negative terms). A positive reward would mean a component went negative."""
        _, records = rollout(make_env())
        assert all(r["reward"] <= 1e-9 for r in records)

    def test_zero_cost_epoch_scores_exactly_zero(self) -> None:
        component = compute_reward(metrics(), WEIGHT_GRID["w_balanced"], ReferenceScales.unit())
        assert component.reward == pytest.approx(0.0)

    def test_each_term_pushes_the_reward_down(self) -> None:
        scales = ReferenceScales.unit()
        weights = WEIGHT_GRID["w_balanced"]
        base = compute_reward(metrics(), weights, scales).reward
        for field, value in (
            ("accrued_s", 600.0), ("left_behind", 5.0), ("energy", 5.0), ("cost", 5.0)
        ):
            worse = compute_reward(metrics(**{field: value}), weights, scales).reward
            assert worse < base, field

    def test_doubling_a_term_doubles_its_contribution(self) -> None:
        scales = ReferenceScales.unit()
        weights = WEIGHT_GRID["w_balanced"]
        single = compute_reward(metrics(energy=10.0), weights, scales).reward
        double = compute_reward(metrics(energy=20.0), weights, scales).reward
        assert double == pytest.approx(2.0 * single)

    def test_weights_scale_their_own_term_only(self) -> None:
        scales = ReferenceScales.unit()
        sample = metrics(accrued_s=600.0, left_behind=1.0, energy=1.0, cost=1.0)
        balanced = compute_reward(sample, WEIGHT_GRID["w_balanced"], scales)
        wait_only = compute_reward(sample, WEIGHT_GRID["w_wait_only"], scales)
        assert wait_only.reward == pytest.approx(-balanced.W_norm)
        assert balanced.reward == pytest.approx(
            -(balanced.W_norm + balanced.U_norm + balanced.E_norm + balanced.C_norm)
        )

    def test_reference_scales_cancel_a_change_of_unit(self) -> None:
        """The point of section 6.4's reference scaling: choosing a different energy unit must
        not silently reweight the objective, because A15 leaves the true unit unresolved."""
        weights = WEIGHT_GRID["w_balanced"]
        a = compute_reward(metrics(energy=30.0), weights, ReferenceScales(1.0, 1.0, 10.0, 1.0))
        b = compute_reward(metrics(energy=3000.0), weights, ReferenceScales(1.0, 1.0, 1000.0, 1.0))
        assert a.reward == pytest.approx(b.reward)


class TestReferenceScales:
    def test_a_zero_scale_is_refused(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            ReferenceScales(1.0, 0.0, 1.0, 1.0)

    def test_a_negative_scale_is_refused(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            ReferenceScales(1.0, 1.0, -2.0, 1.0)

    def test_roundtrip_through_disk_preserves_provenance(self, tmp_path) -> None:
        original = ReferenceScales(1.5, 2.5, 3.5, 4.5, provenance={"source_policy": "constant_1"})
        loaded = ReferenceScales.load(original.save(tmp_path / "reward_scales.json"))
        assert loaded.ref_W == pytest.approx(1.5)
        assert loaded.provenance["source_policy"] == "constant_1"

    def test_the_committed_scales_record_which_policy_produced_them(self) -> None:
        """A reference scale with no recorded source is indistinguishable from a tuned constant."""
        from src.common.paths import REPO_ROOT

        path = REPO_ROOT / "configs" / "reward_scales.json"
        if not path.exists():
            pytest.skip("reward scales not fitted in this checkout")
        payload = json.loads(path.read_text(encoding="utf-8"))
        provenance = payload["provenance"]
        assert provenance["source_policy"]
        assert "git_commit" in provenance
        assert "W_definition" in provenance and "queue integral" in provenance["W_definition"]


class TestWeightGridIsFrozen:
    def test_primary_configuration_is_balanced(self) -> None:
        assert PRIMARY_WEIGHT_ID == "w_balanced"
        assert WEIGHT_GRID[PRIMARY_WEIGHT_ID] == RewardWeights("w_balanced", 1.0, 1.0, 1.0, 1.0)

    def test_single_objective_row_zeroes_the_other_three(self) -> None:
        row = WEIGHT_GRID["w_wait_only"]
        assert (row.alpha, row.beta, row.gamma, row.delta) == (1.0, 0.0, 0.0, 0.0)

    def test_service_and_cost_rows_are_mirror_images(self) -> None:
        service, cost = WEIGHT_GRID["w_service"], WEIGHT_GRID["w_cost"]
        assert (service.alpha, service.gamma) == (cost.gamma, cost.alpha)


class TestNoServiceIsNotOptimal:
    """Regression guard for protocol amendment A-02 / decision D-13.

    Under the superseded reading of ``W_t`` -- the mean wait of the passengers who *boarded* --
    a policy that ran no service at all scored ``r = 0``, the best attainable value, while
    stranding 100 % of demand: nobody boards so there is no mean; no train departs so nobody is
    left behind; nothing moves so energy and cost vanish. Every term was blind to the failure
    simultaneously. These tests fail if that reading ever returns.
    """

    def test_an_empty_epoch_with_a_queue_is_penalised(self) -> None:
        component = compute_reward(
            metrics(accrued_s=90_000.0, mean_wait_s=float("nan")),
            WEIGHT_GRID["w_balanced"], ReferenceScales.unit(),
        )
        assert component.reward < -1000.0

    def test_the_undefined_boarder_mean_does_not_enter_the_reward(self) -> None:
        with_nan = compute_reward(
            metrics(accrued_s=600.0, mean_wait_s=float("nan")),
            WEIGHT_GRID["w_balanced"], ReferenceScales.unit(),
        )
        with_value = compute_reward(
            metrics(accrued_s=600.0, mean_wait_s=120.0),
            WEIGHT_GRID["w_balanced"], ReferenceScales.unit(),
        )
        assert with_nan.reward == pytest.approx(with_value.reward)
        assert np.isfinite(with_nan.reward)

    def test_running_no_service_is_far_worse_than_running_some(self) -> None:
        idle, _ = rollout(make_env(), policy=lambda m: 0)
        served, _ = rollout(make_env(), policy=lambda m: 1)
        assert served["total_reward"] > idle["total_reward"]

    def test_the_gap_is_large_not_marginal(self) -> None:
        """A degenerate objective can also fail by making the difference negligible."""
        idle, _ = rollout(make_env(), policy=lambda m: 0)
        served, _ = rollout(make_env(), policy=lambda m: 1)
        assert abs(idle["total_reward"]) > 10.0 * abs(served["total_reward"])
