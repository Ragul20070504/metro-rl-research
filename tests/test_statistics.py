"""Statistical-analysis tests (M14).

These check the machinery against cases with known answers, because a statistics module that is
merely plausible is the most dangerous kind of bug in a research repository: it produces numbers
that look like evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.statistics import (
    benjamini_hochberg,
    bca_bootstrap_ci,
    cliffs_delta,
    describe,
    hodges_lehmann,
    holm,
    paired_analysis,
    rank_biserial,
    tost_equivalence,
)


class TestDescriptives:
    def test_matches_hand_computed_values(self) -> None:
        result = describe([1.0, 2.0, 3.0, 4.0])
        assert result.n == 4
        assert result.mean == pytest.approx(2.5)
        assert result.median == pytest.approx(2.5)
        assert result.sd == pytest.approx(np.std([1, 2, 3, 4], ddof=1))

    def test_non_finite_values_are_dropped_not_propagated(self) -> None:
        result = describe([1.0, float("nan"), 3.0, float("inf")])
        assert result.n == 2
        assert result.mean == pytest.approx(2.0)

    def test_empty_input_returns_nan_rather_than_raising(self) -> None:
        assert describe([]).n == 0


class TestEffectSizes:
    def test_hodges_lehmann_of_a_symmetric_sample_is_its_centre(self) -> None:
        assert hodges_lehmann(np.array([-2.0, -1.0, 0.0, 1.0, 2.0])) == pytest.approx(0.0)

    def test_hodges_lehmann_tracks_a_shift(self) -> None:
        base = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        assert hodges_lehmann(base + 5.0) == pytest.approx(5.0)

    def test_rank_biserial_is_plus_one_when_all_differences_are_positive(self) -> None:
        assert rank_biserial(np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)

    def test_rank_biserial_is_minus_one_when_all_differences_are_negative(self) -> None:
        assert rank_biserial(np.array([-1.0, -2.0, -3.0])) == pytest.approx(-1.0)

    def test_rank_biserial_is_zero_for_a_balanced_sample(self) -> None:
        assert rank_biserial(np.array([1.0, -1.0, 2.0, -2.0])) == pytest.approx(0.0)

    def test_cliffs_delta_is_plus_one_for_fully_separated_samples(self) -> None:
        assert cliffs_delta([10, 11, 12], [1, 2, 3]) == pytest.approx(1.0)

    def test_cliffs_delta_is_zero_for_identical_samples(self) -> None:
        assert cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


class TestBootstrap:
    def test_interval_brackets_the_point_estimate(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(2.0, 1.0, 40)
        low, high = bca_bootstrap_ci(data)
        assert low < hodges_lehmann(data) < high

    def test_interval_is_reproducible(self) -> None:
        data = np.array([0.1, 0.4, -0.2, 0.9, 0.3, 0.5, -0.1, 0.7])
        assert bca_bootstrap_ci(data) == bca_bootstrap_ci(data)

    def test_a_clear_shift_produces_an_interval_excluding_zero(self) -> None:
        rng = np.random.default_rng(1)
        low, high = bca_bootstrap_ci(rng.normal(5.0, 0.5, 30))
        assert low > 0.0

    def test_tiny_samples_return_nan_rather_than_a_false_interval(self) -> None:
        low, high = bca_bootstrap_ci(np.array([1.0, 2.0]))
        assert np.isnan(low) and np.isnan(high)


class TestPairedAnalysis:
    def test_identical_arms_do_not_reject(self) -> None:
        values = [0.10, 0.12, 0.09, 0.11, 0.13, 0.10, 0.08, 0.12, 0.11, 0.10]
        result = paired_analysis(values, values, label="t", metric="m",
                                 method_a="a", method_b="b")
        assert result.p_value == pytest.approx(1.0)
        assert result.mean_difference == pytest.approx(0.0)
        assert result.direction == "no difference"

    def test_a_consistent_improvement_is_detected(self) -> None:
        rng = np.random.default_rng(3)
        b = rng.normal(0.10, 0.005, 10)
        a = b - 0.02
        result = paired_analysis(a, b, label="t", metric="left_behind_fraction",
                                 method_a="ppo_uq", method_b="ppo_point")
        assert result.p_value < 0.01
        assert result.direction == "ppo_uq better"
        assert result.ci_high < 0

    def test_direction_respects_lower_is_better(self) -> None:
        a = [1.0, 1.1, 1.2, 1.05, 1.15, 1.08, 1.12, 1.03]
        b = [2.0, 2.1, 2.2, 2.05, 2.15, 2.08, 2.12, 2.03]
        lower = paired_analysis(a, b, label="t", metric="m", method_a="a", method_b="b",
                                lower_is_better=True)
        higher = paired_analysis(a, b, label="t", metric="m", method_a="a", method_b="b",
                                 lower_is_better=False)
        assert lower.direction == "a better"
        assert higher.direction == "b better"

    def test_the_test_choice_rule_is_applied_not_chosen(self) -> None:
        """Normal differences take the paired t; a heavy outlier forces the signed-rank test."""
        rng = np.random.default_rng(4)
        b = rng.normal(1.0, 0.1, 12)
        normal = paired_analysis(b + rng.normal(0.05, 0.1, 12), b, label="t", metric="m",
                                 method_a="a", method_b="b")
        skewed_a = b.copy()
        skewed_a[0] += 50.0
        skewed = paired_analysis(skewed_a, b, label="t", metric="m", method_a="a", method_b="b")
        assert normal.test == "paired t-test"
        assert skewed.test == "Wilcoxon signed-rank"

    def test_small_samples_always_use_the_signed_rank_test(self) -> None:
        """Protocol section 8.3: signed-rank when normality is rejected *or n is small*."""
        rng = np.random.default_rng(5)
        b = rng.normal(1.0, 0.1, 6)
        result = paired_analysis(b + 0.2, b, label="t", metric="m", method_a="a", method_b="b")
        assert result.test == "Wilcoxon signed-rank"

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            paired_analysis([1, 2, 3], [1, 2], label="t", metric="m",
                            method_a="a", method_b="b")

    def test_too_few_pairs_are_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            paired_analysis([1, 2], [3, 4], label="t", metric="m",
                            method_a="a", method_b="b")

    def test_reportable_sentence_carries_test_statistic_n_and_p(self) -> None:
        rng = np.random.default_rng(6)
        b = rng.normal(1.0, 0.1, 10)
        sentence = paired_analysis(b - 0.3, b, label="t", metric="m",
                                   method_a="a", method_b="b").sentence()
        for fragment in ("statistic=", "n=", "p=", "Hodges-Lehmann", "95% BCa CI"):
            assert fragment in sentence


class TestEquivalence:
    def test_a_genuinely_null_difference_is_declared_equivalent(self) -> None:
        rng = np.random.default_rng(7)
        result = tost_equivalence(rng.normal(0.0, 0.001, 20), sesoi=0.01)
        assert result["applicable"] and result["equivalent_at_alpha"]

    def test_a_large_difference_is_not_declared_equivalent(self) -> None:
        rng = np.random.default_rng(8)
        result = tost_equivalence(rng.normal(0.05, 0.001, 20), sesoi=0.01)
        assert result["applicable"] and not result["equivalent_at_alpha"]

    def test_equivalence_runs_only_when_the_primary_test_fails_to_reject(self) -> None:
        rng = np.random.default_rng(9)
        b = rng.normal(1.0, 0.1, 10)
        rejected = paired_analysis(b - 5.0, b, label="t", metric="m", method_a="a",
                                   method_b="b", sesoi=0.5)
        not_rejected = paired_analysis(b + rng.normal(0, 0.001, 10), b, label="t", metric="m",
                                       method_a="a", method_b="b", sesoi=0.5)
        assert rejected.equivalence == {}
        assert not_rejected.equivalence["applicable"]


class TestMultiplicity:
    def test_holm_leaves_a_single_p_value_alone(self) -> None:
        assert holm([0.03]) == pytest.approx([0.03])

    def test_holm_matches_a_hand_computation(self) -> None:
        # sorted 0.01, 0.04 -> 2*0.01 = 0.02 ; max(0.02, 1*0.04) = 0.04
        assert holm([0.01, 0.04]) == pytest.approx([0.02, 0.04])

    def test_holm_is_monotone(self) -> None:
        adjusted = holm([0.001, 0.01, 0.03, 0.2])
        assert all(adjusted[i] <= adjusted[i + 1] for i in range(len(adjusted) - 1))

    def test_holm_never_exceeds_one(self) -> None:
        assert all(p <= 1.0 for p in holm([0.5, 0.6, 0.9]))

    def test_benjamini_hochberg_matches_a_hand_computation(self) -> None:
        # m = 4; raw 0.01, 0.02, 0.03, 0.04 -> 0.04, 0.04, 0.04, 0.04
        assert benjamini_hochberg([0.01, 0.02, 0.03, 0.04]) == pytest.approx([0.04] * 4)

    def test_benjamini_hochberg_is_less_conservative_than_holm(self) -> None:
        p = [0.01, 0.02, 0.03, 0.04, 0.05]
        assert all(bh <= hl + 1e-12 for bh, hl in zip(benjamini_hochberg(p), holm(p)))

    def test_empty_input_is_handled(self) -> None:
        assert benjamini_hochberg([]) == []
