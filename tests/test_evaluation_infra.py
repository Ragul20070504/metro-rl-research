"""Tests for the evaluation backbone: cell identity, resumability, and the unit of analysis.

The runner is the module every experimental number passes through, so a defect here would be
invisible in the results and fatal to them. Three properties matter most: a cell's identity must
be stable (or a resumed sweep silently recomputes or skips the wrong work), a successful result
must never be overwritten, and the aggregation must refuse to treat episodes as independent
observations.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.aggregate import (
    METRICS_OF_INTEREST,
    load_frame,
    paired_frame,
    seed_level,
)
from src.evaluation.runner import Cell, cell_dir, load_cell_metrics


class TestCellIdentity:
    def test_key_is_stable_across_construction(self) -> None:
        a = Cell(method="ppo_uq", scenario="surge", eval_seed=3, train_seed=101)
        b = Cell(method="ppo_uq", scenario="surge", eval_seed=3, train_seed=101)
        assert a.key() == b.key()
        assert cell_dir(a) == cell_dir(b)

    def test_every_field_that_changes_the_experiment_changes_the_key(self) -> None:
        base = Cell(method="ppo_uq", scenario="surge", eval_seed=3, train_seed=101)
        variations = [
            Cell(method="ppo_point", scenario="surge", eval_seed=3, train_seed=101),
            Cell(method="ppo_uq", scenario="normal", eval_seed=3, train_seed=101),
            Cell(method="ppo_uq", scenario="surge", eval_seed=4, train_seed=101),
            Cell(method="ppo_uq", scenario="surge", eval_seed=3, train_seed=102),
            Cell(method="ppo_uq", scenario="surge", eval_seed=3, train_seed=101,
                 weight_id="w_cost"),
            Cell(method="ppo_uq", scenario="surge", eval_seed=3, train_seed=101,
                 variant="od_uniform_downstream"),
        ]
        keys = {base.key()} | {c.key() for c in variations}
        assert len(keys) == len(variations) + 1

    def test_a_baseline_without_a_training_seed_is_representable(self) -> None:
        cell = Cell(method="fixed_timetable", scenario="normal", eval_seed=7)
        assert "/na/" in cell.key()

    def test_cells_are_hashable_so_they_can_deduplicate(self) -> None:
        cell = Cell(method="ppo_uq", scenario="surge", eval_seed=1, train_seed=101)
        assert len({cell, cell}) == 1

    def test_missing_metrics_return_none_rather_than_raising(self) -> None:
        assert load_cell_metrics(
            Cell(method="never_run", scenario="normal", eval_seed=999_999)) is None


class TestAggregation:
    @staticmethod
    def _frame(n_train: int = 4, n_eval: int = 5) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        rows = []
        for method, offset in (("ppo_uq", 0.010), ("ppo_point", 0.014)):
            for train_seed in range(101, 101 + n_train):
                level = offset + rng.normal(0, 0.001)
                for eval_seed in range(1, n_eval + 1):
                    row = {
                        "method": method, "scenario": "surge", "train_seed": train_seed,
                        "eval_seed": eval_seed, "weight_id": "w_balanced", "variant": "main",
                        "experiment_id": "EXP-UQ-001",
                        "mask_clip_rate": 0.0, "induction_success_rate": 1.0,
                        "inference_latency_mean_ms": 0.1, "inference_latency_p95_ms": 0.2,
                    }
                    for metric in METRICS_OF_INTEREST:
                        row[metric] = level + rng.normal(0, 0.0005)
                    rows.append(row)
        return pd.DataFrame(rows)

    def test_seed_level_collapses_episodes_to_training_seeds(self) -> None:
        """The core requirement of protocol section 8.2."""
        episodes = self._frame(n_train=4, n_eval=5)
        seeds = seed_level(episodes)
        assert len(episodes) == 40
        assert len(seeds) == 8
        assert set(seeds["n_episodes"]) == {5}

    def test_seed_level_means_match_a_hand_computation(self) -> None:
        episodes = self._frame(n_train=2, n_eval=3)
        seeds = seed_level(episodes)
        for _, row in seeds.iterrows():
            subset = episodes[
                (episodes["method"] == row["method"])
                & (episodes["train_seed"] == row["train_seed"])
            ]
            assert row["left_behind_fraction"] == pytest.approx(
                subset["left_behind_fraction"].mean())

    def test_empty_input_is_handled(self) -> None:
        assert seed_level(pd.DataFrame()).empty

    def test_paired_frame_aligns_on_shared_training_seeds(self) -> None:
        episodes = self._frame()
        seeds = seed_level(episodes)
        pair = paired_frame(seeds, episodes, method_a="ppo_uq", method_b="ppo_point",
                            scenario="surge", metric="left_behind_fraction")
        assert pair is not None and pair.paired
        assert pair.values_a.shape == pair.values_b.shape == (4,)
        assert pair.train_seeds == [101, 102, 103, 104]

    def test_paired_frame_refuses_to_pair_a_baseline_with_no_training_seed(self) -> None:
        """A fixed timetable has no training seed, so there is nothing to pair. Fabricating a
        pairing would be a silent error, so the flag travels with the data."""
        episodes = self._frame()
        baseline = episodes[episodes["method"] == "ppo_point"].copy()
        baseline["method"] = "fixed_timetable"
        baseline["train_seed"] = -1
        combined = pd.concat([episodes, baseline], ignore_index=True)
        seeds = seed_level(combined)
        pair = paired_frame(seeds, combined, method_a="ppo_uq", method_b="fixed_timetable",
                            scenario="surge", metric="left_behind_fraction")
        assert pair is not None
        assert pair.paired is False

    def test_paired_frame_returns_none_when_a_method_is_absent(self) -> None:
        episodes = self._frame()
        seeds = seed_level(episodes)
        assert paired_frame(seeds, episodes, method_a="ppo_uq", method_b="not_a_method",
                            scenario="surge", metric="left_behind_fraction") is None

    def test_episode_values_are_carried_for_distributional_statistics(self) -> None:
        episodes = self._frame()
        seeds = seed_level(episodes)
        pair = paired_frame(seeds, episodes, method_a="ppo_uq", method_b="ppo_point",
                            scenario="surge", metric="left_behind_fraction")
        assert pair is not None
        assert pair.episodes_a.size == 20
        assert pair.episodes_b.size == 20

    def test_load_frame_skips_cells_with_no_result(self) -> None:
        cells = [Cell(method="never_run", scenario="normal", eval_seed=s) for s in range(3)]
        assert load_frame(cells).empty


class TestFigureProvenance:
    def test_every_generated_figure_has_a_source_companion(self) -> None:
        """No experimental number may be hand-entered; the companion file is the evidence."""
        from src.common.paths import RESULTS_DIR

        figures = RESULTS_DIR / "figures"
        if not figures.exists():
            pytest.skip("no figures generated in this checkout")
        pngs = sorted(figures.glob("*.png"))
        if not pngs:
            pytest.skip("no figures generated in this checkout")
        for png in pngs:
            companion = png.with_suffix("").with_suffix(".source.json")
            companion = figures / f"{png.stem}.source.json"
            assert companion.exists(), f"{png.name} has no provenance companion"
            payload = json.loads(companion.read_text(encoding="utf-8"))
            assert "source_files" in payload and "generated_utc" in payload


class TestPolicyWeightSeparation:
    """EXP-SENS-001 scores `w_balanced` policies under other weightings without retraining.

    The reward must follow the *scoring* weight while the checkpoint is still looked up under the
    *training* weight. Conflating the two made all 600 weight-sensitivity cells fail with
    `FileNotFoundError: no trained policy at .../w_cost/...` -- a policy the design deliberately
    never trains.
    """

    def test_policy_weight_defaults_to_the_scoring_weight(self):
        from src.evaluation.runner import Cell

        cell = Cell(method="ppo_uq", scenario="surge", eval_seed=1, train_seed=101)
        assert cell.policy_weight() == "w_balanced" == cell.weight_id

    def test_policy_weight_follows_the_training_weight_when_they_differ(self):
        from src.evaluation.runner import Cell

        cell = Cell(method="ppo_uq", scenario="normal", eval_seed=1, train_seed=101,
                    weight_id="w_cost", train_weight_id="w_balanced")
        assert cell.weight_id == "w_cost", "the reward must still follow the scoring weight"
        assert cell.policy_weight() == "w_balanced"

    def test_the_stored_result_path_is_unaffected(self):
        """`key()` must ignore the new field, or 7,010 stored cells would move."""
        from src.evaluation.runner import Cell

        common = dict(method="ppo_uq", scenario="normal", eval_seed=1, train_seed=101,
                      weight_id="w_cost", variant="weights_w_cost")
        assert Cell(**common).key() == Cell(**common, train_weight_id="w_balanced").key()

    def test_every_weight_ablation_cell_resolves_to_a_trained_policy(self):
        """The regression itself: the checkpoint each registered cell needs must exist."""
        from scripts.run_experiments import weight_ablation_cells, train_seed_list
        from src.rl.train import policy_path

        cells = weight_ablation_cells(train_seed_list(10))
        assert cells, "EXP-SENS-001 must enumerate cells"
        missing = sorted({
            str(policy_path(c.method, c.train_seed, c.policy_weight()))
            for c in cells
            if not policy_path(c.method, c.train_seed, c.policy_weight()).exists()
        })
        assert not missing, f"weight-ablation cells point at policies that do not exist: {missing}"
