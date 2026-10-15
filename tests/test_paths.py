"""Path resolution must be repo-relative and must produce the protocol's results layout."""

from __future__ import annotations

from src.common import paths


def test_repo_root_is_the_directory_containing_configs_and_src() -> None:
    assert (paths.REPO_ROOT / "configs").is_dir()
    assert (paths.REPO_ROOT / "src").is_dir()
    assert (paths.REPO_ROOT / "docs" / "experimental_protocol.md").is_file()


def test_data_directories_sit_under_the_data_root() -> None:
    for directory in (paths.RAW_DIR, paths.INTERIM_DIR, paths.PROCESSED_DIR, paths.EXTERNAL_DIR):
        assert directory.parent == paths.DATA_DIR


def test_run_dir_matches_the_protocol_layout() -> None:
    """Protocol section 8.5: results/<scenario>/<method>/<train_seed>/<eval_seed>/."""
    directory = paths.run_dir("surge", "ppo_uq", 101, 7)
    assert directory.relative_to(paths.RESULTS_DIR).parts == ("surge", "ppo_uq", "101", "7")


def test_run_dir_keeps_uniform_depth_for_baselines_without_a_training_seed() -> None:
    """Non-learning baselines have no training seed; the tree depth must not change."""
    learned = paths.run_dir("normal", "ppo_uq", 101, 3)
    baseline = paths.run_dir("normal", "fixed_timetable", None, 3)
    assert len(baseline.parts) == len(learned.parts)
    assert baseline.parent.name == "na"
