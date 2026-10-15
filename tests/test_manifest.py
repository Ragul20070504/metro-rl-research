"""Run-manifest and append-only guarantees.

Research brief section 25 (traceability) and section 21 (do not overwrite experiments).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.manifest import MANIFEST_NAME, RunManifest, git_info, open_run_dir, package_versions


def _manifest(**overrides) -> RunManifest:
    defaults = dict(
        experiment_id="EXP-UQ-001",
        method="ppo_uq",
        scenario="surge",
        train_seed=101,
        eval_seed=7,
        config_path="configs/ppo.yaml",
        config_hash="deadbeef",
        config_values={"gamma": 0.99},
    )
    defaults.update(overrides)
    return RunManifest.create(**defaults)


def test_git_is_available_and_reports_a_commit() -> None:
    """M2 exit gate: a run manifest must be able to carry a real commit hash.

    If this fails, every result becomes provisional and the reproducibility criterion in
    research brief section 37 cannot be met.
    """
    info = git_info()
    assert info.available, "git not found -- see docs/risk_register.md E-01"
    assert info.commit is not None and len(info.commit) == 40


def test_manifest_captures_the_full_traceability_chain() -> None:
    manifest = _manifest()
    assert manifest.experiment_id == "EXP-UQ-001"
    assert manifest.config_hash == "deadbeef"
    assert manifest.git_commit is not None
    assert manifest.python_version.startswith("3.11")
    assert manifest.packages["torch"] != "not installed"
    assert manifest.packages["stable-baselines3"] != "not installed"
    assert set(manifest.thread_env) == {"PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS"}


def test_manifest_is_provisional_when_the_tree_is_dirty() -> None:
    """A run made from uncommitted code is reproducible only by accident."""
    manifest = _manifest()
    assert manifest.provisional == (manifest.git_dirty or manifest.git_commit is None)


def test_manifest_records_the_sweep_tag_and_unresolved_parameters() -> None:
    """A run that used an ungrounded parameter must say so in its own manifest."""
    manifest = _manifest(
        sweep_tag="sensitivity_sweep",
        unresolved_parameters=["simulator.energy_kwh_per_train_km"],
    )
    assert manifest.sweep_tag == "sensitivity_sweep"
    assert manifest.unresolved_parameters == ["simulator.energy_kwh_per_train_km"]


def test_finish_records_wall_clock(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest.finish()
    assert manifest.finished_utc is not None
    assert manifest.wall_clock_s is not None and manifest.wall_clock_s >= 0.0


def test_manifest_round_trips_through_json(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest.finish()
    written = manifest.write(tmp_path)
    loaded = json.loads(written.read_text(encoding="utf-8"))
    assert loaded["experiment_id"] == "EXP-UQ-001"
    assert loaded["config_values"] == {"gamma": 0.99}


class TestAppendOnlyResults:
    """Protocol section 8.5: do not overwrite experiments."""

    def test_new_directory_is_created(self, tmp_path: Path) -> None:
        target = tmp_path / "surge" / "ppo_uq" / "101" / "7"
        assert open_run_dir(target) == target
        assert target.is_dir()

    def test_reopening_a_completed_run_raises(self, tmp_path: Path) -> None:
        target = open_run_dir(tmp_path / "run")
        _manifest().write(target)
        with pytest.raises(FileExistsError, match="append-only"):
            open_run_dir(target)

    def test_overwrite_requires_an_explicit_opt_in(self, tmp_path: Path) -> None:
        target = open_run_dir(tmp_path / "run")
        _manifest().write(target)
        assert open_run_dir(target, overwrite=True) == target

    def test_a_partial_directory_without_a_manifest_may_be_reused(self, tmp_path: Path) -> None:
        """A crashed run leaves no manifest; retrying it must not need a manual delete."""
        target = open_run_dir(tmp_path / "run")
        (target / "partial.txt").write_text("crashed", encoding="utf-8")
        assert not (target / MANIFEST_NAME).exists()
        assert open_run_dir(target) == target


def test_tracked_packages_are_all_installed() -> None:
    """Every result-affecting package must be recordable; 'not installed' hides a version."""
    missing = [name for name, version in package_versions().items() if version == "not installed"]
    assert not missing, f"packages missing from the environment: {missing}"
