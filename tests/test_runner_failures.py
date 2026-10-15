"""A failed cell must be recorded, retried once, and never silently dropped.

Protocol section 7.4 requires that failures are recorded, diagnosed and retried once, and that a
sweep never ends quietly incomplete. The dangerous failure mode is not a crash -- it is a sweep
that finishes, reports a clean summary, and is missing cells nobody counted. These tests hold the
runner to three properties:

- a cell that fails once and succeeds on retry is recorded as such (`attempt == 2`), so a retried
  result is never mistaken for a first-attempt one;
- the first attempt's traceback survives the retry, because that is what a diagnosis needs;
- a cell that fails twice ends up in a manifest of unrecovered cells rather than in the gaps.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.evaluation import runner
from src.evaluation.runner import Cell


@pytest.fixture()
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    return tmp_path


@pytest.fixture()
def fake_episode(monkeypatch):
    """Replace the environment, policy and rollout with cheap stand-ins."""
    from src.evaluation import policies

    monkeypatch.setattr(runner, "build_env", lambda cell, **kw: SimpleNamespace())
    monkeypatch.setattr(runner, "build_policy", lambda cell, env: SimpleNamespace())
    monkeypatch.setattr(runner, "_git_state", lambda: ("0" * 40, False))
    monkeypatch.setattr(runner, "_hardware", lambda: {"platform": "test"})
    monkeypatch.setattr(runner, "_reward_scales",
                        lambda: SimpleNamespace(provenance={"source": "test"}))
    monkeypatch.setattr(runner, "_obs_scaler", lambda: None)
    monkeypatch.setattr(runner, "_forecast_provider", lambda kind: SimpleNamespace(model_id="test"))
    monkeypatch.setattr(policies, "run_episode", lambda env, policy, eval_seed: SimpleNamespace(
        summary={"left_behind_fraction": 0.01, "inductions": 5.0},
        epochs=[], inference_latency_ms=[1.0], mask_clip_rate=0.0, requested_actions=[1],
    ))


CELL = Cell(method="ppo_uq", scenario="surge", eval_seed=1, train_seed=101)


def _fail_n_times(monkeypatch, n: int) -> dict[str, int]:
    counter = {"calls": 0}

    def build_env(cell, **kw):
        counter["calls"] += 1
        if counter["calls"] <= n:
            raise RuntimeError(f"transient failure {counter['calls']}")
        return SimpleNamespace()

    monkeypatch.setattr(runner, "build_env", build_env)
    return counter


def test_a_first_attempt_success_is_marked_attempt_one(runs_dir, fake_episode):
    metrics = runner.run_cell(CELL)
    assert metrics["attempt"] == 1


def test_a_failure_writes_a_traceback_not_an_exception(runs_dir, fake_episode, monkeypatch):
    _fail_n_times(monkeypatch, 1)
    result = runner.run_cell(CELL)

    assert result["failed"] is True
    payload = json.loads((runner.cell_dir(CELL) / "FAILED.json").read_text(encoding="utf-8"))
    assert payload["attempt"] == 1
    assert "transient failure 1" in payload["traceback"]
    assert runner.load_cell_metrics(CELL) is None, "a failed cell must not look complete"


def test_run_cells_retries_once_and_records_that_it_did(runs_dir, fake_episode, monkeypatch):
    _fail_n_times(monkeypatch, 1)
    runner.run_cells([CELL], workers=1, label="test")

    metrics = runner.load_cell_metrics(CELL)
    assert metrics is not None, "the retry should have recovered this cell"
    assert metrics["attempt"] == 2, "a retried success must be distinguishable from a clean one"


def test_the_first_failure_is_preserved_when_the_retry_succeeds(
        runs_dir, fake_episode, monkeypatch):
    _fail_n_times(monkeypatch, 1)
    runner.run_cells([CELL], workers=1, label="test")

    directory = runner.cell_dir(CELL)
    assert not (directory / "FAILED.json").exists(), "a recovered cell must not still read as failed"
    preserved = json.loads((directory / "FAILED_attempt1.json").read_text(encoding="utf-8"))
    assert "transient failure 1" in preserved["traceback"]


def test_a_cell_that_fails_twice_is_reported_not_dropped(runs_dir, fake_episode, monkeypatch):
    _fail_n_times(monkeypatch, 99)
    runner.run_cells([CELL], workers=1, label="sweep")

    assert runner.load_cell_metrics(CELL) is None
    manifest = json.loads((runs_dir / "FAILED_sweep.json").read_text(encoding="utf-8"))
    assert len(manifest["cells"]) == 1
    assert manifest["cells"][0]["method"] == "ppo_uq"

    directory = runner.cell_dir(CELL)
    assert (directory / "FAILED_attempt1.json").exists()
    assert (directory / "FAILED.json").exists(), "the second failure is the current state"


def test_a_successful_result_is_never_overwritten_by_a_rerun(runs_dir, fake_episode):
    first = runner.run_cell(CELL)
    (runner.cell_dir(CELL) / "metrics.json").write_text(
        json.dumps({**first, "sentinel": 42}), encoding="utf-8")

    again = runner.run_cell(CELL)
    assert again["sentinel"] == 42
    assert again["_cached"] is True


def test_no_failure_manifest_is_written_when_everything_succeeds(runs_dir, fake_episode):
    runner.run_cells([CELL], workers=1, label="sweep")
    assert not (runs_dir / "FAILED_sweep.json").exists()
