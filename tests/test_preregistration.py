"""The pre-registration guard: a confirmatory run must not start against an editable design.

These tests protect two failure modes that would be invisible in the results:

1. A sweep that runs while `experiments/registry.yaml` is still `frozen: false`. The numbers would
   look identical to a properly pre-registered run, and nothing downstream could tell them apart.
2. A sweep that silently uses fewer training seeds than the R-1 power gate asked for. Slicing a
   fixed-length tuple does exactly this, and the only visible symptom is a design that is quietly
   less powerful than the one reported.
"""

from __future__ import annotations

import json

import pytest
import yaml

from scripts import freeze_registry
from scripts.run_experiments import TRAIN_SEEDS, train_seed_list

SKELETON = """\
# STATUS: SKELETON. `frozen: false` - entries are still being written.
meta:
  frozen: false
  frozen_at_commit: null          # filled when frozen
  frozen_utc: null

defaults:
  train_seeds: [101, 102, 103]   # subject to decision item 6
"""


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    path = tmp_path / "registry.yaml"
    path.write_text(SKELETON, encoding="utf-8")
    monkeypatch.setattr(freeze_registry, "REGISTRY", path)
    return path


@pytest.fixture()
def power(tmp_path, monkeypatch):
    def write(n: int, decision: str = "PROCEED with the registered n"):
        path = tmp_path / "power_analysis.json"
        path.write_text(json.dumps({"amended_train_seeds": n, "decision": decision}),
                        encoding="utf-8")
        monkeypatch.setattr(freeze_registry, "POWER", path)
        return path

    return write


# --------------------------------------------------------------------------- the guard


def test_require_frozen_refuses_an_unfrozen_registry(registry):
    with pytest.raises(SystemExit) as raised:
        freeze_registry.require_frozen("the M11 confirmatory sweep")
    assert "unfrozen" in str(raised.value)


def test_require_frozen_accepts_a_frozen_registry(registry, power):
    power(10)
    freeze_registry.freeze()
    assert freeze_registry.require_frozen("a test")["meta"]["frozen"] is True


def test_the_real_registry_is_frozen_before_any_stored_confirmatory_result():
    """No **confirmatory-family** result may exist while the pre-registration is still editable.

    The confirmatory family is exactly two tests (protocol section 7.3), both `ppo_uq` against
    another PPO arm. Those are what the freeze must precede.

    Baseline cells are deliberately *not* counted, and the distinction is not a convenience. The
    registry's open item is the sample size, which R-1 supplies and which applies only to training
    seeds; baselines have no training seed and their results do not depend on it. Baseline
    evaluation is how the baselines are shown to be fair -- protocol section 7.1's prohibition on
    strawmen cannot be checked without running them -- and that check has to happen before the
    comparison, not after.

    This is the looser reading of the rule, so it is worth being explicit about what it permits:
    amendments A-10 through A-13 all corrected baselines *after* seeing baseline results. Each is
    recorded, each was made because a measured outcome was implausible, and each strengthened a
    competitor to the proposed method. The guard below is what prevents the same latitude being
    taken with the arms the hypothesis is actually about.
    """
    from src.evaluation.runner import RUNS_DIR

    from scripts.run_experiments import RL_METHODS

    confirmatory = [
        path for path in RUNS_DIR.rglob("metrics.json")
        if path.parts[-7] == "main"                       # not pilot, demo, dryrun or sensitivity
        and any(f"/{m}/" in path.as_posix() for m in RL_METHODS)
    ]
    if not confirmatory:
        pytest.skip("no confirmatory-family results stored yet")
    assert freeze_registry.is_frozen(), (
        f"{len(confirmatory)} confirmatory-family result(s) exist but the registry is not frozen"
    )


# --------------------------------------------------------------------------- freezing


def test_freeze_refuses_without_a_power_decision(registry, tmp_path, monkeypatch):
    monkeypatch.setattr(freeze_registry, "POWER", tmp_path / "absent.json")
    with pytest.raises(SystemExit) as raised:
        freeze_registry.freeze()
    assert "R-1 power gate" in str(raised.value)


def test_freeze_records_the_gate_decision_verbatim(registry, power):
    power(13, decision="AMEND n upward")
    freeze_registry.freeze()

    meta = yaml.safe_load(registry.read_text(encoding="utf-8"))["meta"]
    assert meta["frozen"] is True
    assert meta["train_seed_count"] == 13
    assert meta["train_seed_decision"] == "AMEND n upward"
    assert meta["frozen_utc"] is not None

    defaults = yaml.safe_load(registry.read_text(encoding="utf-8"))["defaults"]
    assert defaults["train_seeds"] == list(range(101, 114))
    assert "SKELETON" not in registry.read_text(encoding="utf-8")


def test_freeze_is_idempotent(registry, power):
    power(10)
    first = freeze_registry.freeze()
    assert freeze_registry.freeze()["frozen_utc"] == first["frozen_utc"]


def test_a_frozen_registry_is_never_silently_rewritten(registry, power):
    power(10)
    freeze_registry.freeze()
    power(14, decision="AMEND n upward")
    with pytest.raises(SystemExit) as raised:
        freeze_registry.freeze()
    assert "already frozen" in str(raised.value)


# --------------------------------------------------------------------------- sample size


def test_a_smaller_sample_size_takes_a_prefix_of_the_registered_pool():
    assert train_seed_list(4) == TRAIN_SEEDS[:4]


def test_zero_means_the_whole_registered_pool():
    assert train_seed_list(0) == TRAIN_SEEDS


def test_an_amended_sample_size_above_the_pool_is_honoured_not_clipped():
    """The regression this file exists for: `TRAIN_SEEDS[:16]` would return 10 seeds."""
    seeds = train_seed_list(16)
    assert len(seeds) == 16
    assert seeds[: len(TRAIN_SEEDS)] == TRAIN_SEEDS, "the registered pool must stay a prefix"
    assert len(set(seeds)) == 16


def test_pilot_observations_cannot_reach_a_confirmatory_test():
    """Seed-block disjointness is one guard; variant scoping is the other, and it is structural.

    Pilot cells carry `variant='pilot'`, and every paired comparison reads `variant='main'`. Even
    if a pilot seed collided with a registered one, the pilot row could not enter the test.
    """
    import inspect

    import pandas as pd

    from src.evaluation.aggregate import paired_frame

    assert inspect.signature(paired_frame).parameters["variant"].default == "main"

    columns = ["method", "scenario", "weight_id", "variant", "train_seed",
               "left_behind_fraction", "n_episodes"]
    rows = []
    for variant in ("main", "pilot"):
        for method in ("ppo_uq", "ppo_point"):
            for seed in (101, 102, 103, 104):
                rows.append([method, "surge", "w_balanced", variant, seed,
                             0.01 if variant == "main" else 0.99, 30])
    seeds = pd.DataFrame(rows, columns=columns)

    pair = paired_frame(seeds, seeds.assign(eval_seed=1), method_a="ppo_uq",
                        method_b="ppo_point", scenario="surge", metric="left_behind_fraction")
    assert pair is not None and pair.paired
    assert set(pair.values_a) == {0.01}, "a pilot value reached the confirmatory comparison"


def test_the_pilot_pools_are_disjoint_from_every_sweep_pool():
    """No pilot observation may reach a confirmatory result, at any amended sample size."""
    from scripts.run_experiments import (EVAL_SEEDS, PILOT_EVAL_SEEDS, PILOT_TRAIN_SEEDS)

    assert not set(PILOT_TRAIN_SEEDS) & set(train_seed_list(20))
    assert not set(PILOT_EVAL_SEEDS) & set(EVAL_SEEDS)
