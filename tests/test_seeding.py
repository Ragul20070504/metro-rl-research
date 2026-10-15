"""Seeding tests, including the common-random-numbers guarantee.

Protocol section 7.1 requires every method to face an identical demand and disruption
realisation within a ``(scenario, eval_seed)`` cell. ``tests/test_common_random_numbers.py``
will assert that end to end once the simulator exists; these tests assert the property at the
level where it is actually made true.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.seeding import (
    AGENT_STREAMS,
    ENVIRONMENT_STREAMS,
    SCENARIO_NAMES,
    SeedPair,
    agent_streams,
    environment_streams,
    stable_entropy,
)


def test_stable_entropy_is_reproducible_and_discriminating() -> None:
    assert stable_entropy("env", "surge", 7) == stable_entropy("env", "surge", 7)
    assert stable_entropy("env", "surge", 7) != stable_entropy("env", "surge", 8)
    assert stable_entropy("env", "surge", 7) != stable_entropy("env", "normal", 7)


def test_environment_streams_are_reproducible() -> None:
    first = environment_streams("surge", 7)
    second = environment_streams("surge", 7)
    for name in ENVIRONMENT_STREAMS:
        assert np.array_equal(first[name].normal(size=16), second[name].normal(size=16))


def test_environment_streams_cannot_depend_on_the_method() -> None:
    """The CRN guarantee, enforced structurally.

    ``environment_streams`` takes no method argument, so there is no way for the method under
    evaluation to influence the demand or failure realisation. This test pins that signature
    down: adding a method parameter later would break it loudly.
    """
    import inspect

    parameters = set(inspect.signature(environment_streams).parameters)
    assert parameters == {"scenario", "eval_seed"}
    assert "method" not in parameters


def test_environment_streams_differ_across_seeds() -> None:
    surge = environment_streams("surge", 7)["demand"].normal(size=32)
    other_seed = environment_streams("surge", 8)["demand"].normal(size=32)
    assert not np.array_equal(surge, other_seed)


def test_all_scenarios_share_a_base_world_at_the_same_seed() -> None:
    """Decision D-14, and the inversion of an earlier assertion.

    This test previously required ``normal`` and ``surge`` at the same seed to have *different*
    demand streams. That made every cross-scenario comparison unmatched: protocol section 7.4
    defines the degradation ratio as a stressed metric divided by the ``normal`` metric "matched
    on eval_seed", and section 7.2 requires ``combined`` to carry the same surge and the same
    failures as the individual families. Measured on the real network, the realisation noise the
    old keying introduced was large enough to make ``surge`` score *better* than ``normal`` in
    three of five seeds.

    The scenario name is therefore excluded from the entropy, and the scenario's effect enters
    only through which perturbation streams it draws from.
    """
    for stream in ENVIRONMENT_STREAMS:
        draws = {
            scenario: environment_streams(scenario, 7)[stream].normal(size=32)
            for scenario in ("normal", "surge", "disruption", "combined")
        }
        reference = draws["normal"]
        for scenario, values in draws.items():
            assert np.array_equal(values, reference), f"{scenario}/{stream}"


def test_an_unknown_scenario_is_rejected() -> None:
    """The scenario is excluded from the entropy but still validated, so a typo cannot silently
    produce a valid-looking cell."""
    with pytest.raises(ValueError, match="unknown scenario"):
        environment_streams("sirge", 7)


def test_scenario_names_agree_with_the_simulator() -> None:
    from src.simulator.disruptions import SCENARIOS

    assert set(SCENARIO_NAMES) == set(SCENARIOS)


def test_streams_are_mutually_independent() -> None:
    """Consuming one stream must not advance another.

    If demand and failures shared a generator, a scenario that draws more failures would
    shift the demand realisation, silently breaking the paired comparison.
    """
    streams = environment_streams("normal", 1)
    reference = environment_streams("normal", 1)["demand"].normal(size=8)

    streams["failures"].normal(size=1000)  # heavy consumption of a sibling stream
    assert np.array_equal(streams["demand"].normal(size=8), reference)


def test_agent_and_environment_namespaces_are_disjoint() -> None:
    """An agent stream and an environment stream must never coincide by construction."""
    env = environment_streams("normal", 101)["demand"].normal(size=16)
    agent = agent_streams("ppo_uq", 101)["action_sample"].normal(size=16)
    assert not np.array_equal(env, agent)


def test_agent_streams_differ_across_arms_and_seeds() -> None:
    uq = agent_streams("ppo_uq", 101)["policy_init"].normal(size=16)
    point = agent_streams("ppo_point", 101)["policy_init"].normal(size=16)
    other = agent_streams("ppo_uq", 102)["policy_init"].normal(size=16)
    assert not np.array_equal(uq, point)
    assert not np.array_equal(uq, other)


def test_all_declared_streams_are_present() -> None:
    assert set(environment_streams("normal", 1)) == set(ENVIRONMENT_STREAMS)
    assert set(agent_streams("ppo_uq", 101)) == set(AGENT_STREAMS)


class TestSeedPair:
    """Protocol section 8.1: training and evaluation seed pools must be disjoint."""

    train_pool = set(range(101, 111))
    eval_pool = set(range(1, 31))

    def test_valid_pair_passes(self) -> None:
        SeedPair(train_seed=101, eval_seed=7).validate(self.train_pool, self.eval_pool)

    def test_baseline_without_training_seed_passes(self) -> None:
        SeedPair(train_seed=None, eval_seed=7).validate(self.train_pool, self.eval_pool)

    def test_overlapping_pools_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="disjoint"):
            SeedPair(train_seed=5, eval_seed=5).validate({5, 6}, {5, 7})

    def test_seed_outside_its_pool_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="training pool"):
            SeedPair(train_seed=999, eval_seed=7).validate(self.train_pool, self.eval_pool)
        with pytest.raises(ValueError, match="evaluation pool"):
            SeedPair(train_seed=101, eval_seed=999).validate(self.train_pool, self.eval_pool)
