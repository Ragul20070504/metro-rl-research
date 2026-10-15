"""Provenance enforcement.

Protocol section 5.2: every simulation parameter must have provenance, and a config holding an
``unresolved`` parameter must not load outside a tagged sensitivity sweep. Six load-bearing
assumptions are currently unresolved or synthetic (``docs/research_audit.md`` section E), four
of which feed the reward, so these guards do real work.
"""

from __future__ import annotations

import pytest

from src.common.config import ProvenanceError, config_hash, load_config

GOOD = """
provenance_required: true
config_id: test_sim
simulator:
  min_headway_s:
    value: 120
    provenance: simulation_assumption
    note: cross-checked against minimum scheduled headway in GTFS
  running_time_s:
    value: 90
    provenance: real_data:mta_gtfs_static
  failure_rate_per_km:
    value: 0.0001
    provenance: synthetic
    note: rate calibrated against mta_mdbf
"""

MISSING_PROVENANCE = """
provenance_required: true
simulator:
  min_headway_s: 120
"""

UNRESOLVED = """
provenance_required: true
simulator:
  energy_kwh_per_train_km:
    value: 4.0
    provenance: unresolved
    note: no citable figure sourced; see assumption A15
"""

BAD_CLASS = """
provenance_required: true
simulator:
  capacity:
    value: 1000
    provenance: seemed_about_right
"""

BARE_PREFIX = """
provenance_required: true
simulator:
  capacity:
    value: 1000
    provenance: "real_data:"
"""

PERMISSIVE = """
provenance_required: false
ppo:
  learning_rate: 0.0003
  gamma: 0.99
"""


def test_well_formed_config_loads_and_collapses_parameter_nodes(write_config) -> None:
    config = load_config(write_config(GOOD))
    assert config.get("simulator.min_headway_s") == 120
    assert config.get("simulator.running_time_s") == 90
    assert config.provenance["simulator.running_time_s"] == "real_data:mta_gtfs_static"
    assert config.unresolved == []


def test_missing_provenance_is_rejected_in_strict_mode(write_config) -> None:
    with pytest.raises(ProvenanceError, match="state no provenance"):
        load_config(write_config(MISSING_PROVENANCE))


def test_unresolved_parameter_is_blocked_by_default(write_config) -> None:
    """The default path must make an ungrounded parameter impossible to use by accident."""
    with pytest.raises(ProvenanceError, match="unresolved"):
        load_config(write_config(UNRESOLVED))


def test_unresolved_parameter_requires_the_sweep_tag(write_config) -> None:
    """allow_unresolved alone is not enough; the run must be self-identifying."""
    path = write_config(UNRESOLVED)
    with pytest.raises(ProvenanceError, match="sensitivity_sweep"):
        load_config(path, allow_unresolved=True)


def test_unresolved_parameter_loads_under_a_tagged_sweep(write_config) -> None:
    config = load_config(
        write_config(UNRESOLVED), allow_unresolved=True, sweep_tag="sensitivity_sweep"
    )
    assert config.unresolved == ["simulator.energy_kwh_per_train_km"]
    assert config.get("simulator.energy_kwh_per_train_km") == 4.0


def test_provenance_class_outside_the_vocabulary_is_rejected(write_config) -> None:
    with pytest.raises(ProvenanceError, match="not one of"):
        load_config(write_config(BAD_CLASS))


def test_prefixed_provenance_class_requires_a_suffix(write_config) -> None:
    """'real_data:' with no dataset id names no source and must not pass."""
    with pytest.raises(ProvenanceError, match="not one of"):
        load_config(write_config(BARE_PREFIX))


def test_permissive_mode_allows_plain_values_but_still_records_them(write_config) -> None:
    """Configs of non-physical hyperparameters need not carry provenance, but stay visible."""
    config = load_config(write_config(PERMISSIVE))
    assert config.get("ppo.learning_rate") == pytest.approx(0.0003)
    assert "ppo.learning_rate" in config.undocumented


class TestConfigHash:
    """The hash is part of the traceability chain and must depend on content alone."""

    def test_is_insensitive_to_key_order(self) -> None:
        assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})

    def test_changes_when_a_value_changes(self) -> None:
        assert config_hash({"a": 1}) != config_hash({"a": 2})

    def test_is_stable_across_calls(self, write_config) -> None:
        first = load_config(write_config(GOOD, "a.yaml")).hash
        second = load_config(write_config(GOOD, "b.yaml")).hash
        assert first == second, "hash must depend on values, not on the file path"
