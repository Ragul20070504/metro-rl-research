"""Guard exact-byte recovery, isolated experiment design and auditable references."""
import hashlib
import json

import pandas as pd
import pytest

from scripts.recover_raw_data import promote
from scripts import run_replication as replication
from scripts.analyse_followup import _per_budget
from scripts.paper_refs import REFERENCES, cite
from src.common.paths import REPO_ROOT


def test_recovery_never_promotes_changed_bytes_or_overwrites_existing_files(tmp_path):
    candidate, target = tmp_path / "candidate", tmp_path / "original"
    expected = hashlib.sha256(b"original bytes").hexdigest()
    candidate.write_bytes(b"changed publisher bytes")
    assert promote(candidate, target, expected) == "UPSTREAM_CHANGED"
    assert not target.exists() and candidate.exists()
    candidate.write_bytes(b"original bytes")
    assert promote(candidate, target, expected) == "RECOVERED"
    assert target.read_bytes() == b"original bytes" and not candidate.exists()
    candidate.write_bytes(b"a later changed download")
    assert promote(candidate, target, expected) == "ALREADY_VERIFIED"
    assert target.read_bytes() == b"original bytes"
    target.write_bytes(b"unexplained local change")
    with pytest.raises(ValueError, match="Existing file"):
        promote(candidate, target, expected)


def test_replication_has_the_entire_fixed_grid_and_separate_placebo_and_baselines():
    jobs = replication.jobs()
    assert len(jobs) == len(set(jobs)) == 110
    assert sum(job[0] == "ppo_placebo" for job in jobs) == 10
    assert all(job[2] == 60_000 for job in jobs if job[0] == "ppo_placebo")
    cells = replication.cells()
    assert len(cells) == len({cell.key() for cell in cells}) == 13_680
    assert sum(cell.method in replication.BASELINES for cell in cells) == 480
    assert all(replication.BASE in replication.checkpoint(job).parents for job in jobs)


def test_identical_policies_are_a_zero_difference_not_an_invalid_significance_result():
    paired = pd.DataFrame({"ppo_point": [0.1] * 10, "ppo_uq": [0.1] * 10})
    result = _per_budget(paired, 15_000, "surge")
    assert result["p_value"] == 1 and result["direction"] == "no difference"
    assert result["ci_low"] == result["ci_high"] == 0
    json.dumps(result, allow_nan=False)


def test_reference_fields_match_saved_authoritative_records():
    verified = json.loads((REPO_ROOT / "results/reference_verification.json").read_text())
    records = {row["key"]: row["record"] for row in verified["records"]}
    assert len(REFERENCES) == len(records) == 18
    for ref in REFERENCES:
        source = records[ref.key]
        if "citation_metadata" in source:
            fields = dict(source["citation_metadata"])
            assert ref.title == fields["title"] and ref.year == fields["date"][:4]
        else:
            assert ref.title == source["title"][0]
            assert ref.year == str(source["published"]["date-parts"][0][0])
            assert ref.url.endswith(source["DOI"])
        assert ref.status == "full" and "VERIFICATION REQUIRED" not in ref.formatted()
    assert cite("schulman2017") == "(Schulman et al., 2017)"


def test_forecaster_default_respects_the_isolated_models_directory(tmp_path, monkeypatch):
    import src.common.paths as paths
    from src.environment.forecast_provider import load_trained_provider

    monkeypatch.setattr(paths, "MODELS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        load_trained_provider(None)


def test_diagnostic_renderer_cannot_inspect_partial_policy_outcomes(tmp_path, monkeypatch):
    from scripts import render_replication_diagnostics as renderer

    monkeypatch.setattr(renderer, 'ROOT', tmp_path)
    with pytest.raises(RuntimeError, match='analysis is absent'):
        renderer.render()
    (tmp_path / 'results').mkdir()
    (tmp_path / 'results/replication_20261015_analysis.json').write_text(json.dumps(
        {'complete': False, 'policies': 110, 'evaluation_cells': 120}))
    monkeypatch.setattr(renderer, 'verify_frozen', lambda: pytest.fail('Partial-grid guard was bypassed'))
    with pytest.raises(RuntimeError, match='Incomplete replication'):
        renderer.render()
    assert not (tmp_path / 'results/paper_figures').exists()
