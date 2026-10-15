"""Every research page renders, its filters respond, and its numbers match the artefacts.

A dashboard that imports cleanly can still show a reader nothing, or worse, show them a number
that disagrees with the paper. These tests render each page headlessly through Streamlit's own
app-testing harness and assert on what actually reached the page.

The last test is the one that matters most: the figure the dashboard displays for the primary
result must equal the figure in `results/statistical_analysis.json`, so the dashboard cannot
drift from the registered analysis.
"""

from __future__ import annotations

import json

import pytest

from src.common.paths import RESULTS_DIR

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.demo import research_views  # noqa: E402

REPO = str(RESULTS_DIR.parent).replace("\\", "\\\\")
PAGES = [(group, name) for group, entries in research_views.PAGES.items()
         for name, _ in entries]


def _script(group: str, name: str, presets: dict | None = None) -> str:
    body = [
        f"import sys; sys.path.insert(0, r'{RESULTS_DIR.parent}')",
        "import streamlit as st",
        "from src.demo import research_views",
    ]
    for key, value in (presets or {}).items():
        body.append(f"st.session_state[{key!r}] = {value!r}")
    body.append(f"dict(research_views.PAGES[{group!r}])[{name!r}]()")
    return "\n".join(body)


def _run(group: str, name: str, presets: dict | None = None) -> AppTest:
    app = AppTest.from_string(_script(group, name, presets))
    app.run(timeout=180)
    return app


@pytest.mark.parametrize(("group", "name"), PAGES, ids=[n for _, n in PAGES])
def test_every_page_renders_without_error(group: str, name: str) -> None:
    app = _run(group, name)
    assert not app.exception, f"{name} raised: {app.exception[0].value if app.exception else ''}"


@pytest.mark.parametrize(("group", "name"), PAGES, ids=[n for _, n in PAGES])
def test_every_page_shows_something(group: str, name: str) -> None:
    """A page that renders but displays nothing is a broken page that happens not to crash."""
    app = _run(group, name)
    produced = (len(app.markdown) + len(app.dataframe) + len(app.metric)
                + len(app.json) + len(app.code))
    assert produced > 0, f"{name} rendered no content"


def test_no_page_is_missing_its_artefacts() -> None:
    """`missing()` renders a warning naming the artefact; none should fire in a complete run."""
    offenders = []
    for group, name in PAGES:
        app = _run(group, name)
        for warning in app.warning:
            if "is not present" in str(warning.value):
                offenders.append((name, str(warning.value)))
    assert not offenders, f"pages reported missing artefacts: {offenders}"


def test_scenario_filter_changes_the_table() -> None:
    """A filter that does not change the output is decoration, not a control."""
    normal = _run("Results", "Scenario explorer", {"sc_scenario": "normal"})
    surge = _run("Results", "Scenario explorer", {"sc_scenario": "surge"})
    assert not normal.exception and not surge.exception
    first = normal.dataframe[0].value.to_csv()
    second = surge.dataframe[0].value.to_csv()
    assert first != second, "changing the scenario did not change the displayed table"


def test_metric_filter_changes_the_table() -> None:
    baseline = _run("Results", "Baseline explorer",
                    {"bl_metric": "left_behind_fraction", "bl_scenario": "surge"})
    energy = _run("Results", "Baseline explorer",
                  {"bl_metric": "energy_units", "bl_scenario": "surge"})
    assert not baseline.exception and not energy.exception
    assert baseline.dataframe[0].value.to_csv() != energy.dataframe[0].value.to_csv()


def test_primary_result_on_the_page_matches_the_stored_analysis() -> None:
    """The dashboard must not be able to disagree with the registered analysis."""
    stored = json.loads((RESULTS_DIR / "statistical_analysis.json").read_text(encoding="utf-8"))
    test = next(t for t in stored["confirmatory"] if t["experiment_id"] == "EXP-UQ-001")

    app = _run("Overview", "Final verdict")
    assert not app.exception
    text = " ".join(str(m.value) for m in app.markdown)
    assert f"{test['mean_difference']:+.6f}" in text, "the reported difference does not match"
    assert f"{test['p_adjusted']:.6g}" in text, "the reported adjusted p does not match"
    assert "NOT SUPPORTED" in text, "the verdict shown does not match the stored result"


def test_follow_up_page_never_reports_a_result() -> None:
    """An incomplete follow-up must never appear as evidence."""
    app = _run("Provenance", "Follow-up (incomplete)")
    assert not app.exception
    text = " ".join(str(m.value) for m in app.markdown)
    assert "FOLLOW-UP / INCOMPLETE" in text
    errors = " ".join(str(e.value) for e in app.error)
    assert "No result is reported from this study" in errors
