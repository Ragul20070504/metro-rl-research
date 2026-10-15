"""Tests for the Metro Intelligent Dispatch prototype.

Two things are being protected here. The first is that the product works: every page renders,
filters respond, recommendations are producible. The second, and more important, is that the
product **cannot damage the completed research** -- it must read research artefacts and never
write them, must never present discarded or incomplete evidence as a result, and must agree with
the stored analysis wherever it quotes it.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from product.services.research_repository import (
    CONFIRMATORY, FOLLOW_UP, PROTECTED, REPO_ROOT, ResearchArtefactMissing, assert_read_only,
    repository,
)

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from product.ui import pages as product_pages  # noqa: E402

PAGE_IDS = [name for name, _ in product_pages.FLAT]


def _script(name: str, presets: dict | None = None) -> str:
    body = [
        f"import sys; sys.path.insert(0, r'{REPO_ROOT}')",
        "import streamlit as st",
        "from product.ui import pages",
    ]
    for key, value in (presets or {}).items():
        body.append(f"st.session_state[{key!r}] = {value!r}")
    body.append(f"dict(pages.FLAT)[{name!r}]()")
    return "\n".join(body)


def _run(name: str, presets: dict | None = None) -> AppTest:
    app = AppTest.from_string(_script(name, presets))
    app.run(timeout=300)
    return app


# --------------------------------------------------------------------------- research integrity


class TestResearchIntegrity:
    """The product must be incapable of altering the completed study."""

    def test_no_protected_artefact_changes_while_the_product_runs(self):
        before = {name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
                  for name in PROTECTED if (REPO_ROOT / name).is_file()}
        repo = repository()
        repo.primary_result()
        repo.comparison_table("left_behind_fraction", "surge")
        repo.mechanism()
        repo.follow_up_status()
        after = {name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
                 for name in PROTECTED if (REPO_ROOT / name).is_file()}
        assert before == after, "reading research artefacts changed them"
        assert assert_read_only() == []

    def test_product_package_opens_no_research_path_for_writing(self):
        """A static check: nothing under `product/` may write into `results/` or `models/`.

        Cheap and blunt, but it catches the failure mode that matters -- someone later adding a
        convenience `save()` that lands in the research tree.
        """
        offenders = []
        for path in (REPO_ROOT / "product").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for marker in ("write_text(", "to_csv(", "to_parquet(", "savefig(", "open("):
                for line in text.splitlines():
                    if marker in line and ("RESULTS" in line or "MODELS" in line
                                           or "results/" in line or "models/" in line):
                        if "read" in line or "download_button" in line:
                            continue
                        offenders.append(f"{path.name}: {line.strip()[:90]}")
        assert not offenders, f"product code appears to write to research paths: {offenders}"

    def test_the_research_dashboard_entry_point_is_untouched(self):
        """`app.py` on port 8501 is a research deliverable and must not be modified."""
        assert (REPO_ROOT / "app.py").is_file()
        text = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
        assert "research_views" in text, "the research dashboard no longer imports its own views"
        assert "product" not in text.split("\n")[0:40][0], "research app.py imports product code"


# --------------------------------------------------------------------------- repository


class TestRepository:
    def test_verdict_is_derived_from_the_stored_test(self):
        repo = repository()
        test = repo.primary_result()
        expected = ("SUPPORTED" if test["mean_difference"] < 0
                    and test.get("rejects_at_family_alpha") else "NOT SUPPORTED")
        assert repo.h1_verdict() == expected

    def test_primary_result_matches_the_stored_analysis_exactly(self):
        stored = json.loads(
            (REPO_ROOT / "results" / "statistical_analysis.json").read_text(encoding="utf-8"))
        expected = next(t for t in stored["confirmatory"]
                        if t["experiment_id"] == "EXP-UQ-001")
        assert repository().primary_result() == expected

    def test_registered_cells_exclude_pilot_and_sensitivity_variants(self):
        cells = repository().registered_cells()
        assert set(cells.variant.unique()) == {"main"}
        assert not (cells.train_seed >= 900).any(), "a discarded pilot policy reached the product"

    def test_registered_cells_still_include_the_baselines(self):
        """Baselines carry train_seed = -1; a seed filter would silently drop them."""
        methods = set(repository().registered_cells().method)
        assert {"fixed_timetable", "heuristic_threshold", "milp_static"} <= methods

    def test_missing_artefact_raises_rather_than_substituting(self):
        from product.services import research_repository as module

        with pytest.raises(ResearchArtefactMissing):
            module._read_json("results/does_not_exist.json")

    def test_comparison_table_is_ordered_and_complete(self):
        table = repository().comparison_table("left_behind_fraction", "surge")
        assert list(table.columns) == ["method", "n", "mean", "sd"]
        assert table["mean"].is_monotonic_increasing
        assert len(table) >= 5

    def test_follow_up_is_always_reported_as_incomplete_and_unusable(self):
        status = repository().follow_up_status()
        assert status["evidence"] == FOLLOW_UP
        assert status["usable"] is False
        assert status["policies_trained"] < status["policies_planned"]


# --------------------------------------------------------------------------- services


class TestOperations:
    def test_available_methods_exclude_pilot_seeds(self):
        from product.services.operations import available_methods

        seeds = {s for _, _, seeds in available_methods() for s in seeds}
        assert not [s for s in seeds if s >= 900]

    @pytest.mark.parametrize("level,expected", [(0.0, "NOT AVAILABLE"), (0.05, "LOW"),
                                               (0.15, "MEDIUM"), (0.40, "HIGH")])
    def test_uncertainty_banding(self, level, expected):
        from product.services.operations import uncertainty_level

        assert uncertainty_level(1000.0, 1000.0 * level) == expected

    def test_capacity_pressure_bands(self):
        from product.services.operations import capacity_pressure

        assert capacity_pressure(0, 0, 36) == "LOW"
        assert capacity_pressure(2000, 30, 36) == "HIGH"

    @pytest.mark.slow
    def test_a_session_produces_a_feasible_recommendation(self):
        from product.services.operations import SessionService

        session = SessionService("ppo_point", "surge", eval_seed=7, train_seed=101)
        recommendation = session.recommend()
        assert recommendation.action in recommendation.feasible_actions
        assert recommendation.reasons
        assert recommendation.evidence == "PRODUCT DEMONSTRATION"

    @pytest.mark.slow
    def test_stepping_advances_the_episode_and_reports_before_after(self):
        from product.services.operations import SessionService

        session = SessionService("ppo_point", "surge", eval_seed=7, train_seed=101)
        outcome = session.step(12)
        assert outcome.epochs_run == 12
        assert outcome.after.get("epochs_stepped") == 12
        assert isinstance(outcome.delta("inductions"), float)


# --------------------------------------------------------------------------- pages


class TestPages:
    @pytest.mark.parametrize("name", PAGE_IDS)
    def test_page_renders_without_error(self, name):
        app = _run(name)
        assert not app.exception, (
            f"{name} raised: {app.exception[0].value if app.exception else ''}")

    @pytest.mark.parametrize("name", PAGE_IDS)
    def test_page_shows_content(self, name):
        app = _run(name)
        produced = (len(app.markdown) + len(app.dataframe) + len(app.metric)
                    + len(app.json) + len(app.code) + len(app.title))
        assert produced > 0, f"{name} rendered nothing"

    def test_research_pages_declare_an_evidence_class(self):
        for name in ("Research results", "Mechanism", "Statistics", "Follow-up"):
            app = _run(name)
            rendered = " ".join(str(m.value) for m in app.markdown)
            assert any(kind in rendered for kind in
                       (CONFIRMATORY, "EXPLORATORY", "DESCRIPTIVE", FOLLOW_UP)), (
                f"{name} shows research data without an evidence badge")

    def test_research_results_page_matches_the_stored_analysis(self):
        """The product must be unable to disagree with the registered analysis.

        The effect appears in a metric card and the verdict in markdown, so both surfaces are
        searched: an assertion against one alone would pass while the other drifted.
        """
        test = repository().primary_result()
        app = _run("Research results")
        rendered = " ".join(
            [str(m.value) for m in app.markdown]
            + [f"{m.label} {m.value}" for m in app.metric])
        assert f"{test['mean_difference']:+.6f}" in rendered, "the effect size does not match"
        assert f"{test['p_adjusted']:.6g}" in rendered, "the adjusted p-value does not match"
        assert f"{test['ci_low']:.6f}" in rendered, "the confidence interval does not match"
        assert "NOT SUPPORTED" in rendered, "the verdict shown does not match the stored result"

    def test_follow_up_page_never_presents_a_result(self):
        app = _run("Follow-up")
        errors = " ".join(str(e.value) for e in app.error)
        assert "no result is reported" in errors.lower()

    def test_strategy_comparison_metric_filter_changes_the_table(self):
        first = _run("Strategy comparison",
                     {"cmp_metric": "left_behind_fraction", "cmp_scenario": "surge"})
        second = _run("Strategy comparison",
                      {"cmp_metric": "energy_units", "cmp_scenario": "surge"})
        assert not first.exception and not second.exception
        assert first.dataframe[0].value.to_csv() != second.dataframe[0].value.to_csv()

    def test_strategy_comparison_scenario_filter_changes_the_table(self):
        first = _run("Strategy comparison",
                     {"cmp_metric": "left_behind_fraction", "cmp_scenario": "normal"})
        second = _run("Strategy comparison",
                      {"cmp_metric": "left_behind_fraction", "cmp_scenario": "surge"})
        assert first.dataframe[0].value.to_csv() != second.dataframe[0].value.to_csv()

    def test_about_page_states_the_operating_limits(self):
        app = _run("About")
        rendered = " ".join(str(m.value) for m in app.markdown)
        for phrase in ("does not control", "not safety-certified", "NOT SUPPORTED"):
            assert phrase in rendered, f"the About page omits: {phrase}"


class TestAppShell:
    def test_the_app_module_imports_and_declares_a_demo_route(self):
        import product.app as application

        assert callable(application.main)
        names = {name for name, _ in application.PANEL_DEMO}
        assert names <= set(PAGE_IDS), "the panel demo references a page that does not exist"
        assert len(application.PANEL_DEMO) >= 8
