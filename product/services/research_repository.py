"""Read-only access to the completed research artefacts.

This is the **only** route by which the product reads research data, and it is deliberately
one-directional. Every accessor opens a file for reading and returns a copy; nothing in
`product/` opens a research path for writing, and `assert_read_only()` proves it by checking that
the protected artefacts still hash to what they hashed at product start-up.

The research study is frozen. Its conclusion is:

    H1 = NOT SUPPORTED. ppo_uq left more passengers behind than ppo_point.

Numbers are read from the stored analysis at call time rather than restated here, so the product
cannot drift from the paper. The one constant this module defines is *which file* to read.
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
MODELS = REPO_ROOT / "models"

#: Artefacts the product may read and must never alter. Hashed at import to detect any write.
PROTECTED = (
    "experiments/registry.yaml",
    "results/statistical_analysis.json",
    "results/power_analysis.json",
    "results/mechanism_analysis.json",
    "results/sensitivity_analysis.json",
    "results/simulator_validation.json",
    "results/forecasting_metrics.json",
    "results/shap_analysis.json",
    "results/reproducibility_audit.json",
    "models/obs_scaler.json",
    "app.py",
)

#: Evidence classes. Every research figure the product shows carries exactly one.
CONFIRMATORY = "CONFIRMATORY"
EXPLORATORY = "EXPLORATORY"
DESCRIPTIVE = "DESCRIPTIVE"
FOLLOW_UP = "FOLLOW-UP / INCOMPLETE"
PILOT = "PILOT / DISCARDED"
PRODUCT = "PRODUCT DEMONSTRATION"


class ResearchArtefactMissing(RuntimeError):
    """Raised when an artefact the caller needs is absent.

    The product surfaces this to the user rather than substituting a value. A dashboard that
    silently invents a number when a file is missing is worse than one that says it is missing.
    """


@dataclass(frozen=True)
class MethodResult:
    """One method's outcome on one scenario, as the registered study measured it."""

    method: str
    scenario: str
    metric: str
    mean: float
    n: int
    evidence: str = CONFIRMATORY


def _read_json(relative: str) -> Any:
    path = REPO_ROOT / relative
    if not path.is_file():
        raise ResearchArtefactMissing(relative)
    return json.loads(path.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def _hashes() -> dict[str, str]:
    """Fingerprint the protected artefacts once, at first use."""
    out: dict[str, str] = {}
    for name in PROTECTED:
        path = REPO_ROOT / name
        if path.is_file():
            out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def assert_read_only() -> list[str]:
    """Return the protected artefacts whose contents have changed since start-up.

    An empty list is the invariant the product must maintain. This is not decoration: it is the
    check that turns "the product does not write to research files" from a claim into a test.
    """
    baseline = _hashes()
    changed = []
    for name, digest in baseline.items():
        path = REPO_ROOT / name
        if not path.is_file():
            changed.append(f"{name} (deleted)")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            changed.append(name)
    return changed


class ResearchResultsRepository:
    """Every research number the product displays comes through here."""

    # ---------------------------------------------------------------- primary

    def primary_result(self) -> dict[str, Any]:
        """The registered H1 test, located by experiment id rather than list position."""
        stats = _read_json("results/statistical_analysis.json")
        for test in stats.get("confirmatory", []):
            if test.get("experiment_id") == "EXP-UQ-001":
                return dict(test)
        raise ResearchArtefactMissing("EXP-UQ-001 in the confirmatory family")

    def placebo_result(self) -> dict[str, Any]:
        stats = _read_json("results/statistical_analysis.json")
        for test in stats.get("confirmatory", []):
            if test.get("experiment_id") == "EXP-UQ-002":
                return dict(test)
        raise ResearchArtefactMissing("EXP-UQ-002 in the confirmatory family")

    def h1_verdict(self) -> str:
        """Derived from the stored test, never asserted."""
        test = self.primary_result()
        supported = test["mean_difference"] < 0 and test.get("rejects_at_family_alpha")
        return "SUPPORTED" if supported else "NOT SUPPORTED"

    def confirmatory_tests(self) -> list[dict[str, Any]]:
        return list(_read_json("results/statistical_analysis.json").get("confirmatory", []))

    def exploratory_tests(self) -> list[dict[str, Any]]:
        return list(_read_json("results/statistical_analysis.json").get("exploratory", []))

    def power(self) -> dict[str, Any]:
        return _read_json("results/power_analysis.json")

    # ---------------------------------------------------------------- per-method outcomes

    @functools.lru_cache(maxsize=1)  # noqa: B019 - repository is a process-lifetime singleton
    def _seed_level(self) -> pd.DataFrame:
        path = RESULTS / "seed_level.parquet"
        if not path.is_file():
            raise ResearchArtefactMissing("results/seed_level.parquet")
        return pd.read_parquet(path)

    def registered_cells(self) -> pd.DataFrame:
        """Registered confirmatory cells only.

        `variant == 'main'` excludes the discarded pilot (variant `pilot`), the sensitivity
        variants and any follow-up namespace. No `train_seed` filter: that would additionally
        drop the deterministic baselines, which carry `train_seed = -1`.
        """
        df = self._seed_level()
        return df[(df.variant == "main") & (df.weight_id == "w_balanced")].copy()

    def methods(self) -> list[str]:
        return sorted(self.registered_cells().method.unique())

    def metrics(self) -> list[str]:
        skip = {"method", "scenario", "weight_id", "variant", "train_seed", "n_episodes"}
        df = self.registered_cells()
        return [c for c in df.columns
                if c not in skip and pd.api.types.is_numeric_dtype(df[c])]

    def comparison_table(self, metric: str, scenario: str) -> pd.DataFrame:
        """Mean of `metric` per method for one scenario, as the study measured it."""
        df = self.registered_cells()
        if metric not in df.columns:
            raise ResearchArtefactMissing(f"metric {metric!r} in seed_level.parquet")
        subset = df[df.scenario == scenario]
        table = (subset.groupby("method")[metric]
                 .agg(n="count", mean="mean", sd="std").reset_index())
        return table.sort_values("mean").reset_index(drop=True)

    def scenario_matrix(self, metric: str) -> pd.DataFrame:
        df = self.registered_cells()
        scenarios = [s for s in ("normal", "surge", "disruption", "combined")
                     if s in set(df.scenario)]
        return df.pivot_table(index="method", columns="scenario", values=metric,
                              aggfunc="mean")[scenarios]

    def best_method(self, metric: str, scenario: str, lower_is_better: bool = True) -> str:
        table = self.comparison_table(metric, scenario)
        return str(table.iloc[0 if lower_is_better else -1]["method"])

    # ---------------------------------------------------------------- supporting artefacts

    def mechanism(self) -> dict[str, Any]:
        return _read_json("results/mechanism_analysis.json")

    def sensitivity(self) -> dict[str, Any]:
        return _read_json("results/sensitivity_analysis.json")

    def simulator_validation(self) -> dict[str, Any]:
        return _read_json("results/simulator_validation.json")

    def forecasting_metrics(self) -> dict[str, Any]:
        return _read_json("results/forecasting_metrics.json")

    def shap(self) -> dict[str, Any]:
        return _read_json("results/shap_analysis.json")

    def reproducibility(self) -> dict[str, Any]:
        return _read_json("results/reproducibility_audit.json")

    def separation_audit(self) -> dict[str, Any] | None:
        try:
            return _read_json("results/separation_audit.json")
        except ResearchArtefactMissing:
            return None

    def follow_up_status(self) -> dict[str, Any]:
        """The training-budget follow-up, always reported as incomplete.

        Counts are read from disk rather than stored, so the status cannot become stale, and the
        evidence class is fixed at FOLLOW-UP / INCOMPLETE regardless of what those counts say.
        """
        policies = MODELS / "policies"
        namespaces = sorted(p.name for p in policies.glob("budget_*")) if policies.is_dir() else []
        trained = sum(1 for ns in namespaces for _ in (policies / ns).rglob("policy.zip"))
        cells = sum(1 for _ in (RESULTS / "runs").glob("budget_*/**/metrics.json"))
        return {
            "evidence": FOLLOW_UP,
            "budgets_reached": namespaces,
            "policies_trained": trained,
            "policies_planned": 80,
            "cells": cells,
            "cells_planned": 9600,
            "usable": False,
            "reason": ("Stopped before any comparison was possible. The single budget reached is "
                       "asymmetric across arms and no evaluation cells were produced, so no "
                       "paired comparison exists at any budget."),
        }

    # ---------------------------------------------------------------- provenance

    def provenance(self) -> dict[str, Any]:
        audit = None
        try:
            audit = self.reproducibility()
        except ResearchArtefactMissing:
            pass
        return {
            "commit": (audit or {}).get("git_commit", "unavailable"),
            "audit_counts": (audit or {}).get("counts", {}),
            "cells": sum(1 for _ in (RESULTS / "runs").rglob("metrics.json")),
            "policies": sum(1 for _ in (MODELS / "policies" / "w_balanced").rglob("policy.zip"))
            if (MODELS / "policies" / "w_balanced").is_dir() else 0,
            "protected_unchanged": not assert_read_only(),
        }


#: One repository per process. Cheap to construct, but the parquet cache lives on it.
@functools.lru_cache(maxsize=1)
def repository() -> ResearchResultsRepository:
    return ResearchResultsRepository()
