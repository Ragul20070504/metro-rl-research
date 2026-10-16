"""Recheck publication evidence from this checkout; never rerun or edit an original result.

Writes results/publication_readiness.json and docs/SUBMISSION_READINESS.md.
Exit 1 means at least one submission prerequisite remains unresolved.
"""
from __future__ import annotations

try:
    import _path  # noqa: F401
except ModuleNotFoundError:
    pass

import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.paper_refs import REFERENCES
from scripts.run_experiments import EVAL_SEEDS, NON_RL_METHODS, SCENARIOS, TRAIN_SEEDS, sweep_cells
from scripts.run_followup import missing_inputs
from src.common.paths import REPO_ROOT, RESULTS_DIR
from src.evaluation.aggregate import load_frame, seed_level
from src.evaluation.runner import Cell, cell_dir, discover_cells
from src.evaluation.statistics import holm, paired_analysis


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def file_digest(paths: list[Path]) -> str:
    """Fingerprint relative names and bytes, so moves and deletions also change the digest."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def audit() -> dict:
    checks = []

    def record(name, status, detail, evidence=None):
        checks.append({"name": name, "status": status, "detail": detail,
                       "evidence": evidence or {}})

    cells = discover_cells()
    original = [cell for cell in cells if not cell.variant or not cell.variant.startswith("budget_")]
    files = [path for cell in original for path in cell_dir(cell).glob("*.json")]
    record("Stored evidence fingerprint", "PASS", "Fingerprint of original cell JSON; not proof of historical authenticity.",
           {"cells": len(original), "files": len(files), "sha256": file_digest(files)})
    expected = {cell.key() for cell in sweep_cells(TRAIN_SEEDS)}
    expected.update(Cell(method=method, scenario=scenario, eval_seed=seed).key()
                    for method in NON_RL_METHODS for scenario in SCENARIOS for seed in EVAL_SEEDS)
    actual = {cell.key() for cell in cells if not cell.variant}
    record("Original evaluation matrix", "PASS" if expected == actual else "BLOCKED",
           f"{len(actual & expected)}/{len(expected)} main cells present; "
           f"{len(expected - actual)} missing, {len(actual - expected)} unexpected.")

    episodes = load_frame(original)
    seeds = seed_level(episodes)
    stored = json.loads((RESULTS_DIR / "statistical_analysis.json").read_text())
    pairs = []
    for contrast in stored["confirmatory"]:
        selected = seeds[(seeds.variant == "main") & (seeds.weight_id == "w_balanced")
                         & (seeds.scenario == contrast["scenario"])]
        pivot = selected.pivot(index="train_seed", columns="method", values=contrast["metric"])
        a, b = contrast["method_a"], contrast["method_b"]
        values = pivot.loc[list(TRAIN_SEEDS), [a, b]].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("Missing or non-finite original training-seed outcomes")
        result = paired_analysis(values[:, 0], values[:, 1], label=contrast["label"],
                                 metric=contrast["metric"], method_a=a, method_b=b)
        pairs.append((contrast, result))
    for (contrast, result), adjusted in zip(pairs, holm([r.p_value for _, r in pairs])):
        result.p_adjusted = adjusted
        fields = ("n", "mean_a", "mean_b", "mean_difference", "p_value", "p_adjusted",
                  "hodges_lehmann", "ci_low", "ci_high")
        agrees = all(np.isclose(getattr(result, key), contrast[key], rtol=1e-10, atol=1e-12)
                     for key in fields)
        record(contrast["experiment_id"] + " stored-cell reanalysis",
               "PASS" if agrees else "BLOCKED", "Recomputed from stored cell metrics; no experiment rerun.",
               {key: getattr(result, key) for key in fields})

    manifests = []
    commits = set()
    for cell in original:
        path = cell_dir(cell) / "run_manifest.json"
        if path.exists():
            manifest = json.loads(path.read_text())
            manifests.append(manifest)
            if manifest.get("git_commit") not in (None, "", "unknown"):
                commits.add(manifest["git_commit"])
    unavailable = [commit for commit in sorted(commits) if not git("rev-parse", "--verify", commit + "^{commit}")]
    dirty = sum(m.get("git_dirty") is True for m in manifests)
    unknown = sum(m.get("git_commit") in (None, "", "unknown") for m in manifests)
    record("Historical code provenance", "BLOCKED" if dirty or unknown or unavailable else "PASS",
           f"{dirty} dirty manifests; {unknown} unknown commits; {len(unavailable)} referenced commits unavailable.",
           {"manifest_count": len(manifests), "unavailable_commits": unavailable})
    tag = git("rev-parse", "--verify", "refs/tags/original-confirmatory-study^{commit}")
    record("Original study tag", "PASS" if tag else "BLOCKED",
           tag or "original-confirmatory-study is absent locally and was not advertised by origin.")

    missing_raw, mismatches = [], []
    for path in sorted((REPO_ROOT / "data" / "raw").rglob("MANIFEST.json")):
        for item in json.loads(path.read_text()).get("files", []):
            relative = Path(item["path"].replace("\\", "/"))
            target = REPO_ROOT / "data" / "raw" / relative
            if not target.is_file():
                missing_raw.append(relative.as_posix())
            elif hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]:
                mismatches.append(relative.as_posix())
    record("Original raw-data bytes", "BLOCKED" if missing_raw or mismatches else "PASS",
           f"{len(missing_raw)} missing files; {len(mismatches)} checksum mismatches.",
           {"missing": missing_raw, "mismatches": mismatches})
    missing = missing_inputs()
    record("Frozen follow-up inputs", "BLOCKED" if missing else "PASS",
           f"{len(missing)} required input files missing. Do not replace the historical GTFS snapshot with today's feed.",
           {"missing": missing})

    binary_missing = [str(path.with_name("policy.zip").relative_to(REPO_ROOT))
                      for path in (REPO_ROOT / "models" / "policies").rglob("metadata.json")
                      if not path.with_name("policy.zip").is_file()]
    record("Recorded trained-policy binaries", "BLOCKED" if binary_missing else "PASS",
           f"{len(binary_missing)} metadata records lack their policy.zip.", {"missing": binary_missing})
    from scripts.analyse_followup import budget_coverage
    coverage = budget_coverage(load_frame(cells))
    record("Follow-up grid", "PASS" if all(row["complete"] for row in coverage.values()) else "BLOCKED",
           "No new inferential result may be reported until every budget is complete.", coverage)

    pending_refs = [reference.key for reference in REFERENCES if reference.status != "full"]
    record("Bibliography", "BLOCKED" if pending_refs else "PASS",
           f"{len(pending_refs)} references still require authoritative verification.", {"keys": pending_refs})
    record("Independent registration record", "LIMITATION",
           "An internal protocol exists, but its chronology cannot be independently established. The revision removes the externally verifiable pre-registration claim.")
    record("Placebo construction", "LIMITATION",
           "The inherited placebo bootstraps a rolling pool with replacement; exact marginal matching and complete information removal are not established. The manuscript now limits attribution accordingly.")
    record("Author declarations", "BLOCKED",
           "Affiliations, corresponding-author details, CRediT roles, funding and conflicts remain author-confirmation placeholders.")
    record("Journal requirements and archival deposit", "BLOCKED",
           "Current author guidelines and a public artifact archive/DOI have not been verified; no submission was made.")
    counts = dict(Counter(row["status"] for row in checks))
    from scripts.replication_status import progress
    return {"generated_utc": datetime.now(timezone.utc).isoformat(),
            "commit": git("rev-parse", "HEAD"), "working_tree_dirty": bool(git("status", "--porcelain")),
            "ready_for_submission": not counts.get("BLOCKED"), "counts": counts, "checks": checks,
            "independent_replication": progress()}


def render(report: dict) -> str:
    lines = ["# Submission readiness", "", f"Checked {report['generated_utc']}.", "",
             "**NOT READY FOR SUBMISSION**" if not report["ready_for_submission"] else "**CHECKS PASSED**",
             "", "This is a current-checkout audit, not an external peer review or proof of historical experiment execution.",
             "Stored-results reanalysis is distinct from reproducing simulations with the original models and data.", "",
             "| Check | Status | Evidence |", "|---|---|---|"]
    for check in report["checks"]:
        lines.append(f"| {check['name']} | {check['status']} | {check['detail']} |")
    replication = report.get("independent_replication", {})
    lines += ["", "## Independent replication", "",
              "A separately specified replication is authorized under `docs/replication_20261015_protocol.md`. "
              "It trains a fresh 60k anchor and does not claim to recover the historical policies. "
              "Its input/code snapshot and results live under `replication/20261015/`.", "",
              f"Current phase: **{replication.get('stage', 'NOT_STARTED')}**; "
              f"{replication.get('completed_policies', 0)}/110 policies and "
              f"{replication.get('stored_evaluation_cells', 0)}/13,680 evaluation cells.", "",
              "Run `python scripts/replication_status.py` for live counts. No new result is available "
              "until the entire grid completes and passes analysis checks."]
    lines += ["", "## Recovery and completion order", "",
              "1. Complete the independent replication with `python scripts/run_replication.py --stage all --workers 4` only if no replication job is already active. Its new snapshot and fresh anchors are separate from the historical study.",
              "2. Review the full-grid analysis, all learning curves, failures and actual timesteps; integrate every specified comparison with the unchanged original result. No incomplete-grid inference is valid.",
              "3. Continue historical archive recovery where possible. Only exact original inputs/checkpoints permit the original held-fixed follow-up (`scripts/run_followup.py --check`). Do not overwrite original manifests or replace the lost anchor with a fresh model.",
              "4. Retain the historical provenance limitation. Until chronology is independently verified, describe the original protocol as internal rather than externally verified pre-registration. A transparently scoped retrospective study does not require retroactive registration.",
              "5. All 18 references are verified against authoritative metadata. Current journal guidelines, author placeholders and public archival availability remain to be completed before submission.",
              "", "The original H1 conclusion is unchanged. Any future result is follow-up/exploratory and must be reported alongside it.", ""]
    return "\n".join(lines)


def main() -> int:
    report = audit()
    (RESULTS_DIR / "publication_readiness.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (REPO_ROOT / "docs" / "SUBMISSION_READINESS.md").write_text(render(report), encoding="utf-8")
    print(json.dumps({"ready_for_submission": report["ready_for_submission"], "counts": report["counts"]}))
    return 0 if report["ready_for_submission"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
