"""Manuscript claim traceability (protocol section 13's definition of done).

Assigns every manuscript claim one of **SUPPORTED / PARTIALLY SUPPORTED / NOT YET TESTED /
REFUTED**, each with a pointer to the stored result that justifies it. The verdicts are *derived*
from `results/statistical_analysis.json` rather than asserted, so a claim cannot quietly stay
"supported" after the evidence for it changes.

A claim with no stored result is **NOT YET TESTED**. It is never "supported by design".

    python scripts/traceability.py

Writes ``docs/paper_traceability.md``.
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import json
from datetime import datetime, timezone
from typing import Any

from src.common.logging import get_logger
from src.common.paths import REPO_ROOT, RESULTS_DIR

logger = get_logger(__name__)

#: Every hypothesis in docs/research_audit.md, with the experiment that tests it and the
#: comparison that decides it. Claims not decided by a paired test carry `comparison=None` and
#: are resolved from a named artefact instead.
CLAIMS: tuple[dict[str, Any], ...] = (
    {
        "id": "H1", "family": "confirmatory",
        "claim": "A calibrated demand-forecast uncertainty channel reduces unmet demand under "
                 "surge, relative to an otherwise identical point-forecast policy.",
        "experiment": "EXP-UQ-001", "comparison": ("ppo_uq", "ppo_point"), "scenario": "surge",
    },
    {
        "id": "H1c", "family": "confirmatory",
        "claim": "The benefit of the uncertainty channel exceeds that of a dimensionally "
                 "identical but uninformative channel. If this fails, the mechanism claim fails "
                 "regardless of H1.",
        "experiment": "EXP-UQ-002", "comparison": ("ppo_uq", "ppo_placebo"), "scenario": "surge",
    },
    {
        "id": "H2", "family": "exploratory",
        "claim": "Uncertainty-aware PPO is at least as good as point-forecast PPO across all four "
                 "scenarios.",
        "experiment": "EXP-UQ-003", "comparison": ("ppo_uq", "ppo_point"), "scenario": "all",
    },
    {
        "id": "H1b", "family": "exploratory",
        "claim": "The uncertainty-aware policy's dispatch rate increases with forecast dispersion "
                 "after conditioning on the forecast mean; the point policy's does not.",
        "experiment": "EXP-UQ-004", "comparison": None, "scenario": "all",
        "artifact": "results/mechanism_analysis.json",
    },
    {
        "id": "H3", "family": "exploratory",
        "claim": "A multi-objective reward yields better joint outcomes than a waiting-time-only "
                 "reward, at the cost of waiting time alone.",
        "experiment": "EXP-MO-001", "comparison": ("ppo_uq", "ppo_uq_single_obj"),
        "scenario": "all",
    },
    {
        "id": "H4", "family": "exploratory",
        "claim": "Adaptive RL outperforms a fixed timetable.",
        "experiment": "EXP-BASE-001", "comparison": ("ppo_uq", "fixed_timetable"),
        "scenario": "all",
    },
    {
        "id": "H5", "family": "exploratory",
        "claim": "Adaptive RL outperforms a tuned threshold heuristic.",
        "experiment": "EXP-BASE-001", "comparison": ("ppo_uq", "heuristic_threshold"),
        "scenario": "all",
    },
    {
        "id": "H6", "family": "exploratory",
        "claim": "Adaptive RL outperforms a static MILP schedule.",
        "experiment": "EXP-BASE-001", "comparison": ("ppo_uq", "milp_static"), "scenario": "all",
    },
    {
        "id": "H6b", "family": "exploratory",
        "claim": "The advantage of adaptive RL shrinks or disappears against a rolling-horizon "
                 "MILP re-solved on exactly the same forecasts.",
        "experiment": "EXP-BASE-005", "comparison": ("ppo_uq", "milp_rolling"), "scenario": "all",
    },
    {
        "id": "H7", "family": "exploratory",
        "claim": "Predicted train-availability information improves outcomes under disruption.",
        "experiment": "EXP-AVAIL-001", "comparison": ("ppo_uq", "ppo_uq_noavail"),
        "scenario": "disruption",
    },
    {
        "id": "H8-H10", "family": "exploratory",
        "claim": "The relative advantage of adaptive policies grows under surge and failure, and "
                 "degradation is sub-additive under combined events.",
        "experiment": "EXP-ROBUST-001/002/003", "comparison": None, "scenario": "all",
        "artifact": "results/statistical_analysis.json (degradation section)",
    },
    {
        "id": "H11", "family": "descriptive",
        "claim": "SHAP global attributions concentrate on decision-relevant features.",
        "experiment": "EXP-XAI-001", "comparison": None, "scenario": "all",
        "artifact": "results/shap_analysis.json",
    },
    {
        "id": "C-SIM", "family": "methodological",
        "claim": "The simulation environment reproduces the modelled line's published operating "
                 "statistics.",
        "experiment": "M5 validation", "comparison": None,
        "artifact": "results/simulator_validation.json",
    },
    {
        "id": "C-FCST", "family": "methodological",
        "claim": "The forecaster's uncertainty is calibrated well enough for the RL arms to "
                 "consume it.",
        "experiment": "M4 gate", "comparison": None,
        "artifact": "results/forecasting_metrics.json (gate)",
    },
)


def _verdict_from_paired(records: list[dict], comparison, scenario: str) -> tuple[str, str]:
    method_a, method_b = comparison
    matches = [
        r for r in records
        if r.get("method_a") == method_a and r.get("method_b") == method_b
        and (scenario == "all" or r.get("scenario") == scenario)
        and r.get("metric") == "left_behind_fraction"
    ]
    if not matches:
        return "NOT YET TESTED", "no stored comparison"

    key = "p_adjusted" if "p_adjusted" in matches[0] else "p_value"
    significant = [r for r in matches if r.get(key, 1.0) < (
        0.05 if r.get("family") == "confirmatory" else 0.10)]
    favourable = [r for r in significant if r.get("direction") == f"{method_a} better"]
    against = [r for r in significant if r.get("direction") == f"{method_b} better"]

    detail = (f"{len(matches)} comparison(s); {len(favourable)} favour `{method_a}`, "
              f"{len(against)} favour `{method_b}`, "
              f"{len(matches) - len(significant)} not significant")

    if favourable and not against:
        return ("SUPPORTED" if len(favourable) == len(matches) else "PARTIALLY SUPPORTED"), detail
    if against and not favourable:
        return "REFUTED", detail
    if favourable and against:
        return "PARTIALLY SUPPORTED", detail + " (direction differs by scenario)"

    equivalent = [r for r in matches if r.get("equivalence", {}).get("equivalent_at_alpha")]
    if equivalent:
        return "REFUTED", detail + "; TOST establishes equivalence within the SESOI"
    return "NOT YET TESTED", detail + "; inconclusive rather than null"


def main() -> int:
    analysis_path = RESULTS_DIR / "statistical_analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8")) if analysis_path.exists() else {}
    records = list(analysis.get("confirmatory", [])) + list(analysis.get("exploratory", []))
    records = [r for r in records if r.get("status") != "NOT RUN"]

    rows: list[dict[str, Any]] = []
    for claim in CLAIMS:
        if claim.get("comparison"):
            verdict, detail = _verdict_from_paired(
                records, claim["comparison"], claim["scenario"])
            evidence = "results/statistical_analysis.json"
        else:
            artifact = claim.get("artifact", "")
            path = REPO_ROOT / artifact.split(" ")[0] if artifact else None
            if path and path.exists():
                verdict, detail = "SEE ARTEFACT", f"resolved from `{artifact}`"
            else:
                verdict, detail = "NOT YET TESTED", f"`{artifact}` not present"
            evidence = artifact
        rows.append({**claim, "verdict": verdict, "detail": detail, "evidence": evidence})

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1

    lines = [
        "# Manuscript claim traceability",
        "",
        f"Generated by `scripts/traceability.py` on "
        f"{datetime.now(timezone.utc).date().isoformat()} from stored results. Verdicts are "
        "derived, not asserted: a claim whose evidence changes changes verdict on the next run, "
        "and a claim with no stored result is **NOT YET TESTED** rather than assumed.",
        "",
        "Primary endpoint for every paired verdict is `left_behind_fraction`. Confirmatory claims "
        "use Holm-adjusted p < 0.05; exploratory claims use Benjamini-Hochberg adjusted p < 0.10 "
        "and are labelled exploratory wherever they appear.",
        "",
        "| " + " | ".join(f"{k}: {v}" for k, v in sorted(counts.items())) + " |",
        "|" + "---|" * len(counts),
        "",
        "| id | family | verdict | claim | experiment | evidence | detail |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| **{row['id']}** | {row['family']} | **{row['verdict']}** | {row['claim']} | "
            f"{row['experiment']} | `{row['evidence']}` | {row['detail']} |"
        )

    lines += [
        "",
        "## Claims deliberately not made",
        "",
        "- **Real-world deployment performance.** Everything here is simulation; the environment "
        "is a calibrated simulator, not an asset-coupled digital twin (MC-04).",
        "- **Energy or cost in physical units.** Assumptions A15 and A16 are unresolved, so both "
        "are reported in equivalent units and embargoed from headline claims.",
        "- **Train-health prediction accuracy.** Decision D-04 demoted the module to an "
        "information channel; no classifier metric on synthetic labels is reported as a result.",
        "- **Causal interpretation of SHAP.** Attribution describes the model's response on this "
        "input distribution, not the effect of intervening in the world.",
        "",
    ]

    (REPO_ROOT / "docs" / "paper_traceability.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote docs/paper_traceability.md — {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
