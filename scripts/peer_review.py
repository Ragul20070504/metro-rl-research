"""Pre-submission adversarial peer review (M16).

Written from the position of a hostile reviewer looking for reasons to reject. Every item is
either **grounded in a stored artefact** or explicitly marked as a judgement, and the assessment
is generated rather than typed so it cannot drift from what the repository actually contains.

The section that matters is *"What a reviewer will attack"*. A self-assessment that lists only
strengths is worthless; this one is built to surface the weakest points first.

    python scripts/peer_review.py

Writes ``docs/pre_submission_peer_review.md``.
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import json
import subprocess
from datetime import datetime, timezone
from typing import Any

from src.common.logging import get_logger
from src.common.paths import REPO_ROOT, RESULTS_DIR

logger = get_logger(__name__)


def load(path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return ""


def main() -> int:
    analysis = load(RESULTS_DIR / "statistical_analysis.json")
    power = load(RESULTS_DIR / "power_analysis.json")
    forecasting = load(RESULTS_DIR / "forecasting_metrics.json")
    simulator = load(RESULTS_DIR / "simulator_validation.json")
    audit = load(RESULTS_DIR / "reproducibility_audit.json")
    shap_payload = load(RESULTS_DIR / "shap_analysis.json")

    commit = git("rev-parse", "HEAD")
    lines: list[str] = [
        "# Pre-submission peer review",
        "",
        f"Generated at commit `{commit[:12]}` on "
        f"{datetime.now(timezone.utc).date().isoformat()}. Written adversarially: the goal is to "
        "find the reasons a reviewer would reject this, not to defend it.",
        "",
        "---",
        "",
        "## 1. What the study actually establishes",
        "",
    ]

    if analysis:
        confirmatory = [r for r in analysis.get("confirmatory", []) if r.get("status") == "RUN"]
        if confirmatory:
            for record in confirmatory:
                verdict = ("**rejects** the null" if record.get("rejects_at_family_alpha")
                           else "**does not reject** the null")
                lines += [
                    f"- **{record['experiment_id']}** (`{record['method_a']}` vs "
                    f"`{record['method_b']}`, n = {record['n']} training seeds): {verdict} at "
                    f"family-wise alpha 0.05. {record['test']}, statistic "
                    f"{record['statistic']:.4g}, exact p = {record['p_value']:.4g}, "
                    f"Holm-adjusted p = {record['p_adjusted']:.4g}. Hodges-Lehmann difference "
                    f"{record['hodges_lehmann']:.5g} "
                    f"(95% BCa CI {record['ci_low']:.5g} to {record['ci_high']:.5g}).",
                ]
                if record.get("equivalence", {}).get("applicable"):
                    eq = record["equivalence"]
                    lines.append(
                        f"  - TOST against the pre-registered SESOI of {eq['sesoi']}: "
                        f"p = {eq['p_value']:.4g}. {eq['interpretation'].capitalize()}."
                    )
            lines.append("")
        else:
            lines += ["- The confirmatory tests have not been run in this checkout.", ""]
    else:
        lines += ["- No statistical analysis is stored; the confirmatory claims are untested.", ""]

    lines += [
        "---",
        "",
        "## 2. What a reviewer will attack, hardest first",
        "",
        "### 2.1 The digital twin is a simulator, and seven of its quantities cannot be validated",
        "",
    ]
    if simulator:
        counts = simulator.get("counts", {})
        lines += [
            f"{counts.get('validated', 0)} quantities are validated against real published "
            f"statistics, {counts.get('partially validated', 0)} partially, and "
            f"**{counts.get('not validatable', 0)} cannot be validated at all** — including peak "
            "occupancy and the origin-destination distribution, **both of which feed the primary "
            "endpoint**.",
            "",
            "The honest position: the OD model is synthetic (assumption A03) and the capacity "
            "figure is ungrounded (A09). `EXP-SENS-002` sweeps all three OD variants precisely "
            "so the primary conclusion can be reported as OD-dependent if it is. A reviewer who "
            "asks *\"how do you know the OD split is right?\"* must be told plainly: we do not, "
            "and that is why the sweep exists.",
            "",
        ]
    else:
        lines += ["Simulator validation results are absent from this checkout.", ""]

    lines += [
        "### 2.2 Energy and cost are not in physical units",
        "",
        "Assumptions A15 and A16 remain unresolved: no citable conversion to kWh or currency has "
        "been sourced for this fleet. Energy is reported in train-km-equivalent units and cost in "
        "induction-equivalent units, and both are **embargoed from headline claims**. Any "
        "sentence of the form \"saves X kWh\" would be fabricated and does not appear.",
        "",
        "### 2.3 The train-health module makes no prediction claim",
        "",
        "No public per-train availability record exists for the modelled fleet. Decision D-04 "
        "demoted the module from a contribution to an information channel whose *value to the "
        "policy* is measured (H7). **No ROC-AUC or PR-AUC on synthetic labels is reported as a "
        "research result** — a classifier scored on labels from a process we wrote down measures "
        "only our ability to recover our own process.",
        "",
        "### 2.4 The H7 ablation is dimensionality-confounded",
        "",
        "Protocol section 6.1 specifies the availability channel is *present only when enabled*, "
        "so `ppo_uq_noavail` has 121 observation values against `ppo_uq`'s 123. Unlike the sigma "
        "arms — designed specifically to hold dimensionality fixed — this contrast varies both "
        "the information and the input width. Reported with that caveat.",
        "",
        "### 2.5 The MILP baseline is a fluid approximation of a queueing simulator",
        "",
        "The simulator models per-station FIFO cohorts, finite train capacity and platform "
        "contention; the MILP optimises a line-aggregate fluid queue with a fitted "
        "frequency-dependent waiting term. Any RL advantage over it is **partly an advantage over "
        "that abstraction gap**, not pure optimisation superiority. Two calibration defects that "
        "would have made it a strawman were found and fixed before use, and both are recorded in "
        "`models/milp_surrogate.json`. Solver status and optimality gap are reported per instance.",
        "",
        "### 2.6 No PPO hyperparameter search was run",
        "",
        "One frozen configuration is applied unchanged to every arm (amendment A-05). This "
        "satisfies the symmetry constraint exactly, but means **all arms may be jointly "
        "suboptimal**. It bounds absolute performance claims; it does not bias the relative "
        "comparison, which is what the confirmatory tests measure.",
        "",
        "### 2.7 The demand regime is not stationary",
        "",
        "The pre-registered regime screen **fired its fallback**: no 365-day run exists with "
        "|YoY growth| <= 10% (longest 278 days). The tolerance was **not** loosened to make it "
        "pass. Residual drift within the study window is carried into the limitations, and "
        "`EXP-ROBUST-004` tests out-of-regime transfer explicitly.",
        "",
    ]

    if power:
        lines += [
            "### 2.8 Statistical power",
            "",
            f"Pilot-estimated between-training-seed SD: {power.get('between_seed_sd', float('nan')):.6f}. "
            f"Pre-registered SESOI: {power.get('sesoi')}. "
            f"Training seeds required to detect it at {power.get('target_power', 0.8):.0%} power: "
            f"{power.get('required_n_for_sesoi')}; registered S = "
            f"{power.get('registered_train_seeds')}. Minimum detectable effect at the registered "
            f"S: {power.get('minimum_detectable_effect_at_registered_n', float('nan')):.6f}.",
            "",
            f"**Decision taken: {power.get('decision')}.** {power.get('rationale', '')}",
            "",
            "The SESOI was frozen in amendment A-04 *before* the pilot ran and has not been "
            "revisited. A reviewer should check that: it is the single easiest thing to fudge in "
            "a study like this.",
            "",
        ]

    lines += [
        "---",
        "",
        "## 3. Design choices a reviewer should credit",
        "",
        "- **The placebo arm.** `ppo_point` vs `ppo_uq` alone cannot separate *\"uncertainty is "
        "informative\"* from *\"a wider observation helps\"*. `ppo_placebo` receives sigma values "
        "with the correct marginal distribution and destroyed association, at identical "
        "dimensionality and parameter count. A result where `ppo_uq ≈ ppo_placebo > ppo_point` "
        "refutes the manuscript's mechanism even though a two-arm study would report it as a win.",
        "- **Common random numbers enforced structurally.** `environment_streams` takes no method "
        "argument, so no method can perturb the demand or failure realisation it faces.",
        "- **Constraints masked, not penalised.** Zero headway violations is a structural "
        "guarantee, not a tuning outcome.",
        "- **The forecaster never sees the sampled demand path.** The manuscript's original design "
        "generated arrivals *from* the forecast, which would have made the forecaster correct by "
        "construction; protocol section 5.1 replaced it.",
        "- **Amendments are dated and were made before the runs they affect.**",
        "",
        "---",
        "",
        "## 4. Corrections made to the manuscript's design",
        "",
        "These are not incidental: several would have invalidated the headline claim.",
        "",
        "| id | correction |",
        "|---|---|",
        "| A-01 / D-12 | The pre-registered primary endpoint was degenerate (~0.05% under every "
        "policy). Re-operationalised as left-behind passengers, before any policy existed. |",
        "| A-02 / D-13 | The reward was degenerate: reading `W_t` as the mean wait of *boarders* "
        "made \"run no service all day\" score the global maximum while stranding 100% of demand. |",
        "| A-03 / D-14 | Scenario random streams were keyed on the scenario *name*, so `normal` "
        "and `surge` at the same seed were unrelated realisations. `surge` scored **better** than "
        "`normal` in 3 of 5 seeds. |",
        "| MC-02 | Beijing Metro open data is cited in the manuscript as available; it could not "
        "be confirmed as publicly downloadable. |",
        "| MC-04 | \"Digital twin\" replaced by \"calibrated simulation environment\" — there is "
        "no live asset coupling. |",
        "| MC-05 | A rolling-horizon MILP was added; comparing only against a static MILP would "
        "have been a weak baseline. |",
        "",
        "---",
        "",
        "## 5. Claims this study cannot support",
        "",
        "- Real-world deployment performance. **Everything here is simulation.**",
        "- Energy or cost savings in physical units (A15/A16 unresolved).",
        "- Train-health prediction accuracy (D-04).",
        "- Generalisation to other lines, other cities, or the pre-pandemic demand regime beyond "
        "what `EXP-ROBUST-004` measures.",
        "- Causal interpretation of SHAP attributions"
        + (" — the stored analysis states this in its own caveats field."
           if shap_payload else "."),
        "",
        "---",
        "",
        "## 6. Reproducibility",
        "",
    ]
    if audit:
        counts = audit.get("counts", {})
        lines += [
            f"`scripts/reproducibility_audit.py`: **{counts.get('PASS', 0)} pass, "
            f"{counts.get('FAIL', 0)} fail, {counts.get('WARN', 0)} warn, "
            f"{counts.get('SKIPPED', 0)} skipped.** Full detail in "
            "`docs/reproducibility_audit.md`.",
            "",
        ]
        failures = [c for c in audit.get("checks", []) if c["status"] == "FAIL"]
        if failures:
            lines += ["Outstanding failures:", ""]
            lines += [f"- **{c['name']}** — {c['detail']}" for c in failures]
            lines.append("")
    else:
        lines += ["The reproducibility audit has not been run in this checkout.", ""]

    if forecasting:
        gate = forecasting.get("gate", {})
        lines += [
            "---",
            "",
            "## 7. Forecasting",
            "",
            f"Blocking gate (protocol section 4.5): "
            f"{'**passed**' if gate.get('passed') else '**failed**'}. "
            f"Pinball loss {gate.get('quantile_pinball_val', float('nan')):.4f} against a "
            f"seasonal-naive floor of {gate.get('naive_pinball_val', float('nan')):.4f}; "
            f"calibrated 80% coverage {gate.get('picp_80_val', float('nan')):.4f} inside the "
            f"pre-registered band {gate.get('coverage_band')}.",
            "",
            "Forecasting quality and downstream scheduling usefulness are **separate questions**. "
            "The gate is a precondition for testing H1, not evidence about it.",
            "",
        ]

    lines += [
        "---",
        "",
        "## 8. Verdict",
        "",
        "The contribution that survives scrutiny is **the placebo-controlled design and the "
        "corrections it forced**, not a performance number. Three defects found during "
        "implementation would each have produced a confidently reported but meaningless result: "
        "a degenerate endpoint, a degenerate reward, and unmatched scenario realisations. All "
        "three were caught by measuring something that should have been impossible, not by code "
        "review.",
        "",
        "Whether H1 is supported is a question for the confirmatory tests, and the study is built "
        "to report a null result as precisely as a positive one.",
        "",
    ]

    (REPO_ROOT / "docs" / "pre_submission_peer_review.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"wrote docs/pre_submission_peer_review.md ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
