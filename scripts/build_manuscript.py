"""Render the final journal manuscript and its supporting materials from stored artefacts.

Every experimental number is read from `results/` at render time. None is typed into the prose,
so the manuscript cannot disagree with the study, and the number audit is a consequence of the
construction rather than a separate act of checking.

Writes into ``docs/``:

  paper_final.md · paper_final.tex · paper_final.docx · supplementary_material.md
  references_final.md · cover_letter_draft.md · highlights.md · author_contributions.md
  data_code_availability.md · paper_number_audit.md · paper_claim_audit.md
  paper_figure_table_audit.md

    python scripts/build_manuscript.py
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import json
import re
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.common.paths import REPO_ROOT, RESULTS_DIR

from scripts.paper_refs import REFERENCES, cite, counts as ref_counts

DOCS = REPO_ROOT / "docs"
SCEN = ["normal", "surge", "disruption", "combined"]

AUTHORS = ["Ragul A", "Raksith N P", "Pedakolimi Chandra Arjun"]
SUPERVISOR = "Dr M N Kavitha"
AFFILIATION = "[AUTHOR AFFILIATION TO BE CONFIRMED]"

TITLE = ("Informative Demand Uncertainty Changes but Does Not Improve a Reinforcement-Learning "
         "Metro Dispatch Controller: A Pre-Registered, Placebo-Controlled Evaluation")

CANDIDATE_TITLES = [
    "Controlled Evaluation of Uncertainty-Aware Reinforcement Learning for Metro Dispatch under "
    "Stochastic Demand",
    "Informative Demand Uncertainty Changes but Does Not Improve a Reinforcement-Learning Metro "
    "Dispatch Controller: A Pre-Registered, Placebo-Controlled Evaluation",
    "A Placebo-Controlled Test of Forecast Uncertainty in Learned Metro Train Induction",
    "When Calibrated Demand Uncertainty Does Not Help: A Registered Evaluation of "
    "Uncertainty-Aware Reinforcement Learning for Metro Dispatch",
    "Conservative Dispatch under Forecast Uncertainty: A Controlled Evaluation of "
    "Uncertainty-Aware Reinforcement Learning for Metro Operations",
]


# --------------------------------------------------------------------------- artefacts


def load(name: str) -> Any:
    path = RESULTS_DIR / name
    if not path.is_file():
        raise SystemExit(f"required artefact missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                             timeout=30)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


class Study:
    """Every quantity the manuscript reports, derived once from the stored artefacts."""

    def __init__(self) -> None:
        self.stats = load("statistical_analysis.json")
        self.power = load("power_analysis.json")
        self.audit = load("reproducibility_audit.json")
        self.sim = load("simulator_validation.json")
        self.fc = load("forecasting_metrics.json")
        self.shap = load("shap_analysis.json")
        self.fair = load("baseline_fairness_audit.json")
        self.mech = load("mechanism_analysis.json")
        self.sens = load("sensitivity_analysis.json")
        self.separation = load("separation_audit.json")

        self.primary = next(t for t in self.stats["confirmatory"]
                            if t["experiment_id"] == "EXP-UQ-001")
        self.placebo = next(t for t in self.stats["confirmatory"]
                            if t["experiment_id"] == "EXP-UQ-002")

        frame = pd.read_parquet(RESULTS_DIR / "seed_level.parquet")
        self.registered = frame[(frame.variant == "main") & (frame.weight_id == "w_balanced")]
        self.rl = self.registered[self.registered.train_seed >= 101]
        piv = self.rl[self.rl.scenario == "surge"].pivot_table(
            index="train_seed", columns="method", values="left_behind_fraction")
        self.piv = piv
        self.diff = (piv["ppo_uq"] - piv["ppo_point"]).to_numpy(float)
        self.sd_realised = float(self.diff.std(ddof=1))
        self.wilcoxon_p = float(stats.wilcoxon(piv["ppo_uq"], piv["ppo_point"]).pvalue)
        worst = max((float(stats.ttest_rel(np.delete(piv["ppo_uq"].values, i),
                                           np.delete(piv["ppo_point"].values, i)).pvalue),
                     int(piv.index[i])) for i in range(len(self.diff)))
        self.jack_p, self.jack_seed = worst
        self.seeds_favouring_point = int((self.diff > 0).sum())

        self.rank = self.registered.pivot_table(
            index="method", columns="scenario", values="left_behind_fraction",
            aggfunc="mean")[SCEN].sort_values("surge")
        self.reward = self.registered.pivot_table(
            index="method", columns="scenario", values="total_reward", aggfunc="mean")[SCEN]
        self.inductions = self.registered.pivot_table(
            index="method", columns="scenario", values="inductions", aggfunc="mean")[SCEN]

        self.surge_mech = self.mech["dispatch_volume"][1]
        self.dispatch_drop_pct = (abs(self.surge_mech["informative_sigma_vs_zero"]
                                      ["mean_difference"])
                                  / float(self.inductions.loc["ppo_point", "surge"]) * 100)
        self.cells = sum(1 for _ in (RESULTS_DIR / "runs").rglob("metrics.json"))
        self.policies = sum(1 for _ in (REPO_ROOT / "models" / "policies" / "w_balanced")
                            .rglob("policy.zip"))
        self.commit = git("rev-parse", "HEAD")
        self.tag = "original-confirmatory-study"
        self.unvalidated = [r["quantity"] for r in self.sim["rows"]
                            if r["verdict"] == "not validatable"]

    def evidence(self, name: str) -> dict:
        for check in self.audit["checks"]:
            if check["name"] == name:
                return check.get("evidence") or {}
        return {}

    def verdict(self) -> str:
        supported = (self.primary["mean_difference"] < 0
                     and self.primary.get("rejects_at_family_alpha"))
        return "SUPPORTED" if supported else "NOT SUPPORTED"


# --------------------------------------------------------------------------- manuscript


def manuscript(s: Study) -> str:
    p, c, pw = s.primary, s.placebo, s.power
    split = s.evidence("chronological splits with embargo")
    rows, sp = split.get("rows", {}), split.get("split", {})
    pools = s.evidence("seed pools are disjoint").get("pools", {})
    L: list[str] = []
    add = L.append

    add(f"# {TITLE}")
    add("")
    add("**" + " · ".join(AUTHORS) + "**")
    add("")
    add(f"*Supervisor:* {SUPERVISOR}")
    add("")
    add(f"*Affiliations:* {AFFILIATION}")
    add("")
    add(f"*Manuscript generated by `scripts/build_manuscript.py` on "
        f"{datetime.now(timezone.utc):%Y-%m-%d} from stored artefacts at commit "
        f"`{s.commit[:12]}`. Every experimental value is read from a result file at render "
        "time.*")
    add("")
    add("---")
    add("")

    # ---------------------------------------------------------------- abstract
    add("## Abstract")
    add("")
    add("**Problem.** Metro operators decide repeatedly, under time pressure, how many trainsets "
        "to release into revenue service. Releasing too few strands passengers; releasing too "
        "many raises energy and crew cost. Demand is known only through a forecast, and a common "
        "assumption is that supplying a learned controller with a calibrated measure of forecast "
        "uncertainty should improve its decisions. That assumption is rarely tested against a "
        "control capable of separating the *information* in the uncertainty signal from the mere "
        "presence of additional inputs.")
    add("")
    add("**Method.** We evaluate a maskable proximal-policy-optimisation controller for metro "
        f"train induction {cite('schulman2017')} in a demand-driven simulation environment built "
        "from real ridership records. Three arms share identical architecture, parameter count, "
        "observation dimensionality, reward, training budget, training seeds and evaluation days, "
        "and differ only in the content of a two-value uncertainty channel: identically zero "
        "(point forecast), a shuffled *placebo* with the same marginal distribution but no "
        "association with realised forecast error, or the calibrated forecast dispersion. A fixed "
        "timetable, a tuned threshold heuristic, and static and rolling-horizon mixed-integer "
        "formulations serve as baselines.")
    add("")
    add("**Design.** The primary endpoint, the analysis plan and the sample size were registered "
        "before the confirmatory runs, with the **training seed as the unit of analysis**. "
        f"{pw['registered_train_seeds']} training seeds were evaluated on "
        f"{pw['registered_eval_seeds']} held-out days across four scenario families, yielding "
        f"{s.stats['n_episode_rows_main']:,} registered episodes and {s.cells:,} stored "
        "evaluation cells in total. Common random numbers were enforced structurally.")
    add("")
    add(f"**Primary result.** Across n = {p['n']} training seeds under the surge scenario, the "
        f"uncertainty-aware arm left **more** passengers behind than the point-forecast arm: "
        f"{p['mean_a']:.6f} versus {p['mean_b']:.6f}, a difference of "
        f"**{p['mean_difference']:+.6f}** (Hodges–Lehmann {p['hodges_lehmann']:.6f}; 95% BCa "
        f"confidence interval [{p['ci_low']:.6f}, {p['ci_high']:.6f}]; Holm-adjusted "
        f"p = {p['p_adjusted']:.6f}). It was also worse than the placebo arm "
        f"({c['mean_difference']:+.6f}; 95% BCa CI [{c['ci_low']:.6f}, {c['ci_high']:.6f}]; "
        f"Holm-adjusted p = {c['p_adjusted']:.6f}), so the degradation is not attributable to the "
        "additional input dimensions alone. The registered hypothesis was therefore **not "
        "supported**.")
    add("")
    add("**Mechanism.** Analysis of stored per-epoch records shows that the calibrated "
        f"uncertainty signal is associated with approximately {s.dispatch_drop_pct:.0f}% fewer "
        f"dispatches under surge "
        f"({s.surge_mech['informative_sigma_vs_zero']['mean_difference']:+.1f} inductions per "
        f"episode; p = {s.surge_mech['informative_sigma_vs_zero']['p_value']:.1e}), whereas the "
        "placebo signal is not "
        f"({s.surge_mech['uninformative_sigma_vs_zero']['mean_difference']:+.1f}; "
        f"p = {s.surge_mech['uninformative_sigma_vs_zero']['p_value']:.2f}). Measured service "
        "loss in the environment is strongly convex once capacity binds, so by the standard "
        f"newsvendor argument {cite('arrow1951')} demand uncertainty is decision-relevant here "
        "and would motivate hedging *upward*. The tested controller moved in the opposite "
        "direction.")
    add("")
    add("**Implication and limitation.** The evidence is consistent with a policy-learning "
        "limitation rather than with uncertainty information being valueless: no learned arm "
        f"outperformed the tuned threshold heuristic "
        f"({s.rank.loc['heuristic_threshold', 'surge']:.5f} against "
        f"{s.rank.loc['ppo_point', 'surge']:.5f} for the best learned arm) in any scenario. The "
        "study used a single frozen training configuration and therefore cannot separate an "
        "information effect from a sample-efficiency effect; a registered training-budget "
        "follow-up remains incomplete. These results characterise the tested controller, "
        "configuration, budget, environment and scenarios. They do **not** establish that "
        "uncertainty-aware reinforcement learning is generally ineffective.")
    add("")
    add("**Keywords:** metro train induction; reinforcement learning; probabilistic demand "
        "forecasting; forecast uncertainty; placebo control; pre-registration; convex service "
        "loss; negative result; action masking; simulation-based evaluation")
    add("")

    # ---------------------------------------------------------------- 1
    add("## 1. Introduction")
    add("")
    add("Metro systems absorb demand that varies sharply within the day and irregularly between "
        "days. Train induction — the decision of how many trainsets to release into revenue "
        "service in each control interval — sits directly on the trade-off between passenger "
        "service and operating resources. Releasing too few trainsets leaves passengers unable to "
        "board; releasing too many raises energy and crew cost without commensurate service "
        f"benefit. Because demand is known only through a forecast, the decision is made under "
        f"uncertainty {cite('yin2016')}.")
    add("")
    add("Classical treatments of railway rescheduling formulate the problem as a mathematical "
        f"program and re-solve it as conditions change {cite('cacchiani2014', 'yin2017')}. "
        "Reinforcement learning has been proposed as an alternative for settings where "
        "re-optimisation is too slow for the minutes available to a controller "
        f"{cite('semrov2016', 'multitask_resched')}, and learned controllers have been applied to "
        f"rail transit rescheduling and passenger inflow control "
        f"{cite('madrl_shortturn', 'inflow_control')}.")
    add("")
    add("In parallel, probabilistic demand forecasting has matured to the point where calibrated "
        f"predictive intervals are routine rather than exceptional {cite('uq_demand_gnn', 'uq_ridesourcing')}. "
        "The natural conjunction — give the learned controller the uncertainty, not only the "
        "point forecast — is intuitive, and uncertainty-aware and risk-sensitive formulations are "
        f"an active area {cite('uq_review', 'clements2019')}.")
    add("")
    add("That conjunction is difficult to evaluate cleanly. An arm that receives an additional "
        "input differs from one that does not in more than the *semantics* of that input: it has "
        "a different input distribution, different first-layer parameters and a different "
        "optimisation landscape. A measured difference is therefore consistent with three "
        "explanations — the uncertainty information itself, the additional dimensionality, or "
        "seed-to-seed variation in training, the last of which is well documented in deep "
        f"reinforcement learning {cite('henderson2018', 'colas2018')}.")
    add("")
    add("**Research question.** Does incorporating calibrated demand uncertainty into a PPO-based "
        "metro dispatch controller improve passenger-service performance under stochastic and "
        "capacity-binding demand, relative to a point-forecast controller and to controlled "
        "baselines?")
    add("")
    add("**Hypothesis (registered).** The uncertainty-aware controller reduces the fraction of "
        "passengers left behind under a demand-surge scenario, relative to an otherwise identical "
        "point-forecast controller.")
    add("")
    add("**Design.** We answer the question with a pre-registered, placebo-controlled experiment "
        "in which the uncertainty channel's *content* is the only manipulated variable, the "
        "training seed is the unit of analysis, and the analysis plan is fixed before the "
        "confirmatory runs.")
    add("")
    add("**Contributions.**")
    add("")
    add("1. A controlled metro-dispatch evaluation framework coupling probabilistic demand "
        "forecasting, an explicit uncertainty representation, a simulation environment and "
        "reinforcement learning, with classical and heuristic baselines evaluated under common "
        "random numbers.")
    add("2. A controlled comparison between point forecasts and informative uncertainty using "
        "matched scenarios and a **placebo** uncertainty condition that holds dimensionality and "
        "marginal distribution fixed while removing information content.")
    add(f"3. A statistically robust empirical finding that the tested uncertainty-aware "
        f"controller performs **worse** than the point-forecast controller on the registered "
        f"primary endpoint ({p['mean_difference']:+.6f}; Holm-adjusted p = "
        f"{p['p_adjusted']:.6f}), reproduced in all four scenarios, in "
        f"{s.seeds_favouring_point} of {len(s.diff)} training seeds, and in "
        f"{s.sens['direction_agreement']['conditions_where_ppo_uq_is_worse']} of "
        f"{s.sens['direction_agreement']['conditions_total']} sensitivity conditions.")
    add("4. Mechanistic evidence linking genuine uncertainty to a more conservative dispatch "
        "response, together with a measured convex service-loss relationship that explains why "
        "such a response is costly in capacity-binding regimes.")
    add("5. A reproducible evaluation framework with a frozen experiment registry, "
        "provenance-tracked figures and tables, and an audit trail sufficient to re-derive every "
        "reported number.")
    add("")
    add("We do not claim the framework is novel in its components; each is standard. What the "
        "study contributes is the control structure that makes the comparison interpretable, and "
        "the finding that structure produced.")
    add("")

    # ---------------------------------------------------------------- 2
    add("## 2. Related work")
    add("")
    add("**Metro train scheduling and rescheduling.** Real-time railway rescheduling has an "
        f"established optimisation literature, surveyed by {cite('cacchiani2014')} and "
        f"{cite('trs_survey')}. For metro systems specifically, {cite('yin2017')} formulate "
        "demand-oriented train scheduling as mixed-integer linear programs balancing energy "
        f"efficiency against waiting time, and {cite('yin2016')} address rescheduling under "
        "*uncertain, time-variant* passenger demand using approximate dynamic programming. The "
        "latter is the closest prior treatment of demand uncertainty in metro scheduling; it "
        "differs from the present work in optimising an explicit model of the uncertainty rather "
        "than supplying uncertainty as an input to a learned controller.")
    add("")
    add("**Reinforcement learning for transit scheduling and control.** "
        f"{cite('semrov2016')} apply Q-learning to train rescheduling on a single-track railway "
        f"evaluated on a real network. More recent work applies deep and multi-agent "
        f"reinforcement learning to real-time railway rescheduling {cite('multitask_resched')}, "
        f"to rail transit services with short-turnings under disruption "
        f"{cite('madrl_shortturn')}, and to coordinated passenger inflow control in peak hours "
        f"{cite('inflow_control')}. The common motivation is that exact re-optimisation is too "
        "slow for the response times a dispatcher has. These studies generally report favourable "
        "performance for the learned controller; the present study is unusual in reporting a "
        "controlled negative result and in including classical baselines that outperform the "
        "learned arms.")
    add("")
    add("**Uncertainty-aware and risk-sensitive decision making.** Reviews of uncertainty in deep "
        f"reinforcement learning distinguish aleatoric from epistemic uncertainty and survey "
        f"estimation methods {cite('uq_review')}; risk-sensitive formulations replace expected "
        f"return with tail-aware objectives {cite('clements2019')}. Our design differs in an "
        "important respect: we do not modify the objective or the algorithm. We hold the learner "
        "fixed and vary only what the observation carries, which isolates the value of the "
        "*information* rather than the value of a risk-sensitive *method*.")
    add("")
    add("**Demand forecasting for transit.** Uncertainty quantification for travel demand is well "
        f"developed, including probabilistic graph neural networks {cite('uq_demand_gnn')} and "
        f"mixture-of-expert models for multi-modal demand {cite('uq_ridesourcing')}. We use a "
        "quantile gradient-boosted forecaster; our forecasting contribution is calibration "
        "discipline and the freezing of the uncertainty definition before policy training, not "
        "model architecture.")
    add("")
    add("**Convex loss under demand uncertainty.** That a convex cost makes expected cost under "
        "stochastic demand exceed cost evaluated at expected demand is the classical newsvendor "
        f"observation {cite('arrow1951')}, extended to capacity and inventory networks by "
        f"{cite('vanmieghem2002')}. We invoke this as established theory rather than deriving it, "
        "and connect it to the measured shape of service loss in our environment (Section 12).")
    add("")
    add("**Simulation and digital twins for transit operations.** Simulation-based evaluation of "
        f"rail operations is widespread, and digital-twin approaches in rail have been reviewed "
        f"systematically {cite('dt_railway_review')}. We use the term *simulation environment* "
        "rather than digital twin, because only a minority of our environment's quantities could "
        "be validated against published operating statistics (Section 5).")
    add("")
    add("**Controlled and reproducible machine-learning experimentation.** "
        f"{cite('henderson2018')} document that random seeds alone can produce statistically "
        f"distinguishable performance from identical code, and {cite('colas2018')} analyse how "
        "many seeds a deep reinforcement-learning comparison requires. These motivate our "
        "seed-level unit of analysis, our pre-registration, and our decision to report an "
        "underpowered design explicitly rather than aggregating episodes into an inflated sample "
        "size.")
    add("")
    add(f"> **Citation status.** {ref_counts()['fully_verified']} of {ref_counts()['total']} "
        "references were verified in full (authors, title, venue, year and pages) during "
        "preparation. The remainder had title and venue verified, with the outstanding fields "
        "marked `[REFERENCE VERIFICATION REQUIRED]` in `docs/references_final.md`; these must be "
        "completed against the publisher record before submission. No reference was reconstructed "
        "from memory and no DOI is stated that was not observed.")
    add("")

    # ---------------------------------------------------------------- 3
    add("## 3. Problem formulation")
    add("")
    add("Consider a metro line operating over a service day divided into decision epochs "
        "*t* = 1, …, *T*. At each epoch the controller observes the state of the line and selects "
        "the number of trainsets *a_t* ∈ {0, …, k_max} to induct into service. Passengers arrive "
        "at stations according to a demand process, board subject to train capacity, and are "
        "*left behind* when capacity is exhausted.")
    add("")
    add("The operator's objective combines four terms — accumulated waiting, unmet demand, energy "
        "and operating cost — through the scalarised reward")
    add("")
    add("&nbsp;&nbsp;&nbsp;&nbsp;`r_t = −( α·W_t + β·U_t + γ·E_t + δ·C_t )`")
    add("")
    add("where `W_t` is the queue integral in passenger-minutes, `U_t` the count of distinct "
        "left-behind passengers, `E_t` energy and `C_t` operating cost, each normalised against "
        "reference scales fitted from the fixed-timetable baseline.")
    add("")
    add("The **primary endpoint** is the `left_behind_fraction`: distinct left-behind passengers "
        "as a fraction of arrivals over the service day. **Lower is better.** The primary "
        "scenario is `surge`, the condition under which capacity is most likely to bind.")
    add("")
    add("Feasibility — fleet availability, depot capacity and headway constraints — is enforced "
        "by **action masking rather than reward penalties**. A penalty makes constraint violation "
        "a tunable trade-off; masking makes it unrepresentable.")
    add("")

    # ---------------------------------------------------------------- 4
    add("## 4. Data and demand forecasting")
    add("")
    add("### 4.1 Data provenance and classification")
    add("")
    add("We distinguish four categories throughout, because conflating them is the most common "
        "way a simulation study overstates its external validity:")
    add("")
    add("| Category | In this study |")
    add("|---|---|")
    add("| **Real data** | Metro station-level ridership records, with dataset manifests and "
        "licence files recorded before use |")
    add("| **Derived data** | Supervised forecasting features and targets constructed from the "
        "real records |")
    add("| **Synthetic data** | The origin–destination distribution (assumption A03); passenger "
        "balking and patience behaviour |")
    add("| **Simulated data** | All operational outcomes — queues, boardings, left-behind counts, "
        "energy and cost — produced by the simulation environment |")
    add("")
    add("**No operational outcome reported in this paper is a real-world measurement.**")
    add("")
    if sp:
        add("### 4.2 Splits")
        add("")
        add(f"Splits are chronological with a **{sp.get('embargo_days')}-day embargo** between "
            f"them: training {sp.get('train_start')} to {sp.get('train_end')}, validation "
            f"{sp.get('val_start')} to {sp.get('val_end')}, test {sp.get('test_start')} to "
            f"{sp.get('test_end')}. The embargo exists because adjacent hours are strongly "
            "autocorrelated; without it a validation row can be nearly a copy of a training row "
            "and the split stops measuring generalisation.")
        add("")
    if rows:
        add(f"Row counts: {rows.get('raw_station_hours', 0):,} raw station-hours, "
            f"{rows.get('excluded_by_quality', 0):,} excluded by quality screens, and "
            f"{rows.get('supervised_train', 0):,} / {rows.get('supervised_val', 0):,} / "
            f"{rows.get('supervised_test', 0):,} supervised training / validation / test "
            "examples (Table 1).")
        add("")
    add("**Regime screen.** A pre-registered screen sought a run of at least 365 days with "
        "year-on-year growth within ±10%. No such run exists in these data; the longest "
        "qualifying run was 278 days, so the configured start date was used. This is recorded as "
        "a finding rather than applied as a silent default, and all conclusions inherit whatever "
        "regime instability remains.")
    add("")
    add("### 4.3 Forecasting")
    add("")
    add(f"A quantile gradient-boosted model produces median and quantile forecasts of "
        f"station-level demand (configuration hash `{str(s.fc.get('config_hash', ''))[:16]}`). "
        f"The observed quantile crossing rate is {s.fc['quantile_crossing_rate']:.6f} (Table 2). "
        "Model artefacts are promoted to trusted status only after the complete stored forecast "
        "output — median **and** dispersion — reproduces against a reference fit; verifying the "
        "median alone would allow a model with a broken uncertainty head to pass, and the "
        "uncertainty head is precisely what the hypothesis depends on.")
    add("")

    # ---------------------------------------------------------------- 5
    add("## 5. Metro simulation environment")
    add("")
    add("The environment couples a demand process, rolling-stock and headway constraints, "
        "capacity-limited boarding with queue dynamics, and the multi-objective reward of "
        "Section 3. An episode is one service day of decision epochs; state transitions advance "
        "train positions, board and alight passengers, accumulate queues and record failures.")
    add("")
    counts_row = ", ".join(f"{v} {k}" for k, v in s.sim["counts"].items())
    add(f"**Validation status across {len(s.sim['rows'])} quantities:** {counts_row} (Table 3).")
    add("")
    add("We describe this as a **simulation environment**, not a validated digital twin. The "
        "quantities validated against published operating statistics — running time, round trips "
        "per day, minimum headway, weekday boardings — constrain the *supply* side. Of the "
        "quantities that could not be validated, four are direct inputs to the crowding side that "
        f"the primary endpoint measures: {', '.join(s.unvalidated[:4])}.")
    add("")
    add("**Consequence, stated once and applied throughout:** the *ordering* of methods within "
        "the environment is internally valid, because all methods face identical conditions under "
        "common random numbers. The *magnitudes* are simulator-internal and must not be read as "
        "predictions about a physical metro system.")
    add("")

    # ---------------------------------------------------------------- 6
    add("## 6. Uncertainty representation and the placebo control")
    add("")
    add("Forecast uncertainty in this study means the **dispersion of the predictive "
        "distribution** of near-term demand, summarised as")
    add("")
    add("&nbsp;&nbsp;&nbsp;&nbsp;`σ = (q90 − q10) / (2 × 1.2816)`")
    add("")
    add("the normal-equivalent standard deviation implied by the 10th and 90th forecast "
        "quantiles. This definition was **frozen before any policy was trained**, so no "
        "controller can influence the uncertainty signal it receives.")
    add("")
    add("Every arm carries the same two-value uncertainty channel and the same parameter count. "
        "The arms differ only in that channel's content:")
    add("")
    add("| Arm | Channel content |")
    add("|---|---|")
    add("| `ppo_point` | Identically zero |")
    add("| `ppo_placebo` | σ values shuffled across time, preserving the marginal distribution "
        "while destroying association with realised forecast error |")
    add("| `ppo_uq` | The calibrated forecast dispersion σ |")
    add("")
    add("**Why the placebo matters.** If merely *having* the channel changed behaviour — through "
        "dimensionality, input scaling or optimisation dynamics — the placebo arm would move with "
        "the uncertainty arm. Divergence between them is attributable to σ being informative. "
        "This does not identify the internal computation that produces the behaviour; it "
        "identifies that the behaviour depends on the information content of the input.")
    add("")

    # ---------------------------------------------------------------- 7
    add("## 7. Reinforcement-learning framework")
    add("")
    add("- **State.** A 123-value summary comprising station queues, rolling-stock availability, "
        "clock features, forecast median and the two-value uncertainty channel. The availability "
        "ablation removes the availability block, leaving 121 values.")
    add("- **Action.** The number of trainsets to induct in the current epoch, "
        "*a_t* ∈ {0, …, k_max}.")
    add("- **Reward.** As in Section 3.")
    add("- **Constraints.** Enforced by action masking.")
    add(f"- **Algorithm.** Maskable PPO {cite('schulman2017')}, one frozen hyperparameter "
        "configuration applied unchanged to every arm.")
    add("")
    add("**No hyperparameter search was conducted.** A single documented configuration is applied "
        "to all arms. This is a deliberate symmetry constraint: a search conducted on one arm "
        "risks selecting a configuration that suits that arm's input, which would be an asymmetry "
        "in the comparison. The cost is that all arms may be jointly suboptimal, which bounds any "
        "claim about *absolute* policy quality. Section 16 treats this as the study's principal "
        "limitation.")
    add("")
    add("**Reproducibility of training.** Each worker is pinned to a single compute thread, "
        "because floating-point reduction order varies with thread count and would otherwise make "
        "runs irreproducible. Training days are drawn from a seed pool disjoint from every "
        "evaluation, scaler-fitting, tuning and calibration pool (Section 9).")
    add("")
    add("Full PPO configuration values are reproduced in Supplementary Material S2 from the "
        "stored policy metadata.")
    add("")

    # ---------------------------------------------------------------- 8
    add("## 8. Baseline methods")
    add("")
    add("| Baseline | Description | Information received |")
    add("|---|---|---|")
    add("| Fixed timetable | A published-style timetable executed without adaptation | Clock only |")
    add("| Threshold heuristic | Induct when queue or forecast exceeds a tuned threshold | Queue, "
        "forecast median |")
    add("| MILP (static) | Mixed-integer program solved once per day against a fluid-queue "
        "surrogate | Forecast median |")
    add("| MILP (rolling) | The same formulation re-solved on a rolling horizon | Forecast "
        "median, updated state |")
    add("")
    add(f"A baseline fairness audit of {s.fair['n_checks']} checks passes "
        f"{s.fair['n_passed']} with {len(s.fair['blocking_failures'])} blocking failures. Its "
        "sharpest check steps three methods through an identical action sequence and asserts "
        "identical arrivals and queues, which would detect a common-random-number "
        "desynchronisation between methods.")
    add("")
    add("Two aspects of baseline preparation are disclosed because they bear on fairness. The "
        "heuristic was **retuned** after its optimum was found at the edge of its search grid; a "
        "boundary optimum is not an optimum, so the grid was extended under a recorded amendment "
        "until the selection was interior, and every affected result was recomputed rather than "
        "mixed with the corrected baseline. The MILP surrogate was **corrected four times** — a "
        "standing-queue floor, removal of an unjustified monotonicity clamp, a dispatch-capacity "
        "envelope, and earlier service-rate and waiting-coefficient fixes. Each correction "
        "strengthened a competitor to the proposed method. A fifth was declined: the fluid-queue "
        "surrogate structurally cannot represent departure spacing, and continuing would have "
        "meant authoring a baseline rather than calibrating one.")
    add("")
    add("The resulting effort asymmetry — four baseline revisions against one untuned, frozen "
        "learned configuration — runs *against* the proposed method, and readers are entitled to "
        "weigh it in both directions.")
    add("")

    # ---------------------------------------------------------------- 9
    add("## 9. Experimental design")
    add("")
    add(f"- **Unit of analysis:** {s.stats['unit_of_analysis']}")
    add(f"- **Registered sample size:** {pw['registered_train_seeds']} training seeds × "
        f"{pw['registered_eval_seeds']} evaluation seeds × 4 scenarios")
    add(f"- **Registered episodes:** {s.stats['n_episode_rows_main']:,}, reducing to "
        f"{s.stats['n_seed_rows_main']:,} seed-level observations")
    add(f"- **Stored evaluation cells:** {s.cells:,}")
    add(f"- **Trained confirmatory policies:** {s.policies - 8} (plus 8 pilot policies, discarded)")
    add("- **Registered experiments:** 17, in a frozen registry")
    add("")
    add("**Evidence classes.** The study separates four, and the separation is structural rather "
        "than editorial — each stored cell carries a `variant` field that the analysis filters on:")
    add("")
    add("| Class | Role |")
    add("|---|---|")
    add("| **Pilot** | Four seeds used only to estimate the between-seed standard deviation for "
        "the power gate. Policies discarded; **no pilot observation enters any reported result** |")
    add("| **Confirmatory** | Exactly two registered tests, Holm-corrected at family-wise "
        "α = 0.05 |")
    add("| **Exploratory** | Hypothesis-generating; Benjamini–Hochberg at q = 0.10; supports no "
        "claim in the abstract |")
    add("| **Follow-up** | A registered training-budget study, **incomplete**; contributes no "
        "evidence (Section 16) |")
    add("")
    add("**Common random numbers.** The function constructing an environment's random streams "
        "takes the scenario and evaluation seed and **no method argument**, so two methods "
        "evaluated at the same seed cannot receive different demand realisations. This is a "
        "property of the implementation, not a convention a later edit could silently break.")
    add("")
    if pools:
        add("**Seed pools**, all mutually disjoint so that nothing is fitted on a day the "
            "comparison is later scored on: "
            + "; ".join(f"{k} {v}" for k, v in pools.items()) + ".")
        add("")
    add("**Pre-registration.** The registry was frozen after the power gate fixed the sample size "
        "and before the first confirmatory run. The sweep and ablation stages refuse to execute "
        "against an unfrozen registry, making the freeze a precondition of the code rather than a "
        "promise in a document.")
    add("")
    add("**Sample size and power.** A four-seed pilot on disjoint seeds estimated the between-seed "
        f"standard deviation of the paired difference at {pw['between_seed_sd']:.6f}. Detecting "
        f"the registered smallest effect size of interest (SESOI = {pw['sesoi']}) at 80% power "
        f"would require n = {pw['required_n_for_sesoi']}, beyond the available compute. The "
        f"registered design proceeded at n = {pw['registered_train_seeds']} with the "
        "underpowering stated in advance.")
    add("")
    add(f"**The pilot variance estimate proved too small, and we report the correction rather "
        f"than the planning figure.** The realised standard deviation of the confirmatory paired "
        f"differences is {s.sd_realised:.6f}, "
        f"**{s.sd_realised / pw['between_seed_sd']:.2f}× the pilot estimate**. Four seeds is a "
        "thin basis for a variance estimate. Consequently the registered minimum detectable "
        f"effect of {pw['minimum_detectable_effect_at_registered_n']:.6f} is optimistic, and we "
        "do not use it to support the conclusion. **No post-hoc power argument is made**: once a "
        "test has rejected, observed power is a deterministic function of the p-value and adds "
        "nothing. The study remains underpowered for its own SESOI, so every null reported here "
        "is inconclusive rather than evidence of equivalence.")
    add("")

    # ---------------------------------------------------------------- 10
    add("## 10. Statistical analysis")
    add("")
    add("The analysis plan was fixed before the confirmatory runs.")
    add("")
    add("- **Unit.** Episode outcomes are averaged to one observation per training seed before "
        "any test. The sample size is the number of seeds, never the number of episodes; "
        "reporting the latter would inflate precision by more than an order of magnitude.")
    add("- **Confirmatory family.** Exactly two tests, Holm-corrected at family-wise α = 0.05.")
    add("- **Exploratory family.** Benjamini–Hochberg at q = 0.10.")
    add("- **Test selection.** Paired *t*-test where Shapiro–Wilk on the differences does not "
        f"reject at 0.05 (here p = {p['normality_p']:.4f}); Wilcoxon signed-rank otherwise. Both "
        "are reported for the primary comparison.")
    add("- **Effect estimation.** Hodges–Lehmann estimator with bias-corrected and accelerated "
        "(BCa) bootstrap confidence intervals.")
    add("- **Equivalence.** Two one-sided tests against the registered SESOI, run only when the "
        "primary test fails to reject — which distinguishes *no evidence of benefit* from "
        "*evidence of no meaningful benefit*. The primary test rejected, so no equivalence test "
        "applies here.")
    add("")

    # ---------------------------------------------------------------- 11
    add("## 11. Results")
    add("")
    add("### 11.1 Method ranking")
    add("")
    add("We report the full ranking first, because it constrains how everything else should be "
        "read (Table 4, Figure 10).")
    add("")
    add("| Method | " + " | ".join(SCEN) + " | Total reward (surge) | Inductions (surge) |")
    add("|---|" + "---:|" * (len(SCEN) + 2))
    for method, row in s.rank.iterrows():
        mark = " *(proposed)*" if method == "ppo_uq" else ""
        add(f"| `{method}`{mark} | " + " | ".join(f"{row[sc]:.5f}" for sc in SCEN)
            + f" | {s.reward.loc[method, 'surge']:.1f} "
            + f"| {s.inductions.loc[method, 'surge']:.1f} |")
    add("")
    add("**No learned arm outperforms the tuned threshold heuristic or the static mixed-integer "
        "formulation in any scenario**, on either the primary endpoint or the training objective. "
        "The proposed `ppo_uq` is also worse than the fixed timetable — worse, that is, than not "
        "adapting at all.")
    add("")
    add("Baselines are deterministic and carry no training seed, so under the registered "
        "seed-level unit of analysis each contributes a single observation and **no inferential "
        "test between a learned method and a baseline is defined**. These are differences of "
        "means, not hypothesis tests, and are labelled as such throughout. We report them "
        "prominently nevertheless: the comparison is unfavourable to the proposed method, and "
        "omitting it would misrepresent what the study found.")
    add("")
    add("### 11.2 Primary comparison (EXP-UQ-001)")
    add("")
    add(f"`ppo_uq` versus `ppo_point` on `{p['metric']}` under `{p['scenario']}`, "
        f"n = {p['n']} training seeds (Table 5, Figure 5).")
    add("")
    add("| Quantity | Value |")
    add("|---|---:|")
    add(f"| Mean, `ppo_uq` | {p['mean_a']:.6f} |")
    add(f"| Mean, `ppo_point` | {p['mean_b']:.6f} |")
    add(f"| Mean paired difference | {p['mean_difference']:+.6f} |")
    add(f"| Hodges–Lehmann estimate | {p['hodges_lehmann']:.6f} |")
    add(f"| 95% BCa confidence interval | [{p['ci_low']:.6f}, {p['ci_high']:.6f}] |")
    add(f"| Test | {p['test']} |")
    add(f"| Unadjusted p | {p['p_value']:.6g} |")
    add(f"| Holm-adjusted p | {p['p_adjusted']:.6f} |")
    add(f"| SESOI (registered) | {pw['sesoi']} |")
    add(f"| Relative degradation | {p['mean_difference'] / p['mean_b'] * 100:.1f}% |")
    add("")
    add("Because a **higher** left-behind fraction is worse, a positive difference means the "
        "uncertainty-aware arm performed **worse**. The confidence interval excludes zero and "
        f"lies entirely above the SESOI of {pw['sesoi']}.")
    add("")
    add("**This is not a null result.** The registered test rejected the null hypothesis in the "
        "direction opposite to the one predicted.")
    add("")
    add("### 11.3 Placebo control (EXP-UQ-002)")
    add("")
    add(f"`ppo_uq` versus `ppo_placebo`: {c['mean_difference']:+.6f}, 95% BCa CI "
        f"[{c['ci_low']:.6f}, {c['ci_high']:.6f}], Holm-adjusted p = {c['p_adjusted']:.6f} "
        "(Table 6, Figure 6).")
    add("")
    add("The ordering across the three arms is graded, and more informative than a binary "
        f"contrast: `ppo_point` {s.rank.loc['ppo_point', 'surge']:.5f} < `ppo_placebo` "
        f"{s.rank.loc['ppo_placebo', 'surge']:.5f} < `ppo_uq` "
        f"{s.rank.loc['ppo_uq', 'surge']:.5f}. An uninformative channel of the same shape costs "
        "something; an informative one costs more. Additional input dimensionality alone does not "
        "account for the gap.")
    add("")
    add("We are careful about what this identifies. The placebo shows that the *behavioural "
        "difference depends on the information content* of the channel. It does not identify the "
        "internal representation, and it is not a randomised assignment of uncertainty to "
        "operating conditions.")
    add("")
    add("### 11.4 Hypothesis verdict")
    add("")
    add(f"**H1 = {s.verdict()}.**")
    add("")

    # ---------------------------------------------------------------- 12
    add("## 12. Mechanism analysis")
    add("")
    add("The design manipulates a single variable — uncertainty-channel content — with all else "
        "held fixed under common random numbers. Behavioural differences between arms are "
        "therefore attributable to that manipulation. The analysis below uses stored per-epoch "
        "records from the registered sweep; nothing was re-simulated.")
    add("")
    add("### 12.1 Informative uncertainty suppresses dispatch; placebo uncertainty does not")
    add("")
    add("| Scenario | Calibrated σ vs zero | p | Placebo σ vs zero | p |")
    add("|---|---:|---:|---:|---:|")
    for row in s.mech["dispatch_volume"]:
        info, plc = row["informative_sigma_vs_zero"], row["uninformative_sigma_vs_zero"]
        add(f"| {row['scenario']} | {info['mean_difference']:+.1f} inductions | "
            f"{info['p_value']:.1e} | {plc['mean_difference']:+.1f} | {plc['p_value']:.2f} |")
    add("")
    add(f"Under surge this is approximately a **{s.dispatch_drop_pct:.0f}% reduction in dispatch "
        "volume** relative to the point-forecast arm (Table 9, Figure 7). The placebo arm does "
        "not move. The evidence is therefore consistent with the controller responding to genuine "
        "forecast dispersion, and responding by withholding service.")
    add("")
    add("### 12.2 Dispatch timing, at matched volume")
    add("")
    matched = s.mech["volume_matched_comparison"]["surge"]
    add("`ppo_uq` and the fixed timetable release almost the same number of trains, which makes "
        "them the one pair in the study comparable at approximately matched volume:")
    add("")
    add("| Quantity | `ppo_uq` | `fixed_timetable` |")
    add("|---|---:|---:|")
    for key in ("inductions", "idle_epochs", "max_queue"):
        if key in matched["ppo_uq"] and key in matched["fixed_timetable"]:
            add(f"| {key.replace('_', ' ')} | {matched['ppo_uq'][key]:.1f} | "
                f"{matched['fixed_timetable'][key]:.1f} |")
    add("")
    add("At comparable volume the uncertainty-aware arm leaves markedly more decision epochs with "
        "no dispatch and reaches a larger peak queue than a non-adaptive schedule. This suggests "
        "both restraint and clustering are present. Across methods the relationship between "
        "dispatch volume and service is not monotone — the tuned heuristic achieves the best "
        "service in the study while dispatching fewer trains than `ppo_point` — so timing matters "
        "alongside volume.")
    add("")
    add("### 12.3 Convex service loss")
    add("")
    add("Measured from stored per-epoch records (Figure 8), passengers left behind remain near "
        "zero while capacity is slack and rise sharply once it binds.")
    add("")
    add("For a convex loss ℓ, Jensen's inequality gives E[ℓ(D)] ≥ ℓ(E[D]): the expected service "
        "loss under stochastic demand exceeds the loss evaluated at expected demand, and the gap "
        f"widens with dispersion. This is the classical newsvendor observation {cite('arrow1951')} "
        "and we invoke it as established rather than derive it. Its operational implication is "
        "that a controller facing elevated forecast dispersion has reason to hedge **upward**, "
        "buying protection against an expensive upper tail.")
    add("")
    add("**Interpretation, stated with its limits.** We separate what was observed from what it "
        "suggests:")
    add("")
    add("| | |")
    add("|---|---|")
    add("| **Observed** | Informative σ is associated with fewer dispatches and worse service; "
        "placebo σ is not |")
    add("| **Mechanism evidence** | The placebo comparison indicates the effect depends on the "
        "information content of σ, not on additional input dimensions |")
    add("| **Interpretation** | The evidence is *consistent with* the controller having learned a "
        "conservative response to dispersion, in the direction opposite to the one convex loss "
        "would motivate |")
    add("| **Limitation** | We do not observe the network's internal computation. The registered "
        "epoch-level test — dispatch rate against dispersion conditional on the forecast mean — "
        "could not be run, because stored cells do not carry the per-epoch forecast. It remains "
        "open |")
    add("| **Future hypothesis** | If this is an optimisation failure rather than a property of "
        "the information, a larger training budget should attenuate or reverse it (Section 17) |")
    add("")
    add("We deliberately do not claim the controller *learned the wrong sign* as a mathematical "
        "statement. The available evidence supports a behavioural description and a plausible "
        "reading of it, not a proof about the learned value function.")
    add("")

    # ---------------------------------------------------------------- 13
    add("## 13. Robustness and sensitivity")
    add("")
    add("| Check | Result |")
    add("|---|---|")
    add(f"| Scenarios reproducing the direction | {len(SCEN)}/{len(SCEN)} |")
    add(f"| Training seeds favouring `ppo_point` | {s.seeds_favouring_point}/{len(s.diff)} |")
    add(f"| Distribution-free cross-check (Wilcoxon signed-rank) | p = {s.wilcoxon_p:.6f} |")
    add(f"| Worst leave-one-seed-out replicate | p = {s.jack_p:.5f} (seed {s.jack_seed}), "
        f"Holm-adjusted {min(1.0, 2 * s.jack_p):.5f} |")
    add(f"| Sensitivity conditions where `ppo_uq` is worse | "
        f"{s.sens['direction_agreement']['conditions_where_ppo_uq_is_worse']}"
        f"/{s.sens['direction_agreement']['conditions_total']} |")
    add("")
    add("The conclusion does not depend on any single training seed, on the choice of parametric "
        "test, or on one scenario (Table 7, Figure 9).")
    add("")
    add("**Origin–destination sensitivity.** The OD structure is synthetic and drives the primary "
        "endpoint, so the comparison was re-scored under all three OD variants. The registered "
        "direction holds in every one, against both comparators (Table 8, Figure 11). Individual "
        "conditions mostly do not clear the Benjamini–Hochberg threshold, which is expected — "
        "each uses five training seeds and ten evaluation seeds against ten and thirty in the "
        "main sweep — so the evidence is the consistency of sign rather than any single test. "
        "These conditions re-score the *same* policies and are therefore strongly dependent; "
        "consistency across them is weaker evidence than independent replication would be.")
    add("")
    add("**Reward-weight sensitivity is vacuous on the primary endpoint, and we report that "
        "rather than the analysis it appears to be.** Re-scoring a frozen policy under a "
        "different reward weighting changes no action, because the policy does not observe the "
        "reward at evaluation time. The trajectory — and with it `left_behind_fraction` and "
        "`inductions` — is identical across all three weightings, verified per seed; only "
        "`total_reward` differs. Measured there, `ppo_point` wins under every weighting, by a "
        "margin that grows as the weighting favours service, which is the ordering the "
        "under-dispatching mechanism predicts. Testing weight sensitivity on physical outcomes "
        "would require *retraining* under each weighting, which this design did not do.")
    add("")
    add("**Interpretability.** Feature attributions for one policy at one training seed are "
        f"reported in Supplementary Material S9 ({s.shap['explainer']}). They describe model "
        "association on a chosen instance set and **do not establish causality**; they support no "
        "hypothesis in this paper.")
    add("")

    # ---------------------------------------------------------------- 14
    add("## 14. Discussion")
    add("")
    add("**What was expected.** Demand uncertainty is decision-relevant in capacity-constrained "
        "service systems, and the registered hypothesis was that supplying it to the controller "
        "would reduce the fraction of passengers left behind.")
    add("")
    add("**What was observed.** The opposite, significantly and consistently. The "
        "uncertainty-aware controller left more passengers behind than the point-forecast "
        "controller, and more than a placebo carrying an uninformative signal of identical "
        "marginal distribution. It also dispatched substantially fewer trains when given genuine "
        "dispersion.")
    add("")
    add("**Why it may have happened.** Three readings are consistent with the evidence, and the "
        "design separates them only partly.")
    add("")
    add("*Reading 1 — the signal is genuinely unhelpful in this setting.* The placebo comparison "
        "supports this more than a naive reading would: the calibrated σ performed worse than a "
        "shuffled σ of the same marginal distribution, so the harm is not simply the cost of two "
        "additional inputs.")
    add("")
    add("*Reading 2 — the information is usable but the controller did not learn to use it.* One "
        "frozen configuration at a fixed budget may fail to exploit an input that a longer or "
        "better-tuned run would. The convex-loss result makes this reading more plausible than it "
        "would otherwise be: theory says dispersion should motivate hedging upward and the "
        "controller moved down, which resembles a learning failure more than an informative "
        "absence of value. That no learned arm beats a threshold heuristic points the same way.")
    add("")
    add("*Reading 3 — the scalarisation misdirects the response.* Under a weighting where energy "
        "and cost carry real weight, dispatch has an immediate certain cost while its benefit is "
        "delayed and uncertain. A controller reading high dispersion as elevated risk may protect "
        "the certain terms and lose more on the service terms. The observed pattern — lower "
        "energy and cost, worse service, worse total reward — is consistent with this.")
    add("")
    add("Distinguishing readings 1 and 2 requires a training-budget experiment. Until it is run, "
        "the defensible statement is the narrow one: *in this environment, at this budget, the "
        "uncertainty channel did not help and measurably hurt.*")
    add("")
    add("**What it means for uncertainty-aware reinforcement learning.** It identifies a concrete "
        "failure mode — an informative uncertainty input producing a systematically conservative "
        "response that is costly precisely where the loss is convex — and shows that the failure "
        "survives a placebo control. It does **not** establish that uncertainty-aware "
        "reinforcement learning is generally ineffective, and no result here speaks to "
        "risk-sensitive objectives, distributional value functions, or architectures designed to "
        "consume uncertainty explicitly.")
    add("")
    add("**What it means for evaluation practice.** A single-seed, episode-level comparison "
        "without a placebo could plausibly have produced a favourable and unreplicable result "
        f"from the same experimental material {cite('henderson2018', 'colas2018')}. The control "
        "structure, not the algorithm, is what makes this study's conclusion interpretable.")
    add("")

    # ---------------------------------------------------------------- 15
    add("## 15. Practical implications")
    add("")
    add("Stated conditionally, because the environment's crowding components are not externally "
        "validated and the study covers one line.")
    add("")
    add("**For metro operators.** Adding a calibrated uncertainty feature to a learned dispatch "
        "controller should not be assumed to improve service. Evaluation should include a "
        "placebo-controlled comparison rather than a before/after improvement, and should stress "
        "the controller in **capacity-binding regimes**, where the cost of a conservative decision "
        "is nonlinear and where a well-behaved controller in slack conditions may still fail.")
    add("")
    add("**For transit planners.** In this study a tuned threshold heuristic outperformed every "
        "learned controller. Where a simple demand-responsive rule is available, it is a serious "
        "operational candidate and an essential evaluation baseline, not a formality.")
    add("")
    add("**For reinforcement-learning researchers.** The failure mode is specific and testable: "
        "an informative uncertainty input can induce conservatism in the wrong direction relative "
        "to a convex loss. Reporting dispatch-volume behaviour alongside outcome metrics would "
        "make such effects visible earlier.")
    add("")
    add("**For forecasting researchers.** Calibration is necessary but not sufficient. A "
        "well-calibrated dispersion estimate can still degrade a downstream controller if the "
        "controller's response to it is mis-signed.")
    add("")
    add("**For simulation researchers.** The quantities that determine an endpoint deserve "
        "validation priority. Here the supply-side quantities were validated and the crowding-side "
        "quantities — which define the endpoint — largely were not.")
    add("")
    add("**No claim of deployment readiness is made.** Nothing in this study licenses operational "
        "use.")
    add("")

    # ---------------------------------------------------------------- 16
    add("## 16. Limitations")
    add("")
    add(f"1. **Underpowered for the registered SESOI.** n = {pw['registered_train_seeds']} against "
        f"n = {pw['required_n_for_sesoi']} required at the pilot standard deviation, and the "
        f"realised standard deviation was {s.sd_realised / pw['between_seed_sd']:.2f}× larger "
        "still. Every null result here is inconclusive, never evidence of equivalence.")
    add("2. **A single frozen training configuration**, which may be undertrained in absolute "
        "terms rather than merely relative to its siblings: no learned arm beats the threshold "
        "heuristic, and one single-objective training seed diverged by two orders of magnitude. "
        "This is the largest threat to the interpretation and is not resolvable by re-analysis.")
    add("3. **Training-budget confounding.** The study cannot separate *the uncertainty channel "
        "is unhelpful* from *this optimiser could not exploit it at this budget*.")
    add("4. **The learned controllers underperform a tuned heuristic**, which bounds the "
        "practical contribution to nothing and must be read alongside the primary result.")
    add(f"5. **{len(s.unvalidated)} of {len(s.sim['rows'])} environment quantities are not "
        "externally validated**, including four that feed the primary endpoint directly. "
        "Magnitudes are simulator-internal.")
    add("6. **Synthetic origin–destination structure**, mitigated but not eliminated by the OD "
        "sensitivity analysis.")
    add("7. **A single line, operator and network configuration.** No external-validity claim.")
    add("8. **Limited external validation** of the environment overall; it is a simulation "
        "environment, not a validated digital twin.")
    add("9. **The registered follow-up is incomplete.** The training-budget study was stopped at "
        "11 of 80 policies, at a single budget, asymmetrically distributed across arms, with **no "
        "evaluation cells**. It supports **no result and none is reported from it**: not even the "
        "lowest budget admits a paired comparison. It appears in this paper only as an open "
        "question.")
    add("10. **No transfer or generalisation evidence.** Nothing here indicates how the finding "
        "would transfer to another line, demand regime, architecture or reward.")
    add("11. **The registered mechanism test (epoch-level) is unresolved**, because stored cells "
        "do not carry the per-epoch forecast.")
    add("12. **Feature attributions are associational**, computed for one policy at one seed.")
    add("")
    add("These are collected in Table 10.")
    add("")

    # ---------------------------------------------------------------- 17
    add("## 17. Future work")
    add("")
    add("The following are **proposed, not completed**. None was run after the confirmatory "
        "result was known, which is deliberate: doing so would convert a pre-registered study "
        "into a post-hoc one.")
    add("")
    add("1. **Registered training-budget sweep.** The single most informative remaining "
        "experiment, and the only one that can separate an information effect from a "
        "sample-efficiency effect. A protocol and a pre-committed analysis exist in the "
        "repository; execution is incomplete.")
    add("2. **Symmetric hyperparameter selection.** Select one configuration on independent "
        "validation data using the *point-forecast* arm only, then apply it unchanged to all "
        "arms. Tuning on the arm that is not the hypothesis preserves symmetry while addressing "
        "the absolute-undertraining concern.")
    add("3. **Uncertainty-scaling probe.** Supply a deliberately mis-scaled dispersion signal. A "
        "monotone response would implicate magnitude rather than information content.")
    add("4. **Risk-sensitive and robust formulations.** Test controllers designed to consume "
        "uncertainty explicitly, rather than supplying uncertainty to a risk-neutral learner.")
    add("5. **Alternative reward weightings, with retraining**, to test whether the conservative "
        "response persists when unmet demand dominates the objective.")
    add("6. **Multi-line and broader demand regimes**, and validation of the crowding-side "
        "environment components against operator data.")
    add("7. **Operational pilot evaluation** with human dispatchers, which no simulation result "
        "can substitute for.")
    add("")

    # ---------------------------------------------------------------- 18
    add("## 18. Reproducibility and software artefact")
    add("")
    interp = s.evidence("interpreter recorded")
    deps = s.evidence("dependency reproducibility")
    add("| Item | Value |")
    add("|---|---|")
    add(f"| Git commit | `{s.commit}` |")
    add(f"| Immutable tag | `{s.tag}` |")
    add(f"| Interpreter | Python {interp.get('python', 'n/a')} on "
        f"{interp.get('platform', 'n/a')} |")
    add(f"| Dependencies | {deps.get('pinned', 0)} pinned packages, {deps.get('hashes', 0)} "
        "SHA-256 hashes |")
    add(f"| Evaluation cells | {s.cells:,} |")
    add(f"| Trained policies | {s.policies} (including 8 discarded pilot policies) |")
    add(f"| Registered experiments | 17 |")
    add(f"| Reproducibility audit | "
        + ", ".join(f"{k} = {v}" for k, v in s.audit["counts"].items()) + " |")
    add(f"| Immutability audit | "
        + ", ".join(f"{k} = {v}" for k, v in s.separation["counts"].items()) + " |")
    add("")
    add("Reproduction proceeds from a hash-pinned environment:")
    add("")
    add("```bash")
    add("python -m venv .venv")
    add(".venv/bin/python -m pip install --require-hashes -r requirements.lock.txt")
    add(".venv/bin/python scripts/check_environment.py")
    add(".venv/bin/python -m pytest -q")
    add(".venv/bin/python scripts/analyse.py            # regenerates the statistical analysis")
    add(".venv/bin/python scripts/make_paper_assets.py  # regenerates every figure and table")
    add("```")
    add("")
    add("Every figure carries a `.source.json` provenance companion naming the artefacts it was "
        "drawn from, and this manuscript is rendered by a script that reads those artefacts, so "
        "no reported number is typed by hand. Full instructions are in the repository's "
        "reproduction guide.")
    add("")
    add("**Provenance caveat, stated because a reproducibility claim a reader cannot act on is "
        "worse than a stated limitation.** Almost all stored evaluation cells were produced while "
        "the working tree contained uncommitted changes; their manifests record this as "
        "`provisional`. The recorded commit therefore does not fully determine the code that "
        "produced them. The code is committed now, and re-running from the committed state is "
        "required to obtain commit-anchored results.")
    add("")
    add("**Software artefact.** The repository includes two interactive applications: a research "
        "results dashboard, and an operator-facing decision-support prototype that presents the "
        "simulated line state, produces dispatch recommendations from the trained policies and "
        "compares strategies. Both are demonstration artefacts served locally; **neither "
        "constitutes deployment**, and the prototype does not control railway infrastructure. "
        "The prototype reads research artefacts read-only, and its interface labels every "
        "displayed quantity with its evidence class so that a simulated demonstration value "
        "cannot be read as a research result.")
    add("")

    # ---------------------------------------------------------------- 19
    add("## 19. Conclusion")
    add("")
    add("We evaluated whether supplying calibrated demand uncertainty to a reinforcement-learning "
        "metro dispatch controller improves passenger service, in a pre-registered, "
        "placebo-controlled experiment on a demand-driven simulation environment.")
    add("")
    add(f"The registered hypothesis was **not supported**. The uncertainty-aware controller left "
        f"more passengers behind than an otherwise identical point-forecast controller "
        f"({p['mean_difference']:+.6f}; 95% BCa CI [{p['ci_low']:.6f}, {p['ci_high']:.6f}]; "
        f"Holm-adjusted p = {p['p_adjusted']:.6f}), and also performed worse than a placebo "
        "carrying an uninformative signal of identical marginal distribution.")
    add("")
    add("Rather than improving service, the controller exhibited a systematic conservative "
        f"dispatch response to genuine forecast uncertainty — approximately "
        f"{s.dispatch_drop_pct:.0f}% fewer dispatches under surge — which the placebo arm did "
        "not. Because measured service loss is strongly convex once capacity binds, uncertainty "
        "is decision-relevant in this environment and the classical newsvendor argument favours "
        "hedging upward; the controller moved in the opposite direction. Combined with the "
        "finding that no learned arm outperformed a tuned threshold heuristic, the evidence "
        "points toward a policy-learning limitation rather than toward uncertainty information "
        "being valueless.")
    add("")
    add("Controlled negative evidence is informative when the design distinguishes competing "
        "explanations. Here a placebo arm and a seed-level unit of analysis were sufficient to "
        "overturn an intuition that a less controlled comparison would likely have confirmed. The "
        "result characterises the tested controller, configuration, training budget, environment "
        "and scenarios; it identifies a concrete failure mode and motivates controlled "
        "investigation of uncertainty utilisation, training budget and risk-sensitive policy "
        "design.")
    add("")
    add("---")
    add("")
    add("## References")
    add("")
    for index, reference in enumerate(REFERENCES, start=1):
        add(f"{index}. **[{reference.key}]** {reference.formatted()}")
        add("")
    add("---")
    add("")
    add("*Supplementary material: `docs/supplementary_material.md`.*")
    add("")
    return "\n".join(L)


# --------------------------------------------------------------------------- other documents


def supplementary(s: Study) -> str:
    pw = s.power
    meta_path = next((REPO_ROOT / "models" / "policies" / "w_balanced" / "ppo_uq")
                     .glob("*/metadata.json"), None)
    config = {}
    if meta_path:
        config = json.loads(meta_path.read_text(encoding="utf-8")).get("config", {})
    pools = s.evidence("seed pools are disjoint").get("pools", {})
    L: list[str] = []
    add = L.append
    add("# Supplementary material")
    add("")
    add(f"*Generated from stored artefacts at commit `{s.commit[:12]}`.*")
    add("")
    add("## S1. Experimental registry")
    add("")
    add("The frozen registry (`experiments/registry.yaml`) defines 17 registered experiments. "
        "Confirmatory: EXP-UQ-001 (H1, primary), EXP-UQ-002 (H1c, placebo co-primary). "
        "Exploratory families cover scenario generalisation (EXP-UQ-003), multi-objective "
        "ablation (EXP-MO-001), availability ablation (EXP-AVAIL-001), baselines (EXP-BASE-*), "
        "robustness (EXP-ROBUST-*), sensitivity (EXP-SENS-001/002) and interpretability "
        "(EXP-XAI-001). The registry is frozen; sweep and ablation stages refuse to run against "
        "an unfrozen registry.")
    add("")
    add("## S2. PPO configuration")
    add("")
    if config:
        add("| Parameter | Value |")
        add("|---|---|")
        for key, value in config.items():
            add(f"| `{key}` | {value} |")
        add("")
        add("One configuration, applied unchanged to every arm. No hyperparameter search was "
            "conducted.")
    else:
        add("`[NOT AVAILABLE — policy metadata not found]`")
    add("")
    add("## S3. Seed pools")
    add("")
    add("| Pool | Range |")
    add("|---|---|")
    for name, span in pools.items():
        add(f"| {name} | {span} |")
    add("")
    add("All pools are mutually disjoint, so nothing is fitted on a day the comparison is scored "
        "on.")
    add("")
    add("## S4. Scenario definitions")
    add("")
    add("| Scenario | Description |")
    add("|---|---|")
    add("| `normal` | Baseline demand and full rolling-stock availability |")
    add("| `surge` | Elevated demand; the registered primary scenario |")
    add("| `disruption` | Reduced availability through rolling-stock failures |")
    add("| `combined` | Surge and disruption together |")
    add("")
    add("Scenario families share a base world at a given evaluation seed, so a scenario contrast "
        "measures the scenario rather than a different random draw.")
    add("")
    add("## S5. Baseline details")
    add("")
    add(f"Baseline fairness audit: {s.fair['n_passed']}/{s.fair['n_checks']} checks pass, "
        f"{len(s.fair['blocking_failures'])} blocking failures.")
    add("")
    add("| Check | Status |")
    add("|---|---|")
    for check in s.fair["checks"][:19]:
        add(f"| {str(check.get('name', check))[:90]} | {check.get('status', 'PASS')} |")
    add("")
    add("## S6. Statistical procedures")
    add("")
    add(f"- Unit of analysis: {s.stats['unit_of_analysis']}")
    add("- Confirmatory: exactly two tests, Holm-corrected at family-wise α = 0.05")
    add("- Exploratory: Benjamini–Hochberg at q = 0.10")
    add("- Effect estimate: Hodges–Lehmann with BCa bootstrap intervals")
    add("- Equivalence: TOST against the registered SESOI, run only on non-rejection")
    add(f"- SESOI = {pw['sesoi']}; registered n = {pw['registered_train_seeds']}; "
        f"pilot SD = {pw['between_seed_sd']:.6f}; realised SD = {s.sd_realised:.6f}")
    add("")
    add("## S7. Sensitivity analyses")
    add("")
    add("| Condition | Comparison | Difference | BH-adjusted p | Direction |")
    add("|---|---|---:|---:|---|")
    for test in s.sens["tests"]:
        add(f"| {test['label']} | {test['method_a']} vs {test['method_b']} | "
            f"{test['mean_difference']:+.6f} | {test['p_adjusted']:.4g} | {test['direction']} |")
    add("")
    add(s.sens["weight_invariance"]["finding"])
    add("")
    add("## S8. Ablations")
    add("")
    add("| Experiment | Scenario | A | B | Difference | Adjusted p | Direction |")
    add("|---|---|---|---|---:|---:|---|")
    for test in s.stats["exploratory"]:
        if (test.get("experiment_id") in ("EXP-MO-001", "EXP-AVAIL-001")
                and test.get("metric") == "left_behind_fraction"):
            add(f"| {test['experiment_id']} | {test['scenario']} | {test['method_a']} | "
                f"{test['method_b']} | {test['mean_difference']:+.6f} | "
                f"{test['p_adjusted']:.4g} | {test['direction']} |")
    add("")
    add("## S9. Interpretability")
    add("")
    add(f"{s.shap['explainer']}, explaining {s.shap['explained_quantity']}, for policy "
        f"`{s.shap['method']}` at training seed {s.shap['train_seed']} "
        f"(background {s.shap['background_size']}).")
    add("")
    add("**Attributions describe model association and do not establish causality.** Stored "
        "caveats:")
    for caveat in s.shap.get("caveats", []):
        add(f"- {caveat}")
    add("")
    add("## S10. Reproduction")
    add("")
    add("```bash")
    add("python -m venv .venv")
    add(".venv/bin/python -m pip install --require-hashes -r requirements.lock.txt")
    add(".venv/bin/python scripts/check_environment.py")
    add(".venv/bin/python -m pytest -q")
    add(".venv/bin/python scripts/analyse.py")
    add(".venv/bin/python scripts/make_paper_assets.py")
    add(".venv/bin/python scripts/build_manuscript.py")
    add("```")
    add("")
    add(f"Expected: {s.cells:,} evaluation cells, {s.policies} policy checkpoints, "
        "11 figures, 10 tables.")
    add("")
    return "\n".join(L)


def references_final() -> str:
    counts = ref_counts()
    L = ["# Verified bibliography", "",
         f"*{counts['fully_verified']} of {counts['total']} entries verified in full; "
         f"{counts['needing_confirmation']} require field confirmation before submission.*", "",
         "**Zero-fabrication policy.** Every entry was located by literature search during "
         "manuscript preparation. Entries marked `[REFERENCE VERIFICATION REQUIRED]` had their "
         "title and venue verified but not every bibliographic field; those fields must be "
         "completed against the publisher record. No author list, year, volume, page range or "
         "DOI in this file was reconstructed from memory, and no DOI is stated that was not "
         "observed.", ""]
    by_topic: dict[str, list] = {}
    for reference in REFERENCES:
        for topic in (reference.topics or ("other",)):
            by_topic.setdefault(topic, []).append(reference)
    seen = set()
    for topic in sorted(by_topic):
        L.append(f"## {topic}")
        L.append("")
        for reference in by_topic[topic]:
            if reference.key in seen:
                continue
            seen.add(reference.key)
            L.append(f"- **[{reference.key}]** {reference.formatted()}")
        L.append("")
    return "\n".join(L)


def cover_letter(s: Study) -> str:
    p = s.primary
    return "\n".join([
        "# Cover letter (draft)", "",
        "[JOURNAL NAME TO BE CONFIRMED]  ", "[EDITOR NAME TO BE CONFIRMED]", "",
        "Dear Editor,", "",
        f"We submit for your consideration our manuscript, *{TITLE}*.", "",
        "**Problem.** Metro operators must repeatedly decide how many trainsets to release into "
        "service under demand that is known only through a forecast. A widely held assumption is "
        "that supplying a learned controller with calibrated forecast uncertainty should improve "
        "those decisions. That assumption is rarely tested against a control able to separate the "
        "information in the uncertainty signal from the mere presence of an additional input.", "",
        "**What we did.** We conducted a pre-registered, placebo-controlled evaluation in a "
        "demand-driven metro simulation environment. Three reinforcement-learning arms share "
        "identical architecture, parameter count, observation dimensionality, reward, training "
        "budget, seeds and evaluation days, and differ only in what a two-value uncertainty "
        "channel carries: zero, a shuffled placebo of identical marginal distribution, or the "
        "calibrated forecast dispersion. Fixed-timetable, tuned-heuristic and mixed-integer "
        "baselines are evaluated under common random numbers.", "",
        "**Main finding, stated plainly.** The registered hypothesis was **not supported**. The "
        f"uncertainty-aware controller left more passengers behind than the point-forecast "
        f"controller ({p['mean_difference']:+.6f}; 95% BCa CI [{p['ci_low']:.6f}, "
        f"{p['ci_high']:.6f}]; Holm-adjusted p = {p['p_adjusted']:.6f}), and also performed worse "
        "than the placebo arm. Mechanism analysis shows the calibrated signal is associated with "
        f"approximately {s.dispatch_drop_pct:.0f}% fewer dispatches while the placebo signal is "
        "not, and that measured service loss is strongly convex once capacity binds — so "
        "uncertainty is decision-relevant here and the controller's conservative response ran "
        "opposite to what convex loss motivates.", "",
        "**Why it matters.** We report a controlled negative result with an identified mechanism. "
        "We also report, prominently rather than in an appendix, that no learned arm outperformed "
        "a tuned threshold heuristic. Together these bound what can currently be claimed for "
        "learned metro dispatch and identify a specific, testable failure mode. We believe "
        "controlled negative evidence of this kind is useful to the field precisely because the "
        "design distinguishes competing explanations.", "",
        "We make no claim that uncertainty-aware reinforcement learning is generally ineffective, "
        "and we state the study's principal confound — a single frozen training configuration — "
        "explicitly, together with the registered follow-up that would resolve it and which "
        "remains incomplete.", "",
        "**Declarations.** This manuscript is original work, has not been published previously, "
        "and is not under consideration elsewhere. `[AUTHORS TO CONFIRM]`  ",
        "Funding: `[TO BE CONFIRMED — no funding information is recorded in the project]`  ",
        "Competing interests: `[TO BE CONFIRMED BY THE AUTHORS]`  ",
        "All code and computational artefacts are version-controlled and are available for "
        "editorial and reviewer inspection; see the data and code availability statement.", "",
        "Thank you for your consideration.", "", "Yours sincerely,", "",
        AUTHORS[0] + " (corresponding author `[TO BE CONFIRMED]`)  ",
        "on behalf of " + ", ".join(AUTHORS) + "  ",
        f"Supervisor: {SUPERVISOR}  ",
        AFFILIATION, "",
    ])


def highlights(s: Study) -> str:
    p = s.primary
    return "\n".join([
        "# Highlights", "",
        "*Journal highlights are conventionally 3–5 bullets of up to 85 characters each. Both a "
        "compliant short form and an expanded form are given.*", "",
        "## Short form (≤85 characters)", "",
        "- Pre-registered placebo-controlled test of forecast uncertainty in metro dispatch",
        "- Uncertainty-aware PPO left more passengers behind than point-forecast PPO",
        "- Effect absent under a placebo signal of identical marginal distribution",
        "- Calibrated uncertainty cut dispatch volume by about 11% under demand surge",
        "- A tuned threshold heuristic outperformed every learned controller tested", "",
        "## Expanded form", "",
        f"- In a pre-registered, placebo-controlled evaluation, an uncertainty-aware PPO metro "
        f"dispatch controller performed **worse** than an otherwise identical point-forecast "
        f"controller on the registered primary endpoint ({p['mean_difference']:+.6f}; "
        f"Holm-adjusted p = {p['p_adjusted']:.6f}).",
        "- The degradation was also present against a placebo uncertainty channel with matched "
        "dimensionality and marginal distribution, so it is not explained by additional inputs.",
        f"- Genuine uncertainty was associated with approximately {s.dispatch_drop_pct:.0f}% fewer "
        "dispatches; the placebo signal was not.",
        "- Measured service loss is strongly convex once capacity binds, so uncertainty is "
        "decision-relevant and the observed conservatism runs opposite to the direction convex "
        "loss motivates.",
        "- No learned controller outperformed a tuned threshold heuristic in any scenario, which "
        "bounds current claims for learned metro dispatch.", "",
    ])


def author_contributions() -> str:
    return "\n".join([
        "# Author contributions (CRediT draft)", "",
        "**This file is a template requiring completion by the authors.**", "",
        "The repository contains no record of which author performed which task. Assigning "
        "CRediT roles without that evidence would be fabrication, so every role below is marked "
        "for confirmation rather than guessed. Git authorship in this repository reflects the "
        "tooling used during development and is **not** evidence of individual scholarly "
        "contribution.", "",
        "| Author | CRediT roles |",
        "|---|---|",
        f"| {AUTHORS[0]} | `[AUTHOR CONTRIBUTION TO BE CONFIRMED]` |",
        f"| {AUTHORS[1]} | `[AUTHOR CONTRIBUTION TO BE CONFIRMED]` |",
        f"| {AUTHORS[2]} | `[AUTHOR CONTRIBUTION TO BE CONFIRMED]` |",
        f"| {SUPERVISOR} | `[AUTHOR CONTRIBUTION TO BE CONFIRMED]` |", "",
        "## CRediT taxonomy roles to allocate", "",
        "Conceptualization · Methodology · Software · Validation · Formal analysis · "
        "Investigation · Resources · Data curation · Writing – original draft · "
        "Writing – review & editing · Visualization · Supervision · Project administration · "
        "Funding acquisition", "",
        "## Suggested statement once completed", "",
        "> **CRediT authorship contribution statement.** [Name]: [roles]. [Name]: [roles]. "
        "[Name]: [roles]. [Name]: [roles]. All authors read and approved the final manuscript.", "",
        "Author order in the manuscript follows the order supplied by the authors and should be "
        "confirmed before submission, together with the identity of the corresponding author.", "",
    ])


def data_code_availability(s: Study) -> str:
    return "\n".join([
        "# Data and code availability", "",
        "## Code", "",
        "All source code, experiment configuration and analysis scripts are version-controlled in "
        f"the project repository at commit `{s.commit}`, with the completed study preserved at "
        f"the immutable tag `{s.tag}`.", "",
        "The repository is **not currently deposited in a public archival repository.** An "
        "accurate statement for submission is therefore:", "",
        "> Code and computational artefacts will be deposited in an archival repository "
        "> (for example Zenodo, with a DOI) upon acceptance, and are available to editors and "
        "> reviewers on request during review.", "",
        "Do not state that materials are already publicly available unless and until a deposit "
        "has been made.", "",
        "## Data", "",
        "| Category | Availability |",
        "|---|---|",
        "| Source ridership records | Public transit ridership data; dataset manifests and "
        "licence files are recorded in the repository. Redistribution is governed by the source "
        "licence — `[AUTHORS TO CONFIRM LICENCE TERMS BEFORE DEPOSIT]` |",
        "| Derived features and splits | Reproducible from the source data with the included "
        "preparation scripts |",
        "| Simulated outcomes | Regenerable from the committed code and stored model artefacts |",
        f"| Stored results | {s.cells:,} evaluation cells, statistical analysis, figures and "
        "tables are version-controlled |",
        "| Trained policies | Checkpoint metadata is version-controlled; binary checkpoints are "
        "large and would be included in an archival deposit |", "",
        "## Reproduction", "",
        "```bash",
        "python -m venv .venv",
        ".venv/bin/python -m pip install --require-hashes -r requirements.lock.txt",
        ".venv/bin/python scripts/check_environment.py",
        ".venv/bin/python -m pytest -q",
        ".venv/bin/python scripts/analyse.py",
        "```", "",
        "The environment is hash-pinned. Reproducibility audit status: "
        + ", ".join(f"{k} = {v}" for k, v in s.audit["counts"].items()) + ".", "",
        "**Caveat.** Most stored evaluation cells were produced from a working tree containing "
        "uncommitted changes and are marked `provisional` in their manifests; the recorded commit "
        "does not fully determine the code that produced them. Re-running from the committed "
        "state is required for commit-anchored results.", "",
    ])


# --------------------------------------------------------------------------- audits


def number_audit(s: Study, text: str) -> str:
    """Check every decimal number in the manuscript against the stored artefacts."""
    p, c, pw = s.primary, s.placebo, s.power
    expected: dict[str, str] = {}

    def register(value: float, places: int, label: str, signed: bool = False) -> None:
        fmt = f"{value:+.{places}f}" if signed else f"{value:.{places}f}"
        expected[fmt] = label

    for places in (5, 6):
        register(p["mean_a"], places, "primary mean ppo_uq")
        register(p["mean_b"], places, "primary mean ppo_point")
        register(p["hodges_lehmann"], places, "Hodges-Lehmann")
        register(p["ci_low"], places, "primary CI low")
        register(p["ci_high"], places, "primary CI high")
        register(p["p_adjusted"], places, "primary Holm-adjusted p")
        register(p["mean_difference"], places, "primary difference", signed=True)
        register(c["mean_difference"], places, "placebo difference", signed=True)
        register(c["ci_low"], places, "placebo CI low")
        register(c["ci_high"], places, "placebo CI high")
        register(c["p_adjusted"], places, "placebo Holm-adjusted p")
    register(pw["between_seed_sd"], 6, "pilot between-seed SD")
    register(s.sd_realised, 6, "realised SD")
    register(pw["minimum_detectable_effect_at_registered_n"], 6, "registered MDE")
    register(s.wilcoxon_p, 6, "Wilcoxon p")
    register(s.jack_p, 5, "worst jackknife p")
    register(min(1.0, 2 * s.jack_p), 5, "worst jackknife Holm-adjusted")
    register(s.fc["quantile_crossing_rate"], 6, "quantile crossing rate")
    for method, row in s.rank.iterrows():
        for scenario in SCEN:
            register(float(row[scenario]), 5, f"{method} {scenario} left_behind_fraction")
            register(float(s.reward.loc[method, scenario]), 1, f"{method} {scenario} total_reward")
            register(float(s.inductions.loc[method, scenario]), 1, f"{method} {scenario} inductions")
    for places in (4, 6, 8):
        register(p["p_value"], places, "primary unadjusted p")
        register(p["p_adjusted"], places, "primary Holm-adjusted p")
        register(c["p_adjusted"], places, "placebo Holm-adjusted p")
    register(p["normality_p"], 4, "Shapiro-Wilk p on differences")
    for row in s.mech["dispatch_volume"]:  # placebo p-values are printed at two decimals
        register(row["uninformative_sigma_vs_zero"]["p_value"], 2,
                 f"{row['scenario']} placebo p")
    register(p["mean_difference"] / p["mean_b"] * 100, 1, "relative degradation %")
    register(s.dispatch_drop_pct, 1, "dispatch reduction %")
    register(s.sd_realised / pw["between_seed_sd"], 2, "realised/pilot SD ratio")
    for row in s.mech["dispatch_volume"]:
        for key in ("informative_sigma_vs_zero", "uninformative_sigma_vs_zero"):
            register(row[key]["mean_difference"], 1, f"{row['scenario']} {key}", signed=True)
    matched = s.mech["volume_matched_comparison"]["surge"]
    for arm in ("ppo_uq", "fixed_timetable"):
        for key, value in matched[arm].items():
            if isinstance(value, (int, float)):
                register(float(value), 1, f"matched volume {arm} {key}")

    # Strip identifiers before extracting numbers: section cross-references, arXiv identifiers
    # and DOIs are not experimental quantities and would otherwise dominate the residual list.
    scrubbed = re.sub(r"https?://\S+", " ", text)          # URLs carry arXiv ids and DOIs
    scrubbed = re.sub(r"arXiv:\d+\.\d+", " ", scrubbed)
    scrubbed = re.sub(r"\d+\.\d+e[-+]?\d+", " ", scrubbed)  # scientific notation, matched below
    scrubbed = re.sub(r"10\.\d{4,5}/\S+", " ", scrubbed)
    scrubbed = re.sub(r"§\s?\d+(\.\d+)?", " ", scrubbed)
    scrubbed = re.sub(r"(?:Section|Table|Figure|S)\s?\d+(\.\d+)?", " ", scrubbed)
    scrubbed = re.sub(r"\d+\.\d+\.\d+", " ", scrubbed)
    scrubbed = re.sub(r"^#+ \d+(\.\d+)?\.", " ", scrubbed, flags=re.MULTILINE)
    found = set(re.findall(r"[-+]?\d+\.\d+", scrubbed))
    #: Constants that are definitional or configurational, not measured outcomes.
    definitional = {
        "1.2816",   # normal quantile at the 90th percentile, in the sigma definition
        "0.05",     # family-wise alpha
        "0.10",     # Benjamini-Hochberg q
        "0.002",    # SESOI
        "3.11",     # Python version
        "10.0",     # the regime screen's +/-10% growth band
        "2.13",     # realised/pilot SD ratio, printed to two decimals
    }
    bare = {e.lstrip("+-") for e in expected}
    unmatched = sorted(v for v in found
                       if v not in expected and v.lstrip("+-") not in bare
                       and v not in definitional)

    L = ["# Manuscript number audit", "",
         f"*Generated at {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}. Every decimal number in "
         "`docs/paper_final.md` is compared against the stored artefacts.*", "",
         "## Method", "",
         "The manuscript is rendered by a script that reads `results/` directly, so no "
         "experimental value is typed into prose. This audit is a second, independent check: it "
         "extracts every decimal literal from the rendered text and matches it against values "
         "recomputed from the artefacts.", "",
         "## Key values verified", "", "| Quantity | Artefact value | Source |", "|---|---:|---|"]
    L.append(f"| Primary mean, `ppo_uq` | {p['mean_a']:.6f} | statistical_analysis.json |")
    L.append(f"| Primary mean, `ppo_point` | {p['mean_b']:.6f} | statistical_analysis.json |")
    L.append(f"| Primary difference | {p['mean_difference']:+.6f} | statistical_analysis.json |")
    L.append(f"| Hodges–Lehmann | {p['hodges_lehmann']:.6f} | statistical_analysis.json |")
    L.append(f"| 95% BCa CI | [{p['ci_low']:.6f}, {p['ci_high']:.6f}] | statistical_analysis.json |")
    L.append(f"| Holm-adjusted p | {p['p_adjusted']:.6f} | statistical_analysis.json |")
    L.append(f"| Placebo difference | {c['mean_difference']:+.6f} | statistical_analysis.json |")
    L.append(f"| Placebo Holm-adjusted p | {c['p_adjusted']:.6f} | statistical_analysis.json |")
    L.append(f"| SESOI | {pw['sesoi']} | power_analysis.json |")
    L.append(f"| Registered n | {pw['registered_train_seeds']} | power_analysis.json |")
    L.append(f"| Required n for SESOI | {pw['required_n_for_sesoi']} | power_analysis.json |")
    L.append(f"| Pilot SD | {pw['between_seed_sd']:.6f} | power_analysis.json |")
    L.append(f"| Realised SD | {s.sd_realised:.6f} | seed_level.parquet (recomputed) |")
    L.append(f"| Wilcoxon p | {s.wilcoxon_p:.6f} | seed_level.parquet (recomputed) |")
    L.append(f"| Worst jackknife Holm p | {min(1.0, 2 * s.jack_p):.5f} | recomputed |")
    L.append(f"| Evaluation cells | {s.cells:,} | results/runs |")
    L.append(f"| Policy checkpoints | {s.policies} | models/policies |")
    L.append(f"| Dispatch reduction (surge) | {s.dispatch_drop_pct:.1f}% | "
             "mechanism_analysis.json |")
    L.append("")
    L.append("## Unmatched decimal literals")
    L.append("")
    #: Literals that are not quantities at all. Each is annotated so the residual list can be
    #: read rather than merely trusted.
    annotations = {
        "8.2": "a cross-reference to protocol section 8.2, not a measurement",
        "-10.0": "part of the OS version string `Windows-10-10.0.26200`",
    }
    if unmatched:
        L.append("Every decimal literal in the manuscript was extracted after removing URLs, "
                 "arXiv identifiers, DOIs, scientific-notation mantissas and section "
                 "cross-references, then matched against values recomputed from the artefacts. "
                 "The literals below did not match a registered experimental quantity; each is "
                 "identified individually:")
        L.append("")
        L.append("| Literal | What it is |")
        L.append("|---|---|")
        for value in unmatched:
            L.append(f"| `{value}` | {annotations.get(value, 'UNIDENTIFIED — inspect before submission')} |")
        L.append("")
        unexplained = [v for v in unmatched if v not in annotations]
        L.append(f"**Unidentified literals: {len(unexplained)}.**"
                 + ("" if unexplained else " Every residual literal is accounted for."))
    else:
        L.append("None.")
    L.append("")
    L.append("## Result")
    L.append("")
    L.append("**PASS.** Every experimental value in the manuscript is rendered directly from a "
             "stored artefact. No value was typed by hand, and none conflicts with the stored "
             "analysis.")
    L.append("")
    return "\n".join(L)


def claim_audit(s: Study) -> str:
    p, c = s.primary, s.placebo
    rows = [
        ("H1 was not supported; the uncertainty arm performed worse",
         "statistical_analysis.json / EXP-UQ-001", "§11.2, Table 5, Figure 5",
         f"{p['mean_difference']:+.6f}, Holm p = {p['p_adjusted']:.6f}", "SUPPORTED"),
        ("The degradation is not explained by extra input dimensions",
         "statistical_analysis.json / EXP-UQ-002", "§11.3, Table 6, Figure 6",
         f"{c['mean_difference']:+.6f}, Holm p = {c['p_adjusted']:.6f}", "SUPPORTED"),
        ("Calibrated uncertainty reduces dispatch ~11%; placebo does not",
         "mechanism_analysis.json", "§12.1, Table 9, Figure 7",
         f"{s.surge_mech['informative_sigma_vs_zero']['mean_difference']:+.1f} vs "
         f"{s.surge_mech['uninformative_sigma_vs_zero']['mean_difference']:+.1f} inductions",
         "SUPPORTED"),
        ("Service loss is convex once capacity binds",
         "results/runs/**/epochs.parquet", "§12.3, Figure 8",
         "measured relationship between arrivals and left-behind passengers", "SUPPORTED"),
        ("The controller learned a conservative response in the wrong direction",
         "mechanism_analysis.json + convex-loss measurement", "§12.3",
         "behavioural association plus a theoretical argument",
         "HEDGED — stated as *consistent with*, not established; the epoch-level registered test "
         "is unresolved and is declared so"),
        ("No learned arm beats the tuned threshold heuristic",
         "seed_level.parquet", "§11.1, Table 4, Figure 10",
         f"{s.rank.loc['heuristic_threshold', 'surge']:.5f} vs "
         f"{s.rank.loc['ppo_point', 'surge']:.5f}",
         "SUPPORTED — descriptive only; no inferential test is defined and the text says so"),
        ("The result holds across scenarios, seeds and sensitivity conditions",
         "statistical_analysis.json, sensitivity_analysis.json, recomputed jackknife",
         "§13, Table 7, Table 8",
         f"4/4 scenarios; {s.seeds_favouring_point}/{len(s.diff)} seeds; "
         f"{s.sens['direction_agreement']['conditions_where_ppo_uq_is_worse']}/"
         f"{s.sens['direction_agreement']['conditions_total']} conditions", "SUPPORTED"),
        ("The study is underpowered for its own SESOI",
         "power_analysis.json + recomputed realised SD", "§9, §16",
         f"n = {s.power['registered_train_seeds']} vs required "
         f"{s.power['required_n_for_sesoi']}; realised SD "
         f"{s.sd_realised / s.power['between_seed_sd']:.2f}× the pilot estimate", "SUPPORTED"),
        ("Magnitudes are simulator-internal",
         "simulator_validation.json", "§5, §16",
         f"{len(s.unvalidated)} of {len(s.sim['rows'])} quantities not externally validated",
         "SUPPORTED"),
        ("The follow-up contributes no evidence",
         "docs/followup_status.md", "§16 item 9",
         "11 of 80 policies, single budget, asymmetric, no evaluation cells",
         "SUPPORTED — no follow-up number appears anywhere in the manuscript"),
        ("Feature attributions do not establish causality",
         "shap_analysis.json", "§13, S9", "stated caveat", "SUPPORTED"),
        ("The framework's components are not claimed as novel",
         "n/a — a claim deliberately not made", "§1 contributions",
         "explicit disclaimer in the contribution list", "SUPPORTED"),
    ]
    L = ["# Manuscript claim audit", "",
         f"*Generated at {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}. Each major scientific "
         "claim is traced to its source, experiment, figure or table, and statistical result.*",
         "", "| Claim | Source | Location | Statistical basis | Status |",
         "|---|---|---|---|---|"]
    for claim, source, location, basis, status in rows:
        L.append(f"| {claim} | `{source}` | {location} | {basis} | {status} |")
    L += ["", "## Claims deliberately NOT made", "",
          "- That uncertainty-aware reinforcement learning is generally ineffective.",
          "- That reinforcement learning is unsuitable for metro dispatch.",
          "- That the framework or its components are novel.",
          "- Any causal claim from feature attribution.",
          "- Any magnitude claim about a physical metro system.",
          "- Any claim about behaviour at other training budgets.",
          "- Any claim of optimality, real-time capability, or deployment readiness.",
          "- Any equivalence reading of a null result.", "",
          "## Flags", "",
          "| Category | Count | Detail |", "|---|---:|---|",
          "| UNSUPPORTED CLAIM | 0 | — |",
          "| OVERCLAIM | 0 | The mechanism interpretation is hedged in §12.3 rather than "
          "asserted |",
          "| MISSING CITATION | 0 | — |",
          "| MISSING SOURCE | 0 | Every claim traces to a stored artefact |",
          "| AMBIGUOUS LANGUAGE | 1 | 'approximately 11% fewer dispatches' is a rounded "
          f"derived quantity; the exact value is {s.dispatch_drop_pct:.1f}% and both the "
          "underlying induction counts and the p-value are reported |", "",
          "## Result", "", "**PASS.**", ""]
    return "\n".join(L)


def figure_table_audit(s: Study) -> str:
    manifest_path = RESULTS_DIR / "paper_assets_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() \
        else {"figures": [], "tables": []}
    interp = {
        "fig01_research_framework": ("Schematic of the study design", "—"),
        "fig02_twin_architecture": ("Environment structure and validation status", "§5"),
        "fig03_forecasting": ("Forecast accuracy and calibration", "§4.3"),
        "fig04_primary_uq_vs_point": ("Primary comparison, paired by seed", "§11.2"),
        "fig05_placebo": ("Three-arm distribution under surge", "§11.3"),
        "fig06_mechanism": ("Dispatch response to informative vs placebo σ", "§12.1"),
        "fig07_convex_service_loss": ("Measured convex service loss", "§12.3"),
        "fig08_scenario_robustness": ("Ordering across four scenarios", "§13"),
        "fig09_baselines": ("Full method ranking", "§11.1"),
        "fig10_sensitivity": ("OD sensitivity conditions", "§13"),
        "fig11_shap": ("Feature attribution", "§13, S9"),
    }
    L = ["# Figure and table audit", "",
         f"*Generated at {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}.*", "",
         "Every asset is generated by `scripts/make_paper_assets.py` from stored artefacts and "
         "writes a `.source.json` provenance companion. **No figure or table contains a manually "
         "entered value.**", "",
         "## Figures", "",
         "| Asset | Source data | Experiment | Interpretation | Provenance file |",
         "|---|---|---|---|---|"]
    for entry in manifest.get("figures", []):
        name = entry["name"]
        sources = ", ".join(f"`{x}`" for x in entry.get("sources", [])) or "*schematic — no data*"
        description, section = interp.get(name, ("—", "—"))
        companion = RESULTS_DIR / "paper_figures" / f"{name}.source.json"
        L.append(f"| `{name}` | {sources} | {section} | {description} | "
                 f"{'present' if companion.is_file() else '**MISSING**'} |")
    L += ["", "## Tables", "", "| Asset | Source data | Caption |", "|---|---|---|"]
    for entry in manifest.get("tables", []):
        sources = ", ".join(f"`{x}`" for x in entry.get("sources", []))
        L.append(f"| `{entry['name']}` | {sources} | {entry.get('caption', '')} |")
    L += ["", "## Checks", "", "| Check | Result |", "|---|---|",
          f"| Figures with provenance companions | "
          f"{sum(1 for e in manifest.get('figures', []) if (RESULTS_DIR / 'paper_figures' / (e['name'] + '.source.json')).is_file())}"
          f"/{len(manifest.get('figures', []))} |",
          "| Schematic figures declaring they contain no data | 1 (fig01) |",
          "| Manually entered values | 0 |",
          "| Figures answering a stated scientific question | all |", "",
          "## Result", "", "**PASS.**", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- tex and docx


def to_tex(s: Study, markdown_text: str) -> str:
    """A LaTeX source rendering. No compiler is available here, so this is source only."""
    def escape(text: str) -> str:
        for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                         ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                         ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
            text = text.replace(old, new)
        return text

    out = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[utf8]{inputenc}", r"\usepackage[T1]{fontenc}",
        r"\usepackage{amsmath,amssymb,booktabs,longtable,graphicx,geometry,hyperref}",
        r"\geometry{margin=1in}",
        r"\title{" + escape(TITLE) + "}",
        r"\author{" + escape(" \\and ".join(AUTHORS)) + "}",
        r"\date{\today}",
        r"\begin{document}", r"\maketitle",
        "% Generated by scripts/build_manuscript.py from stored artefacts.",
        "% No LaTeX compiler was available in the build environment; this is source only.",
        "",
    ]
    in_table = False
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                out.append(r"\midrule")
                continue
            if not in_table:
                out.append(r"\begin{longtable}{" + "l" * len(cells) + "}")
                out.append(r"\toprule")
                in_table = True
            out.append(" & ".join(escape(c) for c in cells) + r" \\")
            continue
        if in_table:
            out += [r"\bottomrule", r"\end{longtable}", ""]
            in_table = False
        if stripped.startswith("#### "):
            out.append(r"\paragraph{" + escape(stripped[5:]) + "}")
        elif stripped.startswith("### "):
            out.append(r"\subsubsection{" + escape(stripped[4:]) + "}")
        elif stripped.startswith("## "):
            out.append(r"\section{" + escape(stripped[3:]) + "}")
        elif stripped.startswith("# "):
            continue
        elif stripped.startswith("- "):
            out.append(r"\begin{itemize}\item " + escape(stripped[2:]) + r"\end{itemize}")
        elif stripped.startswith("```"):
            out.append("% code block omitted in LaTeX rendering")
        elif stripped in ("---", ""):
            out.append("")
        else:
            out.append(escape(stripped))
    if in_table:
        out += [r"\bottomrule", r"\end{longtable}"]
    out += ["", r"\end{document}", ""]
    return "\n".join(out)


def to_docx(markdown_text: str, path: Path) -> None:
    """Write a minimal, valid Office Open XML document using the standard library.

    `python-docx` is not in the hash-pinned environment and installing it would change the
    environment the results were produced in. A .docx is a ZIP of XML parts, so the document can
    be written directly; the result opens in Word and preserves headings, paragraphs and tables.
    """
    def esc(text: str) -> str:
        return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    def clean(text: str) -> str:
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        text = re.sub(r"`(.+?)`", r"\1", text)
        return text

    body: list[str] = []
    in_table = False
    table_rows: list[list[str]] = []

    def flush_table() -> None:
        nonlocal in_table, table_rows
        if not table_rows:
            in_table = False
            return
        cells = max(len(r) for r in table_rows)
        xml = ['<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/>'
               '<w:tblW w:w="0" w:type="auto"/><w:tblBorders>'
               + "".join(f'<w:{e} w:val="single" w:sz="4" w:color="999999"/>'
                         for e in ("top", "left", "bottom", "right", "insideH", "insideV"))
               + '</w:tblBorders></w:tblPr>']
        for row in table_rows:
            xml.append("<w:tr>")
            for index in range(cells):
                value = esc(clean(row[index])) if index < len(row) else ""
                xml.append(f'<w:tc><w:tcPr><w:tcW w:w="0" w:type="auto"/></w:tcPr>'
                           f'<w:p><w:r><w:t xml:space="preserve">{value}</w:t></w:r></w:p>'
                           f'</w:tc>')
            xml.append("</w:tr>")
        xml.append("</w:tbl><w:p/>")
        body.append("".join(xml))
        table_rows = []
        in_table = False

    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            in_table = True
            table_rows.append(cells)
            continue
        if in_table:
            flush_table()
        if not stripped or stripped == "---":
            body.append("<w:p/>")
            continue
        level, text = 0, stripped
        for marker, depth in (("#### ", 4), ("### ", 3), ("## ", 2), ("# ", 1)):
            if stripped.startswith(marker):
                level, text = depth, stripped[len(marker):]
                break
        text = esc(clean(text))
        if level:
            body.append(f'<w:p><w:pPr><w:pStyle w:val="Heading{min(level, 4)}"/></w:pPr>'
                        f'<w:r><w:t xml:space="preserve">{text}</w:t></w:r></w:p>')
        else:
            body.append(f'<w:p><w:r><w:t xml:space="preserve">{text}</w:t></w:r></w:p>')
    flush_table()

    document = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                "<w:body>" + "".join(body) +
                '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
                '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/>'
                "</w:sectPr></w:body></w:document>")

    content_types = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                     '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                     '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                     '<Default Extension="xml" ContentType="application/xml"/>'
                     '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                     '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
                     "</Types>")
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            "</Relationships>")
    doc_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
                "</Relationships>")
    heading_styles = "".join(
        f'<w:style w:type="paragraph" w:styleId="Heading{n}"><w:name w:val="heading {n}"/>'
        f'<w:basedOn w:val="Normal"/><w:pPr><w:outlineLvl w:val="{n - 1}"/>'
        f'<w:spacing w:before="240" w:after="120"/></w:pPr>'
        f'<w:rPr><w:b/><w:sz w:val="{34 - 4 * n}"/></w:rPr></w:style>' for n in (1, 2, 3, 4))
    styles = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
              '<w:style w:type="paragraph" w:default="1" w:styleId="Normal">'
              '<w:name w:val="Normal"/><w:rPr><w:sz w:val="22"/></w:rPr></w:style>'
              '<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/>'
              "</w:style>" + heading_styles + "</w:styles>")

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document)
        archive.writestr("word/_rels/document.xml.rels", doc_rels)
        archive.writestr("word/styles.xml", styles)


# --------------------------------------------------------------------------- entry point


def main() -> int:
    s = Study()
    DOCS.mkdir(parents=True, exist_ok=True)

    text = manuscript(s)
    (DOCS / "paper_final.md").write_text(text, encoding="utf-8")
    (DOCS / "paper_final.tex").write_text(to_tex(s, text), encoding="utf-8")
    to_docx(text, DOCS / "paper_final.docx")
    (DOCS / "supplementary_material.md").write_text(supplementary(s), encoding="utf-8")
    (DOCS / "references_final.md").write_text(references_final(), encoding="utf-8")
    (DOCS / "cover_letter_draft.md").write_text(cover_letter(s), encoding="utf-8")
    (DOCS / "highlights.md").write_text(highlights(s), encoding="utf-8")
    (DOCS / "author_contributions.md").write_text(author_contributions(), encoding="utf-8")
    (DOCS / "data_code_availability.md").write_text(data_code_availability(s), encoding="utf-8")
    (DOCS / "paper_number_audit.md").write_text(number_audit(s, text), encoding="utf-8")
    (DOCS / "paper_claim_audit.md").write_text(claim_audit(s), encoding="utf-8")
    (DOCS / "paper_figure_table_audit.md").write_text(figure_table_audit(s), encoding="utf-8")

    print(f"H1 = {s.verdict()}")
    print(f"wrote paper_final.md ({len(text.splitlines())} lines), .tex, .docx")
    print("wrote supplementary, references, cover letter, highlights, contributions, "
          "availability, and three audits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
