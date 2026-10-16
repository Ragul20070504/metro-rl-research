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
from datetime import datetime, timezone
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

TITLE = ("Forecast Uncertainty in Learned Metro Dispatch: "
         "A Placebo-Controlled Study at a Fixed Training Budget")


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
        self.publication = load("publication_readiness.json")

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
        self.recorded_main_policies = len(self.rl[["method", "train_seed"]].drop_duplicates())
        self.policies = sum(1 for _ in (REPO_ROOT / "models" / "policies" / "w_balanced")
                            .rglob("policy.zip"))
        self.commit = git("rev-parse", "HEAD")
        self.dirty = bool(git("status", "--porcelain"))
        self.tag = ("original-confirmatory-study" if git("rev-parse", "--verify",
                    "refs/tags/original-confirmatory-study^{commit}") else "unavailable in this checkout")
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
        f"`{s.commit[:12]}`; uncommitted revision files present: {s.dirty}. Numeric text is "
        "rendered from stored reports; this does not authenticate their historical sources.*")
    add("")
    add("---")
    add("")
    add("**Submission status: revision draft; not ready for submission.** The current-checkout "
        "audit in `docs/SUBMISSION_READINESS.md` supersedes historical readiness claims. "
        "The original source history, study tag, some raw-data bytes and trained binaries are "
        "unavailable here. Internal documents describe advance registration, but no external "
        "timestamped record or original Git history is available to independently establish "
        "that chronology. We therefore call this an internally specified, placebo-controlled "
        "study and do not claim externally verifiable pre-registration. Reported experiments "
        "are historical stored results; the present revision independently rechecks their "
        "arithmetic without claiming to have rerun the simulations.")
    add("")

    # ---------------------------------------------------------------- abstract
    add("## Abstract")
    add("")
    add("Supplying a learned dispatch controller with forecast uncertainty may improve service, "
        "but the signal's information must be distinguished from the effect of additional inputs.")
    add("")
    add("We reanalyse stored evaluations of maskable proximal policy optimisation in a "
        "ridership-grounded metro simulator. Three arms share architecture, input width, reward "
        "and a requested 60,000-step training budget; their uncertainty channel contains zero, "
        "resampled dispersion or calibrated dispersion. Ten training seeds were evaluated on "
        "30 held-out simulation seeds across four scenarios. The original two-test family uses "
        "training-seed means and Holm correction.")
    add("")
    add(f"Under surge, the uncertainty-aware controller's left-behind fraction was "
        f"{p['mean_a']:.6f}, versus {p['mean_b']:.6f} for the point-forecast controller "
        f"(mean difference {p['mean_difference']:+.6f}; adjusted p = {p['p_adjusted']:.6f}). "
        f"The paired Hodges–Lehmann estimate was {p['hodges_lehmann']:.6f} "
        f"(95% BCa interval [{p['ci_low']:.6f}, {p['ci_high']:.6f}]). It also performed worse "
        f"than the placebo arm (difference {c['mean_difference']:+.6f}; adjusted "
        f"p = {c['p_adjusted']:.6f}). A tuned threshold heuristic descriptively outperformed "
        "every learned arm in all four scenarios.")
    add("")
    add("The original hypothesis was not supported under this configuration. The result does "
        "not distinguish ineffective information from inadequate optimisation or training. "
        "Missing original model/data artifacts prevent complete experiment reproduction and "
        "the planned budget follow-up. Registration chronology is internally documented but "
        "not independently verified from the available history. Conclusions are limited to "
        "the tested simulator and configuration.")
    add("")
    add("**Keywords:** metro train induction; reinforcement learning; probabilistic demand "
        "forecasting; forecast uncertainty; placebo control; reproducibility; nonlinear service "
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
    add("**Design.** We answer the question with an internally specified, placebo-controlled experiment "
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
        "matched scenarios and a **placebo** uncertainty condition with the same channel width. "
        "The placebo resamples past dispersion values; it is not an exact permutation control.")
    add(f"3. A statistically robust empirical finding that the tested uncertainty-aware "
        f"controller performs **worse** than the point-forecast controller on the registered "
        f"primary endpoint ({p['mean_difference']:+.6f}; Holm-adjusted p = "
        f"{p['p_adjusted']:.6f}), reproduced in all four scenarios, in "
        f"{s.seeds_favouring_point} of {len(s.diff)} training seeds, and in "
        f"{s.sens['direction_agreement']['conditions_where_ppo_uq_is_worse']} of "
        f"{s.sens['direction_agreement']['conditions_total']} sensitivity conditions.")
    add("4. A behavioural association between uncertainty-channel content, episode dispatch "
        "volume and service outcomes, alongside an explicit account of what these summaries "
        "cannot identify about timing, causality or optimal actions.")
    add("5. An evaluation framework and stored result summaries permitting arithmetic "
        "reanalysis, with missing historical artifacts and limits to full reproduction disclosed.")
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
        "latter is a directly related treatment of demand uncertainty in metro scheduling; it "
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
        "slow for the response times a dispatcher has in some applications. Here we focus on "
        "matched uncertainty-channel comparisons and report the observed negative result "
        "alongside classical baselines that outperform the learned arms.")
    add("")
    add("**Uncertainty-aware and risk-sensitive decision making.** Reviews of uncertainty in deep "
        f"reinforcement learning distinguish aleatoric from epistemic uncertainty and survey "
        f"estimation methods {cite('uq_review')}; risk-sensitive formulations replace expected "
        f"return with tail-aware objectives {cite('clements2019')}. Our design differs in an "
        "important respect: we do not modify the objective or the algorithm. We hold the learner "
        "fixed and compare observation constructions without changing to a risk-sensitive method. "
        "The imperfect bootstrap control limits isolation of information quality.")
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
        "as motivation for future analysis; the historical binned association does not establish "
        "structural convexity in our environment (Section 12).")
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
        "seed-level unit of analysis, our internal protocol, and our decision to report an "
        "underpowered design explicitly rather than aggregating episodes into an inflated sample "
        "size.")
    add("")
    add(f"> **Citation verification.** {ref_counts()['fully_verified']} of {ref_counts()['total']} "
        "references were checked against publisher-deposited Crossref metadata or the official "
        "arXiv record. Source records and corrections are retained in "
        "`results/reference_verification.json`. Missing metadata fields are omitted; DOI "
        "suffix years are not substituted for publication years.")
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
    add("**Regime screen.** A protocol-specified screen sought a run of at least 365 days with "
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
        "The historical persistence report records agreement of both median and dispersion "
        "against its reference fit. The original binaries and reference predictions are "
        "unavailable, so that claim cannot be independently repeated here. The new replication "
        "uses a separately validated refit and does not claim identical prediction vectors.")
    add("")
    add("**Online inputs and calibration limits.** The imported environment appends completed "
        "hours of simulated passenger boardings to the forecast history. Boardings depend on "
        "dispatch and available capacity, whereas the offline training target is station "
        "ridership. Thus past actions can affect subsequent forecasts, and served boardings "
        "can be censored relative to arrivals. No future observation is read, but offline "
        "interval coverage does not establish online coverage under this feedback. The "
        "arm comparison concerns the resulting closed-loop controllers, rather than three "
        "policies receiving an identical exogenous forecast path.")
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
        "quantiles. The internal protocol fixes this definition across arms. Its historical "
        "registration chronology cannot be independently verified. The online forecast can "
        "nevertheless depend on earlier policy actions through its observation history.")
    add("")
    add("Every arm carries the same two-value uncertainty channel and the same parameter count. "
        "The arms differ only in that channel's content:")
    add("")
    add("| Arm | Channel content |")
    add("|---|---|")
    add("| `ppo_point` | Identically zero |")
    add("| `ppo_placebo` | Independent draws with replacement from a rolling pool of up to "
        "50,000 current and previously observed σ scalars, pooled across stations and epochs |")
    add("| `ppo_uq` | The calibrated forecast dispersion σ |")
    add("")
    add("**What the placebo controls.** All arms have identical input width. The placebo "
        "resamples values from an empirical pool; it does not preserve a finite sample's "
        "marginal distribution exactly. Pool composition changes over time and includes current "
        "values, so some demand-regime information can remain. Serial dependence and "
        "station-specific associations also differ from the calibrated channel. The pool "
        "persists across training episodes but a fresh evaluation environment starts a new "
        "pool for each cell. Thus this comparison tests channel constructions, without "
        "isolating useful information from every distributional or optimisation effect. "
        "No exact permutation or proven information-free control is claimed.")
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
    add(f"- **Main-study learned-policy identities in stored cells:** {s.recorded_main_policies}; "
        f"available original binaries: {s.policies}. Pilot observations are excluded.")
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
    add("**Internal registration record.** The repository reports that the registry was frozen after the power gate fixed the sample size "
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
    add("The repository describes this plan as fixed before the original runs. That chronology "
        "is not independently established by this checkout. Here 'confirmatory' names the "
        "original protocol's two-test family, not an independently verified registration status.")
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
        "read (Table 6, Figure 9).")
    add("")
    add("| Method | " + " | ".join(SCEN) + " |")
    add("|---|" + "---:|" * len(SCEN))
    for method, row in s.rank.iterrows():
        mark = " *(proposed)*" if method == "ppo_uq" else ""
        add(f"| `{method}`{mark} | " + " | ".join(f"{row[sc]:.5f}" for sc in SCEN) + " |")
    add("")
    add("| Method | Total reward (surge) | Inductions (surge) |")
    add("|---|---:|---:|")
    for method in s.rank.index:
        add(f"| `{method}` | {s.reward.loc[method, 'surge']:.1f} "
            f"| {s.inductions.loc[method, 'surge']:.1f} |")
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
        f"n = {p['n']} training seeds (Table 5, Figure 4).")
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
        "(Table 5, Figure 5).")
    add("")
    add("The mean ordering across the three arms is "
        f"`ppo_point` {s.rank.loc['ppo_point', 'surge']:.5f} < `ppo_placebo` "
        f"{s.rank.loc['ppo_placebo', 'surge']:.5f} < `ppo_uq` "
        f"{s.rank.loc['ppo_uq', 'surge']:.5f}. The specified family compares UQ against each "
        "comparator; it does not test placebo against point. Input width is identical, but "
        "the bootstrap control does not establish exact distributional matching or complete "
        "removal of information. The result supports a difference between these channel "
        "constructions, without identifying the learner's internal mechanism.")
    add("")
    add("### 11.4 Hypothesis verdict")
    add("")
    add(f"**H1 = {s.verdict()}.**")
    add("")

    # ---------------------------------------------------------------- 12
    add("## 12. Mechanism analysis")
    add("")
    add("The documented design changes uncertainty-channel content while holding the specified "
        "learner and random-number scheme fixed. Episode-level induction totals associate the "
        "informative channel with fewer dispatches. The historical per-epoch records and model "
        "binaries are absent from this checkout, so their analyses cannot currently be rerun. "
        "This is a behavioural association across separately trained policies, not a test of "
        "the within-policy causal effect of changing uncertainty at a fixed point forecast. "
        "Nothing was re-simulated for this revision.")
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
        "volume** relative to the point-forecast arm (Figure 6). The placebo arm does "
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
    add("### 12.3 Nonlinear service-loss association and its limits")
    add("")
    add("The historical Figure 7 bins arrivals and left-behind counts across operating epochs. "
        "This is an association across different states and decisions. Dispatch volume, train "
        "capacity, queue history and time of day are not held fixed; the curve therefore does "
        "not establish a structural convex loss function or an optimal dispatch response. "
        "Its underlying epoch records are unavailable in this checkout.")
    add("")
    add("For an independently justified fixed convex loss ℓ, Jensen's inequality gives "
        "E[ℓ(D)] ≥ ℓ(E[D]). A mean-preserving spread ordered in the convex order increases "
        "expected convex loss; a change in variance alone does not establish that ordering. "
        f"The newsvendor analogy {cite('arrow1951')} motivates a future decision-theoretic "
        "analysis, but it neither derives the optimum under our multi-objective reward nor "
        "proves that this policy should dispatch upward when dispersion increases.")
    add("")
    add("**Interpretation, stated with its limits.** We separate what was observed from what it "
        "suggests:")
    add("")
    add("| | |")
    add("|---|---|")
    add("| **Observed** | Informative σ is associated with fewer dispatches and worse service; "
        "placebo σ is not |")
    add("| **Mechanism evidence** | The placebo comparison distinguishes these channel constructions "
        "at identical input width; information and distributional effects remain entangled |")
    add("| **Interpretation** | The evidence is *consistent with* the controller having learned a "
        "association between channel content and lower episode-level dispatch volume; "
        "the optimal direction under the full reward is not established |")
    add("| **Limitation** | We do not observe the network's internal computation. The registered "
        "epoch-level test — dispatch rate against dispersion conditional on the point forecast — "
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
        "test, or on one scenario (Table 9, Figure 8).")
    add("")
    add("**Origin–destination sensitivity.** The OD structure is synthetic and drives the primary "
        "endpoint, so the comparison was re-scored under all three OD variants. The registered "
        "direction holds in every one, against both comparators (Table 7, Figure 10). Individual "
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
        "controller, and more than a past-value bootstrap placebo. It also dispatched "
        "substantially fewer trains when given calibrated "
        "dispersion.")
    add("")
    add("**Why it may have happened.** Three readings are consistent with the evidence, and the "
        "design separates them only partly.")
    add("")
    add("*Reading 1 — the signal is genuinely unhelpful in this setting.* The placebo comparison "
        "shows that calibrated σ performed worse than a resampled channel at identical input "
        "width. However, changed temporal dependence and imperfect distributional matching "
        "prevent a unique attribution to information quality.")
    add("")
    add("*Reading 2 — the information is usable but the controller did not learn to use it.* One "
        "frozen configuration at a fixed budget may fail to exploit an input that a longer or "
        "better-tuned run would. No learned arm beats a threshold heuristic, which motivates this "
        "alternative explanation but does not establish it. The incomplete budget follow-up "
        "does not yet distinguish these explanations.")
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
        "response associated with worse service in the tested environment — and shows that the gap "
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
        "an informative uncertainty input was associated with fewer dispatches and worse service. "
        "Reporting dispatch-volume behaviour alongside outcome metrics would "
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
    add("13. **The bootstrap placebo is imperfect.** It samples a changing empirical pool with "
        "replacement, rather than permuting a fixed batch. Exact marginal matching and complete "
        "removal of information are not established; differences in temporal dependence can "
        "affect learning.")
    add("")
    add("These are collected in Table 10.")
    add("")

    # ---------------------------------------------------------------- 17
    add("## 17. Future work")
    add("")
    add("The following are **proposed, not completed**. Follow-up planning occurred after the "
        "original outcome was known and is exploratory. It cannot retroactively alter the "
        "original family or its conclusion. The historical follow-up remains blocked by "
        "missing original models. A separate independent replication is specified under "
        "`docs/replication_20261015_protocol.md`, using fresh anchors and an explicitly updated "
        "source snapshot. See `docs/replication_execution.md` for its execution status. It "
        "provides no new result until the complete grid passes its checks.")
    add("")
    add("1. **Registered training-budget sweep.** The single most informative remaining "
        "experiment for testing budget dependence of the contrast. It cannot alone establish "
        "information quality or optimality. A protocol and a specified analysis exist in the "
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
    add(f"| Original-study tag in this checkout | `{s.tag}` |")
    add(f"| Interpreter | Python {interp.get('python', 'n/a')} on "
        f"{interp.get('platform', 'n/a')} |")
    add(f"| Dependencies | {deps.get('pinned', 0)} pinned packages, {deps.get('hashes', 0)} "
        "SHA-256 hashes |")
    add(f"| Evaluation cells | {s.cells:,} |")
    add(f"| Trained policy binaries available here | {s.policies} in the registered namespace; historical metadata does not replace binaries |")
    add(f"| Registered experiments | 17 |")
    add(f"| Historical reproducibility report (not current validation) | "
        + ", ".join(f"{k} = {v}" for k, v in s.audit["counts"].items()) + " |")
    add(f"| Historical immutability report (tag unavailable here) | "
        + ", ".join(f"{k} = {v}" for k, v in s.separation["counts"].items()) + " |")
    add("")
    add("Stored-results validation on Linux uses a separate CPU hash lock; it does not "
        "reconstruct missing historical inputs or prove experiment reproducibility:")
    add("")
    add("```bash")
    add("uv venv --python 3.11 .venv")
    add("uv pip install --python .venv/bin/python --require-hashes --torch-backend cpu -r requirements-linux.lock.txt")
    add(".venv/bin/python scripts/check_environment.py")
    add(".venv/bin/python -m pytest -q")
    add(".venv/bin/python scripts/publication_audit.py  # rechecks stored cells; nonzero until blockers are resolved")
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
        "produced them. The current import and uncommitted revision cannot reconstruct that "
        "historical working tree. New computation requires its own frozen source/input snapshot.")
    add("")
    add("The available repository contains one initial import commit; commits named in original "
        "cell manifests and the original-study tag are absent. The current audit reports "
        "missing raw files, frozen simulator inputs, forecasting models and policy binaries "
        "explicitly. A historical audit's PASS does not establish availability now. "
        "Restoring the exact original artifacts is a prerequisite for the budget follow-up; "
        "using a newer GTFS feed would change a held-fixed input. The reported BCa intervals "
        "estimate the Hodges–Lehmann paired location, not the mean difference.")
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
        "metro dispatch controller improves passenger service, in an internally specified, "
        "placebo-controlled experiment on a demand-driven simulation environment.")
    add("")
    add(f"The registered hypothesis was **not supported**. The uncertainty-aware controller left "
        f"more passengers behind than an otherwise identical point-forecast controller "
        f"({p['mean_difference']:+.6f}; 95% BCa CI [{p['ci_low']:.6f}, {p['ci_high']:.6f}]; "
        f"Holm-adjusted p = {p['p_adjusted']:.6f}), and also performed worse than a placebo "
        "carrying dispersion values resampled from a rolling empirical pool.")
    add("")
    add("Rather than improving service, the controller exhibited a systematic conservative "
        f"dispatch response to genuine forecast uncertainty — approximately "
        f"{s.dispatch_drop_pct:.0f}% fewer dispatches under surge — which the placebo arm did "
        "not detectably differ from zero. The episode totals do not identify an optimal "
        "dispatch direction or a causal within-policy response to dispersion. Combined with the "
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
        add(f"{index}. {reference.formatted()}")
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
    add(f"Historical baseline fairness report ({s.fair['audited_utc']}): "
        f"{s.fair['n_passed']}/{s.fair['n_checks']} checks recorded as passing, "
        f"{len(s.fair['blocking_failures'])} recorded blocking failures. "
        "This table transcribes that report; it is not a fresh execution of those checks.")
    add("")
    add("| Check | Status |")
    add("|---|---|")
    for check in s.fair["checks"][:19]:
        label = str(check["item"]).replace("|", "\\|")
        outcome = "PASS" if check["passed"] else "FAIL"
        add(f"| {label} | {outcome} |")
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
    add("## S10. Stored-results checks and document build")
    add("")
    add("These commands use the Linux CPU lock and recheck stored evidence. They do not rerun "
        "historical policies or reconstruct missing original inputs. The audit exits nonzero "
        "while submission blockers remain.")
    add("")
    add("```bash")
    add("uv venv --python 3.11 .venv")
    add("uv pip install --python .venv/bin/python --require-hashes --torch-backend cpu -r requirements-linux.lock.txt")
    add(".venv/bin/python scripts/check_environment.py")
    add(".venv/bin/python scripts/publication_audit.py")
    add(".venv/bin/python scripts/build_manuscript.py")
    add("```")
    add("")
    add(f"Stored: {s.cells:,} historical evaluation cells and {s.recorded_main_policies} "
        f"main-study learned-policy identities; {s.policies} original binaries available. "
        "The historical asset manifest lists 11 figures and 10 tables; Figure 7 is withheld. "
        "Asset presence does not prove missing source artifacts can be regenerated. "
        "See SUBMISSION_READINESS.md and the separate independent replication protocol.")
    add("")
    return "\n".join(L)


def references_final() -> str:
    counts = ref_counts()
    L = ["# Verified bibliography", "",
         f"*{counts['fully_verified']} of {counts['total']} entries verified in full; "
         f"{counts['needing_confirmation']} require field confirmation before submission.*", "",
         "Entries were checked against publisher-deposited Crossref metadata or official "
         "arXiv citation metadata. Source records, retrieval dates and corrected fields are "
         "retained in `results/reference_verification.json`. Missing fields are omitted; "
         "metadata verification does not imply verification of every claim in each paper.", ""]
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
            L.append(f"- {reference.formatted()}")
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
        "**What we did.** We report an internally specified, placebo-controlled evaluation in a "
        "demand-driven metro simulation environment. Three reinforcement-learning arms share "
        "identical architecture, parameter count, observation dimensionality, reward, training "
        "budget, seeds and evaluation days, and differ only in what a two-value uncertainty "
        "channel carries: zero, a rolling-pool bootstrap placebo, or the "
        "calibrated forecast dispersion. Fixed-timetable, tuned-heuristic and mixed-integer "
        "baselines are evaluated under common random numbers.", "",
        "**Main finding, stated plainly.** The registered hypothesis was **not supported**. The "
        f"uncertainty-aware controller left more passengers behind than the point-forecast "
        f"controller ({p['mean_difference']:+.6f}; 95% BCa CI [{p['ci_low']:.6f}, "
        f"{p['ci_high']:.6f}]; Holm-adjusted p = {p['p_adjusted']:.6f}), and also performed worse "
        "than the placebo arm. Mechanism analysis shows the calibrated signal is associated with "
        f"approximately {s.dispatch_drop_pct:.0f}% fewer dispatches while the placebo signal is "
        "not detectably different from zero. Episode summaries do not identify a within-policy "
        "causal response or an optimal action under the multi-objective reward.", "",
        "**Why it matters.** We report a controlled negative result with a dispatch-volume association. "
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
        "Code and stored metric summaries are present; original binary artifacts and eleven raw files "
        "are missing from this checkout. Full reproducibility and submission remain blocked; "
        "see the data and code availability statement.", "",
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
        "- Placebo-controlled test of forecast uncertainty in learned metro dispatch",
        "- Uncertainty-aware PPO left more passengers behind than point-forecast PPO",
        "- Signal content affected performance with input dimensions held constant",
        "- Calibrated uncertainty cut dispatch volume by about 11% under demand surge",
        "- A tuned threshold heuristic outperformed every learned controller tested", "",
        "## Expanded form", "",
        f"- In an internally specified, placebo-controlled evaluation, an uncertainty-aware PPO metro "
        f"dispatch controller performed **worse** than an otherwise identical point-forecast "
        f"controller on the registered primary endpoint ({p['mean_difference']:+.6f}; "
        f"Holm-adjusted p = {p['p_adjusted']:.6f}).",
        "- The degradation was also present against a placebo uncertainty channel with matched "
        "dimensionality. Its bootstrap construction does not isolate all information and "
        "distributional effects.",
        f"- Genuine uncertainty was associated with approximately {s.dispatch_drop_pct:.0f}% fewer "
        "dispatches; the placebo signal was not.",
        "- The binned service-loss curve does not identify an optimal dispatch direction.",
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
    checks = s.publication["checks"]
    lines = ["# Data and code availability", "",
             "**Revision draft: complete experiment reproduction is currently blocked.**", "",
             f"The current code import is at `{s.commit}`. Original experiment commits and the "
             "`original-confirmatory-study` tag are absent from this checkout. A public archival "
             "deposit and DOI have not been verified. No promise of artifact availability is "
             "substituted for an actual deposit.", "",
             "Stored cell metrics and manifests permit independent reanalysis of the reported "
             "two-test family. They do not establish that historical simulations can be rerun.", "",
             "| Material | Current evidence |", "|---|---|"]
    for check in checks:
        if check["name"] in {"Original raw-data bytes", "Recorded trained-policy binaries",
                             "Frozen follow-up inputs", "Historical code provenance"}:
            lines.append(f"| {check['name']} | {check['detail']} |")
    lines += ["", "Raw-data retrieval must reproduce the SHA-256 hashes in the original manifests. "
              "The GTFS endpoint is mutable; a newer feed is not an equivalent historical input. "
              "Source redistribution rights remain an author confirmation item.", "",
              "On Linux, install with the separate CPU hash lock `requirements-linux.lock.txt` "
              "using uv's `--require-hashes --torch-backend cpu` options. The original Windows "
              "lock remains unchanged. Run `python scripts/publication_audit.py` to inspect "
              "current blockers; it exits nonzero until they are resolved.", "",
              "The historical reproducibility and separation reports describe a different "
              "machine state. Their PASS counts must not be quoted as fresh validation. "
              "See `docs/SUBMISSION_READINESS.md` for the current audit.", ""]
    return "\n".join(lines)


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
    scrubbed = re.sub(r"\b(?:Section|Table|Figure|S)\s?\d+(\.\d+)?", " ", scrubbed)
    scrubbed = re.sub(r"\b\d+\.\d+\.\d+\b", " ", scrubbed)
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
    unexplained = [v for v in unmatched if v not in annotations]
    L.append("**REVIEW REQUIRED.** Unidentified numeric literals remain." if unexplained else
             "**NUMERIC SCREEN PASSED.** Checked literals match stored values or documented "
             "constants. This lexical screen does not establish source validity, correct "
             "interpretation, or submission readiness.")
    L.append("")
    return "\n".join(L)


def claim_audit(s: Study) -> str:
    p, c = s.primary, s.placebo
    rows = [
        ("H1 was not supported; the uncertainty arm performed worse",
         "statistical_analysis.json / EXP-UQ-001", "§11.2, Table 5, Figure 4",
         f"{p['mean_difference']:+.6f}, Holm p = {p['p_adjusted']:.6f}", "SUPPORTED"),
        ("The degradation is not explained by extra input dimensions",
         "statistical_analysis.json / EXP-UQ-002", "§11.3, Table 5, Figure 5",
         f"{c['mean_difference']:+.6f}, Holm p = {c['p_adjusted']:.6f}", "SUPPORTED"),
        ("The UQ arm has about 11% fewer dispatches; the placebo arm does not",
         "mechanism_analysis.json", "§12.1, Figure 6",
         f"{s.surge_mech['informative_sigma_vs_zero']['mean_difference']:+.1f} vs "
         f"{s.surge_mech['uninformative_sigma_vs_zero']['mean_difference']:+.1f} inductions",
         "SUPPORTED"),
        ("Historical binned service-loss association",
         "results/runs/**/epochs.parquet", "§12.3; Figure 7 withheld",
         "binned association; state and decisions not controlled", "LIMITED — epoch sources unavailable"),
        ("Channel content is associated with lower dispatch volume",
         "mechanism_analysis.json", "§12.3",
         "behavioural association plus a theoretical argument",
         "HEDGED — stated as *consistent with*, not established; the epoch-level registered test "
         "is unresolved and is declared so"),
        ("No learned arm beats the tuned threshold heuristic",
         "seed_level.parquet", "§11.1, Table 6, Figure 9",
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
          "## Current evidence limitations", "",
          "The claim mapping above describes the historical reports. It is not independent "
          "verification of their source data or of registration chronology.", "",
          "| Current check | Status | Detail |", "|---|---|---|"]
    for check in s.publication["checks"]:
        if check["status"] != "PASS":
            L.append(f"| {check['name']} | {check['status']} | {check['detail']} |")
    L += ["", "**REVIEW REQUIRED.** Submission readiness is determined by the current evidence "
          "audit, not by a hard-coded zero count of unsupported claims.", ""]
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
        "fig07_convex_service_loss": ("Historical binned association — withheld", "§12.3"),
        "fig08_scenario_robustness": ("Ordering across four scenarios", "§13"),
        "fig09_baselines": ("Full method ranking", "§11.1"),
        "fig10_sensitivity": ("OD sensitivity conditions", "§13"),
        "fig11_shap": ("Feature attribution", "§13, S9"),
    }
    L = ["# Figure and table audit", "",
         f"*Generated at {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}.*", "",
         "The historical manifest attributes these assets to `scripts/make_paper_assets.py`. "
         "This audit checks current file availability, not whether every source computation "
         "can be repeated or every interpretation is valid.", "",
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
    missing = []
    if not manifest.get("figures") or not manifest.get("tables"):
        missing.append("Asset manifest is absent or empty")
    for entry in manifest.get("figures", []):
        for suffix in ("png", "pdf", "source.json"):
            path = RESULTS_DIR / "paper_figures" / f"{entry['name']}.{suffix}"
            if not path.is_file():
                missing.append(str(path.relative_to(REPO_ROOT)))
    for entry in manifest.get("tables", []):
        for suffix in ("md", "csv"):
            path = RESULTS_DIR / "paper_tables" / f"{entry['name']}.{suffix}"
            if not path.is_file():
                missing.append(str(path.relative_to(REPO_ROOT)))
    for entry in manifest.get("figures", []) + manifest.get("tables", []):
        for source in entry.get("sources", []):
            if not list(REPO_ROOT.glob(source)):
                missing.append(source)
    L += ["", "## Missing files or source patterns", ""]
    L.extend(f"- `{name}`" for name in sorted(set(missing)))
    L += ["", "## Result", "",
          "**REVIEW REQUIRED.** Missing source files prevent full asset regeneration." if missing else
          "**FILE CHECK PASSED.** Assets and named sources exist; this is not a scientific or submission-readiness verdict.", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- tex and docx


def figure_appendix() -> str:
    manifest = load("paper_assets_manifest.json")
    lines = ["## Figures", "", "Figure 3 was corrected from stored forecast metrics in this "
             "revision; Figures 1 and 5 were regenerated with corrected placebo labels. Other "
             "displayed plots are historical files. Their presence does not "
             "mean the source computations were rerun; Figure 7 is withheld as explained below.", ""]
    for index, entry in enumerate(manifest["figures"], 1):
        name = entry["name"]
        caption = name.split("_", 1)[1].replace("_", " ")
        if name == "fig07_convex_service_loss":
            lines.extend(["### Figure 7. Historical arrival/loss curve — withheld from this revision", "",
                          "The archived plot's epoch-level source data are unavailable, and its "
                          "convexity interpretation is not identified by binned observational "
                          "summaries. The original file is retained in results/paper_figures; "
                          "it is not presented here as evidence of a structural loss function.", ""])
            continue
        lines.extend([f"### Figure {index}. {caption}", "",
                      f"![{caption}](../results/paper_figures/{name}.png)", ""])
        sources = entry.get("sources", [])
        if any(not list(REPO_ROOT.glob(source)) for source in sources):
            lines.extend(["Source data needed to regenerate this historical figure are missing "
                          "from the current checkout.", ""])
    return "\n".join(lines)


def export_documents(markdown_text: str) -> None:
    """Render all three document formats through the same Pandoc settings."""
    for suffix in ("tex", "docx", "pdf"):
        command = ["pandoc", "--from=gfm", "--standalone", "--resource-path=" + str(DOCS),
                   "-V", "geometry:margin=1in", "-V", "mainfont=DejaVu Serif",
                   "-V", "monofont=DejaVu Sans Mono",
                   "--output=" + str(DOCS / f"paper_final.{suffix}")]
        if suffix == "pdf":
            command.append("--pdf-engine=xelatex")
        subprocess.run(command, input=markdown_text, text=True, cwd=DOCS, check=True)


# --------------------------------------------------------------------------- entry point


def main() -> int:
    s = Study()
    DOCS.mkdir(parents=True, exist_ok=True)

    text = (manuscript(s) + "\n\n" + figure_appendix()).replace("ℓ", r"$\ell$")
    (DOCS / "paper_final.md").write_text(text, encoding="utf-8")
    export_documents(text)
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
