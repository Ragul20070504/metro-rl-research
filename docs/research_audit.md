# Research Audit (M0)

**Status:** M0 deliverable. Produced *before* any substantive implementation code.
**Manuscript audited:** `research_spec/Metro_RL_Scheduling_Framework_Final.pdf` — "An Uncertainty-Aware
Adaptive Multi-Objective Reinforcement Learning Framework for Real-Time Metro Train Induction and
Scheduling" (authors not yet named in the draft).
**Audit date:** 2026-08-26.
**Auditor role:** research-methodology reviewer / reproducibility auditor.

## Epistemic labelling convention used throughout this repository

Every substantive statement in `docs/` carries one of these tags. They are never mixed inside a
sentence.

| Tag | Meaning |
|---|---|
| **FACT** | Verifiable from the manuscript text, a cited public source, or a stored artefact in this repo. |
| **OBSERVATION** | Something the auditor noticed about the manuscript or the data; verifiable by re-reading the source. |
| **ASSUMPTION** | A modelling choice not established by data or literature. Must be listed in §E. |
| **HYPOTHESIS** | A claim to be tested by an experiment that could reject it. |
| **EXPERIMENTAL RESULT** | A number produced by an executed run whose raw output is stored under `results/`. |

**FACT.** As of this audit, this repository contains **zero** EXPERIMENTAL RESULT statements. The
manuscript itself states (Abstract, §6, §7, §9) that no empirical results are reported and that the
digital-twin experiments are in progress.

---

## A. Research questions (extracted verbatim)

The manuscript states three research questions in §1 (p. 2). Quoted exactly:

> **(RQ1)** How can demand-forecast uncertainty be integrated into train scheduling?
>
> **(RQ2)** Can an RL agent learn dispatch strategies that robustly balance passenger service and
> operational cost under that uncertainty?
>
> **(RQ3)** How can the resulting policy remain interpretable to operators?

### OBSERVATION — the RQs are not, as written, falsifiable

- **RQ1** and **RQ3** are *design* questions ("how can X be done"). Any implementation answers them
  trivially. They cannot be rejected by an experiment. They must be operationalised before they can
  carry a results section.
- **RQ2** is closer to testable but is under-specified: "robustly balance" names no metric, no
  comparator, and no decision rule.

Proposed operationalisation (to be confirmed — see decision list in the response accompanying this
audit). The original RQ text is **not** replaced in the manuscript; these are the testable forms the
experiments will actually address, and `docs/paper_traceability.md` will map each RQ to its
operationalisation.

| Original RQ | Operationalised, testable form | Where tested |
|---|---|---|
| RQ1 | Does conditioning the dispatch policy on a *calibrated predictive distribution* of demand (mean + dispersion) change policy behaviour and outcomes, relative to conditioning on the point forecast alone, holding architecture, reward, training budget, environment, and evaluation seeds fixed? | EXP-UQ-001…004 |
| RQ2 | Does the multi-objective PPO policy dominate fixed-timetable, heuristic, and static-MILP baselines on a pre-registered primary endpoint under matched conditions, and does its advantage persist under demand surge and train-loss disruption? | EXP-BASE-001…004 |
| RQ3 | Do post-hoc SHAP attributions of the trained policy concentrate on state features that operations research would consider decision-relevant (forecast level, forecast dispersion, queue length, idle fleet), and are the attributions stable across seeds and re-samples? | EXP-XAI-001…003 |

**OBSERVATION.** RQ3 as phrased ("remain interpretable") is a category error that a Q1 reviewer will
attack: applying SHAP to a trained neural policy does not make the policy interpretable; it produces
*post-hoc attributions* of a black-box function, which are not causal and are known to be unstable
under correlated inputs. The forecast mean and forecast dispersion are correlated by construction in
almost any forecaster, which is exactly the regime where Shapley attributions are least identifiable.
See §G-11 and `docs/manuscript_corrections.md` item MC-07.

---

## B. Main scientific hypothesis

**HYPOTHESIS (H1, primary).**

> Explicitly representing passenger-demand forecast uncertainty inside the observation of an adaptive
> RL metro-induction policy improves service robustness under stochastic demand and disruptions,
> relative to an otherwise identical policy conditioned on point forecasts only.

Stated so that it can be **rejected**:

- **H1₀ (null, the position the experiment defaults to):** With training budget, network
  architecture, parameter count, reward function, environment dynamics, demand realisations,
  disruption realisations, and evaluation seeds held identical, the uncertainty-aware policy and the
  point-forecast policy do not differ on the pre-registered primary endpoint.
- **H1_A (alternative):** The uncertainty-aware policy achieves a *lower* value of the primary
  endpoint.

**Pre-registered primary endpoint (frozen in `docs/experimental_protocol.md` §7):** mean fraction of
unmet passenger demand per operating day, under the **demand-surge** scenario family.

Rationale for that choice, traceable to the manuscript rather than chosen post hoc: §7 (p. 7) states
the anticipated mechanism explicitly — *"the uncertainty-aware variant will make comparatively
conservative dispatch decisions when forecast variance is high, reducing unmet demand relative to an
ablation using point forecasts only."* The primary endpoint is therefore the quantity the manuscript
itself predicts will move, in the scenario where it predicts it will move. All other endpoints are
secondary and are multiplicity-controlled.

### H1 is decomposed into a mechanism test and a placebo control

A bare mean-difference test on H1 cannot distinguish three explanations of a positive result:
(i) the policy genuinely exploits forecast dispersion; (ii) the policy benefits from *any* extra
input channels (more parameters exercised, implicit regularisation, richer gradient signal);
(iii) chance across seeds. The design therefore adds:

- **HYPOTHESIS (H1b, mechanism).** The uncertainty-aware policy's dispatch rate increases with
  forecast dispersion σ after conditioning on forecast mean μ. Test: partial rank correlation between
  σ and dispatched-train count given μ, over stored decision logs; and the same statistic for the
  point-forecast policy, which should be ≈ 0 by construction. H1b can fail even if H1 "succeeds",
  which would indicate the gain is not the mechanism the manuscript claims.
- **HYPOTHESIS (H1c, placebo control).** A third arm receives an uncertainty channel that is *shuffled*
  across stations and time within each batch — identical dimensionality, identical marginal
  distribution, zero informational content. If the true-σ arm does not beat the shuffled-σ arm, the
  apparent benefit of "uncertainty awareness" is an artefact of observation dimensionality, not of
  uncertainty. **OBSERVATION:** this control is absent from the manuscript and from all prior work
  cited in Table 1; it is the single strongest defence against the reviewer question *"could the
  improvement simply come from more information?"* (§35 of the research brief).

**OBSERVATION — prior expectation is genuinely uncertain, and plausibly negative.** A PPO agent
trained *inside* the stochastic environment already encounters the demand distribution through
experience and can encode an implicit risk posture in its weights without ever seeing σ. The marginal
information in an explicit σ channel may therefore be small. This audit does **not** assume H1 is
true, and the protocol is powered and pre-registered so that a null result is publishable as a
negative finding rather than treated as a failed run.

---

## C. Secondary hypotheses

Each is stated with its comparator, its endpoint family, and the experiment that tests it.
All are **HYPOTHESIS** until `results/` contains executed runs.

| ID | Hypothesis | Comparison | Scenarios | Experiment |
|---|---|---|---|---|
| H2 | Uncertainty-aware PPO ≥ point-forecast PPO on service endpoints | PPO-UQ vs PPO-Point vs PPO-Placebo | all four | EXP-UQ-001..004 |
| H3 | Multi-objective reward yields better *joint* outcomes than a waiting-time-only reward, at the cost of waiting time alone | PPO-MO vs PPO-SO | all four | EXP-MO-001..004 |
| H4 | Adaptive RL outperforms a fixed timetable, with the gap widening under surge/disruption | PPO-UQ vs FixedTimetable | all four | EXP-BASE-001..004 |
| H5 | Adaptive RL outperforms a threshold heuristic whose thresholds were tuned on training data | PPO-UQ vs Heuristic | all four | EXP-BASE-001..004 |
| H6 | Adaptive RL outperforms a *static* MILP plan under stochastic realisation | PPO-UQ vs MILP-Static | all four | EXP-BASE-001..004 |
| H6b | Adaptive RL's advantage **shrinks or disappears** against a *rolling-horizon* MILP re-solved on the same forecasts | PPO-UQ vs MILP-Rolling | all four | EXP-BASE-005 |
| H7 | Providing predicted train-availability information improves outcomes under the train-disruption family | PPO-UQ+avail vs PPO-UQ−avail | disruption, combined | EXP-AVAIL-001 |
| H8 | Relative advantage of adaptive policies increases under demand surge | interaction term, method × scenario | normal vs surge | EXP-ROBUST-001 |
| H9 | Relative advantage of adaptive policies increases under train failure | interaction term, method × scenario | normal vs disruption | EXP-ROBUST-002 |
| H10 | Under combined events, degradation is sub-additive for adaptive policies and additive-or-worse for pre-computed plans | method × (surge × disruption) | all four | EXP-ROBUST-003 |
| H11 | SHAP global attributions concentrate on decision-relevant features and are stable (rank correlation across seeds ≥ a pre-registered threshold) | descriptive + stability test | all four | EXP-XAI-001..003 |

**OBSERVATION on H6/H6b.** The manuscript proposes only a *static MILP computed offline from average
demand* (§6). Beating a plan that is by construction blind to realised demand is close to
tautological, and §19 of the research brief explicitly forbids comparing "a highly optimized PPO
against an intentionally weak MILP". H6b (rolling-horizon MILP with access to exactly the same
forecasts the RL agent sees) is added as the *fair strong* optimisation baseline. This is a deviation
from the manuscript and is logged as **MC-05** in `docs/manuscript_corrections.md`.

---

## D. Novelty audit

### What the manuscript claims

**FACT (manuscript, Abstract p. 1):** *"To our knowledge, this is the first proposal to jointly
address demand-forecast uncertainty, multi-objective optimization, and decision explainability for
real-time train induction."*
**FACT (manuscript, §2 p. 3):** *"no prior work has simultaneously integrated uncertainty-aware
demand forecasting, multi-objective RL, and explainability for real-time train induction."*

### Component-by-component assessment against the manuscript's own Table 1

| Component | Prior art acknowledged in the manuscript | Novelty margin |
|---|---|---|
| RL for metro scheduling / induction | Ying et al. 2020 [1] (actor–critic, stochastic demand, rolling-stock circulation); Ying et al. 2021 [2] (PPO + constraint masking + flexible composition); Ying et al. 2022 [7] (multi-agent, CTDE) | **None.** Established. |
| Stochastic / uncertain demand in metro RL | [1] explicitly models an MDP with *stochastic demand* | **Thin.** The distinction being drawn is between a *stochastic environment* and an *explicit forecast-uncertainty channel in the observation*. This is a real distinction but a fine one, and the manuscript does not currently articulate it. |
| Multi-objective RL reward for transit timetabling | Pan et al. 2026 [6] (multi-objective PPO, bus, oversaturation) | **None in kind**; novel only in the metro-induction setting with this particular four-term reward. |
| SHAP applied to a DRL scheduling policy | [5] (production scheduling + SHAP, 2024) | **None in kind**; novel only in transit domain. |
| Digital-twin evaluation of transit RL | Simulation environments are standard in [1], [2], [7] | **None.** |

### OBSERVATION — the "first to jointly address" claim is weak and attackable

1. It is a **universal negative** over a large, fast-moving literature. It cannot be verified, only
   falsified, and a single reviewer-supplied counterexample destroys it.
2. Conjunction novelty ("A + B + C have not appeared in one paper") is the weakest form of novelty
   claim in OR/transportation venues. Reviewers routinely reject it as *combinatorial* rather than
   *methodological* novelty.
3. The manuscript's Table 1 already documents that each of A, B and C exists individually in the
   metro/transit RL literature, one of them (multi-objective PPO for transit timetabling, [6]) as
   recently as the manuscript's own citation year.

### What is *genuinely* defensible as a contribution

**OBSERVATION.** The scientifically defensible contribution is not the conjunction; it is a
**methodological question that the cited prior work does not answer**:

> *Does an explicit, calibrated representation of demand-forecast predictive uncertainty in the state
> of an adaptive induction policy provide measurable benefit beyond (a) the implicit risk posture a
> policy learns from training in a stochastic environment, and (b) the generic effect of enlarging
> the observation vector?*

Supporting contributions that are defensible and largely absent from the cited work:

- **C1.** A placebo-controlled ablation design (true-σ vs shuffled-σ vs zero-σ) that isolates the
  informational value of uncertainty from observation dimensionality. **Not present in [1], [2], [6],
  [7].**
- **C2.** A fair-comparison protocol using **common random numbers** across timetable, heuristic,
  static MILP, rolling-horizon MILP, and three RL variants on one simulator, with an automated
  temporal-leakage test suite.
- **C3.** A calibration-first forecasting evaluation (coverage, interval width, pinball loss,
  reliability) *connected to downstream control performance* — i.e. measuring whether better-calibrated
  forecasts produce better dispatch, which is a question about the uncertainty pipeline as a whole
  rather than about either component alone.
- **C4.** Honest reporting of the sim-only ceiling, including a documented statement of which
  simulator parameters could and could not be validated against public data.

**Recommended manuscript action (MC-01):** replace the "first to jointly address" sentence with a
scoped claim of the form *"we are not aware of prior work that isolates the contribution of explicit
forecast-uncertainty representation in metro induction policies under a placebo-controlled ablation"*,
and state the contribution as the *measurement*, not the *assembly*.

---

## E. Assumption audit

Classification key: **RD** real-data-supported · **LIT** literature-supported · **SIM** simulation
assumption (a modelling choice internal to the twin, not claimed to be real) · **SYN** synthetic
assumption (fabricated data-generating process standing in for absent data) · **UNR** unresolved (no
defensible value yet; must be resolved or swept before results are reported).

| # | Domain | Assumption | Class | Basis / consequence |
|---|---|---|---|---|
| A01 | Passenger demand — level | Station-level boardings per hour are taken from MTA Subway Hourly Ridership (entries + transfers) | **RD** | data.ny.gov `wujg-7c2s`. Confirmed to exist; see §F. |
| A02 | Passenger demand — sub-hourly shape | Within-hour arrivals are distributed by a fixed intra-hour profile and realised as a non-homogeneous Poisson process at 5-min resolution | **SIM** | Source data is hourly only. Poisson arrivals are standard in transit simulation [manuscript §5] but the within-hour shape is not observed. Sensitivity: uniform vs. smoothed-spline profile. |
| A03 | Passenger demand — origin–destination | Alighting is drawn from a distance-decay OD split | **SYN** | **The MTA hourly dataset contains entries and transfers only — no exits, no destinations.** No public OD matrix exists for this line. Three alternative OD models must be swept (§G-6). This directly affects occupancy, unmet demand, and therefore the primary endpoint. |
| A04 | Passenger demand — surge | Surge scenario = demand scaled 1.30–1.50× baseline over a contiguous window | **LIT/SIM** | Range taken verbatim from manuscript §6 ("30–50% above baseline"). The *shape* and *duration* of the surge window are SIM. |
| A05 | Passenger behaviour | Passengers board the first arriving train toward their destination if capacity remains; otherwise they wait (no balking, no re-routing, no mode shift) | **SIM** | Manuscript §5 states exactly this. Real crowding behaviour includes balking and boarding-position effects. Listed as a limitation. |
| A06 | Passenger behaviour — abandonment | Passengers never leave the queue | **SIM** | Makes "unmet demand" a pure capacity/headway measure. Alternative (finite patience) must be reported as sensitivity, because it changes the primary endpoint's definition. |
| A07 | Train movement | Inter-station running times from published GTFS static schedule; deterministic plus a bounded noise term | **RD + SIM** | GTFS is authoritative (`data.ny.gov` `fgm6-ccue`). The *noise distribution* is SIM. |
| A08 | Dwell time | Dwell = fixed minimum + per-passenger boarding/alighting term, capped | **SIM/UNR** | GTFS gives scheduled arrival/departure but the boarding-rate coefficient (s/passenger/door) is not observed. **UNR until a citable value is sourced**; otherwise swept. |
| A09 | Train capacity | Cars per train and passengers per car for the fleet serving the modelled line | **UNR** | Fleet composition is public in principle; a citable capacity figure has **not yet been sourced** and must not be invented. Swept until resolved. |
| A10 | Minimum headway | A frozen minimum headway enforced on each segment via action masking | **RD-partial** | Achievable headway on a CBTC-equipped line is documentable, but the operating value used must be sourced or swept, not assumed. |
| A11 | Turnaround time | Fixed terminal turnaround before a train can re-enter service | **SIM/UNR** | Not observable in the public feeds. Swept. |
| A12 | Fleet size | Number of trainsets available at day start | **SIM** | Chosen to be consistent with the peak service level implied by GTFS trip counts; provenance recorded in config. |
| A13 | Train availability / failure process | Per-train failure hazard during the operating day | **SYN, calibrated** | No public per-train availability record exists for the modelled fleet. However, **MTA Subway Mean Distance Between Failures (`e2qc-xgxs`, monthly, by car class)** is real and lets the *aggregate rate* of the synthetic hazard be calibrated rather than invented. The *per-train, within-day* structure remains SYN. |
| A14 | Train-health prediction signal | The policy observes a noisy availability probability per train | **SYN, optionally calibrated** | See §F. The realism of the *noise level* can optionally be anchored to detector performance measured on the real MetroPT-3 dataset (Metro do Porto). Cross-domain transfer is itself an assumption. |
| A15 | Energy | Energy per train-km (and optionally per stop/start) | **UNR** | **No authoritative figure has been sourced.** Energy will not appear in any headline claim until a citable value with provenance exists; until then it enters only as a swept parameter and the energy term is reported but flagged. |
| A16 | Dispatch / fleet cost | Cost per train-induction and per train-hour in service | **UNR** | Same treatment as A15. |
| A17 | Network topology | Single line, both directions, no branching, no shared trackage | **RD** | Chosen precisely because such a line exists (see §F, network selection). Removes the confound of shared-track interference. |
| A18 | Forecast uncertainty | Predictive dispersion comes from quantile regression on real historical demand and is *empirically calibrated on a validation split* | **RD-derived** | This is the load-bearing assumption for H1. If dispersion is not calibrated, H1 is untestable — see §G-4. |
| A19 | Disruption distribution | Number, timing and duration of train losses per episode | **SYN, partially calibrated** | Rate anchored to A13; timing distribution is SIM. Manuscript §6 specifies only "unplanned mid-day loss of one or more trains". |
| A20 | Decision epoch | 5-minute induction decisions over one operating day | **SIM** | Recommended by the research brief §10; consistent with manuscript §3 (discretised epochs). |
| A21 | Observability | The agent observes forecasts and current state only; never realised future demand | **Design constraint, enforced by test** | Required by research brief §4 and §26; enforced by `tests/test_leakage.py`. |

**OBSERVATION.** Assumptions **A03, A08, A09, A11, A15, A16** are currently **UNR/SYN and load-bearing**.
Six of the eight quantities entering the reward function (A03 → unmet demand and occupancy; A08/A09 →
waiting time; A15 → E; A16 → C) are not yet grounded. Reporting a weighted-sum reward as a headline
number before these are resolved would be indefensible. The protocol therefore reports **each reward
component separately and in physical units**, and gates any scalar-reward claim on resolving A15/A16.

---

## F. Data availability audit

> ### ⚠ CORRECTION — 2026-08-26, after direct endpoint probing
>
> This section was written from portal search results. It was then **verified by probing every
> endpoint** (`scripts/probe_sources.py`, evidence in `results/source_probe.json`), and three of its
> statements turned out to be wrong. They are corrected here rather than silently edited, because an
> audit that quietly rewrites its own errors is not an audit.
>
> 1. **`4apg-4kt9` (Customer Journey-Focused Metrics) returns HTTP 403.** The working identifier is
>    **`r7qk-6tcy`**, and it covers **2015-01 → 2026-06**, not the 2020–2024 assumed below.
> 2. **`fgm6-ccue` (GTFS Static) is not served through the tabular API.** The portal entry's only
>    attachment is an overview PDF. The feed itself comes from
>    `https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip` (5.6 MB, **no API key**).
> 3. **"Realised train movements are not retained as an open historical archive" — WRONG.**
>    `sp9g-mzjh` publishes realised **end-to-end running-time percentiles** (25th/50th/75th) by
>    direction, day type and period, and `32ch-sei3` publishes scheduled vs actual train counts.
>    Two rows of the simulator-validation table move from *not validatable* to *validatable*, and
>    running-time dispersion (A07) can be calibrated against a real distribution rather than chosen.
>
> Two findings also came out **better** than assumed: the hourly demand series is continuous from
> **2017-01-13 to 2026-08-13** (three same-schema datasets, not one), which lets the study avoid the
> post-pandemic regime break entirely (decision D-11); and the modelled line's **24 station
> complexes** were confirmed from the stations dataset and independently corroborated by the
> running-times dataset reporting 23 stops end-to-end.
>
> **`docs/data_sources.md` is the authoritative, verified data plan.** The table below is retained as
> the record of what was believed before verification.

Verified by direct lookup on 2026-08-26. URLs below were resolved during this audit; **download date,
row counts, file hashes and dataset version are recorded at ingestion time in
`data/raw/<dataset>/MANIFEST.json` and are not asserted here.**

### F.1 Available and authoritative

| Need | Source | Identifier | Status | What it actually gives / does not give |
|---|---|---|---|---|
| Station-level passenger demand | NY State Open Data (publisher: MTA) — *MTA Subway Hourly Ridership* | `data.ny.gov/d/wujg-7c2s` (2020–2024) and successor *Beginning 2025* | **CONFIRMED to exist** | Fields confirmed: `transit_timestamp, transit_mode, station_complex_id, station_complex, borough, payment_method, fare_class_category, ridership, transfers, latitude, longitude, georeference`. **Hourly. Entries/taps only. No exits, no destinations, no OD.** Attribution: Metropolitan Transportation Authority, https://www.mta.info/open-data. |
| Timetable, travel times, headways, trip counts | NY State Open Data — *MTA General Transit Feed Specification (GTFS) Static Data* | `data.ny.gov/d/fgm6-ccue` | **CONFIRMED to exist** | Scheduled stop times → inter-station running times, scheduled headways, service patterns. Basis for the fixed-timetable baseline. Also available from `mta.info/developers` (that route requires an API key; the data.ny.gov mirror does not). |
| Station inventory / geometry | *MTA Subway Stations* | `data.ny.gov/d/39hk-dx4f` | **CONFIRMED to exist** | Station complex IDs, lines served, coordinates — needed to join ridership (complex-level) to GTFS (stop-level). |
| Realised service quality, for **simulator validation** | *MTA Subway Wait Assessment* (2015–2019; Beginning 2020; Beginning 2025) and *MTA Subway Customer Journey-Focused Metrics* (2020–2024; Beginning 2025) | data.ny.gov | **CONFIRMED to exist** | Wait Assessment = regularity of train spacing at peak timepoints. Customer Journey-Focused Metrics include **Additional Platform Time** (extra platform wait vs schedule) and **Additional Train Time**. These are the correct, real quantities against which simulated waiting time and headway regularity must be checked. |
| Rolling-stock reliability, for **failure-rate calibration** | *MTA Subway Mean Distance Between Failures: Beginning 2015* | `data.ny.gov/d/e2qc-xgxs` | **CONFIRMED to exist** | Monthly MDBF by subway car class; MDBF = miles travelled before a mechanical failure delays final arrival by >5 min 59 s. **Aggregate only — monthly, by car class, not per train, not per day, no sensor traces.** |
| Real metro rolling-stock **sensor** data | *MetroPT-3* (Metro do Porto), UCI ML Repository #791 / Zenodo; described in *Scientific Data* (2022) | DOI `10.24432/C5VW3R` | **CONFIRMED to exist**, licence **CC BY 4.0** | 1,516,948 rows @ 0.1 Hz, Feb–Aug 2020, 15 sensor channels (TP2, TP3, H1, DV_pressure, Reservoirs, Oil_temperature, Motor_current + 8 digital). Ships **unlabelled** with company failure reports for **four air-leak failures**. **Covers the Air Production Unit of a *single* train, not a fleet.** |

### F.2 NOT available — and the consequence for the research claim

| Need | Availability | Consequence |
|---|---|---|
| **Per-train availability / maintenance status for the modelled (NYC) fleet** | **NOT PUBLIC.** No dataset found giving per-trainset daily availability, maintenance-hold status, or failure timestamps for NYC Transit rolling stock. | **The train-health module cannot be validated as a predictor of the availability signal the RL agent consumes.** MDBF constrains only the aggregate rate. **This is a HIGH publication risk (§H-5).** Any ROC-AUC/PR-AUC reported for a classifier trained on a *synthetic* availability process measures the classifier's ability to recover a process we ourselves wrote down — it is a software test, not scientific evidence, and must be labelled as such. |
| **Origin–destination matrix for the modelled line** | **NOT PUBLIC** at the required resolution. The MTA hourly feed records entries and transfers only. | Alighting profile is SYN (A03). Occupancy, crowding, and unmet demand all inherit this. Must be swept and reported as a dominant sensitivity. |
| **Beijing Metro open data** (named in manuscript §5 as a planned source) | **Could not be confirmed as publicly downloadable.** Search indicates AFC access is granted case-by-case under data-use agreements; only limited subsets (e.g. a Line-10 passenger-flow file on figshare) appear openly. | **Factual correction required in the manuscript** — logged as **MC-02**. Do not cite it as an available public source. |
| **NYC MTA turnstile counts** (named in manuscript §5) | The legacy weekly turnstile files have been **superseded** by the hourly ridership dataset above, which is strictly better (hourly, station-complex resolved, fare-class resolved). | Manuscript §5 should cite the current dataset. Logged as **MC-03**. |
| **Traction-energy consumption per train-km** | **NOT SOURCED.** | Assumption A15 stays UNR. Energy claims are embargoed until sourced (§E). |
| **Operating / induction cost per train** | **NOT SOURCED.** | Assumption A16 stays UNR. Same embargo. |
| **Real-time asset telemetry coupling for the modelled line** | **NOT AVAILABLE** (and not obtainable for an external researcher). | The system is a **calibrated simulator**, not a digital twin in the asset-coupled sense. Terminology correction **MC-04**. |
| Weather / events exogenous features (manuscript §4.1) | Obtainable in principle (NOAA/GHCN; public event calendars). Not yet sourced. | Optional; excluded from the frozen forecasting feature set unless explicitly added, to keep the forecast comparison clean. |

### F.3 Network selection

**OBSERVATION.** A single-line prototype is only scientifically clean if the real line is genuinely
isolated — otherwise the simulator ignores shared-track interference that dominates real headway
variability. The **BMT Canarsie Line (L service)** is documented as one of very few NYC non-shuttle
services that hosts a single service on unshared trackage, with a dedicated CBTC-equipped fleet. That
makes it the strongest available candidate: the "single line, no interference" abstraction is a
property of the real system rather than a convenience.

**Caveat (FACT).** Its station count and fleet composition were seen in secondary sources during this
audit and are therefore **not asserted here**; they will be derived at ingestion from *MTA Subway
Stations* and *GTFS Static* and recorded in the dataset manifest. The line has more stations than the
5–10 recommended by the research brief §10 — see decision item 2 in the accompanying response.

---

## G. Threats to validity

Ordered by the auditor's judgement of how much each could change the conclusion.

**G-1. Circular demand generation — CRITICAL, and present in the manuscript as written.**
Manuscript §5 specifies: *"synthetic passenger arrivals generated as Poisson processes whose rate is
set by the demand-forecast model plus noise."* If the simulator's ground-truth arrival rate is
produced by the forecasting model, then the forecaster is correct by construction, forecast error is
whatever noise term we chose, and the predictive dispersion σ is an artefact of our own noise
parameter rather than a property of a real forecasting problem. **H1 would then be untestable: the
central experiment would measure our noise setting, not uncertainty-awareness.** Correction: the
simulator's ground truth must be generated from a *data-derived* demand process fitted to observed
historical ridership on held-out days, and the forecaster must be trained only on the *observable
history*, producing genuine, uncontrolled error. Logged as **MC-06**; this must be fixed before M5.

**G-2. Baseline implementation bias — HIGH.** The manuscript's only optimisation baseline is a static
MILP from *average* demand, which cannot react and is close to a straw man (§C, H6b). Mitigation:
implement both static and rolling-horizon MILP; report solver status, optimality gap, timeout, and
solve time for every instance; give the rolling MILP exactly the forecasts the RL agent sees.

**G-3. Nesting of evaluation runs inside training seeds — HIGH, and routinely mishandled.** With
*S* training seeds × *E* evaluation episodes, the *E* episodes are **not** independent replicates of
"the method": they share one policy. Treating 30 episodes from 1 policy as n = 30 inflates
significance by an order of magnitude. Mitigation: unit of analysis is the training seed; per-seed
means are the primary data; a mixed-effects model with random intercepts for training seed and
evaluation seed provides the secondary analysis. Sample size is decided on *S*, not on *E*.

**G-4. Uncalibrated forecast uncertainty — HIGH.** If σ is systematically wrong, H1 tests a broken
input, and a null result is uninformative about the scientific question. Mitigation: forecasting is
gated — the probabilistic model must reach a pre-registered empirical coverage tolerance on the
validation split before any RL run consumes it; calibration diagnostics are stored and plotted.

**G-5. Synthetic train-health data dominating a "prediction" claim — HIGH.** See §F.2. Mitigation:
reframe the module as an *availability-information ablation* (H7) rather than a train-health
prediction contribution, and label all synthetic-label classifier metrics as software validation.

**G-6. Simulator bias from the OD assumption — HIGH.** A03 is synthetic and drives occupancy and
unmet demand, i.e. the primary endpoint. Mitigation: three OD models swept; the primary conclusion
must hold under all three or be reported as OD-dependent.

**G-7. Sim-to-real gap — HIGH and irreducible.** All conclusions are conditional on the simulator.
No real-world validation is possible in this project. Mitigation: honest framing throughout; the
simulator-validation report states explicitly which parameters were checked against real data and
which could not be.

**G-8. Reward-weight sensitivity — HIGH.** The ranking of methods is a function of (α, β, γ, δ) and
of the normalisation scales. A comparison run at one weight vector is not a finding. Mitigation:
frozen weight grid; results reported per weight configuration; the convex-hull front is explicitly
*not* called Pareto-optimal.

**G-9. RL stochasticity and seed effects — HIGH.** PPO variance across seeds in sparse-ish,
long-horizon control routinely exceeds the effect sizes being claimed. Mitigation: ≥ 10 training
seeds per arm (subject to the compute decision), full training curves stored, per-seed results
published, no seed dropped after the fact.

**G-10. Hyperparameter sensitivity and tuning asymmetry — HIGH.** Tuning the proposed arm harder
than the ablation arm silently manufactures the result. Mitigation: **one** hyperparameter
configuration, selected on the *validation* split using the **point-forecast** arm, then frozen and
applied unchanged to all RL arms. Research brief §17 forbids extra tuning of the uncertainty arm;
this rule operationalises it.

**G-11. SHAP interpretation — MEDIUM/HIGH.** Attributions over correlated features (μ and σ per
station are correlated) are not identifiable in the usual sense, are not causal, and depend on the
background distribution. Mitigation: report attribution *stability* across seeds and background
samples; state non-causality explicitly; do not derive operational recommendations from SHAP alone.

**G-12. Demand-model error and distribution shift — MEDIUM.** Post-2020 NYC ridership is
non-stationary. A chronological split places a structurally different regime in the test period.
Mitigation: chronological split with an explicit stationarity check reported in the data-quality
report; distribution-shift diagnostics between splits.

**G-13. Network simplification — MEDIUM.** Single line, no transfers modelled as network effects, no
depot geography, no crew constraints, no rolling-stock circulation balance beyond turnaround.
**OBSERVATION:** Ying et al. 2020 [1] *does* model rolling-stock circulation; omitting it is a step
back from the cited prior work and reviewers will notice. Document explicitly.

**G-14. Incomplete real-world operational constraints — MEDIUM.** No crew scheduling, no platform
occupancy conflicts, no short-turning, no signal-block modelling below the segment level.

**G-15. Multiple comparisons — MEDIUM.** 8 methods × 4 scenarios × 5+ metrics is a large family.
Uncontrolled, some "significant" result is guaranteed. Mitigation: one pre-registered primary
endpoint with a single test; Holm control within the confirmatory family; everything else labelled
exploratory with BH-controlled FDR.

**G-16. Researcher degrees of freedom — MEDIUM.** Mitigation: `experiments/registry.yaml` is written
and frozen *before* runs, with a `status` field; deviations are appended, never rewritten.

---

## H. Publication-risk assessment

| ID | Risk | Severity | Why | Required action before submission |
|---|---|---|---|---|
| H-1 | Circular demand generation makes the central experiment vacuous (G-1) | **CRITICAL** | Would invalidate every H1 result | Fix the generative design (MC-06) before M5. Non-negotiable. |
| H-2 | Unverified references [3]–[6]; the manuscript's own editorial note states the original "Tan et al. (2024)" and "Liu et al. (2025)" citations could not be verified and were replaced | **CRITICAL** | Unverifiable or incorrect citations are a research-integrity finding, not a formatting issue. Journals desk-reject for this. | Every reference verified against the publisher record (authors, volume, pages, DOI) before submission. Out of scope for the code; tracked in `docs/manuscript_corrections.md`. |
| H-3 | No empirical results yet; all claims are anticipations | **CRITICAL (as a submission)** | The paper cannot support a results-bearing claim today | This project exists to remove this risk. |
| H-4 | "First to jointly address…" universal-negative novelty claim | **HIGH** | One counterexample refutes it; conjunction novelty is weak | MC-01: rescope. |
| H-5 | Train-health module has no real data for the modelled fleet; a classifier scored on self-generated labels | **HIGH** | A reviewer will read this as a fabricated component | MC-08: either reframe as an availability-information ablation, or ground it on MetroPT-3 with explicit cross-domain caveats, or drop it. **Decision required.** |
| H-6 | "Digital twin" used for a simulator with no asset coupling | **HIGH** | Terminology inflation is a known reviewer trigger in this field | MC-04: rename to "calibrated digital-twin-style simulation environment" and define the term where first used. |
| H-7 | Statistical plan as written (default two-tailed t-test, unpaired Wilcoxon rank-sum, no multiplicity control, no treatment of seed nesting) | **HIGH** | Under common random numbers the design is *paired*; rank-sum is the wrong test. Nesting inflates n. | MC-09: adopt the frozen plan in `docs/experimental_protocol.md` §8. |
| H-8 | Beijing Metro open data cited as available; NYC turnstile cited though superseded | **HIGH** | Factually incorrect data-availability statements | MC-02, MC-03. |
| H-9 | Weighted scalar reward described in a way that invites a "Pareto" reading | **MEDIUM** | Weighted sums recover only the convex hull | MC-10: state this explicitly wherever trade-offs are discussed. |
| H-10 | Energy and cost terms with no sourced parameters (A15, A16) | **MEDIUM/HIGH** | Two of four reward terms would rest on invented constants | Source or embargo. No energy/cost headline claim without provenance. |
| H-11 | Rolling-stock circulation omitted although prior work [1] includes it | **MEDIUM** | Reviewers compare directly against [1] | Document as scope limitation; consider adding depot balance. |
| H-12 | Single line, single city, one simulator ⇒ limited external validity | **MEDIUM** | Generalisation claims will be challenged | Restrict conclusions to the studied configuration; add a second line only if time permits. |
| H-13 | Manuscript §7 states the expected direction of results in advance | **LOW/MEDIUM** | Harmless if pre-registered as a hypothesis (which it now is), harmful if it later shapes analysis | Pre-registration in `experiments/registry.yaml` before runs converts this from a bias risk into a strength. |
| H-14 | Author/affiliation placeholders `[Author Name(s)]` unfilled | **LOW** | Administrative | Fill before submission. |

---

## Audit conclusion

**OBSERVATION.** The manuscript is a well-formed *protocol* paper. Its formulation (§3) is coherent
and implementable, its evaluation design (§6) is broadly sound, and it is commendably explicit that no
results exist yet. Three things must change before implementation proceeds, and one before submission:

1. **Before M5 (blocking):** the demand-generation circularity (G-1 / MC-06). As written, the central
   experiment cannot test the central hypothesis.
2. **Before M11 (blocking):** the statistical design (G-3, G-15 / MC-09) and the MILP fairness
   question (G-2 / MC-05).
3. **Before any claim about train health (blocking):** resolve the synthetic-data question (H-5 / MC-08).
4. **Before submission:** reference verification (H-2) and the novelty rescope (H-4).

**HYPOTHESIS status:** nothing in this document is an EXPERIMENTAL RESULT. No number in this
repository may be reported as a result until it has been produced by an executed run stored under
`results/` and registered in `experiments/registry.yaml`.

## Sources consulted during this audit

- [MTA Subway Hourly Ridership: 2020-2024 / Beginning July 2020 — data.ny.gov](https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-Beginning-July-2020/wujg-7c2s)
- [MTA Subway Hourly Ridership: Beginning 2025 — data.gov catalog](https://catalog.data.gov/dataset/mta-subway-hourly-ridership-beginning-2025)
- [MTA General Transit Feed Specification (GTFS) Static Data — data.ny.gov](https://data.ny.gov/Transportation/MTA-General-Transit-Feed-Specification-GTFS-Static/fgm6-ccue)
- [MTA Subway Stations — data.ny.gov](https://data.ny.gov/Transportation/MTA-Subway-Stations/39hk-dx4f)
- [MTA Subway Mean Distance Between Failures: Beginning 2015 — data.ny.gov](https://data.ny.gov/Transportation/MTA-Subway-Mean-Distance-Between-Failures-Beginnin/e2qc-xgxs)
- [MTA Subway Customer Journey-Focused Metrics: 2020-2024 — data.ny.gov](https://data.ny.gov/Transportation/MTA-Subway-Customer-Journey-Focused-Metrics-2020-2/4apg-4kt9)
- [MTA Subway Wait Assessment: Beginning 2020 — data.gov catalog](https://catalog.data.gov/dataset/mta-subway-wait-assessment-beginning-2020)
- [MetroPT-3 Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset)
- [The MetroPT dataset for predictive maintenance — Scientific Data](https://www.nature.com/articles/s41597-022-01877-3)
- [MTA Developer Resources](https://www.mta.info/developers)
