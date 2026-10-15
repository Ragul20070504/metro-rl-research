# Project handover — complete state as of 2026-08-27 16:54

Everything needed to continue without any prior conversation. Written because the working context
may be lost; the repository and git history are the durable record, and this file indexes them.

**Status: PAUSED cleanly at 33/50 confirmatory policies. Nothing is running. Nothing is lost.**

Resume with one command:

```
cd d:\Research\metro-rl-research
.venv\Scripts\python.exe scripts\finish_pipeline.py --from train --skip-wait
```

`RESUME.md` holds the operational detail. This file holds the *scientific* record.

---

## 1. What the project is

A reproducible, experimentally defensible research prototype for the manuscript *"An
Uncertainty-Aware Adaptive Multi-Objective Reinforcement Learning Framework for Real-Time Metro
Train Induction and Scheduling"*.

The central claim under test (**H1**) is that giving an RL policy a *calibrated demand-forecast
uncertainty channel* reduces unmet demand under surge, relative to an otherwise identical
point-forecast policy.

The design is built so that H1 can **fail**, and so that a failure is as reportable as a success.

### The three-arm placebo design

| arm | sigma channel | purpose |
|---|---|---|
| `ppo_point` | zeroed | the control |
| `ppo_placebo` | a draw with the **same marginal distribution** and no association to the truth | separates *responding to a channel* from *responding to information* |
| `ppo_uq` | the calibrated sigma | the proposed method |

All three have identical observation dimensionality and parameter count (24,452). Without the
placebo arm, a positive H1 could be explained by the extra input dimension alone.

### Confirmatory family — exactly two tests

- **EXP-UQ-001 (H1, primary)** — `ppo_uq` vs `ppo_point`
- **EXP-UQ-002 (H1c, co-primary)** — `ppo_uq` vs `ppo_placebo`

Both on `left_behind_fraction`, scenario `surge`, weights `w_balanced`, Holm-corrected at
family-wise alpha 0.05. Everything else is exploratory (Benjamini–Hochberg, q = 0.10) and labelled
as such in every table.

---

## 2. Milestone status

| milestone | status | evidence |
|---|---|---|
| M0–M3 data pipeline | COMPLETE | `results/data_quality_report.md`, `docs/data_sources.md` |
| M4 forecasting | **COMPLETE** | gate passed; 19/20 validation checks; `docs/forecasting.md` |
| M5 digital twin | COMPLETE | `results/simulator_validation.json`, revalidated after A-03 |
| M6 RL environment | COMPLETE | `docs/m6_scientific_gate.md`, `results/env_sanity_report.md` |
| M7 fixed timetable | COMPLETE | 120 cells |
| M8 heuristic | **COMPLETE (corrected)** | threshold 3200, interior; 120 cells |
| M9 MILP | **COMPLETE (corrected)** | static + rolling; 240 cells |
| Baseline fairness gate | **19/19 PASS** | `docs/baseline_fairness_audit.md` |
| R-1 power pilot | **COMPLETE** | `docs/power_analysis.md` |
| Registry freeze | **FROZEN** | commit `03990ba`, S = 10 |
| M11 training | **33/50** | both confirmatory arms 10/10 |
| M11 sweep | NOT STARTED | 6,000 cells |
| M12 ablations | NOT STARTED | machinery ready |
| M13 SHAP | NOT STARTED | machinery ready |
| M14 statistics | NOT STARTED | machinery ready |
| M15 figures | NOT STARTED | machinery ready |
| M16 audit | NOT STARTED | machinery ready |
| M17 dashboard | **BUILT**, stopped during compute | `docs/m17_demo.md` |

### Policies on disk

| arm | complete | needed by |
|---|---|---|
| `ppo_point` | **10/10** | EXP-UQ-001 |
| `ppo_uq` | **10/10** | EXP-UQ-001, EXP-UQ-002 |
| `ppo_placebo` | **10/10** | EXP-UQ-002 |
| `ppo_uq_single_obj` | 3/10 | H3 (exploratory) |
| `ppo_uq_noavail` | 0/10 | H7 (exploratory) |

Plus 8 pilot policies (seeds 901–904) — R-1 artefacts, excluded from all analysis by design.

**Both confirmatory arms are finished.** The 17 outstanding policies serve exploratory hypotheses
only.

### Cells: 560 stored, 0 failed

480 baseline + 80 pilot. Confirmatory sweep not yet begun.

---

## 3. The R-1 power result — read this before interpreting anything

| quantity | value |
|---|---|
| pilot | 4 paired training seeds × 10 evaluation seeds |
| between-seed SD | 0.005298 |
| observed effect (`ppo_uq` − `ppo_point`) | **+0.010077** |
| SESOI (frozen in A-04, before the pilot) | 0.002 |
| alpha per test / target power | 0.025 (Holm) / 0.80 |
| **required n for SESOI** | **69 training seeds** |
| registered n | 10 |
| **minimum detectable effect at n = 10** | **0.005909** |
| decision | proceed with registered n, report as potentially underpowered |

Two facts that must not be lost:

**The study is underpowered for its own SESOI.** Detecting 0.002 needs 69 training seeds — 345
policies, roughly 44 hours of training. Protocol §8.4's honest option was taken: run the
registered design and state the minimum detectable effect. **Lowering the SESOI was not an
option** and was not done.

**The pilot's direction was against H1.** `left_behind_fraction` is lower-is-better, and the
observed effect is *positive*: `ppo_uq` stranded more passengers than `ppo_point` in **all four**
pilot seeds (0.0496 / 0.0343 / 0.0492 / 0.0625 vs 0.0439 / 0.0262 / 0.0405 / 0.0447).

This is **not evidence about H1** — the pilot's registered job is variance estimation only, its
seeds are disjoint from every scored pool, and its policies are discarded. Nothing was changed in
response to it. But whoever reads the confirmatory result should not be surprised by the sign, and
**must not** treat a null or negative result as a failure of the work. A null result here is a
publishable finding, and the TOST equivalence test against the SESOI exists precisely to
distinguish *no evidence of benefit* from *evidence of no meaningful benefit*.

---

## 4. Amendments — the scientific audit trail

All in `docs/experimental_protocol.md` §12. Each was recorded **before** the procedure it governs.

| id | what | why it matters |
|---|---|---|
| **A-01** | primary endpoint = distinct left-behind passengers | operationalises "unmet demand" |
| **A-02** | reward's waiting term = queue integral (passenger-minutes) | the original made "run no service" score the global maximum |
| **A-03** | scenario families share a base world at a given eval seed | before this, `surge` scored *better* than `normal` in 3/5 seeds |
| **A-04** | SESOI = 0.002, frozen before the pilot | prevents lowering it after seeing the data |
| **A-05** | one frozen PPO config, no hyperparameter search | the proposed arm cannot be tuned harder than its ablations |
| **A-06** | rule for extending the training-seed pool | fixed before the pilot was read |
| **A-07** | registry frozen between R-1 and M11 | makes pre-registration checkable, not merely claimed |
| **A-08** | results storage layout differs from frozen §8.5 | recorded rather than left for a reviewer to find |
| **A-09** | one compute thread per worker | thread count changes float reduction order; pins reproducibility |
| **A-10** | heuristic grid extended until its optimum is interior | the tuned threshold sat at the grid boundary |
| **A-11** | MILP surrogate gains a calibrated standing-queue term | surrogate predicted queue 0.5 where simulator realised 1,018 |
| **A-12** | monotone clamp on MILP waiting coefficients removed | the clamp overwrote a measured effect with an assumption |
| **A-13** | dispatch feasibility envelope, and where MILP correction stops | declines a fourth correction that would mean authoring a baseline |

**Every one of A-10 through A-13 strengthened a competitor to the proposed method.** That
direction is deliberate: it is the only direction in which an autonomous correction is safe.

---

## 5. Defects found and fixed — the substantive ones

Each was found by **measuring something that should have been impossible**, not by code review.

1. **Degenerate reward (A-02).** Mean wait of *boarders* made "run no service all day" score 0 —
   the global maximum — while stranding 100% of demand.
2. **Broken CRN (A-03).** The scenario name was in the random-stream entropy, so `normal` and
   `surge` at the same seed were unrelated worlds. The `combined` factorial claim had been false
   since M2.
3. **Fixed timetable ran zero service.** GTFS `direction_id` is an Arrow string; comparing to int
   `0` produced an all-zero timetable.
4. **MILP κ was demand-limited, not capacity-limited** → static MILP stranded 17.6%.
5. **Pure fluid queue prices only capacity** → with spare capacity, frequency looked free.
6. **MILP standing-queue blindness (A-11).** Surrogate and simulator ranked the same two schedules
   *oppositely*. `milp_static` went 0.0567 → 0.0104 → 0.0049 across A-11/A-12/A-13.
7. **Monotone clamp destroyed real information (A-12).** `constant_1` strands 0.44%,
   `constant_2` strands 2.31% for the same number of trains — bunching is genuinely worse, and the
   clamp hid it.
8. **Heuristic tuned to its grid boundary (A-10).** Optimum at the edge means the true optimum is
   probably outside it.
9. **`persist_forecast_models` saved models *before* verifying them**, and `stage_prepare` skips
   persisting when files exist — a failed verification would have been silently promoted to a
   trusted model. Also gated only the median, not sigma — the channel H1 is about.
10. **Validation check 9b passed while asserting nothing.** Its `git()` helper called a bare
    `git`, off PATH on this machine, so every call returned `""` and the check read that as
    "nothing changed".
11. **Every `scripts/*.py` was broken when run as documented** — `ModuleNotFoundError: src`.
12. **Sweep would have silently capped the sample size** at 10 regardless of what R-1 asked for.
13. **Nothing froze the pre-registration** until `scripts/freeze_registry.py` was added.
14. **Runner never retried failed cells**; a sweep could end quietly incomplete.
15. **Torch defaulted to 8 threads per worker** — 11 workers would have contended for 12 cores.

---

## 6. Final baseline results (480 cells, all corrected)

`left_behind_fraction`, mean over 30 evaluation seeds:

| method | normal | surge | disruption | combined | inductions |
|---|---:|---:|---:|---:|---:|
| `fixed_timetable` | 0.0188 | 0.0255 | 0.0221 | 0.0286 | 204 |
| `heuristic_threshold` | 0.0049 | 0.0118 | 0.0101 | 0.0168 | 216 |
| `milp_static` | 0.0049 | 0.0119 | 0.0101 | 0.0171 | 228 |
| `milp_rolling` | 0.0238 | 0.0293 | 0.0269 | 0.0321 | 227 |

`milp_static` and the tuned heuristic are effectively tied and are **strong** baselines. The RL
arms must beat these, not a strawman.

`milp_rolling` remains weakest, for a **structural** reason recorded in A-13: a fluid-queue
surrogate has no representation of departure spacing, so it plans bursts. This narrows what H6b
establishes — it tests *this MILP formulation*, not optimisation-based scheduling in general.

---

## 7. What remains

| stage | command | est. |
|---|---|---|
| Training, 17 policies | `finish_pipeline.py --from train --skip-wait` | ~2.7 h |
| Sweep, 6,000 cells | (chained automatically) | ~3.8 h |
| Ablations | (chained) | ~0.7 h |
| SHAP → statistics → figures → traceability → audit → review | (chained) | ~0.5 h |
| **total** | | **~7.7 h** |

Timings are measured: 105 min per policy at 11 concurrent workers, from three completed waves.

### When M14 finishes, report for `ppo_uq` vs `ppo_point`, `left_behind_fraction`, `surge`:

n · means/medians · effect direction · effect size · Hodges–Lehmann · BCa 95% CI ·
Holm-adjusted p · TOST against SESOI 0.002 · minimum detectable effect (0.005909) ·
**H1 = SUPPORTED / NOT SUPPORTED / INCONCLUSIVE**

Read it from `results/statistical_analysis.json` and `docs/results.md`, both generated. **Never
type an experimental number by hand.**

### Then M17

```
streamlit run app.py          # http://localhost:8501
```

---

## 8. Do NOT change or delete

`models/policies/**` · `results/runs/**` · `models/forecasting/**` · `models/obs_scaler.json` ·
`models/heuristic_threshold.json` · `models/milp_surrogate.json` · `data/processed/**` ·
`experiments/registry.yaml` (**FROZEN**) · `results/power_analysis.json` · `.venv/`

Frozen by protocol: PPO configuration · 60,000-timestep budget · seeds · scenarios · reward
weights · endpoint · baselines · registered sample size.

**Do not tune anything on the basis of a confirmatory result.** Do not change the endpoint,
reward, scenario severity or hyperparameters to obtain significance. Do not delete unfavourable
results.

---

## 9. Key files

| file | what |
|---|---|
| `RESUME.md` | operational resume instructions |
| `docs/experimental_protocol.md` | the frozen protocol + all 13 amendments (§12) |
| `docs/decisions.md` | decisions D-01…D-14, including those escalated to the user |
| `docs/baseline_fairness_audit.md` | the 19/19 fairness gate |
| `docs/power_analysis.md` | R-1 |
| `docs/forecasting.md` | M4 |
| `docs/m17_demo.md` | dashboard |
| `experiments/registry.yaml` | frozen pre-registration |
| `scripts/finish_pipeline.py` | the driver |
| `scripts/analyse.py` | M14 |

**21 test files, 444 tests passing.**

---

## 10. Known limitations to carry into the paper

1. **Underpowered for the SESOI** — MDE 0.0059 vs SESOI 0.002; n = 69 needed, 10 registered.
2. **Baseline effort asymmetry** — the MILP surrogate was revised four times and the heuristic grid
   extended once, while the RL arms run one untuned frozen configuration (A-05). This runs
   *against* the proposed method and is reported, not hidden.
3. **H6b is narrow** — it tests this fluid-queue MILP formulation, not optimisation scheduling
   generally (A-13).
4. **Energy and cost are in equivalent units, not physical ones** — assumptions A15/A16 unresolved;
   embargoed from headline claims.
5. **Simulation only** — a calibrated simulator, not an asset-coupled digital twin (MC-04).
6. **SHAP is associational, never causal.**
7. **Out-of-regime transfer** (pre-pandemic) registered separately as EXP-ROBUST-004; not run.
8. **`quantile_crossing_rate` was 8.48%** pre-repair; quantiles are sorted and the rate recorded.
