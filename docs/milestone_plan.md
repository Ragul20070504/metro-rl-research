# Milestone Plan

Follows brief §30. **Every milestone ends with the same closing ritual:** run the full test suite,
run a minimal experiment, inspect the outputs by eye, write what was *actually* completed (and what
was not) into `docs/progress_log.md`, and commit to Git on the milestone's feature branch.

A milestone is not "done" because code runs. It is done when its **exit gate** below passes.

| M | Deliverable | Exit gate (must pass to proceed) | Depends on | Branch |
|---|---|---|---|---|
| **M0** | `docs/research_audit.md` | ✅ **Complete.** Audit covers RQs, hypotheses, novelty, assumptions, data availability, threats, publication risk | — | `main` |
| **M1** | `docs/experimental_protocol.md`, `docs/data_sources.md`, `docs/risk_register.md`, `docs/manuscript_corrections.md`, this plan, repo skeleton, dependency plan | ✅ **Complete** | M0 | `main` |
| **M1b** | `docs/decisions.md` — all 10 open decisions resolved, with reasoning and accepted consequences | ✅ **Complete.** Taken before any data download or run, so they are pre-registration, not amendments | M1 | `main` |
| **M2** | Environment + repo scaffolding: Python 3.11 env, resolved lockfile, `.gitignore`, `.env.example`, config loader with mandatory `provenance` enforcement, seeding with the CRN guarantee, logging, run-manifest writer with append-only guard, Git repo + branches | 🟡 **Mostly complete.** Done: 3.11.9 env, 18/18 deps import, **10/10 functional stack checks**, git repo with real commit hashes, `src/common/` foundation, **38 tests pass**. Outstanding: hash-pinned `requirements.txt` and `environment.yml`. **E-01 and E-02 both closed** | M1b | `development` |
| **M3** | Real-data ingestion + preprocessing: `src/data/` — download, manifests, validation suite, cleaning, chronological split, feature engineering | All 10 data-quality checks pass or are explicitly waived with a reason; `results/data_quality_report.md` generated; split boundaries frozen in `configs/data.yaml`; `test_raw_immutable` and `test_leakage` (split parts) pass | M2 | `feature/data` |
| **M4** | Demand forecasting: seasonal naive → XGBoost point → XGBoost quantile, calibration, `forecast.parquet` | **Blocking calibration gate** (protocol §4.5): quantile model beats naive on pinball loss on `val` **and** post-calibration 80 % PICP on `val` ∈ [0.76, 0.84]. Full metric table stored | M3 | `feature/forecasting` |
| **M4b** | Train-health / availability module — **scope depends on decision item 4** | Whichever option is chosen, its scientific status is written down before any metric is reported | M3 (+M4) | `feature/forecasting` |
| **M5** | Digital-twin simulator + validation: `src/simulator/`, data-fitted demand process (MC-06 fix), disruptions, energy/cost stubs with `provenance` | `docs/simulator_validation.md` produced, with a per-quantity verdict of `validated` / `partially validated` / `not validatable`; `test_no_circular_demand` passes; failure rate matches `mta_mdbf` within a recorded tolerance | M3, M4 | `feature/simulator` |
| **M6** | RL environment: `src/environment/metro_env.py`, Gymnasium API, action masking, reward components, info dict | Environment/observation/action/reward/constraint/mask tests pass; **the three toy sanity environments pass** (zero demand ⇒ no unnecessary dispatch; all trains unavailable ⇒ dispatch = 0; extreme demand ⇒ dispatch increases while feasible); `check_env` clean | M5 | `feature/environment` |
| **M7** | Heuristic + fixed-timetable baselines | Thresholds tuned on `train`/`val` only (`test_no_test_set_tuning` passes); both run end-to-end on all four scenarios at one seed; CRN test passes across the two methods | M6 | `feature/baselines` |
| **M8** | PPO point-forecast baseline: `MaskablePPO`, frozen hyperparameters, checkpoints, training curves | Training curve rises and stabilises on one seed; reward components sane; **zero headway violations**; per-run wall clock measured (feeds the E-03 compute decision and the §8.4 power analysis) | M6, M7 | `feature/ppo` |
| **M9** | Uncertainty-aware PPO + placebo arm | All three arms verified **byte-identical** in architecture, parameter count, training budget and seeds, differing only in σ-channel content (asserted by test) | M8 | `feature/ppo` |
| **M10** | MILP baselines: `milp_static` (per manuscript) and `milp_rolling` (MC-05) | Formulation documented in `docs/baselines.md`; solver status, optimality gap, solve time recorded per instance; no instance silently treated as optimal | M6 | `feature/baselines` |
| **M11** | Full experiments: 9 methods × 4 scenarios × 30 eval seeds × S training seeds | **Gated on protocol §10 stages 1–4 and the M4 forecasting gate.** `experiments/registry.yaml` frozen before the first run; every run stored under its own non-overwritable path; SESOI and power analysis frozen before the sweep | M4–M10 | `feature/evaluation` |
| **M12** | Ablations: uncertainty (H1/H2), multi-objective vs single-objective (H3), availability channel (H7), reward-weight grid, OD-model sweep | Each ablation traceable to a registry entry and a hypothesis; identical training budgets across compared arms | M11 | `feature/evaluation` |
| **M13** | SHAP: global, scenario-conditional, local, plus stability analysis | Attribution stability reported across seeds and background resamples; non-causality stated in the report itself, not only in the paper | M11 | `feature/shap` |
| **M14** | Statistical analysis: `src/evaluation/analyze.py` | Assumption checks, paired tests, effect sizes with BCa CIs, Holm on the confirmatory family, BH on exploratory, TOST if the primary does not reject, mixed-effects secondary analysis. **Every number machine-generated** | M11, M12 | `feature/evaluation` |
| **M15** | Research-quality figures and tables | Every figure regenerated from `results/` by a single command; no number typed by hand anywhere; figure ↔ result-file mapping recorded | M14 | `feature/evaluation` |
| **M16** | Final reproducibility audit + `docs/paper_traceability.md` + `docs/pre_submission_peer_review.md` | A **fresh clone** reproduces the headline numbers; every manuscript claim classified SUPPORTED / PARTIALLY SUPPORTED / NOT YET TESTED / REFUTED with a pointer to a stored result; the hostile-reviewer document is written and is **not** artificially positive | M15 | `main` |

## Sequencing notes

- **M4b is deliberately decoupled** from the critical path. Whatever is decided for the train-health
  module, it must not delay M5–M9, which carry the central hypothesis.
- **M10 (MILP) runs in parallel with M8–M9.** It depends only on M6.
- **M11 is the only expensive step.** Everything before it is cheap enough to iterate on. That is the
  point of the staging in protocol §10: never spend the sweep budget before the environment and
  reward are known to be correct.
- **The "Prototype complete" statement (brief §37) may only be made after M16 passes**, and only if
  every one of the thirteen conditions in brief §37 holds. If a hypothesis is rejected, that is a
  completed prototype with a negative result, not an incomplete one.
