# PAUSED — how to resume

Paused **2026-08-27 16:51 local** during M11 training, at the request of the operator.

Everything below was verified at the moment of pausing, not assumed.

---

## One command to resume

```
cd d:\Research\metro-rl-research
.venv\Scripts\python.exe scripts\finish_pipeline.py --from train --skip-wait
```

That is all. It re-enters the pipeline at the training stage, **skips every policy and cell that
already exists**, and carries on through sweep → ablations → SHAP → analyse → figures →
traceability → audit → review without further input.

Resuming is safe to run more than once. `train_policy` returns early when `policy.zip` exists, and
`run_cell` returns the stored result when `metrics.json` exists, so nothing completed is ever
recomputed.

---

## Exact state at pause

### Policies — 33 of 50 confirmatory complete

| arm | complete | needed by |
|---|---|---|
| `ppo_point` | **10 / 10** | EXP-UQ-001 (H1, primary) |
| `ppo_uq` | **10 / 10** | EXP-UQ-001, EXP-UQ-002 |
| `ppo_placebo` | **10 / 10** | EXP-UQ-002 (H1c, placebo control) |
| `ppo_uq_single_obj` | 3 / 10 | EXP-MO-001 (H3, exploratory) |
| `ppo_uq_noavail` | 0 / 10 | EXP-AVAIL-001 (H7, exploratory) |

**Both confirmatory-family arms are finished.** The 17 outstanding policies serve exploratory
hypotheses only.

Also on disk: 8 pilot policies (training seeds 901–904). Those are R-1 artefacts, deliberately
excluded from every analysis, and are neither needed nor re-run.

### Cells — 560 stored, 0 failed

| group | cells |
|---|---|
| Baselines (fixed timetable, heuristic, MILP static, MILP rolling) | 480 |
| R-1 pilot (`variant=pilot`, excluded from analysis) | 80 |
| Confirmatory sweep | 0 — not started |

### Integrity checks performed at pause

- 0 undersized or truncated `policy.zip` files.
- 0 policy directories left without a `policy.zip` (no partial artefacts to clean up).
- 0 `FAILED*.json` markers anywhere under `results/runs/`.
- Registry **FROZEN** at commit `03990ba`, `S = 10`.
- All frozen inputs present: forecaster + metadata, observation scaler, heuristic thresholds,
  MILP surrogate, fixed timetable, power analysis, forecasting manifest.

### What the pause cost

Training is **atomic per policy** by design — a policy is written only after its full 60,000
timesteps, because restoring a mid-run checkpoint would not reproduce an uninterrupted policy
bit-for-bit and would compromise reproducibility.

Wave 4 had been running about 10 minutes when it was stopped, so roughly **10 minutes of partial
computation per worker was discarded**. No completed result was lost, and no artefact was
corrupted. Those policies simply start again from the beginning on resume.

---

## Do NOT change or delete before resuming

Deleting any of these forces expensive recomputation, or worse, invalidates the frozen design.

| Path | Why |
|---|---|
| `models/policies/**` | 33 trained policies + 8 pilot. Deleting one costs ~105 min to retrain. |
| `results/runs/**` | 560 completed cells. This is the primary evidence. |
| `models/forecasting/**` | The M4 forecaster the RL arms consume. Regenerating costs ~30 min. |
| `models/obs_scaler.json` | Frozen observation scaling. Changing it invalidates every trained policy. |
| `models/heuristic_threshold.json` | Frozen under A-10 (threshold 3200, verified interior). |
| `models/milp_surrogate.json` | Frozen under A-11/A-12/A-13. |
| `data/processed/**` | Splits, network, timetable, forecasts. |
| `experiments/registry.yaml` | **FROZEN pre-registration.** Editing it after freezing breaks the entire pre-registration guarantee. |
| `results/power_analysis.json` | The R-1 decision the frozen sample size rests on. |
| `results/pipeline_progress.json` | Stage history. |
| `.venv/` | The pinned environment the results were produced in. |

Equally: do **not** change the PPO configuration, the 60,000-timestep budget, seeds, scenarios,
reward weights, the endpoint, the baselines, or the registered sample size. All are frozen, and
several are frozen by recorded amendment.

---

## Is it safe to shut down the PC?

**Yes.** No process is running, no file is open for writing, and every completed artefact is on
disk and committed to git. A clean shutdown, a reboot, or leaving the machine to someone else are
all fine.

Two things worth knowing:

- The machine can be used normally in the meantime. Nothing schedules itself, and nothing resumes
  on boot — resuming is an explicit command.
- If the repository is moved to another machine, the pinned `.venv` must be recreated from
  `requirements.lock.txt`, and `scripts/check_environment.py` should pass its 10 checks before
  resuming. Results produced under a different environment would not be comparable to those
  already stored.

---

## Remaining work after resume

| stage | estimate |
|---|---|
| Training, 17 policies | ~2.7 h |
| Confirmatory sweep, 6,000 cells | ~3.8 h |
| Ablations | ~0.7 h |
| SHAP, statistics, figures, traceability, audit, review | ~0.5 h |
| **total** | **~7.7 h** |

Timings are measured, not estimated: 105 min per policy at 11 concurrent workers, from the three
completed waves.
