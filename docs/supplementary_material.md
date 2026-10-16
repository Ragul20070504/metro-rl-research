# Supplementary material

*Generated from stored artefacts at commit `e053dea4180c`.*

## S1. Experimental registry

The frozen registry (`experiments/registry.yaml`) defines 17 registered experiments. Confirmatory: EXP-UQ-001 (H1, primary), EXP-UQ-002 (H1c, placebo co-primary). Exploratory families cover scenario generalisation (EXP-UQ-003), multi-objective ablation (EXP-MO-001), availability ablation (EXP-AVAIL-001), baselines (EXP-BASE-*), robustness (EXP-ROBUST-*), sensitivity (EXP-SENS-001/002) and interpretability (EXP-XAI-001). The registry is frozen; sweep and ablation stages refuse to run against an unfrozen registry.

## S2. PPO configuration

| Parameter | Value |
|---|---|
| `total_timesteps` | 60000 |
| `n_steps` | 456 |
| `batch_size` | 152 |
| `n_epochs` | 10 |
| `learning_rate` | 0.0003 |
| `gamma` | 0.995 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `ent_coef` | 0.01 |
| `vf_coef` | 0.5 |
| `max_grad_norm` | 0.5 |
| `net_arch` | [64, 64] |

One configuration, applied unchanged to every arm. No hyperparameter search was conducted.

## S3. Seed pools

| Pool | Range |
|---|---|
| evaluation | [1, 30] |
| training | [101, 110] |
| pilot training | [901, 904] |
| pilot evaluation | [41, 50] |
| scaler fitting | [1001, 1020] |
| heuristic tuning | [2001, 2006] |
| surrogate calibration | [3001, 3004] |

All pools are mutually disjoint, so nothing is fitted on a day the comparison is scored on.

## S4. Scenario definitions

| Scenario | Description |
|---|---|
| `normal` | Baseline demand and full rolling-stock availability |
| `surge` | Elevated demand; the registered primary scenario |
| `disruption` | Reduced availability through rolling-stock failures |
| `combined` | Surge and disruption together |

Scenario families share a base world at a given evaluation seed, so a scenario contrast measures the scenario rather than a different random draw.

## S5. Baseline details

Historical baseline fairness report (2026-08-27T04:36:39.843562+00:00): 19/19 checks recorded as passing, 0 recorded blocking failures. This table transcribes that report; it is not a fresh execution of those checks.

| Check | Status |
|---|---|
| C1. every method faces the same network, horizon, fleet, action bound and reward scales | PASS |
| C2. the environment builder cannot specialise the world to the method | PASS |
| C3. common random numbers are structural: the stream factory has no method argument | PASS |
| C3b. the same (scenario, eval_seed) reproduces the same streams | PASS |
| C3d. swapping the forecast provider does not desynchronise the realised world | PASS |
| C3c. scenarios at one eval_seed share a base world (amendment A-03) | PASS |
| C4. the heuristic reads the forecast mean but never sigma | PASS |
| C4b. the rolling MILP reads sigma, as its registration requires | PASS |
| C4c. the static MILP sees only the historical average day | PASS |
| C4d. the three uncertainty arms are distinct and all registered | PASS |
| C5. reading a forecast feature at or after the issue hour raises | PASS |
| C5b. the leakage test suite is present and covers every method's inputs | PASS |
| C6. baseline tuning seeds are disjoint from every scored pool | PASS |
| C6b. the heuristic's selected thresholds are interior to its search grid | PASS |
| C6c. the RL arms share one frozen hyperparameter configuration | PASS |
| C7. every stored method was evaluated on the same evaluation seeds | PASS |
| C7a. every stored cell used one frozen reward_scales | PASS |
| C7b. every stored cell used one frozen obs_scaler | PASS |
| C7c. each method used exactly the forecaster its registration gives it | PASS |

## S6. Statistical procedures

- Unit of analysis: training seed (protocol section 8.2); episode-level n is never reported as the sample size
- Confirmatory: exactly two tests, Holm-corrected at family-wise α = 0.05
- Exploratory: Benjamini–Hochberg at q = 0.10
- Effect estimate: Hodges–Lehmann with BCa bootstrap intervals
- Equivalence: TOST against the registered SESOI, run only on non-rejection
- SESOI = 0.002; registered n = 10; pilot SD = 0.005298; realised SD = 0.011269

## S7. Sensitivity analyses

| Condition | Comparison | Difference | BH-adjusted p | Direction |
|---|---|---:|---:|---|
| EXP-SENS-002 od_cbd_attracted | ppo_uq vs ppo_point | +0.008559 | 0.1727 | ppo_point better |
| EXP-SENS-002 od_cbd_attracted | ppo_uq vs ppo_placebo | +0.005674 | 0.2047 | ppo_placebo better |
| EXP-SENS-002 od_distance_decay | ppo_uq vs ppo_point | +0.014126 | 0.1727 | ppo_point better |
| EXP-SENS-002 od_distance_decay | ppo_uq vs ppo_placebo | +0.008375 | 0.184 | ppo_placebo better |
| EXP-SENS-002 od_uniform_downstream | ppo_uq vs ppo_point | +0.023265 | 0.1727 | ppo_point better |
| EXP-SENS-002 od_uniform_downstream | ppo_uq vs ppo_placebo | +0.012574 | 0.1727 | ppo_placebo better |
| EXP-SENS-001 weights_w_cost / normal | ppo_uq vs ppo_point | -31.852804 | 0.5685 | ppo_point better |
| EXP-SENS-001 weights_w_cost / surge | ppo_uq vs ppo_point | -30.708567 | 0.5685 | ppo_point better |
| EXP-SENS-001 weights_w_service / normal | ppo_uq vs ppo_point | -508.944445 | 0.1727 | ppo_point better |
| EXP-SENS-001 weights_w_service / surge | ppo_uq vs ppo_point | -488.591581 | 0.1727 | ppo_point better |
| EXP-SENS-001 weights_w_unmet_heavy / normal | ppo_uq vs ppo_point | -604.107252 | 0.1727 | ppo_point better |
| EXP-SENS-001 weights_w_unmet_heavy / surge | ppo_uq vs ppo_point | -581.869363 | 0.1727 | ppo_point better |

EXP-SENS-001 cannot test reward-weight sensitivity on the primary endpoint. Re-scoring a frozen policy under a different reward weighting changes no action: the policy does not observe the reward at evaluation time, so the trajectory, and therefore `left_behind_fraction` and `inductions`, are identical across all three weightings -- verified per seed, not merely on average. Only `total_reward` differs, and it is an arithmetic re-weighting of components already stored.

## S8. Ablations

| Experiment | Scenario | A | B | Difference | Adjusted p | Direction |
|---|---|---|---|---:|---:|---|
| EXP-MO-001 | normal | ppo_uq | ppo_uq_single_obj | -0.111136 | 0.01647 | ppo_uq better |
| EXP-MO-001 | surge | ppo_uq | ppo_uq_single_obj | -0.115261 | 0.007812 | ppo_uq better |
| EXP-MO-001 | disruption | ppo_uq | ppo_uq_single_obj | -0.110953 | 0.007812 | ppo_uq better |
| EXP-MO-001 | combined | ppo_uq | ppo_uq_single_obj | -0.114320 | 0.007812 | ppo_uq better |
| EXP-AVAIL-001 | normal | ppo_uq | ppo_uq_noavail | +0.002258 | 0.6255 | ppo_uq_noavail better |
| EXP-AVAIL-001 | surge | ppo_uq | ppo_uq_noavail | +0.002628 | 0.5254 | ppo_uq_noavail better |
| EXP-AVAIL-001 | disruption | ppo_uq | ppo_uq_noavail | +0.002436 | 0.6059 | ppo_uq_noavail better |
| EXP-AVAIL-001 | combined | ppo_uq | ppo_uq_noavail | +0.002013 | 0.6441 | ppo_uq_noavail better |

## S9. Interpretability

shap.PermutationExplainer (model-agnostic), explaining P(dispatch >= 1 trainset) under the policy, for policy `ppo_uq` at training seed 103 (background 60).

**Attributions describe model association and do not establish causality.** Stored caveats:
- SHAP attribution is associational, not causal: it decomposes this model's output on this input distribution and says nothing about what changing a channel in the world would do.
- A high attribution to a sigma channel is evidence the policy's action responds to it, not evidence that responding improves outcomes. That question is H1's, and only the confirmatory test answers it.
- Masks are not applied inside the explainer, so infrastructure constraints are not attributed to the policy's reasoning.
- Attributions come from one training seed; EXP-XAI-002 measures their stability across seeds and background resamples.

## S10. Stored-results checks and document build

These commands use the Linux CPU lock and recheck stored evidence. They do not rerun historical policies or reconstruct missing original inputs. The audit exits nonzero while submission blockers remain.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --require-hashes --torch-backend cpu -r requirements-linux.lock.txt
.venv/bin/python scripts/check_environment.py
.venv/bin/python scripts/publication_audit.py
.venv/bin/python scripts/build_manuscript.py
```

Stored: 7,610 historical evaluation cells and 50 main-study learned-policy identities; 0 original binaries available. The historical asset manifest lists 11 figures and 10 tables; Figure 7 is withheld. Asset presence does not prove missing source artifacts can be regenerated. See SUBMISSION_READINESS.md and the separate independent replication protocol.
