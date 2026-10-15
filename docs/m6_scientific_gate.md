# M6 scientific gate

**Status: M6 COMPLETE.** All fifteen completion criteria met.
**Stop point reached — M7–M16 are not authorised by this document.**

| | |
|---|---|
| Git commit | `02109e9` on `development` |
| Tests | **301 passing**, 0 failing, 0 skipped |
| Date | 2026-08-26 |
| Python / platform | 3.11.9, Windows 11 (win32), CPU only |
| Key packages | gymnasium 1.3.0, stable-baselines3 2.9.0, sb3-contrib 2.9.0, torch 2.13.0+cpu |

This repository still contains **no reportable experimental result**. Everything below is
environment validation.

---

## 1. State definition

**123 values** with the availability channel on; **121** with it off. Full per-feature table —
name, meaning, units, source, time availability, observed-vs-forecast, normalisation, uncertainty
— is in [`docs/rl_environment.md`](rl_environment.md) §1, generated from
`src/environment/observation.py` and guarded against drift by `tests/test_docs_match_code.py`.

| block | size | index |
|---|---:|---|
| `forecast_mu_h1` | 24 | 0:24 |
| `forecast_sigma_h1` | 24 | 24:48 |
| `forecast_mu_line_h2h3` | 2 | 48:50 |
| `forecast_sigma_line_h2h3` | 2 | 50:52 |
| `queue_length` | 24 | 52:76 |
| `boardings_prev_epoch` | 24 | 76:100 |
| `fleet_counts` | 3 | 100:103 |
| `availability_signal` | 2 | 103:105 |
| `trains_per_segment` | 8 | 105:113 |
| `occupancy_summary` | 2 | 113:115 |
| `time_index` | 3 | 115:118 |
| `disruption_flags` | 3 | 118:121 |
| `headway_state` | 2 | 121:123 |

The forecast is recomputed **inside the episode** from realised boardings, never looked up from a
precomputed table — protocol §5.1 requires the forecaster to see only observable history, and the
simulated day is a fresh draw from the ground-truth process that no stored forecast corresponds
to. All access goes through `HistoryBuffer`, whose accessors *raise* rather than return any hour
at or after the issue time.

### The three uncertainty arms

Identical dimensionality, architecture, parameter count, budget, seeds and environment. Only σ
content differs: `ppo_point` = constant 0, `ppo_placebo` = correct marginal with the association
destroyed, `ppo_uq` = calibrated σ̂. A result in which `ppo_uq ≈ ppo_placebo > ppo_point` refutes
the manuscript's mechanism even though a two-arm study would report it as a win.

---

## 2. Action definition

`Discrete(3)` — induct 0, 1 or 2 trainsets over the next 5-minute interval. `k_max = 2` because a
terminal at 150 s minimum headway can clear at most 300/150 = 2 departures per interval.

Masked unless: `k ≤` idle trainsets, minimum headway respected, terminal platform free, trainset
available. **Action 0 always feasible**, so the mask is never empty. The mask is contiguous from
zero (feasibility in `k` is monotone).

---

## 3. Reward definition

`r_t = −( α·W_t + β·U_t + γ·E_t + δ·C_t )`, each term divided by a fixed reference scale.

| term | quantity | units |
|---|---|---|
| `W_t` | passenger-minutes of platform waiting accrued in the epoch (queue integral) | passenger-minutes |
| `U_t` | distinct passengers left behind by a departing full train (A-01) | passengers |
| `E_t` | energy | train-km-equivalent units (A15 unresolved) |
| `C_t` | dispatch and fleet cost | induction-equivalent units (A16 unresolved) |

Reference scales, fitted from `constant_1` over 8 episodes on seeds 1001–1008 (disjoint from both
the training pool 101–110 and the evaluation pool 1–30):
`ref_W = 2729.64`, `ref_U = 6.63`, `ref_E = 45.36`, `ref_C = 1.83` per epoch.
**Marked `provisional: true`** — protocol §6.4 names the fixed-timetable baseline as the reference
policy, and that baseline is an M7 deliverable.

Primary weight configuration `w_balanced` (1, 1, 1, 1). Grid frozen in code, not config, so a run
cannot invent an unregistered row.

---

## 4. Enforced constraints

| constraint | enforcement | verified by |
|---|---|---|
| Minimum headway (150 s) between departures | structural — held at the platform | `headway_violations == 0` under all four scenarios |
| Terminal platform holds one train | mask + simulator refusal | `test_an_occupied_terminal_platform_blocks_induction` |
| Cannot induct more than the idle fleet | mask | `test_mask_never_exceeds_the_idle_fleet` |
| Unavailable trainsets cannot be inducted | mask | `test_all_trains_unavailable_means_dispatch_is_impossible` |
| Train capacity never exceeded | simulator boarding limit | `test_occupancy_is_never_negative_and_never_exceeds_capacity` |
| Turnaround completed before re-induction | lifecycle — a trainset reaches the depot only after a full round trip | `test_simulator.py` |
| Passenger conservation | explicit alighting count | `test_passenger_conservation` |

Constraints are enforced by **masking, not reward penalties** (manuscript §3.4). An attempted
masked action raises `InfeasibleActionError`; simulator time, epoch, totals, queues and train
statuses are all asserted unchanged afterwards, and no epoch record is written. A silent no-op
would let an unmasked policy appear compliant while ignoring the constraint, and the resulting
zero headway-violation count would mean nothing.

**Requested ≠ delivered.** The mask may permit 2 while the infrastructure delivers 1; the second
induction expires if the terminal never frees. That surfaces as protocol §7.4's *induction
success rate* rather than being silently granted.

---

## 5. Leakage test results

All pass. Protocol §9.2 items 1–7 plus the M6 additions.

| test | result |
|---|---|
| Reading at or after the issue hour raises | **pass** |
| Window ending beyond realised hours raises | **pass** |
| Insufficient history refused | **pass** |
| ×1000 change to every hour ≥ issue time leaves the forecast bit-identical | **pass** |
| *Positive control*: changing a past hour **does** move the forecast | **pass** |
| ×100 change to every later hour of the day leaves the observation bit-identical | **pass** |
| *Positive control*: a change already realised **does** move the observation | **pass** |
| No observation block declares future availability | **pass** |
| Availability signal is drawn at episode start and never updated when a failure occurs | **pass** |
| Disruption flags are zero before any failure has occurred | **pass** |
| Reward reference scales fitted off the evaluation seeds | **pass** |

The positive controls matter as much as the negative ones: without them an environment that
ignored its inputs entirely would pass every leakage test.

---

## 6. Hand-calculated deterministic scenario

`tests/test_env_hand_calculation.py` — 2 stations, 2 trainsets, deterministic arrivals (exactly 1
per station per 30 s step), no failures, no disruptions. The full derivation is in the module
docstring; **21 assertions**, all passing.

| quantity | derived by hand | simulator |
|---|---:|---:|
| Epoch 1 arrivals | 20 | 20 |
| Epoch 1 queue integral | 3,300 passenger-seconds | 3,300 |
| Epoch 1 queue at end | 20 | 20 |
| **Epoch 1 reward** | **−55.0 exactly** | **−55.0** |
| Epoch 2 boarders | 12 | 12 |
| Epoch 2 mean wait of boarders | 1650/12 = 137.5 s | 137.5 |
| Epoch 2 queue integral | 6,090 passenger-seconds | 6,090 |
| Epoch 2 energy | 1.0 km × 1.0 + 0.35 = 1.35 | 1.35 |
| Epoch 2 cost | 1.0 + (10 × 30/3600) × 0.5 = 1.0416667 | 1.0416667 |
| **Epoch 2 reward** | **−103.8916667 exactly** | **−103.8916667** |

The first draft of this test used 3 stations and asserted 12 boarders in epoch 2; the simulator
returned 21. **The simulator was right** — the derivation had forgotten that a trainset inducted
at station 0 reaches station 1 mid-epoch and boards that platform too. The scenario was reduced
to 2 stations, where station 1 is the direction-0 terminus and its forward platform is empty by
construction, making the arithmetic unambiguous. This is recorded because it is the test doing
its job in the direction that matters: catching the *analyst*, not only the code.

Two regression assertions confirm the fixture fails when it should: doubling the arrival rate
doubles the queue integral, and a wrong energy constant changes the measured energy.

---

## 7. Simulator and environment invariants

| invariant | result |
|---|---|
| Passenger conservation (arrivals = alighted + onboard + residual) | **holds**, to 1e-6 |
| Train conservation (fleet size and ids constant) | **holds** |
| Occupancy never negative, never above capacity | **holds** under saturation |
| Queue sizes never negative | **holds** |
| Headway violations = 0 | **holds** under all four scenarios |
| `left_behind_fraction` ∈ [0, 1] | **holds** at capacities 100/400/800 |
| Rewards finite throughout | **holds** |
| Observations finite throughout | **holds** |
| Episode terminates after exactly 228 epochs, `truncated` always False | **holds** |
| Records cover every epoch, in order | **holds** |
| Same seed → identical trajectory | **holds**, per-epoch rewards equal |
| Different seeds → different realisations | **holds** |
| All three arms see byte-identical demand, failures, surge | **holds** |
| Changing `train_seed` leaves the environment unchanged | **holds** |
| Gymnasium `check_env` | **passes** |

### Sanity environments (protocol §10 stage 2)

| gate | result |
|---|---|
| All trainsets unavailable ⇒ dispatch impossible | mask = `[True, False, False]`; 0 inductions |
| Zero demand ⇒ running service is strictly worse | idle reward 0.0 > busy reward |
| Extreme demand ⇒ running service is strictly better | busy reward > idle reward |
| Doing nothing is **not** reward-optimal | served ≫ idle; idle is >10× worse |

---

## 8. PPO smoke-test results

`tests/test_ppo_smoke.py`, 6 tests, all passing. Tiny budget (128 timesteps, 5 stations, 3-hour
day, 16×16 network) — **software validation only, no result here is reportable.**

| check | result |
|---|---|
| MaskablePPO initialises against this observation and action space | pass |
| Rollout collection and gradient steps run | pass |
| Training changes policy parameters | pass |
| Checkpoint saves and reloads | pass |
| Trained policy evaluates without ever selecting a masked action | pass |
| All three arms train with an **identical parameter count** | pass |

No claim is made about policy quality. An assertion that a 128-step run improves the return would
be a test that passes on noise.

---

## 9. Deterministic baseline sanity results

`results/env_sanity_report.{md,json}` — six deterministic non-RL probe policies on the real
24-station network, 5 evaluation seeds each. **These are not the pre-registered baselines**
(`fixed_timetable`, `heuristic_threshold`, `milp_static`, `milp_rolling` arrive at M7–M9).

Scenario `normal`:

| policy | reward | inductions | wait (pass-h) | p95 wait | left-behind frac | energy | cost | served |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `constant_0_no_service` | −56,301 | 0 | 2,561,353 | — | 0.0000 | 0 | 0 | 0.000 |
| `constant_1` | **−822** | 228 | 10,226 | 210 s | 0.0033 | 10,344 | 416 | 1.000 |
| `constant_2` | −1,548 | 236 | 12,221 | 330 s | 0.0190 | 10,673 | 431 | 1.000 |
| `queue_threshold_100` | −838 | 226 | 10,499 | 210 s | 0.0036 | 10,275 | 413 | 1.000 |
| `queue_threshold_400` | −2,884 | 188 | 18,442 | 894 s | 0.0495 | 8,652 | 346 | 0.997 |
| `queue_threshold_1000` | −8,661 | 163 | 45,061 | 2,376 s | 0.1731 | 7,518 | 300 | 0.997 |

The environment responds monotonically and with the correct sign to changes in dispatch
behaviour, across a 68× range of reward. Under `surge`, every policy scores worse than under
`normal`, as it must.

---

## 10. Current forecasting results (M4)

**M4 is not yet validated. The full fit is still running** (started before this milestone; the
12-candidate quantile grid fits 7 quantiles per candidate on 1,126,584 training rows).

What is established:

| | |
|---|---|
| Blocking gate (protocol §4.5), smoke run | **passed** — pinball 13.34 vs naive 23.30; calibrated PICP₈₀ = 0.7998, inside the pre-registered [0.76, 0.84] band |
| Point-model selection, smoke run | `{lr 0.1, depth 8, min_child_weight 20}`, RMSE 124.89 |
| Point model vs seasonal naive, smoke run | MASE 1.002 vs 1.000 — **no better than the reference floor** |

The smoke run used truncated splits and **is not reportable**. If the point model remains no
better than seasonal naive on the full fit, that is a **finding**, to be reported as one — the
protocol does not require the forecaster to be good, only that its uncertainty be calibrated
before H1 is tested.

**Consequence for M6:** every environment validation above used the `naive_seasonal` in-episode
provider. `TrainedForecastProvider` is implemented and its feature construction mirrors
`src/data/features.py` column for column, but it has not been exercised against fitted models.
Swapping providers changes the *content* of the forecast channels and nothing else about the
environment — no observation index, constraint, reward term or invariant depends on which
provider supplies μ and σ̂.

---

## 11. Known limitations

1. **Observation scaler unfitted.** Protocol §6.1 requires constants from `train`-period days;
   the natural generating policy is the M7 fixed timetable. Runs are currently unnormalised and
   every manifest records `obs_scaler: identity`.
2. **Reward reference scales provisional**, fitted from `constant_1` rather than the fixed
   timetable. Flagged `provisional: true` in the file and asserted by test.
3. **H7 ablation is dimensionality-confounded.** Protocol §6.1 says the availability channel is
   *present only when enabled*, so `ppo_uq_noavail` has 121 inputs against `ppo_uq`'s 123. Unlike
   the σ arms — designed specifically to hold dimensionality fixed — the H7 contrast varies both
   information and input width. Report with that caveat, or re-run zero-filled.
4. **Availability signal noise is unresolved** (A14). No public per-train availability record
   exists for the modelled fleet, so `noise_sd` and `frailty_log_sd` have no defensible value.
   `configs/env.yaml` refuses to load them outside a tagged sensitivity sweep.
5. **Line-aggregate σ assumes station independence** (A-ENV-01), understating true line-level
   spread. Monotone in the per-station σ, so the information available to the policy is unchanged.
6. **The action space is effectively close to binary.** Action 2 is dominated by action 1 in every
   configuration tested — bunching, exactly as A-01 predicted.
7. **Seven M5 quantities remain unvalidatable**, including peak occupancy and the OD distribution,
   both of which feed the primary endpoint. The OD and capacity sweeps stay mandatory.
8. **Energy and cost are not in physical units** (A15/A16 unresolved) and remain embargoed from
   headline claims.

---

## 12. Unresolved scientific risks

**R-1 — The headroom for any method may be small, and the power analysis is stale.**
On the real network under `w_balanced`, `constant_1` scores −822 and the best variant tried scores
−838. The plausible operating region is narrow; badly-tuned policies are several times worse, but
the gap between *sensible* policies is a few percent. Protocol §8.4's power analysis was written
before any of this was measurable. **Before the M11 sweep is authorised, the between-seed standard
deviation must be estimated from real short runs (protocol §10 stage 3) and the power analysis
redone against it.** If the study is underpowered for the effect size actually available, that
must be stated before running, not discovered afterwards. This is the single most consequential
open item.

**R-2 — Reference scales and observation scaler both depend on an M7 artefact.** Both are
currently derived from `constant_1`. They are applied identically to every arm, so no arm is
advantaged, but the reward's *scale* — and therefore the relative weight α…δ carry — will shift
when they are refitted from the fixed timetable. Any run made before that refit must be repeated.

**R-3 — The trained forecaster has never been driven in-episode.** `TrainedForecastProvider`'s
online feature construction is asserted to mirror the offline frame, but until the M4 fit lands it
has not been run against real models. A mismatch would degrade every forecast the agent sees
without failing any current test.

**R-4 — The primary endpoint penalises bunching** (A-01, restated). A policy achieving shorter
waits can leave more passengers behind. Waiting-time percentiles are reported alongside and no
conclusion may rest on `left_behind_fraction` alone.

**R-5 — Two defects of this class were found in M6 alone**, one dating to M2 and undetected
through three milestones and 140 tests. Both were caught by *measuring something that should have
been impossible* — a do-nothing policy scoring the maximum reward, and surge outperforming normal
— not by code review. The corresponding risk is that a third such defect exists and no probe has
yet been pointed at it. The M7–M9 baselines are the next such probe: a MILP that beats the
simulator's physical limits, or a fixed timetable that cannot reproduce the published service
pattern, would expose it.

---

## 13. Completion criteria

| # | criterion | status |
|---|---|---|
| 1 | Gymnasium environment works | **pass** — `check_env` |
| 2 | Observation space validated | **pass** |
| 3 | Action space validated | **pass** |
| 4 | Action distinguishability tested | **pass** |
| 5 | Future-information leakage tests pass | **pass** — 24 tests |
| 6 | Constraints tested (valid and invalid actions) | **pass** |
| 7 | Reward decomposition tested | **pass** — exact reconstruction, every weight row |
| 8 | A-01 endpoint correctly implemented | **pass** — bounded, distinct passengers, unreverted |
| 9 | Deterministic hand-calculated scenario passes | **pass** — 21 assertions |
| 10 | Reproducibility test passes | **pass** |
| 11 | Simulator/environment invariants pass | **pass** |
| 12 | Tiny PPO smoke test passes | **pass** |
| 13 | Deterministic baseline sanity test passes | **pass** |
| 14 | Full test suite passes | **pass** — 301/301 |
| 15 | Documentation updated | **pass** — `rl_environment.md`, A-02, A-03, D-13, D-14 |
| 16 | Git commit created | **pass** — `02109e9` |

---

## 14. Reproducibility

```bash
git checkout 02109e9
python -m venv .venv && .venv/Scripts/pip install -r requirements.lock.txt  # hashes pinned
set PYTHONPATH=<repo root>

# full suite (301 tests)
python -m pytest tests/ -q

# hand-calculated scenario alone
python -m pytest tests/test_env_hand_calculation.py -q

# regenerate the reward reference scales (deterministic, seeds 1001-1008)
python scripts/fit_reward_scales.py --days 8

# regenerate the environment responsiveness report (deterministic, seeds 1-5)
python scripts/env_sanity_report.py

# re-run the M5 simulator validation against published statistics
python -m src.simulator.validate
```

Every artefact above is deterministic given the committed inputs: all stochasticity flows through
named streams derived from `eval_seed` alone, and `environment_streams` takes no method argument.

---

## 15. What is NOT authorised by this document

M7 (fixed timetable), M8 (heuristic), M9 (MILP), M10 (RL baselines) and M11 (full sweep) are
**not** started and are not authorised here. Before M11 in particular:

- the M4 full fit must complete and be validated;
- the reward reference scales and observation scaler must be refitted from the fixed-timetable
  baseline and the provisional flags cleared;
- **R-1 must be resolved**: the between-seed standard deviation measured on real short runs and
  protocol §8.4's power analysis redone against it.
