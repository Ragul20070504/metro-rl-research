# RL environment (M6)

Implements protocol section 6. One episode is one operating day (05:00–24:00) on the modelled
24-station line, with an induction decision every 5 minutes — **228 decision epochs per episode**.

Source of truth for the layout is `src/environment/observation.py`, not this document.
`tests/test_docs_match_code.py` asserts that every block below matches the code, so the two
cannot drift apart.

---

## 1. State

**123 values** with the availability channel on (`ppo_uq`, the proposed method); **121** with it
off (`ppo_uq_noavail`, the H7 ablation). Block order reproduces protocol section 6.1 items 1–13
and must never be reordered — `models/obs_scaler.json` and every trained policy are indexed by
position.

| block | size | index | units | source | kind | normalised | uncertainty |
|---|---:|---|---|---|---|---|---|
| `forecast_mu_h1` | 24 | 0:24 | passengers/hour | forecast | forecast | yes | no |
| `forecast_sigma_h1` | 24 | 24:48 | passengers/hour | forecast | forecast | yes | **yes** |
| `forecast_mu_line_h2h3` | 2 | 48:50 | passengers/hour | forecast | forecast | yes | no |
| `forecast_sigma_line_h2h3` | 2 | 50:52 | passengers/hour | forecast | forecast | yes | **yes** |
| `queue_length` | 24 | 52:76 | passengers | simulator | observed | yes | no |
| `boardings_prev_epoch` | 24 | 76:100 | passengers | simulator | observed | yes | no |
| `fleet_counts` | 3 | 100:103 | trainsets | simulator | observed | yes | no |
| `availability_signal` | 2 | 103:105 | probability | simulator | observed | no | no |
| `trains_per_segment` | 8 | 105:113 | trainsets | simulator | observed | yes | no |
| `occupancy_summary` | 2 | 113:115 | load/capacity | simulator | observed | no | no |
| `time_index` | 3 | 115:118 | dimensionless | clock | observed | no | no |
| `disruption_flags` | 3 | 118:121 | indicator/count/indicator | scenario | observed | yes | no |
| `headway_state` | 2 | 121:123 | seconds | simulator | observed | yes | no |

### Time availability — the leakage-critical column

Every block declares when its value becomes knowable, and the declaration is machine-readable
(`FeatureBlock.time_availability`). None may say "future"; `tests/test_leakage.py` asserts it.

| block | available |
|---|---|
| `forecast_mu_h1`, `forecast_sigma_h1`, `forecast_mu_line_h2h3`, `forecast_sigma_line_h2h3` | issued at `t` from history **strictly earlier** than `t` |
| `queue_length`, `fleet_counts`, `trains_per_segment`, `occupancy_summary`, `time_index`, `headway_state` | realised at or before `t` |
| `boardings_prev_epoch` | realised during `[t−1, t)`; complete at `t` |
| `availability_signal` | drawn at episode start; constant within the episode |
| `disruption_flags` | set by events realised at or before `t` |

**The agent never observes realised future demand.** The forecast is recomputed *inside* the
episode from realised boardings, through `HistoryBuffer`, whose accessors raise rather than
return any hour at or after the issue time. Precomputed forecasts keyed on the simulated day
would not work in any case: the day is a fresh draw from the ground-truth process (protocol
section 5.1), so no stored forecast corresponds to it.

Two empirical leakage tests back the structural guarantee:

- inject a ×100 change into every *later* hour of the operating day → the observation at `t` is
  bit-identical;
- positive control: change something that has *already* happened → the observation moves.

Without the positive control, an environment that ignored its inputs entirely would pass.

### Per-feature detail

- **`forecast_mu_h1` / `forecast_sigma_h1`** — per-station mean and Gaussian-equivalent
  dispersion for the next hour. σ̂ = (q₉₀ − q₁₀) / (2 × 1.2816), with the single scalar
  interval-width multiplier fitted on `val` applied unchanged (protocol section 4.3).
- **`forecast_*_line_h2h3`** — line aggregate at h=2, h=3. Means add. Dispersions are combined
  **in quadrature**, i.e. assuming independence across stations. Station demands are certainly
  correlated, so this *understates* the true line-level spread. Recorded as **assumption
  A-ENV-01**. It is a monotone function of the per-station σ either way, so the information
  available to the policy is unchanged; only the scale is.
- **`queue_length`** — passengers waiting on both platforms of a station.
- **`boardings_prev_epoch`** — realised boardings in the previous 5-minute epoch.
- **`fleet_counts`** — idle at depot / in service / unavailable (failed or on a maintenance hold).
- **`availability_signal`** — mean and minimum noisy availability probability over idle
  trainsets. **Synthetic** (assumption A14, decision D-04): a noisy readout of a latent per-train
  health state drawn at episode start, which *causes* the failure hazard. It is causal, not
  anticipatory — it does not know which train will actually fail. The latent frailty multiplier
  is log-normal normalised to **mean exactly 1**, so the fleet-aggregate failure rate stays equal
  to the value calibrated against published MDBF in M5; `tests/test_environment.py` asserts the
  mean over 60 episodes. No claim of train-health prediction performance is made anywhere.
- **`trains_per_segment`** — in-service trainsets in each of K=4 frozen line segments, direction 0
  then direction 1.
- **`occupancy_summary`** — mean and maximum load factor over in-service trainsets.
- **`time_index`** — sin and cos of time of day, plus fraction of epochs remaining.
- **`disruption_flags`** — `active_failure`, `n_trains_lost_today`, `active_surge`. All three
  describe events that have **already** occurred, or an elevated-demand window the operator is
  **already** in. None announces a future event.
- **`headway_state`** — seconds since the last induction, and the smallest headway realised so
  far today between consecutive departures anywhere on the line.

### Normalisation

Frozen affine scaling, `(x − centre) / scale`, with constants fitted on **`train`-period days
only** and stored in `models/obs_scaler.json`. Never recomputed on `val`/`test`. SB3's
`VecNormalize` running statistics are deliberately **not** used for observations: a running
normaliser adapts differently under each arm and would confound the comparison.

Blocks flagged `normalised=False` — probabilities, load ratios, sin/cos — are already on a sane
scale and are left untouched so the policy's view of them stays interpretable.

An unfitted scaler is the identity. Legitimate for smoke and unit tests; `strict_scaler: true`
makes the environment refuse to start on one, and reportable runs set it.

> **Status.** The scaler is not yet fitted. Protocol section 6.1 requires the constants to come
> from `train`-period days; the natural generating policy is the fixed-timetable baseline, which
> is an **M7** deliverable. `scripts/fit_obs_scaler.py` fits from any policy and records which
> one. Until M7 the environment runs unnormalised, and every run manifest records
> `obs_scaler: identity`.

---

## 2. The three uncertainty arms

All three have **identical observation dimensionality, architecture, parameter count, training
budget, seeds and environment**. Only the *content* of the two σ blocks differs.

| arm | σ channels contain |
|---|---|
| `ppo_point` | constant 0 — informationally empty, shape and parameter count preserved |
| `ppo_placebo` | σ̂ values with the correct marginal distribution but the (station, epoch) association destroyed |
| `ppo_uq` | the calibrated σ̂ from `xgb_quantile` |

**Why the placebo exists.** `ppo_point` versus `ppo_uq` alone cannot distinguish *"uncertainty is
informative"* from *"a wider observation helps"* — the second arm simply has more numbers in it.
A result in which `ppo_uq ≈ ppo_placebo > ppo_point` **refutes** the manuscript's mechanism even
though a two-arm study would report it as a win. That outcome is a legitimate finding and the
protocol requires it be reported as such.

**Implementation deviation, recorded.** Protocol section 6.2 says the placebo σ is *"shuffled
across stations and epochs within each rollout batch"*. The implementation instead draws a
bootstrap sample from a bounded pool (50,000 entries) of every σ value seen so far. Both preserve
the marginal exactly and both destroy the association; the bootstrap is O(1) per element, whereas
permuting the pool on each of 228 epochs per episode would dominate rollout time. Asserted
directly in `tests/test_environment.py`: the pooled mean and standard deviation match the truth
to within 5 % and 10 %, and the element-wise correlation with the truth falls below 0.10.

The pool persists **across episodes**, because a rollout batch does. Clearing it per episode
would confine the shuffle to one day and let the agent recover the within-day σ trajectory.

**Common random numbers survive the transform.** The placebo shuffle draws from an *agent-side*
stream keyed on `(method, train_seed)`, never from the environment streams keyed on `eval_seed`.
`tests/test_environment.py` asserts that all three arms see byte-identical demand, failures and
surge, and that changing `train_seed` leaves the environment untouched.

---

## 3. Action

`Discrete(3)` — induct 0, 1 or 2 trainsets during the next 5-minute interval.

`k_max = 2` is not arbitrary: at a 150 s minimum headway a terminal can clear at most
300 / 150 = 2 departures per decision interval, so a larger `k_max` would add actions the mask
could never enable.

### Feasibility masking

Action `k` is masked out unless **all** hold: `k ≤` idle trainsets; the minimum headway on the
departure segment is respected; the terminal platform is free; and each inducted trainset is
predicted-available (when that channel is on) or actually available. **Action 0 is always
feasible**, so the mask is never empty and the policy can never be trapped. The mask is also
contiguous from zero — feasibility in `k` is monotone — which `tests/test_action_masking.py`
asserts.

An attempted masked action **raises `InfeasibleActionError`** rather than silently no-opping, and
simulator state is unchanged afterwards (time, epoch, totals, queues and train statuses are all
asserted equal). A silent no-op would let an unmasked policy appear compliant while ignoring the
constraint, and the resulting zero headway-violation count would mean nothing.

Because masking is structural rather than a reward penalty, **headway violations are zero by
construction**, asserted under all four scenario families. A non-zero value is a bug, not a
trade-off.

### Requested versus delivered

`feasible_inductions()` may permit 2 while the infrastructure delivers 1: the second induction is
attempted on each internal step and expires at the end of the interval if the terminal never
frees up. That is correct behaviour — the policy asked for service the infrastructure could not
deliver — and the shortfall surfaces as protocol section 7.4's **induction success rate** rather
than being silently granted.

### Action distinguishability

The M5 defect in which all inductions fired at the epoch boundary made actions 1 and 2
behaviourally identical, collapsing a three-way action space to a binary one with nothing in the
results to show it. Regression tests now assert that the three actions differ in dispatch count,
fleet state, energy, cost, passenger outcomes and reward.

**Measured, and worth stating plainly: action 2 is dominated by action 1 in every configuration
tested.** Asking for two inductions inside one 300 s interval produces a closely spaced pair
followed by a long gap rather than two evenly spaced departures. The same number of round trips
completes, but headways are less regular, so the 95th-percentile wait rises (210 s → 330 s on the
real network) and roughly six times as many passengers are passed over by a full train. This is
the bunching limitation recorded in protocol amendment **A-01**, appearing in the simulator
exactly as that amendment predicted. It is a property of the modelled infrastructure, not a
defect — but it means the *effective* action space is close to binary, and any policy that learns
to use action 2 heavily is doing something wrong.

---

## 4. Reward

```
r_t = −( α·W_t + β·U_t + γ·E_t + δ·C_t )
```

| term | quantity | units | amendment |
|---|---|---|---|
| `W_t` | passenger-minutes of platform waiting **accrued during epoch t** (the queue integral) | passenger-minutes | **A-02** (decision D-13) |
| `U_t` | distinct passengers left behind by a departing full train | passengers | **A-01** (decision D-12) |
| `E_t` | energy | train-km-equivalent units (A15 unresolved) | — |
| `C_t` | dispatch and fleet cost | induction-equivalent units (A16 unresolved) | — |

Each term is divided by a fixed **reference scale** (`configs/reward_scales.json`) computed once
from a reference policy over `train`-period days and never recomputed, so α…δ act on
dimensionless quantities. A useful side effect: the unresolved energy and cost constants largely
cancel, because each appears in both its term and its own reference scale.

> **Status.** Protocol section 6.4 names the fixed-timetable baseline as the reference policy;
> that baseline is an **M7** deliverable. The committed file is fitted from `constant_1` over
> 8 episodes on seeds 1001–1008 — disjoint from both the training pool (101–110) and the
> evaluation pool (1–30), so no component of the objective is tuned on the days the comparison is
> scored — and is marked `provisional: true`. It must be regenerated with
> `--policy fixed_timetable` before any reportable run.

**Component logging is mandatory.** Every epoch records `W_t, U_t, E_t, C_t` in physical units,
the four normalised terms and `r_t`. `tests/test_reward.py` asserts the stored components
reconstruct `r_t` exactly, for every row of the frozen weight grid and across a whole episode.
No experiment reports the scalar reward alone.

`VecNormalize(norm_reward=True)` is **off**, so reward scale stays interpretable and identical
across arms.

### Why `W_t` is the queue integral (amendment A-02)

`W_t` was first implemented as the mean wait of passengers who *boarded* in the epoch. That made
the reward degenerate. Measured on the real network:

| policy | W (mean over boarders) | U | E | C | reward |
|---|---:|---:|---:|---:|---:|
| do nothing, all day | 0.0 | 0.0 | 0 | 0 | **0.0 — global maximum** |
| minimal service | 455 min | 1,631 | 10,349 | 416 | large negative |

Running no service scored the best attainable reward while stranding 100 % of demand: nobody
boards so there is no mean; no train departs so nobody is left behind; nothing moves so energy
and cost vanish. All four terms went blind simultaneously.

The mean over boarders also contradicts protocol section 6.4's own wording — it reports waiting
that accumulated in *earlier* epochs, not waiting *"accrued in epoch t"*.

Candidate readings, do-nothing versus minimal service:

| reading | do nothing | minimal | ratio | degenerate? |
|---|---:|---:|---:|---|
| mean over boarders *(superseded)* | 0 | 455 | 0× | **yes** |
| accrued ÷ passengers waiting | 1,123 | 387 | 2.9× | **yes** — E+C = 0 still wins |
| **accrued passenger-minutes (adopted)** | 151,399,865 | 607,853 | **249×** | no |
| accrued ÷ passengers arriving | 142,924 | 675 | 212× | no |

The queue integral is the standard transit-OR objective, is defined whether or not anybody
boards, and has no zero-denominator edge case. `tests/test_reward.py` and
`tests/test_environment.py` both carry regression guards that fail if the superseded reading
returns. The mean over boarders is still logged, as a diagnostic.

### Weight grid

Frozen in code, not config, so a run cannot invent a row that was never pre-registered.

| id | α | β | γ | δ | role |
|---|---|---|---|---|---|
| `w_balanced` | 1.0 | 1.0 | 1.0 | 1.0 | **primary** — the confirmatory H1 test |
| `w_service` | 2.0 | 2.0 | 0.5 | 0.5 | service-priority operator |
| `w_cost` | 0.5 | 0.5 | 2.0 | 2.0 | cost-priority operator |
| `w_wait_only` | 1.0 | 0.0 | 0.0 | 0.0 | single-objective arm for H3 |
| `w_unmet_heavy` | 1.0 | 3.0 | 1.0 | 1.0 | stress on the primary endpoint's term |

---

## 5. Episode structure

`reset()` draws a scenario instance from `eval_seed`, resets the simulator, draws latent train
health, installs pre-episode history and resets the forecaster.

`step(action)` validates the mask, advances one 5-minute epoch (10 internal 30 s steps), computes
the reward, appends the epoch record and reassembles the observation.

`terminated` is True after 228 epochs. `truncated` is always False — the episode has a natural
horizon (the operating day), so there is nothing to truncate.

`info` carries `components`, `epoch_metrics`, `action_mask`, `scenario`, `eval_seed`, and on the
final step `episode_summary`. Non-finite values in `info` are rendered as `None`: the mean wait
over boarders is genuinely undefined in an epoch where nobody boarded, and `NaN != NaN` would
make Gymnasium's checker report this deterministic environment as non-deterministic. The honest
NaN is preserved in `episode_records()`, which is what downstream analysis reads.

---

## 6. Determinism and common random numbers

Environment streams come from `environment_streams(scenario, eval_seed)`, which **takes no method
argument** — the demand and failure realisations cannot depend on which policy is running.

Since M6 the **scenario name is excluded from the stream entropy** (decision D-14). It used to be
part of it, which meant `normal` seed 1 and `surge` seed 1 were unrelated realisations — matched
in name only — while protocol section 7.4 defines the degradation ratio as a stressed metric
divided by the `normal` metric *matched on* `eval_seed`. The confound was large enough to invert
the sign: `surge` scored *better* than `normal` in three of five seeds. With the scenario name
removed, all four families share one base world per seed, and because surge draws and failure
draws use separate streams, `combined` reuses exactly the surge of `surge` and the failures of
`disruption`. See `docs/decisions.md` D-14.

Pre-episode history is drawn from a dedicated `history` stream, so how much history the forecaster
needs cannot shift the realised arrivals.

---

## 7. Known limitations

1. **The observation scaler is unfitted**, pending the M7 fixed-timetable baseline. Runs are
   currently unnormalised.
2. **The reward reference scales are provisional**, fitted from `constant_1` rather than the
   fixed-timetable baseline. Marked `provisional: true` in the file.
3. **The H7 ablation is dimensionality-confounded.** Protocol section 6.1 specifies the
   availability channel is *present only when enabled*, so `ppo_uq_noavail` has 121 inputs
   against `ppo_uq`'s 123. Unlike the σ arms — which are explicitly designed to hold
   dimensionality fixed — the H7 contrast therefore varies both the information and the input
   width. Any H7 result must be reported with that caveat, or the ablation should be re-run with
   the channel zero-filled instead of removed.
4. **The availability signal's noise level is unresolved** (assumption A14). No public per-train
   availability record exists for the modelled fleet, so `noise_sd` and `frailty_log_sd` have no
   defensible value and must be swept.
5. **Line-aggregate σ assumes independence across stations** (A-ENV-01), understating the true
   line-level spread.
6. **The action space is effectively close to binary** — action 2 is dominated by action 1 in
   every configuration tested.
7. **The headroom for any method looks small.** On the real network under `w_balanced`, a
   constant "induct 1 per epoch" policy scores −822 and the best threshold variant tried scores
   −838; the worst sensible policies are several times worse. If the achievable improvement is a
   few percent, the effect size for H1 is small, and the power analysis in protocol section 8.4
   must be redone against a between-seed standard deviation measured on real runs before the full
   sweep is authorised. This is flagged in `docs/m6_scientific_gate.md` as an unresolved
   scientific risk.
8. **The in-episode forecaster used for all M6 validation is `naive_seasonal`**, not the trained
   `xgb_quantile`. `TrainedForecastProvider` is implemented and its feature construction mirrors
   `src/data/features.py`, but the M4 full fit had not completed when M6 was validated. Swapping
   the provider changes the *content* of the forecast channels and nothing else about the
   environment.
