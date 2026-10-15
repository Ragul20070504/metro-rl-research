# Experimental Protocol (M1) — FROZEN SPECIFICATION

**Status:** M1 deliverable. This document is the pre-registration. It is written **before**
implementation and **before** any experiment is run.

**Freeze rule.** Once this document is committed and `experiments/registry.yaml` is marked
`frozen: true`, the specifications in §§3–8 may not be changed to improve a result. Any change after
freeze is recorded as a dated **amendment** in §12, with the reason, made *before* the affected runs
are executed. Amendments made after seeing the affected results are not permitted; if one becomes
unavoidable, the affected experiment is re-registered with a new `experiment_id` and the original is
retained with `status: superseded`.

**Values marked `TBD@Mx`** are quantities that cannot be responsibly fixed today (they depend on data
that has not been downloaded, or on a compute pilot that has not run). Each has a named milestone by
which it must be frozen, and each must be frozen **before** the first run that consumes it. `TBD@Mx`
is not a licence to tune after seeing results.

---

## 1. Scope and unit of study

**Configuration under study:** one isolated metro line, both directions, single operating day.
**Primary candidate:** the BMT Canarsie Line (L service), NYC Transit — selected because it is
documented as hosting a single service on unshared trackage with a dedicated fleet, so the
"single line, no inter-service interference" abstraction is a property of the real system rather than
a modelling convenience (`docs/research_audit.md` §F.3).

Two frozen network configurations:

| Config id | Purpose | Stations | Notes |
|---|---|---|---|
| `net_toy_10` | Smoke tests, unit tests, sanity environments, CI | 10, synthetic | Never used for reported results. |
| `net_l_full` | **All reported experiments** | Full station list of the modelled line, derived from *MTA Subway Stations* + GTFS at ingestion | Station count is **not asserted here**; it is written to `data/processed/network/MANIFEST.json`. |

**DECIDED 2026-08-26 (D-02): no `net_l_reduced` configuration is created.** Truncating a real line
severs passenger trips at the cut and would force a further synthetic assumption on top of the
already synthetic OD model (A03). The line was chosen precisely because its isolation is a property
of the real system; discarding that to hit a station count would trade real validity for nominal
compliance with the brief's "5–10 stations" recommendation. That recommendation exists to control
compute, which is handled instead by the staged gating in §10. See `docs/decisions.md`.

---

## 2. Datasets

Every dataset below is ingested by `src/data/download.py` into `data/raw/<dataset_id>/`, which is
**immutable** thereafter. At ingestion the loader writes `data/raw/<dataset_id>/MANIFEST.json`
containing: `source_url`, `download_utc`, `sha256` per file, `row_count`, `column_list`,
`socrata_view_id`, `dataset_last_updated` (from the source metadata), `license`, `attribution`,
`retrieval_tool_version`. **No dataset field is asserted in documentation until it appears in a
manifest.**

All identifiers below were **probed and verified on 2026-08-26** by `scripts/probe_sources.py`;
evidence in `results/source_probe.json`. Full detail, limitations and the two identifiers the M0
audit got wrong are in `docs/data_sources.md`.

| dataset_id | Identifier | Verified coverage | Role | Known limitations |
|---|---|---|---|---|
| `mta_hourly_ridership_2017` | `t69i-h2me` | 2017-01-13 → 2019-12-31 | Demand — pre-pandemic regime (out-of-regime robustness check) | as below |
| `mta_hourly_ridership_2020` | `wujg-7c2s` | 2020-01-01 → 2024-12-31 | Demand ground truth; forecasting target | **Hourly**; **entries + transfers only, no exits, no OD**; station-*complex* granularity; estimates, not raw counts; numeric fields arrive as strings |
| `mta_hourly_ridership_2025` | `5wq4-mkjj` | 2025-01-01 → 2026-08-13 | Demand — most recent regime | as above |
| `mta_gtfs_static` | `rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip` | snapshot, 5.6 MB, no API key | Inter-station running times, scheduled headways, trip counts, `fixed_timetable` baseline | Scheduled not realised; snapshot only, no archive of older editions — hash must be pinned |
| `mta_subway_stations` | `39hk-dx4f` | current | Complex↔stop join, line membership, ordering | — |
| `mta_mdbf` | `e2qc-xgxs` | 2015-01 → 2026-06 | **Calibrates the aggregate failure rate** of the synthetic availability process (R143 confirmed under division B) | Monthly, by car class; not per train, not per day |
| `mta_cjfm` | `r7qk-6tcy` | 2015-01 → 2026-06 | **Validation**: Additional Platform Time / Additional Train Time, split peak vs offpeak | Aggregated to month × line × period |
| `mta_wait_assessment` | `s666-h6b7` | 2015-01 → 2026-06 | **Validation**: headway regularity | Selected timepoints only |
| `mta_service_delivered` | `32ch-sei3` | 2015-01 → 2026-06 | **Validation**: scheduled vs actual trains; real reference for dispatch success rate | Monthly; published `service_delivered` field is mis-scaled — recompute from the counts |
| `mta_running_times` | `sp9g-mzjh` | 2019-01 → 2026-06 | **Validation**: realised end-to-end runtime **distribution** (25th/50th/75th percentiles) by direction, day type and period | Aggregated to month; no per-train trajectories |
| `mta_terminal_otp` | `f6rf-2a3t` | 2015-01 → 2026-06 | **Validation**: punctuality (secondary) | Monthly |
| `metropt3` *(conditional)* | UCI #791, DOI 10.24432/C5VW3R, **CC BY 4.0** | Feb–Aug 2020 | Only under D-04's optional extension: estimate an achievable failure-detector operating point | **One train's Air Production Unit, Metro do Porto; unlabelled + 4 documented air-leak failures.** Cross-domain transfer is an explicit assumption |

**Substitutions relative to the manuscript** (§5 of the manuscript names "NYC MTA turnstile counts"
and "published Beijing Metro open data"):

- *Turnstile counts* → replaced by `mta_hourly_ridership`, which supersedes them and is strictly
  higher resolution. Recorded as **MC-03**.
- *Beijing Metro open data* → **dropped**. Could not be confirmed as publicly downloadable; access
  appears to be case-by-case under data-use agreement. Recorded as **MC-02**. No substitute is
  needed because the NYC source is sufficient for a single-line study.

**Licence handling.** Each manifest records the licence text/URL as published at download time. NY
Open Data terms and MTA attribution requirements are reproduced in `data/raw/<id>/LICENSE.txt`.
Raw data files are **git-ignored**; manifests and hashes are committed, so a fresh clone can verify
that it re-downloaded the same bytes.

---

## 3. Data pipeline and splits — FROZEN

Pipeline: `raw → validate → clean → resample/align → feature-engineer → split → model-ready parquet`.
`data/raw/` is never written to after ingestion (enforced by `tests/test_raw_immutable.py`, which
re-checks the manifest hashes).

### 3.1 Temporal split (no random splitting anywhere)

Split is **chronological, by calendar date, at the day boundary**, with an embargo gap.

| Split | Purpose | Rule |
|---|---|---|
| `train` | Forecaster fitting; simulator calibration; heuristic threshold tuning; RL training-day pool | Earliest available date … `T_train_end` |
| `gap_1` | Leakage embargo | ≥ 7 days, discarded |
| `val` | Forecaster hyperparameters; calibration check; **all** RL/PPO hyperparameter selection; MILP tuning | `T_val_start` … `T_val_end` |
| `gap_2` | Leakage embargo | ≥ 7 days, discarded |
| `test` | **Final evaluation only.** Touched once, after everything is frozen. | `T_test_start` … end |

`T_*` boundaries: **TBD@M3**, frozen in `configs/data.yaml` once the ingested date range is known,
chosen by a fixed rule (proportions ≈ 70/15/15 of available days, rounded to whole weeks, boundaries
on a Sunday/Monday break) and **not** by inspecting model performance.

**Regime handling — strengthened after source verification (amendment D-11).** The demand series is
strongly non-stationary across 2020–2022. The M0 audit treated this as a threat to be managed
within a single 2020–2024 dataset. Verification on 2026-08-26 showed the hourly series is actually
continuous from **2017-01-13 to 2026-08-13** across three same-schema datasets, so the design
changes from *managing* the regime break to *avoiding* it:

- The **primary study window sits wholly inside the post-recovery regime**, whose start date is
  `TBD@M3`, chosen by a pre-registered stationarity screen (rolling 28-day mean of line-level daily
  ridership within ±X % of the trailing year; screen and parameters in `configs/data.yaml`).
- The screen is fitted on the **training** portion only and never looks at `test`.
- The **2017-01-13 → 2019-12-31 pre-pandemic period is reserved as an out-of-regime robustness
  check**, not merged into training. It is used, if at all, only to report how far conclusions
  transfer across regimes — registered separately so it cannot become an unlogged extra test set.

This removes the confound in which a structural regime change is mistaken for forecast error, which
would have propagated straight into the uncertainty channel and therefore into H1.

### 3.2 Data validation tests (must pass before any modelling)

`tests/test_data_quality.py` — each an assertion, each reported in `results/data_quality_report.md`:

1. no missing `transit_timestamp`, `station_complex_id`, `ridership`
2. no exact duplicate (timestamp, complex, payment_method, fare_class) rows
3. timestamps parse, are timezone-resolved, fall inside the declared coverage, and contain no
   impossible values (e.g. DST-gap local times)
4. `ridership >= 0`, `transfers >= 0`
5. outlier screen: per-station hourly values beyond a frozen robust bound (median ± k·MAD, `k`
   in `configs/data.yaml`) are **flagged and counted, never silently deleted**
6. station identifiers all resolve against `mta_subway_stations`; unmatched complexes are reported
7. temporal continuity: for each station, count missing hours per day; days with > `max_missing_hours`
   are excluded and the exclusion list is written to `data/interim/excluded_days.csv`
8. split integrity: `train`, `val`, `test` date sets are disjoint and correctly ordered, and the
   embargo gaps are non-empty
9. distribution shift: report (not gate) PSI / KS statistics for demand between `train`, `val`, `test`
10. **leakage screen:** every engineered feature column is checked by name and by construction
    against a whitelist of allowed lags; any feature whose computation window includes t ≥ prediction
    time fails the test

---

## 4. Demand forecasting — FROZEN

### 4.1 Task definition

- **Target:** boardings per station-complex per hour, on the modelled line, direction-agnostic
  (the source data is not direction-resolved — this is a stated limitation; direction split in the
  simulator is governed by assumption A03).
- **Time resolution:** 1 hour (matches the source; **do not** claim 5-minute forecasting accuracy
  we cannot evaluate).
- **Horizon:** h ∈ {1, 2, 3} hours ahead, issued at the top of each hour.
- **Regime:** rolling-origin (walk-forward) evaluation; the model may use data strictly earlier
  than the forecast issue time.

### 4.2 Input features (frozen whitelist)

Lags of the target: h+1 … h+6 hours; same-hour lags at 24 h, 48 h, 168 h; rolling means over 3 h,
24 h, 168 h computed on strictly past data; hour-of-day (sin/cos); day-of-week; is-weekend;
is-public-holiday (from a static calendar); station-complex id (categorical); station position index
along the line. **No weather, no events** in the frozen set — the manuscript proposes them (§4.1) but
adding unvalidated exogenous features widens the leakage surface and is not needed for H1. Logged as
a documented scope reduction, **MC-11**.

### 4.3 Models compared (all three are mandatory)

| id | Model | Role |
|---|---|---|
| `naive_seasonal` | Seasonal naive: ŷ(s, t+h) = y(s, t+h−168 h) | Reference floor |
| `xgb_point` | XGBoost regressor, squared error | Point-forecast baseline; supplies μ to the **PPO-Point** arm |
| `xgb_quantile` | XGBoost, pinball loss, quantiles {0.10, 0.50, 0.90} (+ {0.05, 0.25, 0.75, 0.95} for calibration diagnostics) | Probabilistic model; supplies μ and σ to the **PPO-UQ** arm |

Uncertainty representation frozen: **σ̂ = (q₀.₉₀ − q₀.₁₀) / (2 · z₀.₉₀)** with z₀.₉₀ = 1.2816, i.e. a
Gaussian-equivalent scale from the central 80 % interval. The RL state receives (μ = q₀.₅₀, σ̂).
This is a deliberate, documented simplification of a full predictive distribution; the alternative of
feeding raw quantiles is recorded as a future ablation, not run.

**Post-hoc calibration:** a single scalar interval-width multiplier fitted on `val` by minimising the
absolute deviation of empirical 80 % coverage from 0.80. Fitted on `val` only, applied unchanged to
`test`.

**Hyperparameters:** a frozen small grid (`configs/forecasting.yaml`), selected on `val` by pinball
loss for `xgb_quantile` and RMSE for `xgb_point`, identical search budget for both. Grid contents
`TBD@M4`, frozen before fitting.

### 4.4 Metrics (all reported; point-accuracy alone is insufficient)

Point: MAE, RMSE, **MASE** (scale-free, robust to the near-zero night-time counts that make MAPE
undefined/explosive — MAPE and SMAPE are reported *only* for hours with target ≥ a frozen floor, and
that floor is stated). Probabilistic: pinball loss per quantile and averaged; **PICP** (80 % and
90 %); **MPIW** and normalised MPIW; **Winkler/interval score**; **CRPS** (approximated from the
quantile set); reliability diagram data; **ECE** over predicted quantile levels.

### 4.5 Gate (blocking)

**RL training may not consume forecasts until:** (i) `xgb_quantile` beats `naive_seasonal` on pinball
loss on `val`, **and** (ii) post-calibration empirical 80 % coverage on `val` lies in
[0.76, 0.84]. If (ii) fails, the uncertainty channel is not trustworthy and H1 cannot be tested; the
failure is recorded and the forecaster is revised **before** any RL run, not after seeing RL results.

### 4.6 Stored output

`data/processed/forecasts/forecast.parquet` with columns:
`issue_time, target_time, horizon_h, station_id, split, actual, p10, p50, p90, mean, sigma,
model_id, model_version, calibrated (bool)`.

---

## 5. Digital-twin simulator — FROZEN

**Terminology (MC-04).** Referred to throughout as a **calibrated simulation environment
(digital-twin-style)**. It is not coupled to a live asset and the term "digital twin" is used only
with that qualifier.

### 5.1 Demand generation — the correction that makes H1 testable

**This supersedes manuscript §5**, which specifies arrivals "generated as Poisson processes whose
rate is set by the demand-forecast model plus noise". That design is circular (`research_audit.md`
§G-1): it makes the forecaster correct by construction and reduces σ to a parameter we chose.

**Frozen replacement:**

1. A **ground-truth demand process** `D_true` is fitted to *observed* `mta_hourly_ridership` on
   `train` days: a per-station, per-hour-of-week intensity with a fitted over-dispersion term
   (negative-binomial or Poisson-with-random-effect; family selected on `train`/`val` by fit
   diagnostics, frozen at M5).
2. An episode draws a **day** and samples an hourly intensity path from `D_true`, then realises
   5-minute arrivals as a non-homogeneous Poisson process using the frozen intra-hour profile (A02).
3. The **forecaster never sees** the sampled path. It receives only the observable history
   (previous realised hours of the same episode plus the historical training data) and produces
   (μ, σ̂). Forecast error therefore arises from the genuine difficulty of the task.
4. `tests/test_no_circular_demand.py` asserts that the simulator's demand module holds no reference
   to any forecasting artefact, and that perturbing the forecaster's parameters leaves the realised
   demand stream bit-identical under a fixed seed.

### 5.2 Frozen simulator components and provenance

Every parameter lives in `configs/simulator.yaml` with a mandatory `provenance` field taking one of:
`real_data:<dataset_id>`, `literature:<citation>`, `simulation_assumption`, `synthetic`, `unresolved`.
A config with any `unresolved` parameter **fails to load** unless the run is explicitly tagged
`sensitivity_sweep`.

| Component | Specification | Provenance class |
|---|---|---|
| Station topology & order | From `mta_subway_stations` + `mta_gtfs_static` | `real_data` |
| Inter-station running times | Median scheduled segment time from GTFS `stop_times`, per direction, per time band | `real_data` |
| Running-time noise | Multiplicative lognormal, σ frozen at M5 | `simulation_assumption` (swept) |
| Dwell time | `dwell = d_min + c_board · boarders + c_alight · alighters`, capped at `d_max` | `d_min`, `d_max` from GTFS-implied dwell; `c_*` **unresolved** → swept (A08) |
| Train capacity | seats+standees per trainset | **unresolved** (A09) → swept until a citable figure with provenance exists |
| Fleet size | Trainsets available at 05:00 | `simulation_assumption`, set consistent with GTFS peak trip counts; recorded |
| Minimum headway | Enforced per segment via action masking | `simulation_assumption` (swept), cross-checked against GTFS minimum observed scheduled headway |
| Turnaround time | Fixed terminal turnaround | `simulation_assumption` (swept) |
| Passenger arrivals | §5.1 | `real_data`-fitted + `simulation_assumption` |
| Boarding | FIFO by arrival at platform, board first train toward destination with residual capacity | `simulation_assumption` (manuscript §5) |
| Alighting / OD | Distance-decay OD split, three variants swept: `uniform_downstream`, `distance_decay`, `cbd_attracted` | **synthetic** (A03) — dominant sensitivity |
| Passenger patience | Infinite by default; `finite_patience` variant swept | `simulation_assumption` (A06) |
| Train failures | Poisson hazard per train-km; rate calibrated so simulated failures/1000 train-km match `mta_mdbf` for the relevant car class and period | **synthetic, rate-calibrated** (A13) |
| Maintenance holds | Fixed fraction of fleet unavailable at day start | `simulation_assumption` |
| Demand surge | ×U(1.30, 1.50) over a contiguous window; window start/length drawn from a frozen distribution | manuscript §6 for the magnitude; `simulation_assumption` for the shape |
| Energy model | `E = e_km · train_km + e_stop · stops` | **unresolved** (A15) → swept; embargoed from headline claims |
| Cost model | `C = c_induct · inductions + c_hour · train_hours` | **unresolved** (A16) → swept; embargoed from headline claims |
| Decision interval | 5 min | brief §10 / manuscript §3 |
| Operating day | 05:00–24:00 local, one episode = one day | `simulation_assumption` |

### 5.3 Simulator validation (M5 gate) → `docs/simulator_validation.md`

Simulated statistics under the **fixed-timetable** baseline on `train`-period days are compared with
real published statistics. Each row reports the real value, the simulated value, the absolute and
relative error, and a **verdict**: `validated` / `partially validated` / **`not validatable`**.

| Quantity | Real reference | Expected verdict |
|---|---|---|
| Hourly line-level demand profile | `mta_hourly_ridership_*` (held-out days) | validatable |
| Cross-station demand distribution | `mta_hourly_ridership_*` | validatable |
| Scheduled headway distribution | `mta_gtfs_static` | validatable |
| Headway regularity / evenness at peak | `mta_wait_assessment` (`s666-h6b7`) | validatable |
| Excess passenger platform wait | `mta_cjfm` Additional Platform Time, peak vs offpeak | validatable |
| Excess in-vehicle time | `mta_cjfm` Additional Train Time | validatable |
| **End-to-end running time — mean *and dispersion*** | `mta_running_times` (`sp9g-mzjh`): 25th/50th/75th percentiles by direction, day type, period | **validatable — upgraded** |
| **Trains actually run vs scheduled (dispatch success rate)** | `mta_service_delivered` (`32ch-sei3`), ratio recomputed from the counts | **validatable — new** |
| Terminal punctuality | `mta_terminal_otp` (`f6rf-2a3t`) | validatable (secondary) |
| Train-failure frequency | `mta_mdbf` (`e2qc-xgxs`), R143 / division B | validatable (aggregate rate only) |
| Train capacity / occupancy | — | **not validatable** — no public occupancy data |
| Dwell-time distribution | — | **not validatable** at the needed resolution |
| Passenger waiting behaviour (balking, patience) | — | **not validatable** |
| Per-train trajectories | — | **not validatable** — GTFS-RT has no historical archive |

**OBSERVATION — this table improved after the M0 audit, and one M0 claim was wrong.** The audit
stated that realised train movements are not publicly retained, and classified running-time noise
(A07) as an unvalidatable simulation assumption. `sp9g-mzjh` publishes realised end-to-end runtime
*percentiles*, so the simulator's running-time dispersion can be calibrated against a real
distribution rather than chosen. `32ch-sei3` likewise supplies a real comparator for dispatch
success rate, which the audit had none for.

Four rows still come back **not validatable**, and they are reported as such rather than hidden;
brief §11 requires exactly this. Note that the two most consequential unvalidatable quantities —
occupancy and passenger patience — are precisely the ones feeding the primary endpoint through
assumption A03, which is why the OD sweep (`EXP-SENS-002`) is not optional.

---

## 6. RL environment, action space and reward — FROZEN

### 6.1 Observation (`src/environment/metro_env.py`)

Per decision epoch t, the observation vector is the concatenation, in this fixed order:

1. per-station forecast mean μ(s, h=1) — N_stations values
2. per-station forecast dispersion σ̂(s, h=1) — N_stations values *(see arm definitions below)*
3. line-aggregate forecast mean for h=2, h=3 — 2 values
4. line-aggregate forecast dispersion for h=2, h=3 — 2 values *(arm-dependent)*
5. per-station current platform queue length — N_stations values
6. per-station realised boardings in the previous epoch — N_stations values
7. fleet counts: n_idle, n_in_service, n_unavailable — 3 values
8. train-availability summary: mean and min predicted availability probability over idle trains —
   2 values *(present only when the availability channel is enabled, H7)*
9. rolling-stock position summary: number of trains in each of K frozen line segments, per direction
10. occupancy summary: mean and max occupancy ratio over in-service trains — 2 values
11. time index: sin/cos of time-of-day, plus epochs-remaining fraction — 3 values
12. disruption flags: active_failure (0/1), n_trains_lost_today, active_surge (0/1) — 3 values
13. headway state: time since last induction, current minimum observed headway — 2 values

**Normalisation:** frozen affine scaling with constants computed **on `train` days only** and stored
in `models/obs_scaler.json`. Not recomputed on `val`/`test`. `VecNormalize` running statistics are
**not** used for observations, to avoid an arm-dependent normalisation confound.

**The agent never observes realised future demand.** Enforced by `tests/test_leakage.py` (§9).

### 6.2 The three uncertainty arms — the critical design detail

All three arms have **identical observation dimensionality, identical architecture, identical
parameter count, identical training budget, identical seeds, identical environment**. Only the
*content* of the σ channels (items 2 and 4) differs:

| Arm | σ channels contain |
|---|---|
| `PPO-Point` | constant 0 (informationally empty; preserves shape and parameter count) |
| `PPO-Placebo` | σ̂ values **shuffled** across stations and epochs within each rollout batch (correct marginal distribution, destroyed information) |
| `PPO-UQ` | the calibrated σ̂ from `xgb_quantile` |

**Rationale.** `PPO-Point` vs `PPO-UQ` alone cannot separate "uncertainty is informative" from
"a wider observation helps". `PPO-Placebo` is the control that separates them. A result in which
`PPO-UQ ≈ PPO-Placebo > PPO-Point` refutes the manuscript's mechanism even if it looks like a win.

### 6.3 Action space

`Discrete(K_max + 1)` — the number of trainsets to induct during the next 5-minute interval, per
line. `K_max` frozen in `configs/env.yaml` (`TBD@M6`, small: expected 2–3). Multi-line extension
keeps the manuscript's `[action_line_1, action_line_2, …]` form via `MultiDiscrete`, unused in the
frozen single-line study.

**Feasible-action masking** (manuscript §3.4 requires masking, not penalties). A candidate action *k*
is masked out unless **all** hold: k ≤ available idle trainsets; inducting k respects minimum headway
on the departure segment; each inducted trainset has completed turnaround; each inducted trainset is
predicted-available (when the availability channel is on) or actually available; platform/terminal
capacity is not exceeded. Action 0 is always feasible, so the mask is never empty.
`tests/test_action_masking.py` asserts: no masked action is ever executed; an attempted masked action
raises rather than silently no-ops; and the simulator state is unchanged after a rejected action.

### 6.4 Reward

Exactly the manuscript's form (§3.3):

```
r_t = -( α·Ŵ_t + β·U_t + γ·E_t + δ·C_t )
```

- `Ŵ_t` — **amended by A-02:** passenger-minutes of platform waiting accrued during epoch t, by
  everyone on a platform whether served or not (the queue integral). The superseded reading —
  the mean wait of passengers who *boarded* in the epoch — made "run no service all day" the
  reward-optimal policy. See §12 A-02.
- `U_t` — unmet demand in epoch t (passengers left behind)
- `E_t` — energy in epoch t (kWh)
- `C_t` — dispatch/fleet cost in epoch t (cost units)

**Unit problem (not addressed in the manuscript).** The four terms have incommensurable units, so
α…δ are not interpretable as priorities unless the terms are normalised first. Frozen solution: each
term is divided by a **fixed reference scale** `ref_W, ref_U, ref_E, ref_C` computed once from the
fixed-timetable baseline over `train`-period days, stored in `configs/reward_scales.json`, and never
recomputed. α…δ then act on dimensionless quantities.

**Component logging is mandatory.** Every step writes `W_t, U_t, E_t, C_t` in physical units, the
normalised terms, and `r_t`, to the episode record. `tests/test_reward.py` asserts that the stored
components reconstruct `r_t` exactly. **No experiment reports the scalar reward alone.**

**Reward normalisation:** SB3 `VecNormalize(norm_reward=True)` is **off** by default so that reward
scale stays interpretable and identical across arms; if enabled it is enabled identically for all
arms and recorded.

### 6.5 Reward-weight configurations (frozen grid)

Weights are never chosen and left unjustified (brief §15). The frozen grid, run for the RL arms:

| id | α | β | γ | δ | Interpretation |
|---|---|---|---|---|---|
| `w_balanced` | 1.0 | 1.0 | 1.0 | 1.0 | equal weight on normalised terms — **the primary configuration** |
| `w_service` | 2.0 | 2.0 | 0.5 | 0.5 | service-priority operator |
| `w_cost` | 0.5 | 0.5 | 2.0 | 2.0 | cost-priority operator |
| `w_wait_only` | 1.0 | 0.0 | 0.0 | 0.0 | **single-objective arm** for H3 |
| `w_unmet_heavy` | 1.0 | 3.0 | 1.0 | 1.0 | stress on the primary endpoint's term |

The primary H1 test is run at `w_balanced`; the other configurations are the sensitivity analysis and
the input to the trade-off surface. The resulting surface is described as a **convex-hull
approximation obtained by weighted scalarisation** and is explicitly **not** called a Pareto front
(brief §15, MC-10).

---

## 7. Methods compared, scenarios, and the primary endpoint

### 7.1 Methods (all on the identical simulator, identical conditions)

| method_id | Description | Information available to it |
|---|---|---|
| `fixed_timetable` | Deterministic induction schedule reproducing the GTFS scheduled service pattern | none beyond the clock |
| `heuristic_threshold` | Induct when forecast demand or queue exceeds a threshold; thresholds tuned by grid search **on `train` days only**, selected on `val` | same μ as `PPO-Point` |
| `milp_static` | Offline MILP over average demand, solved once per day-type; executed open-loop (manuscript §6) | historical average demand |
| `milp_rolling` | Rolling-horizon MILP re-solved every epoch over the forecast horizon (**added**, MC-05) | exactly the same μ, σ̂ the RL arms see |
| `ppo_point` | PPO, σ channels = 0 | μ |
| `ppo_placebo` | PPO, σ channels shuffled | μ + uninformative σ |
| `ppo_uq` | PPO, calibrated σ̂ | μ + σ̂ — **the proposed method** |
| `ppo_uq_single_obj` | `ppo_uq` trained under `w_wait_only` | μ + σ̂ |
| `ppo_uq_noavail` | `ppo_uq` without the availability channel (H7) | μ + σ̂, no availability |

**Fairness rules (brief §4), enforced by test:**
- Identical initial conditions, demand realisation, disruption realisation, fleet, network, and
  horizon for every method within a given `(scenario, eval_seed)` cell — **common random numbers**.
  `tests/test_common_random_numbers.py` asserts that the realised demand and failure streams for a
  given `(scenario, eval_seed)` are byte-identical across all nine methods.
- No method receives information unavailable to the others except as the arm definition states.
- Baseline tuning uses `train`/`val` only; `tests/test_no_test_set_tuning.py` asserts that no tuning
  routine reads a `test`-split file.
- **MILP is not deliberately weakened**: the same solver (OR-Tools CP-SAT / MIP), a documented
  formulation, a per-instance time limit recorded in the results, and reported solver status and
  optimality gap for every instance. Instances that hit the time limit are reported as such, never
  silently treated as optimal.

### 7.2 Scenario families (exactly four, per manuscript §6)

| scenario_id | Definition |
|---|---|
| `normal` | Demand sampled from `D_true` for a `test`-period weekday; no injected disruption |
| `surge` | `normal` + demand ×U(1.30, 1.50) over a contiguous window sampled from the frozen window distribution |
| `disruption` | `normal` + unplanned mid-day loss of `n_fail` trainsets, `n_fail` ≥ 1, at times drawn from the frozen disruption distribution |
| `combined` | `surge` and `disruption` together, drawn jointly with the same seed |

Scenario instances are **generated reproducibly from `eval_seed`**, never hard-coded.

### 7.3 Primary endpoint (pre-registered, single)

> **Mean fraction of unmet passenger demand over the operating day, under the `surge` scenario,
> at reward configuration `w_balanced`, comparing `ppo_uq` against `ppo_point`.**
>
> **Operationalised (amendment A-01) as `left_behind_fraction`:** distinct passengers passed
> over by at least one full train, divided by arrivals. Bounded in [0, 1]. The alternative
> reading — end-of-day residual queue — was measured to be degenerate (≈ 0.05 % under every
> policy) and is retained as a secondary metric. See §12 A-01 for the full reasoning and for the
> stated limitation of this endpoint.

One confirmatory test. Everything else is secondary or exploratory (§8.4).
Justification is in `docs/research_audit.md` §B and is traceable to manuscript §7.

**Co-primary supporting test (also confirmatory, Holm-controlled with the above):**
`ppo_uq` vs `ppo_placebo` on the same endpoint. If this fails, the mechanism claim fails regardless
of the `ppo_point` comparison.

### 7.4 Full metric set (recorded for every run)

- **Passenger service:** mean / median / p95 waiting time; unmet demand (count and fraction);
  service level (fraction of passengers served within a frozen wait threshold); max queue length.
- **Operations:** trains dispatched; induction success rate; headway violations (must be 0 under
  masking — a non-zero value is a bug, asserted in tests); fleet utilisation; mean load factor.
- **Energy / cost:** energy (kWh); dispatch cost; total operational cost — all flagged
  `provenance: unresolved` until A15/A16 are sourced.
- **Robustness:** degradation ratio (metric under `surge`/`disruption`/`combined` ÷ same metric under
  `normal`, matched on `eval_seed`); recovery time after a failure event (epochs until queue returns
  within a frozen band of the pre-event trajectory).
- **Computation:** policy inference latency per decision (mean, p95); wall-clock training time; MILP
  solve time per instance, solver status, optimality gap.

---

## 8. Experimental design and statistical analysis — FROZEN

### 8.1 Seeds

| Purpose | Seeds | Rule |
|---|---|---|
| Training | `101 … 100+S`, S = 10 (subject to decision item 6) | Used only for policy training |
| Evaluation | `1 … 30` | **Disjoint from training seeds.** Fixed across all methods and scenarios; drives demand, disruption and any stochastic tie-breaking |
| Data-splitting | fixed constant | Split is chronological, so this only affects tie-breaking |

`tests/test_seed_disjointness.py` asserts the two sets never intersect.
`tests/test_reproducibility.py` asserts that re-running a stored `(method, scenario, train_seed,
eval_seed)` reproduces `metrics.json` bit-identically.

### 8.2 Unit of analysis — the point most often got wrong

Evaluation produces `S × 30` episodes per (method, scenario). **The 30 episodes from one trained
policy are not 30 independent observations of the method** — they share a policy. Frozen rules:

- **Primary analysis:** compute each training seed's mean over its 30 CRN-matched evaluation
  episodes; the primary sample is therefore **n = S per arm**, paired across arms by training seed
  *and* by evaluation seed (both arms share both).
- **Secondary analysis:** a linear mixed-effects model on episode-level outcomes with fixed effect
  for method and crossed random intercepts for training seed and evaluation seed.
- Reporting an episode-level n as if it were the sample size is **prohibited**.

### 8.3 Tests

1. **Check assumptions first.** Shapiro–Wilk on the paired differences; Q–Q plot stored.
2. **Paired design.** Because of common random numbers the arms are matched; **paired** tests only.
   Wilcoxon **signed-rank** (not rank-sum — MC-09) when normality is rejected or n is small; paired
   *t* when it is not rejected. The choice rule is fixed here, not after seeing the data.
3. **Effect size, always.** Hodges–Lehmann median paired difference with a BCa bootstrap 95 % CI
   (10 000 resamples, fixed seed); matched-pairs rank-biserial correlation; Cliff's δ for the
   episode-level distribution.
4. **Descriptives, always.** mean, median, SD, IQR, 95 % CI, n — for every (method, scenario, metric).
5. **Exact p-values** reported; `p < 0.05` alone is never written, and the phrase "statistically
   significant" never appears without a named test, its statistic, its n, and its p.
6. **Multiplicity.** Confirmatory family = the two tests in §7.3, Holm-corrected at family-wise
   α = 0.05. All other comparisons are **exploratory**, reported with Benjamini–Hochberg FDR control
   at q = 0.10 and labelled exploratory in every table.
7. **Scenario interaction (H8–H10).** Aligned-rank-transform two-way analysis on
   method × scenario, or a mixed model on the degradation ratio; the interaction term is the
   quantity of interest, not the marginal means.
8. **Equivalence, not just difference.** If the primary test fails to reject, a TOST equivalence test
   is run against a pre-registered smallest effect size of interest (`SESOI`, `TBD@M11`, frozen before
   the runs) so the paper can distinguish "no evidence of benefit" from "evidence of no meaningful
   benefit". **A null result is a publishable finding here, and the protocol is built to state it
   precisely.**

### 8.4 Power

**TBD@M11, frozen before the sweep.** After the single-seed pilot (§10 stage 3), the observed
between-seed SD is used to compute the detectable effect at S = 10, n = 30. If the design is
underpowered for the SESOI, the honest options are: raise S (compute permitting) or state the
detectable effect size in the paper. **Lowering the SESOI after seeing the pilot is not an option.**

### 8.5 Storage

`results/<scenario>/<method>/<train_seed>/<eval_seed>/metrics.json` — never overwritten. Writing to
an existing path raises. Alongside: `episode_log.parquet` (per-step components, actions, masks,
observations), `run_manifest.json` (§9).

---

## 9. Reproducibility, provenance and leakage control

### 9.1 Run manifest (written for every run)

`git_commit`, `git_dirty` (a dirty tree marks the run `provisional`), `config_hash` and full resolved
config, `dataset_manifest_hashes`, `experiment_id`, `train_seed`, `eval_seed`, `python_version`,
`platform`, `cpu`, `gpu`, `package_versions` (full `pip freeze`), `random_state_snapshot`,
`start/end_utc`, `wall_clock_s`. Traceability chain: **result → run_manifest → experiment_id →
registry.yaml → config → git commit → dataset manifest**.

### 9.2 Leakage tests (`tests/test_leakage.py`) — all blocking

1. Forecast features contain no value with timestamp ≥ issue time (checked by construction on a
   synthetic series with a known future spike: the spike must not move any forecast).
2. The RL observation is a pure function of simulator state at or before t (checked by injecting a
   large future demand change and asserting the observation at t is unchanged).
3. No tuning routine reads a `test`-split artefact (import-time file-access audit).
4. Heuristic thresholds and MILP parameters are derived only from `train`/`val` artefacts.
5. Observation and reward normalisation constants are derived only from `train`.
6. Simulator calibration uses `train`-period days; final evaluation uses `test`-period days;
   the two date sets are asserted disjoint.
7. Evaluation seeds never appear in training.

### 9.3 Experiment registry

`experiments/registry.yaml`, written before runs, with per-experiment
`experiment_id, hypothesis, scenario, methods, seeds, dataset, config, expected_output, status`
(`registered → running → complete → failed → superseded`). Failures stay in the registry with
`status: failed` and a `failure_note`. Nothing is deleted.

---

## 10. Compute staging (brief §39) — gates, not suggestions

| Stage | Content | Gate to proceed |
|---|---|---|
| 1 | `net_toy_10`, 100 steps, deterministic | env/action/reward/mask unit tests pass |
| 2 | Sanity environments: zero demand ⇒ dispatch ≈ 0; all trains unavailable ⇒ dispatch = 0; extreme demand ⇒ dispatch increases while feasible | all three pass |
| 3 | Single-seed short PPO run on `net_l_full` | training curve rises; components sane; no headway violations; between-seed SD estimated for §8.4 |
| 4 | Full-budget single-seed run per arm | reward components and behaviour inspected |
| 5 | **Multi-seed sweep** — S training seeds × 9 methods × 4 scenarios × 30 eval seeds | **gated on stages 1–4 and on the §4.5 forecasting gate** |

The multi-seed sweep is not started until every prior gate has passed and been recorded.

---

## 11. Explainability protocol (M13)

Run **only after** policies are trained and frozen. `shap.KernelExplainer` (or `DeepExplainer` where
the policy structure permits — choice recorded) applied to the policy's action logits, with a
background set drawn from `train`-period states. Outputs: global mean |SHAP| per feature group;
scenario-conditional global importance (`normal`, `surge`, `disruption`, `combined`); local
explanations for a frozen set of decision instances (highest-dispatch epochs, post-failure epochs);
**stability analysis** — Spearman rank correlation of global importance across training seeds and
across background resamples. Explicit statements in the report: attributions are **not causal**;
μ and σ are correlated so attributions between them are not uniquely identifiable; results are
conditional on the background distribution.

---

## 12. Amendments log

*Append-only. Each entry is dated and states whether it was made before or after the runs it
affects. An amendment made after seeing the affected results is not permitted.*

---

### A-01 — Operationalisation of the primary endpoint · 2026-08-26

**Made BEFORE any affected run.** No policy exists, no RL training has occurred, and no H1
comparison has been computed. This is therefore pre-registration, not a post-hoc adjustment.

**What changed.** §7.3 pre-registers *"mean fraction of unmet passenger demand over the
operating day"*. That phrase admits two readings, and the implementation initially took the
first:

1. *end-of-day residual queue* ÷ arrivals — passengers never served all day;
2. *distinct passengers left behind by at least one full train* ÷ arrivals.

The primary endpoint is now **reading 2**.

**Why.** Measured on the validated simulator, reading 1 is **degenerate**: it sits at ≈ 0.05 %
under every sensible policy, including under surge, because the modelled line has adequate
capacity for its real demand and a 1.3–1.5× surge over a 1.5–3 h window raises *daily* arrivals
by only ~4 %. An endpoint with no variance cannot reject or support H1 — the test would be
uninformative by construction rather than by finding.

Reading 2 is the standard transit meaning of unmet demand ("left-behind passengers"), is what
manuscript §7's *"conservative dispatch … reducing unmet demand"* most plausibly refers to, and
discriminates roughly tenfold between policies (0.23 % vs 2.23 % under `normal`; 0.51 % vs
2.83 % under `surge`). Scenario definitions are unchanged and remain exactly as the manuscript
specifies.

**Why not raw denial counts.** A passenger passed over by three successive trains contributes
three denial events, so `denials ÷ arrivals` is a *rate* that can exceed 1, not a fraction. The
simulator therefore tracks distinct passengers via a per-cohort flag. Both are logged:
`left_behind_fraction` (bounded, primary) and `denial_rate_per_passenger` (secondary).

**Stated limitation of the new endpoint.** It counts a passenger as left behind whether the
next train arrives in 30 seconds or 10 minutes, so it penalises train *bunching* as well as
genuine capacity shortfall. Measured behaviour reflects this: a policy inducting 2 trains per
epoch leaves more passengers behind than one inducting 1, while achieving *shorter* mean waits.
Mean, median and 95th-percentile waiting time are reported alongside for exactly this reason,
and no conclusion may rest on the left-behind fraction alone.

**Affected artefacts.** `docs/experimental_protocol.md` §7.3, §7.4; `experiments/registry.yaml`
(`EXP-UQ-001`, `EXP-UQ-002`, `EXP-SENS-002`); `src/simulator/{entities,engine}.py`. Recorded as
decision **D-12**.

---

### A-02 — Operationalisation of the reward's waiting term `Ŵ_t` · 2026-08-26

**Made BEFORE any affected run.** No policy exists and no RL training has occurred. The change
was found by a deterministic probe over hand-written policies, not by inspecting an experimental
comparison.

**What changed.** §6.4 defines `Ŵ_t` as *"mean passenger waiting time **accrued in epoch t**
(minutes/passenger)"*. Those two clauses point at different quantities and cannot both be
satisfied. The implementation first took "mean passenger waiting time" over the passengers who
*boarded* in epoch t. `Ŵ_t` is now the **queue integral**: passenger-minutes of platform waiting
accrued during the epoch by everyone on a platform, served or not.

**Why.** The first reading made the reward degenerate. Measured on the real network, a policy
that ran **no service at all** scored `r = 0` — the global maximum of a non-positive reward —
while stranding 100 % of demand. Nobody boards, so the mean is undefined; no train departs, so
`U_t = 0`; nothing moves, so `E_t = C_t = 0`. All four terms went blind at once. Every RL result
computed under that objective would have been meaningless.

The mean over boarders also contradicts §6.4's own wording: it reports waiting that accumulated
in *earlier* epochs and reports nothing for passengers still queuing, so it is not "accrued in
epoch t" at all.

**Alternatives measured** (do-nothing vs minimal service, whole day, real network):

| reading | do nothing | minimal service | ratio | degenerate |
|---|---:|---:|---:|---|
| mean over boarders (superseded) | 0 | 455 min | 0× | **yes** |
| accrued ÷ passengers waiting | 1,123 | 387 | 2.9× | **yes** — E+C = 0 still wins |
| **accrued passenger-minutes (adopted)** | 151,399,865 | 607,853 | 249× | no |
| accrued ÷ passengers arriving | 142,924 | 675 | 212× | no |

The queue integral is the standard transit-OR objective, is defined whether or not anybody
boards, and has no zero-denominator case (the per-arrival variant is undefined in the late-evening
epochs where arrivals reach zero while a queue persists). It departs from the literal
"(minutes/passenger)" parenthetical; the reference scaling of §6.4 divides the term by a fixed
constant in the same units, so α's interpretability is unaffected.

**Not changed.** `U_t` keeps the A-01 operationalisation, the primary endpoint is untouched, and
the weight grid is untouched. Only the waiting term's definition moved.

**Stated limitation.** The queue integral is extensive: it scales with line size and demand
volume, so its magnitude is not comparable across networks. Within this study every comparison is
made on one line at matched demand realisations, so the property does not affect any registered
test. The mean wait over boarders remains logged as a diagnostic, and mean, median and 95th
percentile waiting times are reported for every run regardless.

**Affected artefacts.** `docs/experimental_protocol.md` §6.4; `src/environment/reward.py`;
`src/simulator/engine.py` (queue-integral accumulator); `configs/reward_scales.json`. Recorded as
decision **D-13**.

---

### A-03 — Scenario families share a base world at a given `eval_seed` · 2026-08-26

**Made BEFORE any affected run.** Found by the M6 environment responsiveness report, before any
method comparison existed.

**What changed.** Nothing in the written protocol. This amendment records that the
*implementation* did not satisfy §7.2 and §7.4 as written, and what was done about it.

`environment_streams(scenario, eval_seed)` included the **scenario name** in the random-stream
entropy, so `normal` seed 1 and `surge` seed 1 were entirely unrelated demand realisations —
matched in name only. §7.4 defines the degradation ratio as a stressed metric divided by the
`normal` metric *"matched on `eval_seed`"*, and §7.2 requires `combined` to be drawn jointly with
the individual families at the same seed. Neither held.

**Measured consequence.** On the real network under a constant policy, `surge` scored *better*
than `normal` in **three of five** evaluation seeds — the realisation noise the keying introduced
was larger than the surge effect it was supposed to isolate, and large enough to invert its sign.

**Fix.** The scenario name is excluded from the stream entropy; `eval_seed` is the sole source of
environment entropy. All four families now share one base world per seed and differ only by which
perturbation streams they draw from. Because surge draws and failure draws use separate streams,
`combined` reuses exactly the surge of `surge` and the failures of `disruption`, which is the
factorial contrast `EXP-ROBUST-003` requires. After the fix, `surge` is worse than or equal to
`normal` in five of five seeds. The scenario argument is retained and validated, so a typo cannot
silently produce a valid-looking cell.

**Common random numbers across methods (§7.1) were never affected** — that guarantee is
structural (`environment_streams` takes no method argument) and is unchanged.

**Affected artefacts.** `src/common/seeding.py`; `src/simulator/disruptions.py` (a docstring that
asserted the property that did not hold); `tests/test_seeding.py` (one assertion inverted);
`docs/simulator_validation.md` and `results/simulator_validation.json` (M5 re-run under the new
seeding: 4 validated, 3 partially validated, 7 not validatable, **0 mismatches**, unchanged
verdicts); `configs/reward_scales.json`; `results/env_sanity_report.*`. Recorded as decision
**D-14**.

---

### A-04 — Smallest effect size of interest, frozen · 2026-08-27

**Made BEFORE the pilot ran and before any confirmatory comparison existed.** No RL policy had
been trained against the M4 forecaster at the time of writing. §8.3 item 8 and §8.4 both leave
this value `TBD@M11, frozen before the runs`; this entry is that freezing, not a change.

**Value.** `SESOI = 0.002` absolute on `left_behind_fraction`.

**Grounding.** Chosen from *baseline* measurements only — no RL arm was involved:

- The fixed-timetable baseline leaves **1.8 %** of passengers behind under `surge` on the real
  network. The SESOI is roughly a tenth of that.
- On a line carrying ~285,000 weekday passengers, 0.002 is about **570 passengers a day** who
  either do or do not watch a full train leave without them. That is an operationally noticeable
  change rather than a statistical artefact.
- The measured spread between *sensible* deterministic policies runs from 0.0033 (`constant_1`) to
  0.019 (`constant_2`). The SESOI sits below that spread, so it does not require a method to beat
  the whole baseline range before the result counts.

**Why this is not a permissive threshold.** A smaller SESOI would make equivalence impossible to
establish and effectively guarantee an "inconclusive" verdict; a larger one would let a genuine
improvement be dismissed. 0.002 is the smallest value that remains operationally interpretable.

**Consequence.** If the confirmatory test rejects, the effect is reported with its Hodges–Lehmann
estimate and BCa interval. If it does not reject, a TOST equivalence test against this value
decides between *no evidence of benefit* and *evidence of no meaningful benefit*. Per §8.4,
**lowering it after seeing the pilot is not an option**, and it is not revisited.

**Affected experiments.** `EXP-UQ-001`, `EXP-UQ-002` (the confirmatory family), and every
exploratory comparison that reports an equivalence test.

---

### A-05 — Frozen PPO configuration; no hyperparameter search · 2026-08-27

**Made BEFORE any policy was trained.**

**What changed.** Nothing in the registered design. This records how the standing constraint in
`docs/decisions.md` — *"One hyperparameter configuration, selected on `val` using the
point-forecast arm, frozen and applied unchanged to all RL arms"* — is satisfied.

**Decision.** No PPO hyperparameter search is run. A single documented configuration
(`src/rl/train.py: PPOConfig`) is applied unchanged to every arm.

**Why.** The constraint exists to prevent the proposed arm being tuned harder than its ablations,
and one untuned configuration satisfies that exactly. A search conducted on the point arm would
consume the compute budget without improving the comparison's validity, and would risk selecting a
configuration suited to that arm's narrower input — an asymmetry running *against* the proposed
method, but an asymmetry all the same.

**Scientific consequence.** All arms may be jointly suboptimal, which bounds any claim about the
*absolute* quality of the learned policies. It does not bias the *relative* comparison, which is
what H1 and H1c test. Reported as a limitation.

---

### A-06 — Rule for extending the training-seed pool, fixed before the pilot was read · 2026-08-27

**Made BEFORE the R-1 pilot produced a decision, and before any confirmatory run.**

**What changed.** Nothing in the registered design. This fixes, in advance, *which* seeds are used
if the R-1 power gate amends the sample size **upward** — §8.4's "raise S (compute permitting)".

**Decision.** The registered pool is `train_seeds = [101 … 110]`. If the gate decides `S > 10`, the
pool is extended contiguously upward — `[101 … 100 + S]` — and never resampled, reshuffled or
reselected. `scripts/run_experiments.py::train_seed_list` implements exactly this and logs the
extension; `tests/test_preregistration.py` holds it to the rule.

**Why now.** §8.4 permits amending `S` upward by the smallest defensible increase, but says nothing
about which additional seeds to draw. Left open, that choice would be made *after* the pilot's
effect estimate was visible, and seed selection is precisely the kind of degree of freedom that can
move a marginal result. Fixing the rule before the pilot is read removes the choice entirely.

**Scientific consequence.** None on the estimand. It removes an unforced researcher degree of
freedom and makes the extended design reproducible from `S` alone. The extension is arithmetic, so
a reader can reconstruct the exact seed set from the reported sample size.

**Also recorded.** Before this amendment, `stage_sweep` computed its seeds as `TRAIN_SEEDS[:S]`
against a fixed 10-element tuple. Had the gate asked for `S > 10`, that expression would have
returned 10 seeds while every downstream document reported `S`. The defect was found and fixed
before the pilot ran, so no stored result is affected; it is recorded here because a silent
sample-size truncation is exactly the failure this log exists to make impossible to hide.

---

### A-07 — The pre-registration is frozen between the R-1 gate and M11 · 2026-08-27

**Made BEFORE any confirmatory run.**

**What changed.** Nothing in the registered design. This fixes *when* `experiments/registry.yaml`
becomes immutable, and makes it enforceable rather than a convention.

**Decision.** The registry is frozen by `scripts/freeze_registry.py` immediately after the R-1 power
gate records its decision and immediately before the M11 sweep. Freezing stamps `frozen: true`, the
git commit, the UTC timestamp, and the sample size the gate decided. `require_frozen()` is a hard
precondition of both `--stage sweep` and `--stage ablations`: neither will start against an
unfrozen registry.

**Why this point in the sequence.** Everything in the registry except the sample size is already
fixed; §8.4 deliberately leaves the power analysis as `TBD@M11, frozen before the sweep`, and R-1's
registered job is to supply it. Freezing earlier would mean freezing a placeholder; freezing later
would mean the first confirmatory result was produced against an editable design.

**Scientific consequence.** None on any estimand. It converts "the design was fixed in advance"
from a claim into a checkable property of the repository: the freeze commit precedes every
confirmatory result file, and `tests/test_preregistration.py` fails if a confirmatory result exists
while the registry is unfrozen.

---

### A-08 — Results storage layout differs from §8.5 as frozen · 2026-08-27

**Recorded BEFORE any confirmatory result was stored.** §8.5 is a FROZEN section, and the
implementation does not match it literally. Recording the difference is the point of this log.

**What §8.5 says.** `results/<scenario>/<method>/<train_seed>/<eval_seed>/metrics.json`, alongside
`episode_log.parquet` and `run_manifest.json`; *"never overwritten. Writing to an existing path
raises."*

**What the code does.** `results/runs/<variant>/<scenario>/<method>/<weight_id>/<train_seed>/<eval_seed>/`,
containing `metrics.json`, `epochs.parquet` and `run_manifest.json`
(`src/evaluation/runner.py::cell_dir`).

**The four differences, and why each exists.**

1. **`results/runs/` root.** `results/` also holds the analysis artefacts (`statistical_analysis.json`,
   figures, reports). Separating per-cell results from derived artefacts keeps "delete and
   regenerate the analysis" from touching the primary records.
2. **A leading `variant` segment.** Sensitivity and pilot cells (`od_*`, `weights_*`, `pilot`) must
   be *structurally* unable to be swept up into the confirmatory tree by a glob. The segment is
   `main` for registered cells, so the confirmatory subtree is exactly `results/runs/main/…`.
3. **A `weight_id` segment.** EXP-SENS-001 evaluates the same policy under several reward
   weightings. Without this segment those cells would collide on one path, and the collision would
   resolve as a silent overwrite.
4. **`epochs.parquet`, not `episode_log.parquet`.** A rename only; the contents are the per-epoch
   record §8.5 requires.

**On "writing to an existing path raises".** `run_cell` returns the stored result instead of
raising, which is what makes an interrupted sweep resumable rather than restartable. The property
§8.5 is protecting — *a successful result is never overwritten by a rerun of the sweep* — holds and
is enforced by test (`tests/test_runner_failures.py::test_a_successful_result_is_never_overwritten_by_a_rerun`).
Overwriting requires an explicit `overwrite=True`, which no stage passes.

**Scientific consequence.** None on any estimand. Every path component is part of the cell key, so
no two distinct experimental conditions can share a directory — which is the guarantee §8.5 exists
to give.

---

### A-09 — Compute threads pinned to one per worker · 2026-08-27

**Made BEFORE any policy in the registered design was trained.** The only policies trained before
this point were the throwaway dry-run policies, which are not a result and have been discarded.

**What changed.** Every evaluation and training worker now calls `src/common/compute.py`'s
`single_threaded()`, which pins PyTorch, OpenMP and the BLAS pools to one thread in that process.

**Why.** Two reasons, and the second is the one that touches the science.

1. *Oversubscription.* The sweep runs one process per core. PyTorch defaults to 8 threads per
   process on this 12-core machine, so 11 workers would contend with 88 threads for 12 cores. The
   networks are two 64-unit layers — far too small for intra-op parallelism to repay its own
   synchronisation.
2. *Reproducibility.* Torch splits CPU reductions across threads, and floating-point addition is
   not associative, so thread count can change results in the low bits. Thread count is a property
   of the machine, not of the experiment. §9 requires a run to be reproducible; pinning to one
   thread is what makes that achievable on a machine with a different core count.

**Scientific consequence.** None on any estimand, and it applies identically to every arm, so it
cannot favour one. It does mean results are not bit-comparable with the discarded dry-run
policies, which were trained multi-threaded — those are not a result and nothing references them.
The change is applied uniformly and before the pilot, so no confirmatory comparison spans the
boundary.

---

### A-10 — The heuristic threshold grid is extended until its optimum is interior · 2026-08-27

**Made BEFORE the re-tuning run, and before any confirmatory result exists.** The stopping rule
below is stated in full here so that it cannot be adjusted after seeing which threshold wins.

**What was found.** The first tuning run over the frozen grid
`QUEUE_THRESHOLDS = (0, 50, 100, 200, 400, 800)` selected **800 — the largest value in the grid**.
The three highest-scoring grid points all sat at that boundary. An optimum on the edge of a search
region is evidence that the true optimum lies outside it, so the tuned heuristic was very probably
weaker than the heuristic this grid was meant to find.

**Why this is not a minor tuning detail.** Protocol §7.1 forbids a strawman baseline. H4, H5 and
H6 are claims that adaptive RL beats these baselines, and an under-tuned heuristic inflates every
one of them. The bias runs **in favour of the proposed method**, which is the worst direction for
an error of this kind to run.

**Decision — the grid is extended upward, under a stopping rule fixed in advance:**

1. Extend `QUEUE_THRESHOLDS` by successive doubling from the current maximum: 1600, 3200, 6400, …
2. Re-tune over the extended grid and stop when the selected `queue_threshold` is **strictly
   interior** — that is, not the largest value tested.
3. Stop unconditionally once the smallest untested threshold exceeds the **largest station queue
   observed across the tuning episodes**. Beyond that point the rule `queue > threshold` can never
   fire, so every larger threshold produces an identical policy and the "boundary" is not a real
   one. This bound is measured from the tuning episodes themselves, not chosen.
4. The same interiority check is applied to `FORECAST_THRESHOLDS`. Its selected value (4000) is
   already interior to `(0, 2000, 4000, 6000, 8000)`, so it is not extended.

Nothing else changes: the same tuning seeds (2001–2006, disjoint from every scored pool), the same
scenarios, the same selection criterion, the same information available to the heuristic (forecast
mean and observed queue; never sigma).

**Scientific consequence.** It strengthens a *competitor* to the proposed method, so it can only
make H4–H6 harder to support. No estimand changes, and no confirmatory test's definition changes.
Every stored `heuristic_threshold` cell produced under the boundary solution is **deleted and
re-run**; corrected and uncorrected heuristic results are never mixed, and the corrected heuristic
is the sole heuristic baseline in every reported comparison.

**Also recorded.** `tune_heuristic` ran its grid sequentially, which is why the first tuning took
3.9 hours for episodes that cost ~16 seconds each in parallel. The re-tune parallelises the grid.
That is a scheduling change only: each grid point is an independent set of episodes with its own
seeded environment, so parallel evaluation cannot alter any score.

---

### A-11 — The MILP surrogate gains a calibrated standing-queue term · 2026-08-27

**Made BEFORE the corrected MILP is used in any confirmatory comparison.** Every stored
`milp_static` and `milp_rolling` cell produced under the uncorrected surrogate is deleted and
re-run; corrected and uncorrected MILP results are never mixed.

**What was found.** Across all four scenarios the static MILP stranded roughly three times as many
passengers as a plain fixed timetable (5.7–7.9% versus 1.9–2.9%) while dispatching 28% fewer
trains (164 planned versus 228). An optimisation baseline that loses to a naive schedule is not a
baseline.

**The cause, measured rather than guessed.** The MILP solves its model to optimality; the model is
wrong. Scoring both schedules under the surrogate objective and under the simulator:

| | MILP plan | fixed timetable |
|---|---:|---:|
| surrogate mean predicted queue | 0.5 | 728.8 |
| simulator mean realised queue | 1017.9 | — |
| surrogate total cost | 447.2 | 8316.9 |
| simulator reward | −1474.2 | −1103.7 |

The surrogate and the simulator rank the two schedules **oppositely**. The fluid recursion
`q_t = max(0, q_{t-1} + d_t − κ·n_t)` drives the queue to zero whenever capacity exceeds demand,
so it predicted a mean queue of 0.5 passengers where the simulator realised 1,018 — a factor of
1,856. With no queue to speak of, the MILP's objective was dominated by induction and service cost
(283.7 + 81.8) against a queue penalty of 5.9, and under-dispatching looked nearly free.

**Why the fluid model is wrong here.** Passengers arrive continuously; trains arrive discretely.
There is always a queue between train arrivals, and its size is set by the headway — that is,
by the number of trainsets in service — not by whether aggregate capacity exceeds aggregate
demand. A fluid queue has no representation of this irreducible standing queue. It is not a
calibration error in κ: refitting κ on every probe trajectory moves it only from 94.9 to 87.4.

**Measured dependence** (probe policies, pooled, least squares through the origin):

| trainsets in service | 1–2 | 6–8 | 10–12 | 16–18 | 20–24 | 24–30 |
|---|---:|---:|---:|---:|---:|---:|
| standing queue per arrival | 2.21 | 2.79 | 0.99 | 0.49 | 0.31 | 0.26 |

**Decision.** The surrogate gains a **standing-queue floor indexed on trainsets in service**,
`q_t ≥ φ[n_t] · d_t`, with `φ` calibrated from the same probe episodes that already calibrate κ,
the round trip, energy and the waiting coefficients. It is represented exactly in CP-SAT by
one-hot indicators on bucketed `n_t`, the same device already used for the action-specific waiting
term. Buckets are the resolution the probe data supports; no functional form is assumed, and a
`c/n` fit was rejected because it explained little of the variance (R² = 0.07).

**Scientific consequence.** It strengthens a *competitor* to the proposed method, so it can only
make H6 and H6b harder to support. No estimand changes and no confirmatory test's definition
changes. This is the third correction to this surrogate — κ was refitted against the realised
queue path, then action-specific waiting coefficients were added — and each was made because the
uncorrected model produced a baseline that protocol §7.1 forbids. The pattern is recorded here
because a reader is entitled to know how much modelling effort the baseline received relative to
the proposed method: substantially more.

---

### A-12 — The monotone clamp on the MILP's waiting coefficients is removed · 2026-08-27

**Made BEFORE the corrected MILP is used in any confirmatory comparison.** All `milp_static` and
`milp_rolling` cells are re-run; corrected and uncorrected results are never mixed.

**What was found.** After A-11 the static MILP became a strong baseline, but the *rolling* MILP
became the worst method of the four — worse than a plain fixed timetable — while dispatching the
most trains (232 against 204) and having 48% of its requested inductions clipped by the action
mask. It was requesting two trainsets in nearly every decision interval.

**The cause.** `_fit_wait_coefficients` forced the per-action waiting coefficients to be
non-increasing in `k`, on the reasoning that *"waiting must not increase with more service, or the
MILP would learn to withhold trains"*. That conflates two different quantities:

- More service **over time** — a higher induction rate — genuinely reduces waiting. This is priced
  by the queue recursion and, since A-11, by the standing-queue floor. The clamp was never what
  protected it.
- More service **at one instant** — two trainsets in a single 5-minute interval — bunches
  departures. Two trains leaving together do not halve the headway; they create one gap and then a
  longer one.

The simulator is unambiguous about the second, measured over four probe seeds:

| probe | inductions | left_behind_fraction | total reward |
|---|---:|---:|---:|
| `constant_1` | 228.0 | 0.0044 | −740.9 |
| `constant_2` | 233.5 | 0.0231 | −1027.7 |

Essentially the same number of trains, **five times the stranding**. The unclamped fit reflects
this: `w2 = 0.680` against `w1 = 0.633`. The clamp rewrote `w2` to `0.633`, so the MILP saw a
second trainset as delivering extra capacity at no additional waiting cost, and always wanted two.

**Decision.** The monotone clamp is removed. The non-negativity clip is kept, and each coefficient
continues to be fitted only from epochs that actually used that action. The protection the clamp
was believed to provide is supplied by the queue term and the A-11 standing-queue floor, which is
exactly what A-11 established.

**Scientific consequence.** It strengthens two competitors to the proposed method, so it can only
make H6 and H6b harder to support. No estimand changes. This is the fourth correction to the MILP
surrogate; §7.1's prohibition on strawman baselines is the reason each was made, and the
cumulative modelling effort spent on the baseline — far exceeding that spent on the RL arms, which
run one untuned frozen configuration under A-05 — is itself reported as a limitation.

**A note on what this does not license.** The clamp was removed because a measurement contradicted
it, not because a baseline was performing inconveniently. The direction of the correction is
against the proposed method in both cases, which is the only direction in which such a change is
safe to make autonomously.

---

### A-13 — Sustainable dispatch capacity, and where the MILP correction stops · 2026-08-27

**Made BEFORE the corrected MILP is used in any confirmatory comparison.**

**Change.** The surrogate gains a feasibility envelope: over any window of `W` epochs, planned
inductions may not exceed the most the environment has ever been observed to deliver in a window
that long, measured from the always-request-two probe
(`{8: 14, 12: 18, 20: 26, 36: 42}`, i.e. 1.75 down to 1.17 per epoch). `max_induct_per_decision`
bounds a single interval, but `feasible_inductions` returns zero while the terminal platform is
occupied or the minimum headway has not elapsed, so that instantaneous bound is not sustainable.
This is a feasibility constraint measured from the environment, in the same category as the
fleet-size constraint, and it is the *observed maximum* — the loosest bound consistent with what
the environment has actually delivered, so it cannot bind tighter than physics.

**Effect.** Clipping of the rolling MILP's requests fell from 48% to 43%. Outcomes barely moved.

**Where this stops, and why.** After A-11, A-12 and A-13 the *static* MILP is a strong baseline:
it strands 1.04% under `normal` against the fixed timetable's 1.88%, and the surrogate and the
simulator now agree on which of two schedules is better. The *rolling* MILP remains weaker than a
fixed timetable, and the diagnosed reason is structural rather than a defect:

> A fluid-queue surrogate has no representation of **departure spacing**. It prices capacity and,
> since A-11, standing queue — but two trainsets dispatched in one interval and two dispatched in
> consecutive intervals are indistinguishable to it. The simulator distinguishes them sharply:
> `constant_1` strands 0.44% and scores −740.9, while `constant_2` strands 2.31% and scores
> −1027.7, for essentially the same number of trains. The rolling MILP re-plans every interval,
> always wants the larger action, and therefore executes a bursty schedule.

Per-action waiting coefficients (A-12) are the right *form* for this but cannot carry it: the harm
from a burst lands in the following blocked intervals, not in the interval that burst, and a
coefficient indexed on the current action cannot represent a cost that arrives later.

**No further correction is made.** Continuing until the rolling MILP wins would mean designing a
better dispatch algorithm and calling it a baseline. The honest report is that a rolling-horizon
MILP re-solved on the same forecasts underperforms here **because the fluid abstraction cannot see
bunching**, and that this limits what H6b establishes: it tests this MILP formulation, not
optimisation-based scheduling in general.

**Scientific consequence.** H6b's interpretation is narrowed and this limitation is reported
wherever H6b appears. The three corrections that were made all strengthened the baselines and so
work against the proposed method; declining to make a fourth is the point at which strengthening a
baseline would become authoring one.

---

## 13. Definition of done for this protocol

This protocol is satisfied only when every `TBD@Mx` above has been resolved and frozen at its named
milestone, every gate in §10 has passed and been recorded, and `docs/paper_traceability.md` assigns
every manuscript claim one of SUPPORTED / PARTIALLY SUPPORTED / NOT YET TESTED / REFUTED with a
pointer to a stored result.
