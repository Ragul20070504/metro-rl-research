# Milestone Status

Updated after every milestone. **This repository contains no reportable experimental results.**

| Milestone | Status |
|---|---|
| M0 Research audit | COMPLETE |
| M1 Experimental protocol | COMPLETE |
| M1b Decisions (D-01…D-11) | COMPLETE |
| M2 Environment + foundation | COMPLETE |
| M3 Data ingestion | **COMPLETE** |
| M4 Forecasting | **PARTIAL** — implemented and smoke-passed; full fit running |
| M5 Digital twin | **COMPLETE** — re-validated under the D-14 seeding fix |
| M6 RL environment | **COMPLETE** — see `docs/m6_scientific_gate.md` |
| M7 Fixed timetable | NOT STARTED |
| M8 Heuristic | NOT STARTED |
| M9 MILP | NOT STARTED |
| M10 RL baselines | NOT STARTED |
| M11 Full sweep | BLOCKED — see open decision below |
| M12 Ablations | NOT STARTED |
| M13 SHAP | NOT STARTED |
| M14 Statistical analysis | NOT STARTED |
| M15 Figures and tables | NOT STARTED |
| M16 Reproducibility audit | NOT STARTED |

Tests: **301 passing**, 0 failing.

---

## M3 — DATA INGESTION · COMPLETE

**Implemented:** `src/data/{sources,download,validate,splits,features,gtfs,prepare}.py`.

**Artefacts:** 115 month partitions across 3 datasets; 9 `MANIFEST.json` + `LICENSE.txt`
(committed); `results/data_quality_report.{md,json}`; `results/split_report.md`;
`data/processed/{demand_hourly,supervised_train,supervised_val,supervised_test}.parquet`;
`data/processed/split_manifest.json`.

**Executed:** 1,344,528 in-regime station-hour rows (2020-01-01 → 2026-08-13). All ten quality
checks pass, **0 blocking failures**. Study window 2024-01-01 → 2026-08-13 (956 days); split
train 2024-01-01→2025-10-20 / val →2026-03-23 / test →2026-08-13 with 7-day embargoes.
Supervised rows: 1,126,584 / 254,016 / 235,008. PSI train↔val 0.005, train↔test 0.025.

**Known issues:**
- The pre-registered regime screen **fired its fallback**: no 365-day run with |YoY growth| ≤ 10%
  exists (longest 278 days). Residual drift in the study window is +7.1% median. **Tolerance was
  not loosened**; diagnostics are machine-generated and labelled a diagnostic, not a licence to
  retune. Carried into limitations.
- 14 negative values in the source, all in `transfers` (never `ridership`). Recorded as an
  upstream defect; not blocking because nothing downstream consumes that column.
- Upstream `service_delivered` field confirmed mis-scaled by exactly 100×; pipeline recomputes it.

**Next:** M4 full fit.

---

## M4 — FORECASTING · PARTIAL

**Implemented:** `src/forecasting/{metrics,models,calibrate,train}.py`; `configs/forecasting.yaml`.

**Executed:** smoke run only (truncated splits, **not reportable**). It passed the blocking gate —
pinball 13.34 vs naive 23.30, calibrated PICP₈₀ = 0.7998 inside the pre-registered [0.76, 0.84]
band. Full fit on 1.13M training rows is running; the point model selected
`{lr 0.1, depth 8, min_child_weight 20}` at RMSE 124.89.

**Known issues:**
- Two calibration defects found by tests and fixed in the code: the search grid was linear where
  the correction is multiplicative (so it could return a factor *worse* than no correction), and a
  factor pinned to a search endpoint was accepted silently. Now geometric, guarded, and flagged
  via `at_boundary`.
- In the smoke run the point model was **no better than seasonal naive** (MASE 1.002 vs 1.000).
  If that persists on the full fit it is a finding, not a failure, and will be reported.

**Next:** complete the full fit; write `docs/forecasting.md`.

---

## M5 — DIGITAL TWIN · COMPLETE

**Implemented:** `src/simulator/{network,build_network,demand,fit_demand,od,entities,disruptions,engine,validate}.py`;
`configs/simulator.yaml` with strict provenance enforcement.

**Artefacts:** `data/processed/network/{network,station_order,demand_process}.json`;
`docs/simulator_validation.md`; `results/simulator_validation.json`.

**Grounded in real data:** 24 stations and segment times from GTFS; 150 s minimum headway measured
from the published timetable; fleet size 36 derived from the published 277 weekday trips per
direction; demand fitted to 471 weekday + 188 weekend observed days (99.3 % of station-hour cells
over-dispersed, empirically justifying negative binomial over Poisson); failure hazard calibrated
to real MDBF (124,133 car-miles, R143, division B).

**Validation:** 4 validated, 3 partially validated, **7 not validatable**, 0 mismatches.
End-to-end runtime 37.00 min vs published 37.11 (0.3 %); boardings 280,257 vs 285,004 (1.7 %);
round trips 252 vs 277 (9 %).

**Known issues:**
- Seven quantities cannot be validated at all, including peak occupancy and the OD distribution —
  **both of which feed the primary endpoint.** The OD and capacity sweeps are therefore mandatory.
- The failure-rate row is only *partially* validated: five episodes cannot resolve a 0.33/day rate.
- Five simulator defects were found and fixed during M5 (dwell double-counting, stranded termini,
  collapsed action space, non-boarding held trains, under-sized fleet). Each has a regression test.

**Next:** M6 RL environment.

**Re-validated at M6** under the D-14 seeding fix (the scenario name was removed from the random
stream entropy). Verdicts unchanged: 4 validated, 3 partially validated, 7 not validatable,
**0 mismatches**. Weekday boardings moved 280,257 → 280,936 against a real 285,004 (1.7 % → 1.4 %).
The failure-rate row moved 0.00 → 1.00 failures/day against a real 0.331; both are draws from the
same calibrated hazard and five episodes cannot resolve that rate, so the row stays *partially*
validated, as before.

---

## M6 — RL ENVIRONMENT · COMPLETE

**Implemented:** `src/environment/{observation,forecast_provider,arms,action_mask,reward,metro_env}.py`;
`configs/env.yaml` (strict provenance); `scripts/{fit_reward_scales,env_sanity_report}.py`.

**Artefacts:** `docs/rl_environment.md`; `docs/m6_scientific_gate.md`; `configs/reward_scales.json`;
`results/env_sanity_report.{md,json}`.

**Validated:** 123-value observation (121 without the availability channel); `Discrete(3)` with
feasibility masking; multi-objective reward with exact component reconstruction. Gymnasium
`check_env` passes. **161 new tests**, all sixteen completion criteria met. A 2-station
deterministic scenario is derived by hand in the test docstring and asserted term by term,
including epoch rewards of exactly −55.0 and −103.8916667.

**Two protocol amendments, both made before any policy existed:**
- **A-02 / D-13** — the reward was degenerate. With `W_t` read as the mean wait of passengers who
  *boarded*, a policy running no service all day scored `r = 0`, the global maximum, while
  stranding 100 % of demand. `W_t` is now the queue integral (249× separation). The primary
  endpoint is untouched.
- **A-03 / D-14** — scenario streams were keyed on the scenario *name*, so `normal` and `surge` at
  the same `eval_seed` were unrelated realisations, though protocol §7.4 requires them matched.
  Measured: `surge` scored *better* than `normal` in 3 of 5 seeds. Now worse-or-equal in 5 of 5.
  The `combined` factorial claim in `disruptions.py` had been false since M2.

**Known issues:** the observation scaler is unfitted and the reward reference scales are
provisional — both depend on the M7 fixed-timetable baseline. The H7 ablation is
dimensionality-confounded by the protocol's own wording. Action 2 is dominated by action 1 in
every configuration tested (bunching), so the action space is effectively close to binary.

**Blocking risk carried forward (R-1):** on the real network under `w_balanced`, `constant_1`
scores −822 and the best variant tried scores −838. If the headroom for any method really is a few
percent, protocol §8.4's power analysis — written before any of this was measurable — is stale.
**It must be redone against a between-seed SD measured on real short runs before M11 is
authorised.**

**Next:** complete and validate the M4 full fit; then M7.

---

## RESOLVED — the primary-endpoint decision (A-01 / D-12)

**Settled 2026-08-26.** The endpoint was redefined as `left_behind_fraction` — distinct passengers
passed over by at least one full train, divided by arrivals, bounded in [0, 1]. Implemented,
tested and recorded as protocol amendment **A-01** and decision **D-12**, before any policy
existed. The original reading (end-of-day residual queue) is retained as a secondary metric. The
bunching limitation is documented and no conclusion may rest on the endpoint alone.

The original statement of the problem is kept below, unedited, as the record of why.

---

## OPEN ITEM NOW BLOCKING M11 — statistical power (R-1)

Protocol §8.4's power analysis was written at M1, before the environment existed and before any
effect size was measurable. The M6 responsiveness report suggests the headroom between *sensible*
policies is a few percent: on the real network under `w_balanced`, `constant_1` scores −822 and
the best variant tried scores −838, while badly-tuned policies are several times worse.

**Before M11 is authorised**, the between-seed standard deviation must be estimated from real
short runs (protocol §10 stage 3) and the power analysis redone against it. If the study is
underpowered for the effect size actually available, that must be stated *before* running, not
discovered after. Full statement in `docs/m6_scientific_gate.md` §12, risk **R-1**.

---

## ORIGINAL RECORD — the primary-endpoint problem (resolved by A-01 / D-12)

**The pre-registered primary endpoint is degenerate at the calibrated parameter values.**

Protocol §7.3 pre-registers *"mean fraction of unmet passenger demand over the operating day,
under the surge scenario"*. Measured on the validated simulator, end-of-day residual queue is
**≈ 0.05 % under every sensible policy**, including under surge — there is essentially no variance
for the H1 test to detect an effect in.

Cause: the modelled line has adequate capacity for its real demand, and the surge window
(1.3–1.5× for 1.5–3 h) raises *daily* arrivals by only ~4 %. Unmet demand becomes material only
above roughly 2× whole-day demand.

Measured evidence (whole-day demand multiplier, action = 1):

| ×demand | unmet % | mean wait (min) | denied boardings |
|---|---|---|---|
| 1.0 | 0.03 | 1.70 | 1,631 |
| 1.5 | 0.04 | 6.49 | 406,878 |
| 2.0 | 2.80 | 18.37 | 2,227,618 |
| 3.0 | 10.34 | 22.52 | 7,978,153 |

Both *waiting time* and *denied boardings* (passengers who watched a full train leave) are highly
responsive across the whole range; end-of-day residual queue is not.

This is a decision about the **primary metric**, which the execution directive reserves for the
user. It is recorded here **before any H1 comparison has been run** — no policy exists yet — so
resolving it now is pre-registration, not a post-hoc choice. Options are set out in the
accompanying report; nothing further on M11 proceeds until it is settled. M6–M10 are unaffected,
because every reward component is logged separately regardless.
