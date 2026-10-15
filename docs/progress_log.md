# Progress Log

What was **actually** completed at each milestone, including what was attempted and failed.
Research brief §30: at the end of every milestone, run tests, run a minimal experiment, inspect
outputs, document what was actually completed, commit.

This file records fact, not intent. If a milestone is partially done, it says so.

---

## M0 — Research audit · **COMPLETE** · 2026-08-26

**Produced:** `docs/research_audit.md`.

**What was done:** manuscript read in full; RQ1–RQ3 extracted verbatim and operationalised;
primary hypothesis stated with its null plus a mechanism test and a placebo control; novelty audited
against the manuscript's own Table 1; 21 assumptions classified; data availability verified by direct
lookup; 16 threats to validity; publication-risk table.

**Verified externally (direct lookup, 2026-08-26):** MTA Subway Hourly Ridership (`wujg-7c2s`, field
list confirmed via the Socrata metadata API), GTFS Static (`fgm6-ccue`), Subway Stations (`39hk-dx4f`),
Mean Distance Between Failures (`e2qc-xgxs`), Customer Journey-Focused Metrics (`4apg-4kt9`), Wait
Assessment, and MetroPT-3 (UCI #791, CC BY 4.0, 1,516,948 rows, 15 channels, 4 documented failures).

**Found to be NOT available**, contrary to manuscript §5: publicly downloadable Beijing Metro AFC
data; per-train availability records for any NYC fleet; an OD matrix for the modelled line; sourced
energy and cost parameters.

**Not done at M0:** no code, no data downloaded, no results. Correct — brief §41.

---

## M1 — Experimental protocol · **COMPLETE** · 2026-08-26

**Produced:** `docs/experimental_protocol.md` (pre-registration), `docs/data_sources.md`,
`docs/manuscript_corrections.md`, `docs/risk_register.md`, `docs/milestone_plan.md`,
`docs/dependency_plan.md`, `docs/repository_structure.md`, `experiments/registry.yaml` (skeleton,
`frozen: false`), repository skeleton, `.gitignore`, `.env.example`, `requirements.in`.

**Open at close of M1:** five blocking manuscript corrections and ten decision items.

**Commit:** `5377fec`.

---

## M1b — Decisions · **COMPLETE** · 2026-08-26

All ten decision items confirmed as recommended. Recorded in `docs/decisions.md` with reasoning and
accepted consequences. Taken **before** any data download, training or experiment, so they are part
of the pre-registration; `experimental_protocol.md` §12 (amendments) remains empty.

Two of them were resolved by doing rather than deciding: D-07 (Python) and D-08 (git) — see M2.

---

## M2 — Repository architecture and environment · **IN PROGRESS** · 2026-08-26

### Completed and verified

**Interpreter (closes risk E-02).** Python 3.11.9 installed via the Python Install Manager
(`py install 3.11`); `.venv/` created. The system 3.14.7 interpreter is untouched.

**Dependencies.** All 18 packages from `requirements.in` installed and imported cleanly. Resolved
versions are tabulated in `docs/dependency_plan.md`. Two notable resolutions: torch is a **CPU
build** (helps evaluation determinism), and pandas resolved to **3.0.5** / numpy to **2.4.6**, both
majors newer than most published transit-RL code targets.

**Functional stack verification.** `scripts/check_environment.py` — **10/10 pass**. This goes beyond
imports: it asserts that MaskablePPO trains and never selects a masked action, that CP-SAT reports
status and wall time, that XGBoost multi-quantile regression fits with acceptable quantile crossing
(0.5 %), that SciPy's BCa bootstrap and Wilcoxon signed-rank work, and that seeded torch forward
passes are bit-identical.

**Git (closes risk E-01).** Git 2.55.0 was already installed at `C:\Program Files\Git\cmd\git.exe`,
merely absent from `PATH` — which is why the M0 audit reported it missing. **No installation was
needed; the M0 finding was wrong on the facts, and is corrected here rather than quietly dropped.**
Repository initialised on `main`; author identity set repo-scoped; `.gitattributes` normalises line
endings to LF so a clone on Linux/macOS is byte-identical.

**Foundation modules** (`src/common/`), each written to make a protocol rule structurally
enforceable rather than a matter of discipline:

| Module | What it makes impossible |
|---|---|
| `paths.py` | Hard-coded absolute paths; a non-uniform `results/` tree |
| `seeding.py` | Breaking common random numbers — `environment_streams(scenario, eval_seed)` takes **no method argument**, so the method under evaluation cannot influence the demand or failure realisation. A test pins the signature down |
| `config.py` | Using a parameter with no stated provenance, or an `unresolved` parameter, outside an explicitly tagged sensitivity sweep |
| `manifest.py` | Losing the traceability chain; **overwriting a stored run** (`open_run_dir` raises if a manifest already exists) |
| `logging.py` | Losing a run's console output |

**Tests: 38 pass, 0 fail** (`pytest`). Coverage at this milestone is the infrastructure only:
path layout, seeding and the CRN guarantee, provenance enforcement (8 cases including the two
escape-hatch conditions), and manifest/append-only behaviour.

**One defect found and fixed during M2:** in `manifest.py`, a dataclass field named `platform`
shadowed the `platform` module for subsequent field definitions in the same class body, raising
`AttributeError` at import. Caught by test collection, fixed by aliasing the import.

**Lockfile.** `pip-compile --generate-hashes` was run directly against `requirements.in` and produced
no output after ~25 minutes; it was stopped. It performs a full resolution *and* downloads every
candidate distribution in order to hash it, and the torch candidate set makes that impractical here.

Approach changed, and the replacement is **better provenance, not a compromise**: `requirements.txt`
is now `pip freeze` of the environment that actually passed the 10 functional checks and the 38
tests. It pins exactly what was verified to work, rather than a fresh resolution that might differ.
The header records that provenance and the timestamp. `environment.yml` mirrors the same pins for
conda users, installing through pip rather than conda-forge so that the verified resolution is not
silently replaced by different builds of the same packages. A hash-verified `requirements.lock.txt`
is generated separately from the fully-pinned input, where resolution is trivial.

**Git branches** created per brief §31: `main`, `development`, and `feature/{data, forecasting,
simulator, environment, ppo, baselines, evaluation, shap}`.

---

## M3 precursor — data-source verification · **COMPLETE** · 2026-08-26

Run before committing to ingestion, on the principle that a portal URL is not evidence a resource can
be read. `scripts/probe_sources.py` probes every planned source, captures its field list and temporal
coverage, and confirms the modelled line actually appears in it. Evidence: `results/source_probe.json`.
**11 usable sources verified.**

### Corrections to the M0 audit — recorded, not silently fixed

| M0 claim | Reality |
|---|---|
| `4apg-4kt9` is the Customer Journey-Focused Metrics dataset | **HTTP 403.** Correct id is `r7qk-6tcy`, covering 2015-01 → 2026-06 rather than the assumed 2020–2024 |
| `fgm6-ccue` serves GTFS Static | **HTTP 403** — not served through the tabular API; its only attachment is an overview PDF. The feed comes from `rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip`, 5.6 MB, **no API key** |
| "Realised train movements are not retained as an open historical archive" | **Wrong.** `sp9g-mzjh` publishes realised end-to-end runtime **percentiles**; `32ch-sei3` publishes scheduled vs actual train counts |

### Findings that improved the design

- **Demand is continuous 2017-01-13 → 2026-08-13** across three same-schema datasets
  (`t69i-h2me`, `wujg-7c2s`, `5wq4-mkjj`), not the single 2020–2024 table assumed. The study window
  can therefore sit wholly inside one regime instead of straddling the post-pandemic break — which
  matters for H1 specifically, because a regime break inflates forecast dispersion for reasons
  unrelated to short-horizon demand uncertainty. Decision **D-11**; `EXP-ROBUST-004` registered so
  the pre-pandemic period cannot quietly become an extra test set.
- **Volume is tractable:** filtered to the line's 24 complexes, `wujg-7c2s` holds **6,893,715 rows**.
  An unfiltered `count(*)` times out, which is itself a useful constraint on how to query it.
- **24 station complexes** confirmed from `39hk-dx4f`, and independently corroborated by
  `sp9g-mzjh` reporting 23 stops over 9.84–10.15 miles end-to-end.
- **Failure calibration is concrete:** MDBF publishes `number_of_failures`, `total_miles` and
  `number_of_cars` for car class **R143** under **division B** (e.g. 2026-06: 212 cars, 934,940
  miles, 13 failures), so the synthetic hazard is scaled to a directly computed rate.
- **Simulator validation strengthened:** running-time *dispersion* and dispatch success rate move
  from unvalidatable to validatable. Four rows remain unvalidatable and are reported as such.
- **Data-quality defect found upstream:** `32ch-sei3`'s published `service_delivered` field is
  mis-scaled by 1/100 (2028 / 2220 = 0.9135, but the field reads 0.00913). The pipeline recomputes
  the ratio from the counts and does not trust the published field; a quality assertion is specified.

### Not yet done at M2
- Config schema files under `configs/` — these are written at the milestone that owns each one
  (M3 data, M4 forecasting, M5 simulator, M6 environment, M8 PPO, M10 MILP), because a config
  written before its consumer would have invented parameter values.
- No data downloaded, no model trained, no experiment run. **This repository still contains zero
  experimental results.**
