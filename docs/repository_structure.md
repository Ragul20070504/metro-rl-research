# Proposed Repository Structure

Follows brief §5. Directories that exist today are marked ✅; the rest are planned and are created at
the milestone shown. No substantive implementation code has been written yet (brief §41).

```
metro-rl-research/
├── research_spec/                     ✅  Metro_RL_Scheduling_Framework_Final.pdf (immutable copy of the manuscript)
├── docs/                              ✅
│   ├── research_audit.md              ✅  M0
│   ├── experimental_protocol.md       ✅  M1 — the pre-registration; frozen
│   ├── data_sources.md                ✅  M1
│   ├── manuscript_corrections.md      ✅  M1
│   ├── risk_register.md               ✅  M1
│   ├── milestone_plan.md              ✅  M1
│   ├── dependency_plan.md             ✅  M1
│   ├── repository_structure.md        ✅  M1 (this file)
│   ├── progress_log.md                    M2  what was ACTUALLY completed per milestone, incl. failures
│   ├── environment_report.md              M2  interpreter, pip freeze, platform, threading
│   ├── data.md                            M3  pipeline, features, splits, quality findings
│   ├── simulator.md                       M5  design + every parameter's provenance
│   ├── simulator_validation.md            M5  real vs simulated, with per-quantity verdicts
│   ├── rl_environment.md                  M6  state/action/reward/masking, exactly as implemented
│   ├── baselines.md                       M7/M10  incl. full MILP formulation, objective, constraints
│   ├── experiments.md                     M11
│   ├── statistical_analysis.md            M14
│   ├── explainability.md                  M13
│   ├── limitations.md                     M16  the accepted risks, unsoftened
│   ├── reproducibility.md                 M16  fresh-clone reproduction procedure + evidence
│   ├── paper_traceability.md              M16  every claim → SUPPORTED / PARTIALLY / NOT YET TESTED / REFUTED
│   └── pre_submission_peer_review.md      M16  hostile reviewer; deliberately not positive
├── configs/                           ✅
│   ├── data.yaml                          M3  split boundaries, quality thresholds, stationarity screen
│   ├── forecasting.yaml                   M4  features, quantiles, hyperparameter grid, calibration
│   ├── network_l_full.yaml                M5  topology, running times, headway — every key has `provenance:`
│   ├── network_toy10.yaml                 M5  synthetic 10-station config, tests only
│   ├── simulator.yaml                     M5  dwell, capacity, arrivals, failures, energy, cost
│   ├── scenarios.yaml                     M5  normal / surge / disruption / combined generators
│   ├── env.yaml                           M6  observation layout, K_max, masking rules, normalisation
│   ├── reward_weights.yaml                M6  the frozen 5-configuration weight grid
│   ├── ppo.yaml                           M8  architecture, hyperparameters, budget, seeds
│   ├── milp.yaml                          M10 horizon, time limit, solver options
│   └── evaluation.yaml                    M11 methods × scenarios × seeds, metric list
├── data/
│   ├── raw/                           ✅  IMMUTABLE. one dir per dataset_id + MANIFEST.json + LICENSE.txt. git-ignored (manifests committed)
│   ├── interim/                       ✅  cleaned/joined intermediates, excluded_days.csv
│   ├── processed/                     ✅  model-ready parquet, forecasts/forecast.parquet, network/
│   └── external/                      ✅  conditional third-party data (e.g. metropt3)
├── src/
│   ├── data/                          ✅  download.py · validate.py · clean.py · features.py · split.py · prepare.py
│   ├── forecasting/                   ✅  naive.py · xgb_point.py · xgb_quantile.py · calibrate.py · metrics.py · train.py
│   ├── train_health/                  ✅  scope pending decision item 4 (synthetic_availability.py and/or metropt3_detector.py)
│   ├── simulator/                     ✅  network.py · trains.py · passengers.py · demand_process.py · disruptions.py · energy_cost.py · engine.py · validate.py
│   ├── environment/                   ✅  metro_env.py · observation.py · action_mask.py · reward.py
│   ├── rl/                            ✅  train.py · policies.py · arms.py (point/placebo/uq) · evaluate.py
│   ├── baselines/                     ✅  fixed_timetable.py · heuristic.py · milp_static.py · milp_rolling.py
│   ├── evaluation/                    ✅  run.py · metrics.py · statistics.py · analyze.py
│   ├── explainability/                ✅  shap_analysis.py · stability.py
│   ├── visualization/                 ✅  figures.py (one function per brief §33 figure) · tables.py
│   └── common/                            M2  config.py · seeding.py · manifest.py · logging.py · paths.py
├── experiments/                       ✅
│   ├── registry.yaml                  ✅  pre-registration; frozen before M11
│   └── run_all.py                         M11
├── tests/                             ✅  see the test map below
├── notebooks/                         ✅  exploration only — never the source of a reported number
├── models/                            ✅  checkpoints, obs_scaler.json, forecaster artefacts (git-ignored)
├── results/                           ✅  results/<scenario>/<method>/<train_seed>/<eval_seed>/{metrics.json, episode_log.parquet, run_manifest.json} — never overwritten
├── logs/                              ✅  training curves, solver logs, stdout captures
├── scripts/                           ✅  thin CLI wrappers; all logic lives in src/
├── requirements.in                    ✅  direct dependencies, unpinned by design
├── requirements.txt                       M2  resolved + hash-pinned lockfile
├── environment.yml                        M2  conda equivalent
├── .gitignore                         ✅
├── .env.example                       ✅
└── README.md                          ✅
```

## Test map (brief §29)

| File | Covers |
|---|---|
| `tests/test_data_quality.py` | The 10 checks in protocol §3.2 |
| `tests/test_raw_immutable.py` | `data/raw/` hashes still match their manifests |
| `tests/test_leakage.py` | The 7 leakage checks in protocol §9.2 |
| `tests/test_no_test_set_tuning.py` | No tuning routine reads a `test`-split artefact |
| `tests/test_no_circular_demand.py` | Simulator demand is independent of the forecaster (MC-06) |
| `tests/test_simulator.py` | Conservation of passengers, capacity never exceeded, headway respected, deterministic under seed |
| `tests/test_env_api.py` | Gymnasium `check_env`, reset/step contracts, info dict keys |
| `tests/test_observation_space.py` | Layout, bounds, dtype, normalisation constants from `train` only, arm-dependent σ channels |
| `tests/test_action_space.py` / `test_action_masking.py` | Mask never empty; masked actions never execute; rejected actions raise and leave state unchanged |
| `tests/test_reward.py` | Stored components reconstruct `r_t` exactly; component signs; normalisation scales fixed |
| `tests/test_constraints.py` | Fleet availability, turnaround, headway, capacity |
| `tests/test_sanity_envs.py` | **Zero demand ⇒ no unnecessary dispatch · all trains unavailable ⇒ dispatch = 0 · extreme demand ⇒ dispatch increases while feasible** |
| `tests/test_common_random_numbers.py` | Demand and failure streams byte-identical across all methods for a given `(scenario, eval_seed)` |
| `tests/test_seed_disjointness.py` | Training and evaluation seed sets never intersect |
| `tests/test_reproducibility.py` | Re-running a stored run reproduces `metrics.json` bit-identically |
| `tests/test_arm_equivalence.py` | The three PPO arms have identical architecture, parameter count, budget and seeds |

## Structural rules

- **No hard-coded experimental parameters** anywhere in `src/`. Every number that could change a
  result lives in `configs/` with a `provenance:` field; a config containing an `unresolved`
  provenance fails to load unless the run is tagged `sensitivity_sweep`.
- **No hard-coded paths.** Resolved through `src/common/paths.py` from the repo root.
- **`notebooks/` may not produce a reported number.** Every figure and table in the paper is
  regenerated by `src/visualization/` from files in `results/`.
- **`results/` is append-only.** Writing to an existing run path raises.
- **`scripts/` contains no logic** — only argument parsing that calls into `src/`.

## Reproduction commands (final form, brief §40)

```
python -m src.data.prepare                 --config configs/data.yaml
python -m src.forecasting.train            --config configs/forecasting.yaml
python -m src.simulator.validate           --config configs/simulator.yaml
python -m src.rl.train                     --config configs/ppo.yaml --arm uq --seed 101
python -m experiments.run_all              --config configs/evaluation.yaml
python -m src.evaluation.analyze           --config configs/evaluation.yaml
python -m src.explainability.shap_analysis --config configs/evaluation.yaml
python -m src.visualization.figures        --config configs/evaluation.yaml
```
