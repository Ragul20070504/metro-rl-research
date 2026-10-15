# Dependency Plan

Rule (brief §6): every dependency must have a reason. Nothing is added for convenience or fashion.

## Interpreter

**Target: Python 3.11. RESOLVED — 3.11.9, installed and verified 2026-08-26.**

The system interpreter is 3.14.7, which is ahead of assured wheel support for PyTorch,
Stable-Baselines3, sb3-contrib, SHAP, XGBoost and OR-Tools (risk E-02). Rather than argue about
whether it would work, the environment was built and exercised:

```
py install 3.11                                          # Python Install Manager -> 3.11.9 (64-bit)
%LOCALAPPDATA%\Python\pythoncore-3.11-64\python.exe -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.in
.venv\Scripts\python.exe scripts\check_environment.py    # 10/10 functional checks pass
```

The system 3.14 interpreter is untouched and is not used by this project.

### Resolved versions (2026-08-26, Python 3.11.9, Windows AMD64)

| Package | Version | | Package | Version |
|---|---|---|---|---|
| numpy | 2.4.6 | | torch | **2.13.0+cpu** |
| pandas | 3.0.5 | | gymnasium | 1.3.0 |
| pyarrow | 25.0.1 | | stable-baselines3 | 2.9.0 |
| scipy | 1.17.1 | | sb3-contrib | 2.9.0 |
| scikit-learn | 1.9.0 | | ortools | 9.15.6755 |
| statsmodels | 0.14.6 | | shap | 0.51.0 |
| xgboost | 3.2.0 | | matplotlib | 3.11.1 |
| pyyaml | 6.0.3 | | pytest | 9.1.1 |

**OBSERVATION.** Torch resolved to a **CPU build**. That is a benefit here rather than a compromise:
evaluation runs on CPU anyway (protocol determinism configuration), and it removes GPU
non-determinism from the reproducibility argument. Training throughput for an MLP policy on this
problem is not GPU-bound.

**OBSERVATION.** pandas resolved to **3.0.x** and numpy to **2.4.x**, both majors with API changes
relative to the 1.x/1.x era that most published transit-RL code targets. Module code is written
against these versions and the lockfile pins them, so this is recorded, not a latent surprise.

### Functional verification, not just imports

`scripts/check_environment.py` exercises the specific capability each dependency was chosen for,
because importing a package proves it installed, not that it works. All ten pass:

| Check | What it actually proves |
|---|---|
| numpy seeding | `SeedSequence.spawn` streams are reproducible **and** mutually independent — the basis of common random numbers |
| pandas/parquet | exact round-trip, needed for `forecast.parquet` and episode logs |
| scipy paired tests | Wilcoxon signed-rank, Shapiro–Wilk and BCa bootstrap all functional (protocol §8.3) |
| statsmodels multiplicity | Holm and Benjamini–Hochberg corrections functional |
| xgboost quantile | multi-quantile pinball regression fits with 0.5 % quantile crossing |
| torch determinism | identical seeds produce identical CPU forward passes |
| **MaskablePPO** | trains, and **never selects a masked action** — if masking silently failed, invalid dispatches would corrupt every experiment |
| ortools CP-SAT | solves, and **status, objective and wall time are recoverable**, as the protocol requires for every MILP instance |
| shap | attributions recover the known dominant feature |
| matplotlib | Agg backend renders headless, so figures generate in batch |

## Core dependencies and their justification

| Package | Why it is here | Named in brief §6? |
|---|---|---|
| `numpy` | Numerical core; simulator state arrays; RNG (`Generator`/`SeedSequence`) for reproducible, independently seeded streams | yes |
| `pandas` | Tabular ingestion, resampling, chronological splitting, results aggregation | yes |
| `pyarrow` | Parquet I/O for `forecast.parquet` and per-step episode logs — required by brief §8; pandas needs an engine | implied |
| `scipy` | Statistical tests (Shapiro–Wilk, Wilcoxon signed-rank, paired *t*), distributions, bootstrap | yes |
| `scikit-learn` | Metrics (ROC-AUC, PR-AUC, Brier), calibration curves, preprocessing utilities, `TimeSeriesSplit`-style walk-forward helpers | yes |
| `xgboost` | Point forecaster and quantile (pinball-loss) forecaster; the manuscript specifies XGBoost for train health | yes |
| `torch` | Backend for Stable-Baselines3 policies | yes |
| `gymnasium` | Environment API required by brief §10 and §12 | yes |
| `stable-baselines3` | PPO implementation — brief §16 requires a well-tested library rather than a hand-rolled PPO | yes |
| **`sb3-contrib`** | **`MaskablePPO`.** Vanilla SB3 `PPO` cannot do action masking, and the manuscript (§3.4) mandates masking rather than penalties. This dependency is required by the *research design*, not by convenience | **added — justified** |
| `shap` | Post-hoc attribution of the trained policy (manuscript §4.4, brief §24) | yes |
| `ortools` | MILP baselines (`milp_static`, `milp_rolling`); the manuscript names OR-Tools explicitly | yes |
| `matplotlib` | All figures | yes |
| `pyyaml` | YAML configuration | yes |
| `pytest` | Test suite: unit, simulator, environment, observation, action, reward, constraint, leakage, reproducibility | yes |
| `statsmodels` | Mixed-effects model for the secondary (episode-level) analysis, multiple-comparison procedures (Holm, BH), and diagnostics that SciPy does not provide | **added — justified by protocol §8.2/§8.3** |
| `tqdm` | Progress reporting for long sweeps; keeps long-running jobs observable without bespoke logging | added — minor |

## Deliberately **not** included

| Package | Why not |
|---|---|
| `seaborn` | Brief §6 permits it "only if genuinely useful". Matplotlib covers every figure in brief §33. Not added. |
| `mlflow` | Same clause. The run-manifest + `results/<scenario>/<method>/<seed>/` layout already satisfies brief §21 and §25 traceability, and MLflow would add a service dependency to a fresh-clone reproduction. Not added. |
| `torch-geometric` / any GNN stack | Brief §8 and MC-13: start simple. A 24-station line with hourly counts does not justify a GNN. Would be added only if the quantile model fails its calibration gate for a reason a spatial model would fix. |
| `optuna` / any HPO framework | Protocol §4.3 and §6.2 freeze a small explicit grid deliberately — automated search over the RL arms is precisely the mechanism by which asymmetric tuning (R-09) creeps in. |
| `ray[rllib]` | SB3 + `SubprocVecEnv` is sufficient at this scale and is far easier for another researcher to reproduce. |
| `wandb` / any hosted tracker | Would require credentials and a network service in the reproduction path. |
| `tensorflow` | Not needed; `torch` is the SB3 backend. |

## Version pinning and lockfile

**No versions are pinned in this document.** Pinning numbers that have not been resolved by an actual
install would be fabricated provenance — the same failure mode this project exists to avoid.

At **M2** the following are produced and committed:

- `requirements.in` — the direct dependencies above, with the minimum constraints implied by the
  interpreter choice (already written, unpinned by design).
- `requirements.txt` — the **fully resolved, hash-pinned** output of `pip-compile --generate-hashes`
  (or `uv pip compile`) run in the 3.11 environment. This is the lockfile.
- `environment.yml` — conda equivalent, for users who prefer conda; pinned to the same resolved
  versions.
- `docs/environment_report.md` — captured `pip freeze`, interpreter version, platform, CPU/GPU,
  BLAS threading settings. The same information is embedded in every run manifest.

## Determinism configuration (recorded at M2)

- Fixed `PYTHONHASHSEED`.
- One `numpy.random.SeedSequence` per run, spawned into independent child streams for demand,
  disruptions, policy sampling and tie-breaking, so streams stay independent and reproducible under
  common random numbers.
- `torch.use_deterministic_algorithms(True)` where supported; fixed intra-op thread count;
  `OMP_NUM_THREADS`/`MKL_NUM_THREADS` fixed and recorded.
- Evaluation runs CPU-only, so results do not depend on GPU non-determinism.
- Any residual non-determinism found at M2 is measured and its tolerance recorded, rather than
  asserted to be zero.

## External tooling

| Tool | Status | Note |
|---|---|---|
| **Git** | **Present — 2.55.0**, at `C:\Program Files\Git\cmd\git.exe` | It was never missing, only absent from `PATH`, which is why the initial audit could not see it. No installation was required. `src/common/manifest.py` probes the standard Windows install locations so run manifests capture a real commit hash even when git is off `PATH`; `tests/test_manifest.py` asserts a 40-character commit. Author identity is set **repo-scoped** (`git config --local`), not globally. |
| A C/C++ toolchain | Not needed | Every package above resolved to a prebuilt Windows wheel for 3.11; no source build occurred. |
