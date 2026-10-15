# Risk Register

Scope: risks to the *scientific validity*, *reproducibility*, and *deliverability* of this research
prototype. Scientific-validity risks are cross-referenced to `research_audit.md` §G/§H.

Severity: **CRITICAL** (invalidates the research claim) · **HIGH** · **MEDIUM** · **LOW**.
Status: `open` · `mitigated` · `accepted` (documented, not fixable) · `closed`.

---

## A. Scientific-validity risks

| ID | Risk | Sev | Likelihood | Mitigation | Owner milestone | Status |
|---|---|---|---|---|---|---|
| R-01 | Circular demand generation (simulator arrivals driven by the forecaster) makes H1 untestable — `audit §G-1` | **CRITICAL** | Certain if the manuscript is implemented literally | Replace with a data-fitted ground-truth demand process; forecaster sees observable history only; guarded by `tests/test_no_circular_demand.py` | M5 | **decided 2026-08-26 (D-01)** — correction accepted; implementation pending at M5 |
| R-02 | **H1 is simply false** — the σ channel provides no measurable benefit | HIGH | **Genuinely plausible**: a PPO agent trained inside a stochastic environment already learns an implicit risk posture | This is not a risk to *validity*, only to a positive result. Protocol includes a TOST equivalence test and a pre-registered SESOI so a null can be reported precisely as a finding. **No re-tuning of the UQ arm to force a win.** | M11 | accepted by design |
| R-03 | A positive H1 result is an artefact of larger observation dimensionality, not of uncertainty | HIGH | Moderate | `ppo_placebo` arm (shuffled σ, identical shape and marginals) as a confirmatory co-primary comparison | M9 | mitigated by design |
| R-04 | Forecast σ is miscalibrated ⇒ H1 tests a broken input, and a null is uninformative | HIGH | Moderate | Blocking calibration gate: 80 % PICP on `val` must fall in [0.76, 0.84] post-calibration before any RL run consumes forecasts | M4 | mitigated by design |
| R-05 | Evaluation episodes nested inside training seeds treated as independent ⇒ inflated significance | HIGH | High (this is the single most common error in RL papers) | Training seed is the unit of analysis; mixed-effects secondary analysis; episode-level n never reported as the sample size | M14 | mitigated by design |
| R-06 | Synthetic OD/alighting model (A03) drives the primary endpoint | HIGH | High | Three OD variants swept; the primary conclusion must hold under all three or be reported as OD-dependent | M5, M11 | open |
| R-07 | MILP straw man — beating a demand-blind static plan is near-tautological | HIGH | High if implemented as the manuscript specifies | Add `milp_rolling` with exactly the RL agent's forecasts; report solver status, gap, timeout for every instance | M10 | **decided 2026-08-26 (D-03)** — `milp_rolling` registered in advance as `EXP-BASE-005` so a negative outcome cannot be quietly dropped |
| R-08 | Reward weights determine the method ranking | HIGH | High | Frozen 5-configuration weight grid; results reported per configuration; convex-hull framing, never "Pareto-optimal" | M12 | mitigated by design |
| R-09 | Asymmetric hyperparameter tuning silently manufactures the result | HIGH | Moderate | One configuration, selected on `val` using the **point-forecast** arm, frozen and applied unchanged to all RL arms | M8 | mitigated by design |
| R-10 | PPO seed variance exceeds the effect size being claimed | HIGH | Moderate–high | ≥10 training seeds; per-seed results published; power analysis after the pilot; no seed dropped post hoc | M8, M11 | open |
| R-11 | Train-health module is scientifically empty (classifier scored on our own synthetic labels) | HIGH | Certain if presented as a prediction contribution | **Decided 2026-08-26 (D-04): option (a).** Reframed as an availability-information ablation (`H7` / `EXP-AVAIL-001`); **no train-health prediction performance is reported as a research result.** MetroPT-3 grounding retained as an optional, off-critical-path extension at M4b | M4b/M9 | **decided — mitigated by reframing** |
| R-12 | Unresolved energy (A15) and cost (A16) parameters underpin two of four reward terms | MEDIUM/HIGH | Certain today | Marked `unresolved`; config refuses to load outside a tagged sensitivity sweep; **no energy/cost headline claim without sourced provenance** | M5 | open |
| R-13 | Multiple comparisons across 9 methods × 4 scenarios × many metrics | MEDIUM | High | Two-test confirmatory family with Holm; everything else exploratory with BH FDR and labelled as such | M14 | mitigated by design |
| R-14 | Sim-to-real gap — all conclusions conditional on the simulator | HIGH | Certain | Irreducible. Honest framing; `simulator_validation.md` states which parameters could and could not be validated | M5 | accepted |
| R-15 | Post-2020 demand non-stationarity puts a structurally different regime in `test` | MEDIUM | Moderate | Pre-registered stationarity screen on `train` only; distribution-shift diagnostics reported between splits | M3 | open |
| R-16 | SHAP attributions unstable/non-identifiable over correlated μ and σ | MEDIUM/HIGH | High | Report cross-seed and cross-background stability; state non-causality; never derive operational recommendations from SHAP alone | M13 | mitigated by design |
| R-17 | Rolling-stock circulation omitted although prior work [1] models it | MEDIUM | Certain | Documented scope limitation (MC-14); depot balance added only if time allows | M5 | accepted |
| R-18 | Single line, single city, single simulator ⇒ weak external validity | MEDIUM | Certain | Conclusions restricted to the studied configuration; second line only if time permits | M16 | accepted |
| R-19 | Researcher degrees of freedom / post-hoc endpoint selection | MEDIUM | Moderate | `experiments/registry.yaml` frozen before runs; append-only amendments log with before/after-runs flag | M11 | mitigated by design |
| R-20 | Unverified references [3]–[6] (research-integrity finding) | **CRITICAL for submission** | Certain today | Author must verify every reference against the publisher record. Out of scope for the code; tracked in MC-15 | pre-submission | open |
| R-21 | Manuscript's "first to jointly address" novelty claim refuted by a single reviewer counterexample | HIGH | Moderate | Rescope to a measurement claim (MC-01) | pre-submission | **decided 2026-08-26 (D-09)** — rescoped; manuscript text still to be edited by the author |

---

## B. Engineering / delivery risks

| ID | Risk | Sev | Mitigation | Status |
|---|---|---|---|---|
| E-01 | Git appeared to be absent (`git` not on PATH). Brief §31 requires git discipline from the beginning; without a commit hash in the run manifest, no run is fully traceable. | HIGH (was blocking M2) | **Resolved 2026-08-26 (D-08): git was already installed at `C:\Program Files\Git\cmd\git.exe`, merely off PATH — no installation needed.** Repository initialised on `main`; author identity set **repo-scoped** (`--local`) so it does not leak into other projects; `.gitattributes` normalises line endings to LF so a clone on Linux/macOS is byte-identical. `src/common/manifest.py` locates git even when it is off PATH, and asserts a 40-character commit in `tests/test_manifest.py`. | **closed** |
| E-02 | Environment Python is **3.14.7**. PyTorch, Stable-Baselines3, sb3-contrib, SHAP, XGBoost and OR-Tools wheel availability on 3.14 is not assured, and SB3 pins a supported range. A broken install discovered at M8 would waste the whole build. | HIGH | **Resolved 2026-08-26 (D-07) by execution, not argument.** Python 3.11.9 installed via the Python Install Manager; `.venv/` created; all 18 dependencies install and import; `scripts/check_environment.py` exercises the specific capability each one was chosen for (MaskablePPO actually refusing a masked action, CP-SAT reporting status and wall time, XGBoost multi-quantile fitting, SciPy BCa bootstrap, deterministic seeded torch) — **10/10 pass**. Torch resolved to a CPU build, which also helps evaluation determinism. | **closed** |
| E-10 | A password was pasted into the working session. | **HIGH (security)** | Not used, not stored, not written to any file in this repository. Git author identity needs only a name and email; any future remote must authenticate with a PAT or SSH key. **The exposed password should be rotated by the account owner.** | **open — owner action** |
| E-03 | Compute budget for the multi-seed sweep. 9 methods × 4 scenarios × 30 eval seeds × S training seeds, plus PPO training per arm per weight configuration, is the dominant cost and is unbounded until a pilot measures it. | HIGH | Staged gating (protocol §10). Pilot at stage 3 measures per-run wall clock; the sweep size is then confirmed against the available budget **before** it starts, not during. | open — decision item 6 |
| E-04 | MILP tractability. A full-day, per-epoch induction model may not solve to optimality within a usable time limit. | MEDIUM | Time limit per instance recorded; solver status and optimality gap stored per instance; the static approximation is defined explicitly; timeouts reported, never silently treated as optimal. | open |
| E-05 | Upstream dataset drift — Socrata datasets are updated, and the "Beginning 2025" split of the ridership series means the analysis window may span two dataset ids. | MEDIUM | Manifest records `dataset_last_updated` and per-file hashes; ingestion of both series is explicit; a hash mismatch on re-download fails loudly rather than silently changing results. | open |
| E-06 | Action-masking dependency: vanilla SB3 `PPO` cannot mask actions, but the manuscript requires masking (§3.4). | MEDIUM | Use `sb3-contrib` `MaskablePPO`; the dependency is justified by a manuscript requirement, not convenience. | mitigated |
| E-07 | Storage growth — per-step episode logs across the full sweep. | LOW/MEDIUM | Parquet with compression; per-step logs retained for the confirmatory experiments and sampled elsewhere; retention policy recorded in the protocol amendments if it changes. | open |
| E-08 | Non-determinism from threading/BLAS/GPU making `test_reproducibility` fail. | MEDIUM | Fixed thread counts, deterministic torch flags, CPU-only for evaluation, seeds recorded in the run manifest. Reproducibility asserted on evaluation, and on training within the tolerance recorded at M2. | open |
| E-09 | SHAP on a policy network is slow (KernelSHAP scales badly in observation dimension ~90). | MEDIUM | Background set subsampled with a recorded size; `DeepExplainer` used where the policy structure permits; the choice and its cost recorded. | open |

---

## C. Risks explicitly accepted (documented, not fixable within this project)

- **R-14** sim-to-real gap — no real-world deployment or shadow-mode data is obtainable.
- **R-17** reduced rolling-stock circulation modelling relative to prior work [1].
- **R-18** single-line, single-city external validity.
- Absence of public OD data (R-06 is mitigated by sweeping, not solved).
- Absence of per-train availability data for the modelled fleet (R-11 is reframed, not solved).

Each of these appears in `docs/limitations.md` and in the paper's limitations section, in these
words, rather than being softened.
