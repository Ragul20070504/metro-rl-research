# FOLLOWUP-UQ-TRAINING — pre-registered follow-up protocol

**Status: PRE-REGISTRATION. Written before any follow-up computation was run.**

This study is **not** part of M0–M17 and does not modify `experiments/registry.yaml`. The original
confirmatory study is immutable and is preserved at git tag `original-confirmatory-study`
(commit `c9b92e49`). Its conclusion stands regardless of anything found here:

> **Under the registered 60,000-step experiment, H1 was NOT SUPPORTED.** `ppo_uq` left more
> passengers behind than `ppo_point` under surge: +0.013742, 95% BCa CI [0.006570, 0.018523],
> Holm-adjusted p = 0.007741.

No result from this follow-up may be merged into, relabelled as, or reported in place of a
confirmatory result. Follow-up cells are written under a separate variant namespace so the
separation is structural rather than editorial.

---

## 1. Why this study exists

Two findings from the completed study, both documented in `docs/q1_q2_readiness_audit.md`, make
the training budget the dominant unresolved confound:

1. **No reinforcement-learning arm beats a tuned threshold heuristic.** In all four scenarios, on
   both the primary endpoint and the training objective, `heuristic_threshold` and `milp_static`
   outperform every learned arm. `ppo_uq` is also worse than the fixed timetable. A PPO
   configuration that cannot beat a threshold rule is plausibly undertrained in absolute terms,
   not merely relative to its siblings.
2. **One frozen configuration, one budget.** Amendment A-05 froze a single PPO configuration for
   every arm at 60,000 timesteps. This protects the comparison from selective tuning, and it is
   why the original study cannot distinguish *the uncertainty channel is unhelpful* from *this
   optimiser cannot exploit it at this budget*.

The mechanism analysis (`results/mechanism_analysis.json`) sharpens the question rather than
settling it. The calibrated sigma causes a large, consistent reduction in dispatch volume
(−14 to −26 trains per episode, p ≤ 4×10⁻⁴ in every scenario), while a placebo sigma of identical
marginal distribution does not (+1.9, p ≈ 0.2–0.3). So the policy is *responding* to genuine
forecast dispersion, and responding by withholding service. Whether that response would persist,
attenuate or reverse with more training is exactly what an unconfounded budget grid can answer and
nothing in the stored data can.

## 2. Research question

**Does the relative performance of uncertainty-aware PPO against point-forecast PPO depend on the
training budget?**

Stated as a testable interaction: is the paired difference `ppo_uq − ppo_point` on
`left_behind_fraction` constant in training budget, or does it vary systematically with it?

This is **not** "can `ppo_uq` be made to win". A budget at which `ppo_uq` wins is only meaningful
if the trend is systematic; a single favourable point inside a noisy grid is not a result, and
§8 pre-commits to reading the grid as a whole.

## 3. Design

| element | value |
|---|---|
| Arms | `ppo_point`, `ppo_uq` — exactly two |
| Training budgets | **15k, 30k, 60k\*, 120k, 240k** timesteps (\* reused from the original study) |
| Training seeds | 101–110 (the registered pool, 10 seeds) |
| Evaluation seeds | 1–30 (the registered pool) |
| Scenarios | normal, surge, disruption, combined |
| Primary endpoint | `left_behind_fraction` under `surge` |
| Unit of analysis | training seed |
| Total policies | 2 arms × 5 budgets × 10 seeds = 100, of which **20 already exist** |
| Total evaluation cells | 2 × 5 × 10 × 4 × 30 = 12,000, of which **2,400 already exist** |

### 3.1 Why this grid rather than the suggested 20k/40k/60k/100k/150k/250k

Documented **before execution**, as required.

- **Geometric spacing (×2) rather than mixed linear.** Compute-scaling effects in RL are
  approximately linear in log-budget, so a grid should be equally spaced in log. A linear grid
  spends resolution in a region where little changes and leaves the top of the range
  under-sampled.
- **16× total range on five points**, versus 12.5× on six for the suggested grid. Wider range is
  what distinguishes "the gap closes" from "the gap is flat"; extra interior points are not.
- **60k is retained exactly** — same seeds, same configuration, same artefacts — so the follow-up
  is anchored to the original study at a shared point rather than merely near it. The registered
  result is a datum inside the grid, not a separate thing being compared to it.
- **Cost.** Derived from measured wall-clock in the stored policy metadata (96.8 s per 1,000
  timesteps, mean over the 50 confirmatory policies): this grid needs 8.1M new training steps
  (≈20 h wall at 11 workers) plus 9,600 new evaluation cells (≈6–11 h). The suggested grid needs
  11.2M steps (≈27 h) plus 12,000 cells (≈8–13 h). The chosen grid answers the same question for
  roughly 25% less compute.

**The grid is fixed now and will not be extended after seeing results.** If the trend is
unresolved at 240k, that is reported as unresolved; adding a budget because 240k looked
promising would reintroduce exactly the post-hoc freedom this protocol exists to remove.

### 3.2 Held fixed across every budget

Architecture, parameter count, optimiser, learning rate, batch size, `n_steps`, `n_epochs`,
gamma, GAE lambda, clip range, entropy and value coefficients, gradient clipping, reward
weights and scales, observation scaler, forecasting model and frozen sigma definition,
environment configuration, simulator configuration, training-day seed pool, evaluation seeds,
scenario rotation, and common-random-number streams.

**Only `total_timesteps` changes.** No hyperparameter is retuned per budget. No checkpoint is
selected within a run: the policy evaluated is the one at the budget, full stop. No seed,
scenario or budget may be dropped after the fact.

## 4. Statistical analysis

Pre-specified in full.

**Primary test — the interaction.** Per training seed *s* and budget *b*, form the paired
difference `d(s,b) = left_behind_fraction[ppo_uq] − left_behind_fraction[ppo_point]` under surge.
Fit a linear mixed model `d ~ log2(budget) + (1 | seed)`. The primary quantity is the slope on
`log2(budget)`, with a Wald test at α = 0.05.

- Slope significantly **negative** → the disadvantage shrinks with budget.
- Slope significantly **positive** → it grows.
- Slope not distinguishable from zero → **no budget dependence detected**, with the confidence
  interval reported so the reader can see what magnitude of trend the design could have excluded.

**Secondary, per budget.** At each budget, a paired test of `ppo_uq` vs `ppo_point` on the primary
endpoint: paired *t* if Shapiro–Wilk on the differences does not reject at 0.05, Wilcoxon signed-
rank otherwise; Hodges–Lehmann estimate with BCa bootstrap confidence intervals. Five tests,
Holm-corrected within this family. Reported for every budget whatever the direction.

**Reported for every budget**: n, mean, median, SD, paired difference, CI, and the per-scenario
breakdown across all four scenarios.

**Multiplicity.** The interaction test is the single primary test of this study and is not
corrected. Per-budget tests are a family of five under Holm. Scenario breakdowns and all mechanism
quantities are exploratory under Benjamini–Hochberg at q = 0.10.

**Power.** The realised between-seed SD of the paired difference in the original study was
0.011269 — **2.13× the pilot estimate** that the original power analysis used. At n = 10 and this
SD, the per-budget MDE is ≈0.0127. This follow-up is therefore **underpowered for per-budget
comparisons of SESOI magnitude** and is powered only for effects of roughly the size already
observed. The interaction test borrows strength across all five budgets and is correspondingly
better powered, which is why it is the primary. This limitation is stated now, before execution,
and will be repeated in any report.

## 5. Stopping rules

- The grid runs to completion. There is **no interim analysis** and no early stopping for a
  favourable trend; the primary test is computed once, after all cells are stored.
- Execution stops if the pinned environment fails `scripts/check_environment.py`, if a stage
  cannot complete after one recorded retry, or if any held-fixed quantity is found to differ
  across budgets. In each case the failure is documented and no partial result is reported as if
  it were the study.
- If compute is exhausted before the grid completes, the **completed budgets are reported as an
  incomplete grid**, explicitly labelled, and the interaction test is either omitted or reported
  with its reduced range stated. Missing budgets are never imputed or quietly dropped.

## 6. Interpretation rules, fixed in advance

1. Whatever this study finds, **the original conclusion is reported unchanged alongside it**: at
   the registered 60,000-step budget, H1 was NOT SUPPORTED.
2. A negative slope that reaches significance supports *the disadvantage is budget-dependent*. It
   does **not** establish that uncertainty information is beneficial; that would require an
   `ppo_uq` advantage at some budget, tested and surviving correction.
3. A flat slope supports *the disadvantage is not explained by training budget over this range*,
   which strengthens the original finding. It does not prove budget-independence beyond 240k.
4. A positive slope is reported as such.
5. Every result here is **FOLLOW-UP / EXPLORATORY**. This document is a genuine prior
   registration, so per-budget tests may be described as pre-specified — but the study was
   designed after the confirmatory result was known, and that ordering is disclosed wherever it
   is reported. It can never be labelled confirmatory.
6. No claim of generalisation beyond this line, simulator, reward and architecture.

## 7. Separation from the original study

- Follow-up cells carry `variant = "budget_<N>"`, so they cannot satisfy the `variant == "main"`
  filter that every confirmatory and exploratory test applies.
- Follow-up policies are written under `models/policies_followup/`, leaving
  `models/policies/` untouched.
- `experiments/registry.yaml` is **not** modified; the frozen registry describes the original
  study only.
- Results are written to `results/followup/` and analysed by a separate script. The original
  `results/statistical_analysis.json` is not regenerated with follow-up data in scope.
- The 60k arm reuses the original policies read-only; it does not retrain or overwrite them.

## 8. What this study cannot settle

- Whether a *tuned* configuration at any budget would behave differently. Only budget varies here;
  a hyperparameter search would reintroduce the selective-tuning problem A-05 was written to
  prevent.
- Whether the RL approach can beat the tuned heuristic. That comparison remains descriptive,
  because the baselines are deterministic and carry no training seed.
- The epoch-level form of H1b — whether dispatch rate rises with forecast dispersion conditional
  on the forecast mean — which needs the per-epoch forecast re-derived from the frozen forecaster.
  It is orthogonal to budget and is listed here so it is not forgotten.
- Anything about the physical system: four of the seven simulator quantities that feed
  `left_behind_fraction` are not externally validated.

## 9. Authorisation

This protocol is complete and fixed. **Execution requires explicit authorisation**, because the
grid costs an estimated 26–31 hours of wall-clock on the study machine. Nothing in §3 may be
changed after execution begins; if the design must change, this document is superseded by a new
pre-registration with the change and its reason recorded, before any further run.

---

## Amendment F-01 — storage mechanism, recorded before execution

**Recorded 2026-08-28, before any follow-up computation ran.**

Section 7 specified `models/policies_followup/` and `results/followup/`. Execution instead uses:

- policies at `models/policies/budget_<N>/<method>/<seed>/policy.zip`
- cells at `variant = "budget_<N>"`, stored under the existing runs tree

**Reason.** The repository already has a tested mechanism for exactly this separation: `Cell`
carries `train_weight_id` (which selects the checkpoint) independently of `weight_id` (which
selects the reward), and `variant` is embedded in every stored cell's path and read by every
analysis filter. Reusing it means the follow-up needs no new storage code. A parallel directory
tree that no existing loader, analysis or audit knows about would be new, untested surface between
the follow-up and the registered study — the opposite of what the separation is for.

**What is unchanged.** The separation is not weakened; it is strengthened. `variant` is part of a
cell's identity rather than merely its location, so a follow-up cell cannot reach a confirmatory
table even if a directory were moved. Verified before launch: 9,600 follow-up cells, of which
**zero** carry `variant == "main"`, all carry `weight_id == "w_balanced"` (reward identical to the
registered study), and all resolve their checkpoints under a `budget_<N>` namespace. The
registered `w_balanced` policies are opened read-only for the 60,000-step point and never
rewritten.

**Scope.** This amendment changes where artefacts are written. It changes no budget, seed,
arm, scenario, endpoint, statistical method, stopping rule or interpretation rule in sections 2–6.
