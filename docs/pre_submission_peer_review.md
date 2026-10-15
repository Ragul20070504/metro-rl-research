# Pre-submission peer review

Generated at commit `ca3d5f182a1c` on 2026-08-28. Written adversarially: the goal is to find the reasons a reviewer would reject this, not to defend it.

---

## 1. What the study actually establishes

- **EXP-UQ-001** (`ppo_uq` vs `ppo_point`, n = 10 training seeds): **rejects** the null at family-wise alpha 0.05. paired t-test, statistic 3.856, exact p = 0.003871, Holm-adjusted p = 0.007741. Hodges-Lehmann difference 0.013588 (95% BCa CI 0.0065701 to 0.018523).
- **EXP-UQ-002** (`ppo_uq` vs `ppo_placebo`, n = 10 training seeds): **rejects** the null at family-wise alpha 0.05. paired t-test, statistic 3.152, exact p = 0.01171, Holm-adjusted p = 0.01171. Hodges-Lehmann difference 0.0078149 (95% BCa CI 0.0028132 to 0.014461).

---

## 2. What a reviewer will attack, hardest first

### 2.1 The digital twin is a simulator, and seven of its quantities cannot be validated

4 quantities are validated against real published statistics, 3 partially, and **7 cannot be validated at all** — including peak occupancy and the origin-destination distribution, **both of which feed the primary endpoint**.

The honest position: the OD model is synthetic (assumption A03) and the capacity figure is ungrounded (A09). `EXP-SENS-002` sweeps all three OD variants precisely so the primary conclusion can be reported as OD-dependent if it is. A reviewer who asks *"how do you know the OD split is right?"* must be told plainly: we do not, and that is why the sweep exists.

### 2.2 Energy and cost are not in physical units

Assumptions A15 and A16 remain unresolved: no citable conversion to kWh or currency has been sourced for this fleet. Energy is reported in train-km-equivalent units and cost in induction-equivalent units, and both are **embargoed from headline claims**. Any sentence of the form "saves X kWh" would be fabricated and does not appear.

### 2.3 The train-health module makes no prediction claim

No public per-train availability record exists for the modelled fleet. Decision D-04 demoted the module from a contribution to an information channel whose *value to the policy* is measured (H7). **No ROC-AUC or PR-AUC on synthetic labels is reported as a research result** — a classifier scored on labels from a process we wrote down measures only our ability to recover our own process.

### 2.4 The H7 ablation is dimensionality-confounded

Protocol section 6.1 specifies the availability channel is *present only when enabled*, so `ppo_uq_noavail` has 121 observation values against `ppo_uq`'s 123. Unlike the sigma arms — designed specifically to hold dimensionality fixed — this contrast varies both the information and the input width. Reported with that caveat.

### 2.5 The MILP baseline is a fluid approximation of a queueing simulator

The simulator models per-station FIFO cohorts, finite train capacity and platform contention; the MILP optimises a line-aggregate fluid queue with a fitted frequency-dependent waiting term. Any RL advantage over it is **partly an advantage over that abstraction gap**, not pure optimisation superiority. Two calibration defects that would have made it a strawman were found and fixed before use, and both are recorded in `models/milp_surrogate.json`. Solver status and optimality gap are reported per instance.

### 2.6 No PPO hyperparameter search was run

One frozen configuration is applied unchanged to every arm (amendment A-05). This satisfies the symmetry constraint exactly, but means **all arms may be jointly suboptimal**. It bounds absolute performance claims; it does not bias the relative comparison, which is what the confirmatory tests measure.

### 2.7 The demand regime is not stationary

The pre-registered regime screen **fired its fallback**: no 365-day run exists with |YoY growth| <= 10% (longest 278 days). The tolerance was **not** loosened to make it pass. Residual drift within the study window is carried into the limitations, and `EXP-ROBUST-004` tests out-of-regime transfer explicitly.

### 2.8 Statistical power

Pilot-estimated between-training-seed SD: 0.005298. Pre-registered SESOI: 0.002. Training seeds required to detect it at 80% power: 69; registered S = 10. Minimum detectable effect at the registered S: 0.005909.

**Decision taken: PROCEED with the registered n and report as potentially underpowered.** detecting the SESOI would need S = 69, which is beyond the compute available. Protocol section 8.4's honest option is taken: run the registered design and state the minimum detectable effect in the paper. Lowering the SESOI is not an option.

The SESOI was frozen in amendment A-04 *before* the pilot ran and has not been revisited. A reviewer should check that: it is the single easiest thing to fudge in a study like this.

---

## 3. Design choices a reviewer should credit

- **The placebo arm.** `ppo_point` vs `ppo_uq` alone cannot separate *"uncertainty is informative"* from *"a wider observation helps"*. `ppo_placebo` receives sigma values with the correct marginal distribution and destroyed association, at identical dimensionality and parameter count. A result where `ppo_uq ≈ ppo_placebo > ppo_point` refutes the manuscript's mechanism even though a two-arm study would report it as a win.
- **Common random numbers enforced structurally.** `environment_streams` takes no method argument, so no method can perturb the demand or failure realisation it faces.
- **Constraints masked, not penalised.** Zero headway violations is a structural guarantee, not a tuning outcome.
- **The forecaster never sees the sampled demand path.** The manuscript's original design generated arrivals *from* the forecast, which would have made the forecaster correct by construction; protocol section 5.1 replaced it.
- **Amendments are dated and were made before the runs they affect.**

---

## 4. Corrections made to the manuscript's design

These are not incidental: several would have invalidated the headline claim.

| id | correction |
|---|---|
| A-01 / D-12 | The pre-registered primary endpoint was degenerate (~0.05% under every policy). Re-operationalised as left-behind passengers, before any policy existed. |
| A-02 / D-13 | The reward was degenerate: reading `W_t` as the mean wait of *boarders* made "run no service all day" score the global maximum while stranding 100% of demand. |
| A-03 / D-14 | Scenario random streams were keyed on the scenario *name*, so `normal` and `surge` at the same seed were unrelated realisations. `surge` scored **better** than `normal` in 3 of 5 seeds. |
| MC-02 | Beijing Metro open data is cited in the manuscript as available; it could not be confirmed as publicly downloadable. |
| MC-04 | "Digital twin" replaced by "calibrated simulation environment" — there is no live asset coupling. |
| MC-05 | A rolling-horizon MILP was added; comparing only against a static MILP would have been a weak baseline. |

---

## 5. Claims this study cannot support

- Real-world deployment performance. **Everything here is simulation.**
- Energy or cost savings in physical units (A15/A16 unresolved).
- Train-health prediction accuracy (D-04).
- Generalisation to other lines, other cities, or the pre-pandemic demand regime beyond what `EXP-ROBUST-004` measures.
- Causal interpretation of SHAP attributions — the stored analysis states this in its own caveats field.

---

## 6. Reproducibility

`scripts/reproducibility_audit.py`: **22 pass, 0 fail, 2 warn, 0 skipped.** Full detail in `docs/reproducibility_audit.md`.

---

## 7. Forecasting

Blocking gate (protocol section 4.5): **passed**. Pinball loss 15.2931 against a seasonal-naive floor of 47.9578; calibrated 80% coverage 0.8017 inside the pre-registered band [0.76, 0.84].

Forecasting quality and downstream scheduling usefulness are **separate questions**. The gate is a precondition for testing H1, not evidence about it.

---

## 8. Verdict

The contribution that survives scrutiny is **the placebo-controlled design and the corrections it forced**, not a performance number. Three defects found during implementation would each have produced a confidently reported but meaningless result: a degenerate endpoint, a degenerate reward, and unmatched scenario realisations. All three were caught by measuring something that should have been impossible, not by code review.

Whether H1 is supported is a question for the confirmatory tests, and the study is built to report a null result as precisely as a positive one.
