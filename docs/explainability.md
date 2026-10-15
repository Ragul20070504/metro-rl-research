# Explainability (M13)

**SHAP attribution is associational, not causal.** It decomposes this policy's output on this input distribution. It shows which observation channels the policy's action responds to; it does not show that changing a channel in the world would change outcomes, and it is not evidence for H1. Only the confirmatory tests answer that.

Generated 2026-08-28 from `ppo_uq` at training seed 101, explained quantity: P(dispatch >= 1 trainset) under the policy, explainer: shap.PermutationExplainer (model-agnostic).

## Global attribution (top 15)

| rank | feature | mean \|SHAP\| | mean SHAP |
|---:|---|---:|---:|
| 1 | `fleet_counts[2]` | 0.016095 | -0.0017693 |
| 2 | `disruption_flags[0]` | 0.014743 | -0.0020213 |
| 3 | `disruption_flags[1]` | 0.013782 | -0.0012498 |
| 4 | `queue_length[618]` | 0.0077077 | 0.0011857 |
| 5 | `trains_per_segment[0]` | 0.0071149 | -3.6156e-05 |
| 6 | `queue_length[601]` | 0.005792 | -0.00031239 |
| 7 | `queue_length[137]` | 0.0051953 | 0.00060378 |
| 8 | `queue_length[118]` | 0.0051723 | -0.00017304 |
| 9 | `forecast_sigma_h1[120]` | 0.0049955 | -1.108e-05 |
| 10 | `boardings_prev_epoch[124]` | 0.0049468 | -0.00025684 |
| 11 | `queue_length[122]` | 0.0049176 | 0.0004606 |
| 12 | `forecast_sigma_h1[127]` | 0.00486 | 0.000705 |
| 13 | `forecast_sigma_line_h2h3[1]` | 0.0047537 | -0.00029919 |
| 14 | `boardings_prev_epoch[127]` | 0.0047138 | 0.00014614 |
| 15 | `queue_length[123]` | 0.0047002 | -0.00019228 |

## Attribution to the uncertainty channels

The two sigma blocks account for **16.8%** of total absolute attribution.

What this does and does not mean: a high share is evidence that the policy's dispatch probability *responds* to forecast dispersion. It is **not** evidence that responding improves outcomes — a policy could respond strongly to a channel and be worse for it. The `ppo_uq` vs `ppo_placebo` comparison is what separates those two possibilities, because the placebo arm responds to a channel with the same marginal distribution and no information.

## Stability across training seeds (EXP-XAI-002)

Mean pairwise Spearman rank correlation of global attributions: **0.292** across 3 training seeds.

| seed A | seed B | Spearman rho |
|---:|---:|---:|
| 101 | 102 | 0.3889 |
| 101 | 103 | 0.2545 |
| 102 | 103 | 0.2317 |

High rank correlation means the attribution describes the method; low correlation means it describes the particular training seed and should not be generalised.

## Scenario-specific attribution

| scenario | top feature | mean \|SHAP\| |
|---|---|---:|
| normal | `disruption_flags[0]` | 0.011592 |
| surge | `disruption_flags[0]` | 0.016135 |
| disruption | `fleet_counts[2]` | 0.023293 |
| combined | `fleet_counts[2]` | 0.021912 |

## Caveats

- SHAP attribution is associational, not causal: it decomposes this model's output on this input distribution and says nothing about what changing a channel in the world would do.
- A high attribution to a sigma channel is evidence the policy's action responds to it, not evidence that responding improves outcomes. That question is H1's, and only the confirmatory test answers it.
- Masks are not applied inside the explainer, so infrastructure constraints are not attributed to the policy's reasoning.
- Attributions come from one training seed; EXP-XAI-002 measures their stability across seeds and background resamples.
