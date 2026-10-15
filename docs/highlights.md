# Highlights

*Journal highlights are conventionally 3–5 bullets of up to 85 characters each. Both a compliant short form and an expanded form are given.*

## Short form (≤85 characters)

- Pre-registered placebo-controlled test of forecast uncertainty in metro dispatch
- Uncertainty-aware PPO left more passengers behind than point-forecast PPO
- Effect absent under a placebo signal of identical marginal distribution
- Calibrated uncertainty cut dispatch volume by about 11% under demand surge
- A tuned threshold heuristic outperformed every learned controller tested

## Expanded form

- In a pre-registered, placebo-controlled evaluation, an uncertainty-aware PPO metro dispatch controller performed **worse** than an otherwise identical point-forecast controller on the registered primary endpoint (+0.013742; Holm-adjusted p = 0.007741).
- The degradation was also present against a placebo uncertainty channel with matched dimensionality and marginal distribution, so it is not explained by additional inputs.
- Genuine uncertainty was associated with approximately 11% fewer dispatches; the placebo signal was not.
- Measured service loss is strongly convex once capacity binds, so uncertainty is decision-relevant and the observed conservatism runs opposite to the direction convex loss motivates.
- No learned controller outperformed a tuned threshold heuristic in any scenario, which bounds current claims for learned metro dispatch.
