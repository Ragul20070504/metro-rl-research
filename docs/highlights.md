# Highlights

*Journal highlights are conventionally 3–5 bullets of up to 85 characters each. Both a compliant short form and an expanded form are given.*

## Short form (≤85 characters)

- Placebo-controlled test of forecast uncertainty in learned metro dispatch
- Uncertainty-aware PPO left more passengers behind than point-forecast PPO
- Signal content affected performance with input dimensions held constant
- Calibrated uncertainty cut dispatch volume by about 11% under demand surge
- A tuned threshold heuristic outperformed every learned controller tested

## Expanded form

- In an internally specified, placebo-controlled evaluation, an uncertainty-aware PPO metro dispatch controller performed **worse** than an otherwise identical point-forecast controller on the registered primary endpoint (+0.013742; Holm-adjusted p = 0.007741).
- The degradation was also present against a placebo uncertainty channel with matched dimensionality. Its bootstrap construction does not isolate all information and distributional effects.
- Genuine uncertainty was associated with approximately 11% fewer dispatches; the placebo signal was not.
- The binned service-loss curve does not identify an optimal dispatch direction.
- No learned controller outperformed a tuned threshold heuristic in any scenario, which bounds current claims for learned metro dispatch.
