"""Demand forecasting with uncertainty quantification (M4).

Implemented independently of the RL components, so the forecaster can be evaluated on its own
terms before anything consumes it. The uncertainty representation frozen by the protocol --
``sigma = (q90 - q10) / (2 * 1.2816)`` -- is what the uncertainty-aware policy arm observes and
what the placebo arm shuffles.
"""
