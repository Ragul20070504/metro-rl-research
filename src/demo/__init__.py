"""M17 interactive demonstration layer.

This package is a **demonstration**, not part of the experiment. It consumes the validated M0-M16
artefacts -- the fitted forecaster, the calibrated simulator, the frozen RL environment, the
trained policies, the tuned baselines -- and never modifies, retrains or retunes any of them.

Three rules hold throughout, and the tests in `tests/test_demo.py` enforce them:

1. **No second implementation.** Every action comes from the actual policy object, every metric
   from the actual simulator, every reward component from `src.environment.reward`. The dashboard
   renders state; it never computes an outcome of its own.
2. **No future information.** The policy sees exactly the observation the environment builds for
   it. Realised demand is displayed only after the epoch that produced it has been stepped.
3. **No contamination.** Demonstration runs write to `results/demo/` and nothing else. The
   confirmatory tree under `results/runs/` is never opened for writing.
"""

from src.demo.session import (
    DemoSession,
    PolicyOption,
    SCENARIOS,
    available_policies,
)

__all__ = ["DemoSession", "PolicyOption", "SCENARIOS", "available_policies"]
