"""Post-hoc interval calibration (protocol section 4.3).

A single scalar multiplier rescales every predictive interval about its median so that
empirical 80% coverage on the **validation** split matches the nominal 0.80. Fitted on
validation only and applied unchanged to test.

Why a single scalar rather than something richer: the quantity H1 depends on is whether the
dispersion fed to the policy is *trustworthy*. A one-parameter correction fixes systematic
over- or under-dispersion without giving the uncertainty channel extra expressive power that
the point-forecast arm does not have. A per-station or per-hour correction would begin to
encode information beyond dispersion, and would confound the ablation the whole study rests on.

Rescaling about the median also leaves the point forecast untouched, so calibration cannot
change the mean the point-forecast arm sees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common.logging import get_logger
from src.forecasting.metrics import coverage

logger = get_logger(__name__)


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of fitting the interval-width multiplier."""

    factor: float
    coverage_before: float
    coverage_after: float
    target: float
    n_candidates: int
    #: True when the selected factor sits on an endpoint of the search range, meaning the
    #: range binds and the true optimum may lie outside it. Never allowed to pass silently.
    at_boundary: bool = False

    @property
    def improved(self) -> bool:
        return abs(self.coverage_after - self.target) <= abs(self.coverage_before - self.target)

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "factor": self.factor,
            "coverage_before": self.coverage_before,
            "coverage_after": self.coverage_after,
            "target": self.target,
            "n_candidates": self.n_candidates,
            "improved": self.improved,
            "at_search_boundary": self.at_boundary,
        }


def apply_factor(median: np.ndarray, lower: np.ndarray, upper: np.ndarray, factor: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Rescale an interval about its median by ``factor``, clipped at zero."""
    lower_scaled = np.clip(median + (lower - median) * factor, 0.0, None)
    upper_scaled = np.clip(median + (upper - median) * factor, 0.0, None)
    return lower_scaled, upper_scaled


def fit_interval_scale(
    actual: np.ndarray,
    median: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    target_coverage: float = 0.80,
    search_min: float = 0.01,
    search_max: float = 10.0,
    n_candidates: int = 193,
) -> CalibrationResult:
    """Find the multiplier whose 80% interval coverage is closest to nominal.

    A grid search rather than a root-find: empirical coverage is a step function of the
    multiplier (it changes only when a point crosses a boundary), so derivative-based methods
    are unreliable and a dense grid is both simpler and more robust. Ties are broken toward
    the *narrower* interval, since a wider interval can always reach any coverage and must not
    be preferred for free.

    Two details that a first implementation got wrong, caught by
    ``tests/test_forecasting.py``:

    * The grid is **geometric**, not linear. The correction is multiplicative, so a linear
      grid over [0.2, 5.0] puts almost all its resolution above 1.0 and cannot express the
      small factors a badly over-dispersed forecaster needs. A severely over-wide interval
      required ~0.06 and the linear grid bottomed out at 0.2.
    * The result is **never worse than no correction**. If no candidate improves on
      ``factor = 1.0``, the identity is returned. Otherwise a search-range limitation could
      silently degrade calibration, which would then feed a miscalibrated sigma into H1.
    """
    actual = np.asarray(actual, dtype=float)
    median = np.asarray(median, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    before = coverage(actual, lower, upper)
    if not np.isfinite(before):
        return CalibrationResult(1.0, before, before, target_coverage, 0)

    candidates = np.geomspace(max(search_min, 1e-6), search_max, n_candidates)
    # Always include the identity so "no correction" is a reachable outcome.
    candidates = np.unique(np.append(candidates, 1.0))

    best_factor = 1.0
    best_error = abs(before - target_coverage)
    best_coverage = before
    for factor in candidates:
        lo, hi = apply_factor(median, lower, upper, float(factor))
        achieved = coverage(actual, lo, hi)
        if not np.isfinite(achieved):
            continue
        error = abs(achieved - target_coverage)
        if error < best_error - 1e-12 or (abs(error - best_error) <= 1e-12 and factor < best_factor):
            best_error, best_factor, best_coverage = error, float(factor), achieved

    if best_error > abs(before - target_coverage) + 1e-12:
        logger.warning("no calibration factor improved coverage; falling back to the identity")
        best_factor, best_coverage = 1.0, before

    at_boundary = bool(
        best_factor != 1.0
        and (
            np.isclose(best_factor, candidates[0], rtol=1e-6)
            or np.isclose(best_factor, candidates[-1], rtol=1e-6)
        )
    )
    if at_boundary:
        logger.warning(
            "calibration factor %.4f sits on a search-range endpoint [%.4f, %.4f]: the range "
            "binds and the true optimum may lie outside it. Coverage reached %.4f against a "
            "target of %.2f. This is recorded in the manifest, not silently accepted.",
            best_factor, candidates[0], candidates[-1], best_coverage, target_coverage,
        )

    result = CalibrationResult(
        factor=best_factor,
        coverage_before=before,
        coverage_after=best_coverage,
        target=target_coverage,
        n_candidates=int(n_candidates),
        at_boundary=at_boundary,
    )
    logger.info("interval calibration: factor %.3f, coverage %.3f -> %.3f (target %.2f)",
                result.factor, result.coverage_before, result.coverage_after, target_coverage)
    return result
