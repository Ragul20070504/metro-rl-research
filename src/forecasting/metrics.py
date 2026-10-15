"""Forecast evaluation metrics (protocol section 4.4).

Point accuracy alone is explicitly insufficient here: the hypothesis under test concerns the
*uncertainty* the forecaster emits, so calibration and interval quality are first-class
results, not diagnostics.

Two deliberate choices worth stating:

* **MASE, not MAPE, as the headline scale-free metric.** Overnight station-hours are legitimately
  zero or near-zero, which makes MAPE undefined or explosive and would let night hours dominate
  a percentage score. MAPE and SMAPE are still reported, but only over rows above a stated
  floor, and the floor is reported alongside them.
* **CRPS approximated from the quantile set.** The models emit quantiles rather than a closed-form
  density, so CRPS is computed as twice the mean pinball loss across a quantile grid, which is
  the standard discrete approximation and is exact in the limit of a dense grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "mae", "rmse", "mase", "smape", "mape",
    "pinball_loss", "coverage", "mean_interval_width", "winkler_score",
    "crps_from_quantiles", "quantile_calibration_error", "reliability_table",
    "PointMetrics", "ProbabilisticMetrics", "evaluate_point", "evaluate_probabilistic",
]


def _finite_pair(actual: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop non-finite entries pairwise so a single NaN cannot poison a whole metric."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    keep = np.isfinite(actual) & np.isfinite(predicted)
    return actual[keep], predicted[keep]


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error."""
    actual, predicted = _finite_pair(actual, predicted)
    return float(np.mean(np.abs(actual - predicted))) if actual.size else float("nan")


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error."""
    actual, predicted = _finite_pair(actual, predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2))) if actual.size else float("nan")


def mase(actual: np.ndarray, predicted: np.ndarray, naive_errors: np.ndarray) -> float:
    """Mean absolute scaled error.

    Scaled by the in-sample mean absolute error of the seasonal naive reference, so a value
    below 1 means the model beats seasonal naive. Scale-free and defined at zero demand, which
    is why it is the headline metric here.
    """
    actual, predicted = _finite_pair(actual, predicted)
    scale = float(np.mean(np.abs(naive_errors[np.isfinite(naive_errors)])))
    if not np.isfinite(scale) or scale <= 0 or actual.size == 0:
        return float("nan")
    return float(np.mean(np.abs(actual - predicted)) / scale)


def smape(actual: np.ndarray, predicted: np.ndarray, floor: float = 1.0) -> float:
    """Symmetric MAPE, over rows whose actual value is at least ``floor``."""
    actual, predicted = _finite_pair(actual, predicted)
    keep = actual >= floor
    if not keep.any():
        return float("nan")
    a, p = actual[keep], predicted[keep]
    denominator = (np.abs(a) + np.abs(p)) / 2.0
    return float(np.mean(np.abs(a - p) / np.where(denominator == 0, np.nan, denominator)) * 100.0)


def mape(actual: np.ndarray, predicted: np.ndarray, floor: float = 1.0) -> float:
    """MAPE over rows whose actual value is at least ``floor``.

    The floor is not cosmetic: without it this metric is undefined on the many genuinely-zero
    overnight station-hours in this series.
    """
    actual, predicted = _finite_pair(actual, predicted)
    keep = actual >= floor
    if not keep.any():
        return float("nan")
    return float(np.mean(np.abs((actual[keep] - predicted[keep]) / actual[keep])) * 100.0)


def pinball_loss(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    """Pinball (quantile) loss at one quantile level."""
    actual, predicted = _finite_pair(actual, predicted)
    if actual.size == 0:
        return float("nan")
    delta = actual - predicted
    return float(np.mean(np.maximum(quantile * delta, (quantile - 1.0) * delta)))


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction-interval coverage probability (PICP): fraction of actuals inside [lower, upper]."""
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    keep = np.isfinite(actual) & np.isfinite(lower) & np.isfinite(upper)
    if not keep.any():
        return float("nan")
    return float(np.mean((actual[keep] >= lower[keep]) & (actual[keep] <= upper[keep])))


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean prediction-interval width (MPIW).

    Always reported next to coverage: an interval can reach any coverage by becoming wide
    enough, so coverage on its own is not evidence of a good forecaster.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    keep = np.isfinite(lower) & np.isfinite(upper)
    return float(np.mean(upper[keep] - lower[keep])) if keep.any() else float("nan")


def winkler_score(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
    """Winkler interval score for a central ``1 - alpha`` interval.

    Penalises width and misses jointly, so it is the single number that cannot be gamed by
    widening or narrowing alone. Lower is better.
    """
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    keep = np.isfinite(actual) & np.isfinite(lower) & np.isfinite(upper)
    if not keep.any():
        return float("nan")
    a, lo, hi = actual[keep], lower[keep], upper[keep]
    width = hi - lo
    penalty = np.where(a < lo, 2.0 / alpha * (lo - a), 0.0) + np.where(a > hi, 2.0 / alpha * (a - hi), 0.0)
    return float(np.mean(width + penalty))


def crps_from_quantiles(actual: np.ndarray, quantile_predictions: dict[float, np.ndarray]) -> float:
    """CRPS approximated as twice the mean pinball loss over the quantile grid."""
    if not quantile_predictions:
        return float("nan")
    losses = [pinball_loss(actual, predictions, q) for q, predictions in sorted(quantile_predictions.items())]
    finite = [value for value in losses if np.isfinite(value)]
    return float(2.0 * np.mean(finite)) if finite else float("nan")


def reliability_table(actual: np.ndarray, quantile_predictions: dict[float, np.ndarray]) -> list[dict[str, float]]:
    """Nominal vs empirical exceedance for each quantile level.

    This is the raw material for a reliability diagram, stored so the figure is regenerated
    from data rather than drawn by hand.
    """
    rows: list[dict[str, float]] = []
    for level, predictions in sorted(quantile_predictions.items()):
        a, p = _finite_pair(actual, predictions)
        if a.size == 0:
            continue
        rows.append({
            "nominal": float(level),
            "empirical": float(np.mean(a <= p)),
            "n": int(a.size),
        })
    return rows


def quantile_calibration_error(actual: np.ndarray, quantile_predictions: dict[float, np.ndarray]) -> float:
    """Expected calibration error: mean |nominal - empirical| across quantile levels."""
    table = reliability_table(actual, quantile_predictions)
    if not table:
        return float("nan")
    return float(np.mean([abs(row["nominal"] - row["empirical"]) for row in table]))


@dataclass
class PointMetrics:
    """Point-forecast accuracy."""

    mae: float
    rmse: float
    mase: float
    smape: float
    mape: float
    smape_floor: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mae": self.mae, "rmse": self.rmse, "mase": self.mase,
            "smape": self.smape, "mape": self.mape,
            "smape_mape_floor": self.smape_floor, "n": self.n,
        }


@dataclass
class ProbabilisticMetrics:
    """Interval and distributional quality."""

    pinball_by_quantile: dict[float, float] = field(default_factory=dict)
    pinball_mean: float = float("nan")
    picp_80: float = float("nan")
    picp_90: float = float("nan")
    mpiw_80: float = float("nan")
    mpiw_80_normalised: float = float("nan")
    winkler_80: float = float("nan")
    crps: float = float("nan")
    ece: float = float("nan")
    reliability: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pinball_by_quantile": {str(k): v for k, v in self.pinball_by_quantile.items()},
            "pinball_mean": self.pinball_mean,
            "picp_80": self.picp_80, "picp_90": self.picp_90,
            "mpiw_80": self.mpiw_80, "mpiw_80_normalised": self.mpiw_80_normalised,
            "winkler_80": self.winkler_80, "crps": self.crps, "ece": self.ece,
            "reliability": self.reliability,
        }


def evaluate_point(
    actual: np.ndarray, predicted: np.ndarray, naive_errors: np.ndarray, floor: float = 1.0
) -> PointMetrics:
    """Compute the point-accuracy block."""
    a, p = _finite_pair(actual, predicted)
    return PointMetrics(
        mae=mae(actual, predicted),
        rmse=rmse(actual, predicted),
        mase=mase(actual, predicted, naive_errors),
        smape=smape(actual, predicted, floor),
        mape=mape(actual, predicted, floor),
        smape_floor=floor,
        n=int(a.size),
    )


def evaluate_probabilistic(
    actual: np.ndarray, quantile_predictions: dict[float, np.ndarray]
) -> ProbabilisticMetrics:
    """Compute the interval/distributional block from a quantile forecast."""
    metrics = ProbabilisticMetrics()
    if not quantile_predictions:
        return metrics

    metrics.pinball_by_quantile = {
        float(q): pinball_loss(actual, predictions, q) for q, predictions in sorted(quantile_predictions.items())
    }
    finite = [v for v in metrics.pinball_by_quantile.values() if np.isfinite(v)]
    metrics.pinball_mean = float(np.mean(finite)) if finite else float("nan")

    if 0.10 in quantile_predictions and 0.90 in quantile_predictions:
        lower, upper = quantile_predictions[0.10], quantile_predictions[0.90]
        metrics.picp_80 = coverage(actual, lower, upper)
        metrics.mpiw_80 = mean_interval_width(lower, upper)
        metrics.winkler_80 = winkler_score(actual, lower, upper, alpha=0.20)
        mean_actual = float(np.nanmean(np.asarray(actual, dtype=float)))
        metrics.mpiw_80_normalised = (
            metrics.mpiw_80 / mean_actual if np.isfinite(mean_actual) and mean_actual > 0 else float("nan")
        )
    if 0.05 in quantile_predictions and 0.95 in quantile_predictions:
        metrics.picp_90 = coverage(actual, quantile_predictions[0.05], quantile_predictions[0.95])

    metrics.crps = crps_from_quantiles(actual, quantile_predictions)
    metrics.ece = quantile_calibration_error(actual, quantile_predictions)
    metrics.reliability = reliability_table(actual, quantile_predictions)
    return metrics
