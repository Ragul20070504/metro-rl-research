"""Forecasting models: seasonal naive, XGBoost point, XGBoost quantile (protocol section 4.3).

The progression is deliberately conservative (brief section 8, correction MC-13): start with a
reference floor, then a strong tabular baseline, then the probabilistic model that supplies the
uncertainty channel H1 is about. No graph network, no Bayesian LSTM -- 24 stations of hourly
counts do not justify them, and an unnecessarily elaborate forecaster would make it harder,
not easier, to attribute any downstream effect to uncertainty representation.

The uncertainty representation is frozen by the protocol::

    sigma_hat = (q90 - q10) / (2 * z90),   z90 = 1.2816

i.e. a Gaussian-equivalent scale from the central 80% interval. This is a documented
simplification of the full predictive distribution; feeding raw quantiles into the RL state is
recorded as a future ablation and is not run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.common.logging import get_logger

logger = get_logger(__name__)

#: Standard normal quantile at 0.90; converts an 80% interval half-width to a sigma.
Z90 = 1.2816

#: Quantiles the probabilistic model emits. The outer pair supports 90% coverage
#: diagnostics; the inner pair defines the frozen sigma.
DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def sigma_from_interval(q10: np.ndarray, q90: np.ndarray) -> np.ndarray:
    """Convert an 80% central interval to a Gaussian-equivalent standard deviation.

    Clipped at zero because quantile crossing (q90 < q10) is possible in gradient-boosted
    quantile regression and a negative dispersion is meaningless. Crossings are counted and
    reported by the trainer rather than silently absorbed.
    """
    return np.clip((np.asarray(q90, float) - np.asarray(q10, float)) / (2.0 * Z90), 0.0, None)


@dataclass
class ForecastOutput:
    """Predictions for one evaluation block.

    Attributes:
        mean: Point forecast fed to the point-forecast RL arm.
        quantiles: Level -> predictions. Empty for point-only models.
        sigma: Frozen dispersion, or zeros for point-only models.
    """

    mean: np.ndarray
    quantiles: dict[float, np.ndarray] = field(default_factory=dict)
    sigma: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.sigma is None:
            self.sigma = np.zeros_like(self.mean, dtype=float)


class Forecaster(ABC):
    """Common interface so the trainer treats every model identically."""

    model_id: str
    is_probabilistic: bool = False

    @abstractmethod
    def fit(self, features: pd.DataFrame, target: np.ndarray) -> "Forecaster":
        """Fit on training rows."""

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> ForecastOutput:
        """Predict for the given rows."""

    def config(self) -> dict[str, Any]:
        """Hyperparameters, recorded alongside the stored predictions."""
        return {}


class SeasonalNaive(Forecaster):
    """Reference floor: predict the value at the same clock hour one week earlier.

    Uses the ``y_seasonal_168`` feature, which the feature builder has already constructed
    leakage-safely, rather than re-deriving the lag here. Reusing the same column guarantees
    the baseline and the learned models see exactly the same history.
    """

    model_id = "naive_seasonal"

    def __init__(self, lag_column: str = "y_seasonal_168") -> None:
        self.lag_column = lag_column

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> "SeasonalNaive":
        if self.lag_column not in features.columns:
            raise KeyError(f"{self.lag_column!r} is required by SeasonalNaive but is absent")
        return self

    def predict(self, features: pd.DataFrame) -> ForecastOutput:
        return ForecastOutput(mean=features[self.lag_column].to_numpy(dtype=float))

    def config(self) -> dict[str, Any]:
        return {"lag_column": self.lag_column, "seasonal_period_h": 168}


class XGBPointForecaster(Forecaster):
    """Squared-error XGBoost. Supplies the mean to the point-forecast RL arm."""

    model_id = "xgb_point"

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "random_state": 0,
            "n_jobs": 4,
            "tree_method": "hist",
            **params,
        }
        self._model: Any = None
        self._columns: list[str] = []

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> "XGBPointForecaster":
        import xgboost as xgb

        self._columns = list(features.columns)
        self._model = xgb.XGBRegressor(**self.params)
        self._model.fit(features, target, verbose=False)
        return self

    def predict(self, features: pd.DataFrame) -> ForecastOutput:
        if self._model is None:
            raise RuntimeError("XGBPointForecaster.predict called before fit")
        predictions = np.clip(self._model.predict(features[self._columns]), 0.0, None)
        return ForecastOutput(mean=np.asarray(predictions, dtype=float))

    def config(self) -> dict[str, Any]:
        return dict(self.params)


class XGBQuantileForecaster(Forecaster):
    """Pinball-loss XGBoost emitting a quantile set; source of the uncertainty channel.

    XGBoost's multi-quantile objective fits all levels in one model, which keeps the levels
    mutually consistent far better than fitting them independently. Residual crossings are
    repaired by sorting each row's quantiles, and the crossing rate is recorded so the repair
    is visible rather than hidden.
    """

    model_id = "xgb_quantile"
    is_probabilistic = True

    def __init__(self, quantiles: tuple[float, ...] = DEFAULT_QUANTILES, **params: Any) -> None:
        self.quantiles = tuple(sorted(quantiles))
        self.params: dict[str, Any] = {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_lambda": 1.0,
            "objective": "reg:quantileerror",
            "random_state": 0,
            "n_jobs": 4,
            "tree_method": "hist",
            **params,
        }
        self._model: Any = None
        self._columns: list[str] = []
        self.crossing_rate: float = float("nan")
        #: Multiplicative interval-width correction, fitted on validation only.
        self.calibration_factor: float = 1.0

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> "XGBQuantileForecaster":
        import xgboost as xgb

        self._columns = list(features.columns)
        self._model = xgb.XGBRegressor(quantile_alpha=np.array(self.quantiles), **self.params)
        self._model.fit(features, target, verbose=False)
        return self

    def _raw_quantiles(self, features: pd.DataFrame) -> np.ndarray:
        predictions = np.atleast_2d(self._model.predict(features[self._columns]))
        if predictions.shape[0] == len(self.quantiles) and predictions.shape[1] == len(features):
            predictions = predictions.T
        return np.asarray(predictions, dtype=float)

    def predict(self, features: pd.DataFrame) -> ForecastOutput:
        if self._model is None:
            raise RuntimeError("XGBQuantileForecaster.predict called before fit")

        raw = self._raw_quantiles(features)
        self.crossing_rate = float(np.mean(np.diff(raw, axis=1) < 0))
        ordered = np.sort(raw, axis=1)  # enforce monotone quantiles
        ordered = np.clip(ordered, 0.0, None)

        median_index = self.quantiles.index(0.50)
        median = ordered[:, median_index].copy()

        if self.calibration_factor != 1.0:
            # Rescale each quantile's offset from the median. Widening or narrowing about the
            # median leaves the point forecast untouched, so calibration cannot change the
            # mean the point-forecast arm would have seen.
            ordered = median[:, None] + (ordered - median[:, None]) * self.calibration_factor
            ordered = np.clip(ordered, 0.0, None)

        quantiles = {level: ordered[:, index] for index, level in enumerate(self.quantiles)}
        sigma = sigma_from_interval(quantiles[0.10], quantiles[0.90])
        return ForecastOutput(mean=median, quantiles=quantiles, sigma=sigma)

    def config(self) -> dict[str, Any]:
        return {
            **self.params,
            "quantiles": list(self.quantiles),
            "calibration_factor": self.calibration_factor,
            "crossing_rate": self.crossing_rate,
            "sigma_definition": "(q90 - q10) / (2 * 1.2816)",
        }


MODEL_REGISTRY: dict[str, type[Forecaster]] = {
    SeasonalNaive.model_id: SeasonalNaive,
    XGBPointForecaster.model_id: XGBPointForecaster,
    XGBQuantileForecaster.model_id: XGBQuantileForecaster,
}
