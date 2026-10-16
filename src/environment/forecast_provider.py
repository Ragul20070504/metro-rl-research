"""In-episode demand forecasting for the RL environment (protocol section 5.1 item 3).

**Why this module exists at all.** The manuscript generates passenger arrivals from the demand
*forecast* (manuscript section 5). That is circular: it makes the forecaster correct by
construction and turns sigma into a parameter we chose rather than a quantity we measured
(`docs/research_audit.md` section G-1). Protocol section 5.1 replaces it, and item 3 is the
operative sentence:

    The forecaster never sees the sampled path. It receives only the observable history
    (previous realised hours of the same episode plus the historical training data) and
    produces (mu, sigma-hat).

So the forecast the agent observes at epoch `t` must be recomputed *inside the episode*, from
realised boardings up to `t` and from real history before the episode began. It cannot be
looked up from a precomputed table keyed on the simulated day, because the simulated day is a
fresh draw from the ground-truth process that no stored forecast corresponds to.

That is also what makes the leakage test meaningful: a provider that reads any hour at or after
the issue time will be caught by `tests/test_leakage.py`, which injects a large future spike and
asserts the observation at `t` does not move.

Two providers are supplied:

* :class:`SeasonalNaiveProvider` -- needs no trained artefact, so unit tests and the toy network
  can exercise the whole environment. Its sigma is the dispersion of recent same-hour-of-week
  observations, a purely historical estimate.
* :class:`TrainedForecastProvider` -- wraps the M4 `xgb_point` / `xgb_quantile` models and the
  frozen calibration factor. This is the one the experiments use.

Both build features through the same :class:`HistoryBuffer`, so the leakage guarantee is a
property of the buffer's API rather than of each provider's discipline: :meth:`HistoryBuffer.window_ending_at`
refuses to return any hour at or after the issue time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.common.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "HistoryBuffer",
    "ForecastProvider",
    "SeasonalNaiveProvider",
    "TrainedForecastProvider",
    "load_trained_provider",
    "Forecast",
    "SIGMA_FROM_INTERVAL_Z",
]

#: sigma-hat = (q90 - q10) / (2 * z90), frozen in protocol section 4.3.
SIGMA_FROM_INTERVAL_Z = 1.2816

#: Hours of history the frozen feature whitelist needs (168 h seasonal lag + a 168 h rolling
#: window means the deepest reach back from an issue time is 168 hours; the seasonal lag is
#: taken relative to the *target* time, which is up to 3 h later, so 168 is sufficient with the
#: target-time offset applied).
REQUIRED_HISTORY_HOURS = 168


@dataclass(frozen=True)
class Forecast:
    """Per-station forecast at the three frozen horizons.

    Attributes:
        mu: ``(n_stations, 3)`` conditional mean boardings per hour, horizons 1..3.
        sigma: ``(n_stations, 3)`` Gaussian-equivalent dispersion, same shape.
        issue_hour_index: Absolute hour index the forecast was issued at, for auditing.
    """

    mu: np.ndarray
    sigma: np.ndarray
    issue_hour_index: int


class HistoryBuffer:
    """Observable hourly boardings per station, indexed by absolute hour.

    The buffer holds pre-episode history (real observed data for the days preceding the
    simulated day) followed by hours realised within the episode. ``origin_index`` is the
    absolute index of the first *episode* hour, so history occupies negative offsets.

    The one method that reads data, :meth:`window_ending_at`, takes the issue hour and returns
    only strictly-earlier values. Every feature in the frozen whitelist is built from it, which
    is why no provider can leak the future by accident.
    """

    def __init__(self, n_stations: int) -> None:
        self.n_stations = n_stations
        self._values: list[np.ndarray] = []
        self._history_hours: int = 0

    def set_history(self, history: np.ndarray) -> None:
        """Install pre-episode history, shape ``(hours, n_stations)``, oldest hour first."""
        history = np.asarray(history, dtype=float)
        if history.ndim != 2 or history.shape[1] != self.n_stations:
            raise ValueError(
                f"history must be (hours, {self.n_stations}); got {history.shape}"
            )
        if history.shape[0] < REQUIRED_HISTORY_HOURS:
            raise ValueError(
                f"the frozen feature whitelist reaches back {REQUIRED_HISTORY_HOURS} hours; "
                f"only {history.shape[0]} hours of history were supplied"
            )
        self._values = [row.copy() for row in history]
        self._history_hours = history.shape[0]

    @property
    def origin_index(self) -> int:
        """Absolute index of the first episode hour."""
        return self._history_hours

    @property
    def n_hours(self) -> int:
        return len(self._values)

    def append_hour(self, boardings: np.ndarray) -> None:
        """Record one completed hour of realised boardings."""
        boardings = np.asarray(boardings, dtype=float)
        if boardings.shape != (self.n_stations,):
            raise ValueError(f"expected ({self.n_stations},), got {boardings.shape}")
        self._values.append(boardings.copy())

    def window_ending_at(self, issue_hour: int, length: int) -> np.ndarray:
        """The ``length`` hours **strictly before** ``issue_hour``, oldest first.

        Returns shape ``(length, n_stations)``. Hours before the start of the buffer are
        zero-filled, matching :func:`src.data.features.to_hourly_grid`, which treats an absent
        station-hour as a recorded zero rather than as missing.

        Raises:
            IndexError: if ``issue_hour`` exceeds the number of hours realised so far. Asking
                for a window that ends in the future is the leakage failure mode this class
                exists to prevent, so it is an error rather than a silent truncation.
        """
        if issue_hour > self.n_hours:
            raise IndexError(
                f"issue_hour {issue_hour} is beyond the {self.n_hours} realised hours; "
                "the forecaster may not read hours at or after the issue time"
            )
        start = issue_hour - length
        rows: list[np.ndarray] = []
        for index in range(start, issue_hour):
            if index < 0:
                rows.append(np.zeros(self.n_stations))
            else:
                rows.append(self._values[index])
        return np.asarray(rows, dtype=float)

    def value_at(self, hour_index: int, issue_hour: int) -> np.ndarray:
        """A single past hour, with the same strictly-earlier guarantee."""
        if hour_index >= issue_hour:
            raise IndexError(
                f"hour {hour_index} is at or after issue hour {issue_hour}: this is future "
                "information and must not enter a forecast feature"
            )
        if hour_index < 0 or hour_index >= self.n_hours:
            return np.zeros(self.n_stations)
        return self._values[hour_index]


class ForecastProvider(Protocol):
    """What the environment needs from a forecaster."""

    n_stations: int

    def reset(self, buffer: HistoryBuffer, day_start_hour: int, is_weekend: bool) -> None:
        ...

    def forecast(self, issue_hour: int) -> Forecast:
        ...


class SeasonalNaiveProvider:
    """Seasonal-naive mean with an empirical same-hour-of-week dispersion.

    ``mu(s, h) = y(s, target_time - 168 h)``, matching the `naive_seasonal` reference floor of
    protocol section 4.3. Sigma is the standard deviation of the ``n_weeks`` most recent
    observations at the same hour of week, which is a genuine historical estimate rather than a
    parameter we chose -- important, because a provider whose sigma we invented would reintroduce
    exactly the circularity protocol section 5.1 removes.

    This provider requires no trained artefact, so the environment is fully testable before M4's
    full fit lands. It is **not** used for any reported experiment.
    """

    model_id = "naive_seasonal"

    def __init__(self, n_stations: int, n_weeks: int = 4, sigma_floor: float = 1.0) -> None:
        self.n_stations = n_stations
        self.n_weeks = n_weeks
        self.sigma_floor = sigma_floor
        self._buffer: HistoryBuffer | None = None

    def reset(self, buffer: HistoryBuffer, day_start_hour: int, is_weekend: bool) -> None:
        self._buffer = buffer

    def forecast(self, issue_hour: int) -> Forecast:
        if self._buffer is None:
            raise RuntimeError("reset() must be called before forecast()")
        buffer = self._buffer

        mu = np.zeros((self.n_stations, 3))
        sigma = np.zeros((self.n_stations, 3))
        for horizon in (1, 2, 3):
            target = issue_hour + horizon
            samples = []
            for week in range(1, self.n_weeks + 1):
                index = target - 168 * week
                # `value_at` enforces the strictly-earlier rule. A seasonal lag taken relative
                # to a target time up to 3 h ahead could in principle land at or after the
                # issue hour if the lag were too short; 168 h makes that impossible, and the
                # guard means a future change to the lag cannot silently introduce leakage.
                samples.append(buffer.value_at(index, issue_hour))
            stacked = np.asarray(samples)
            mu[:, horizon - 1] = stacked[0]
            spread = stacked.std(axis=0, ddof=1) if len(samples) > 1 else np.zeros(self.n_stations)
            sigma[:, horizon - 1] = np.maximum(spread, self.sigma_floor)

        return Forecast(mu=mu, sigma=sigma, issue_hour_index=issue_hour)


class TrainedForecastProvider:
    """Wraps the M4 `xgb_quantile` model and the frozen post-hoc calibration factor.

    Builds the frozen feature whitelist of protocol section 4.2 online from
    :class:`HistoryBuffer`, in the column order fixed by
    :attr:`src.data.features.FeatureSpec.feature_names`, so an episode-time feature vector is
    byte-comparable with the offline training frame. `tests/test_environment.py` asserts that
    correspondence rather than trusting it.

    sigma-hat is the frozen Gaussian-equivalent scale ``(q90 - q10) / (2 * z90)`` with the
    single scalar interval-width multiplier fitted on `val` applied unchanged (section 4.3).
    """

    model_id = "xgb_quantile"

    def __init__(
        self,
        model,
        station_index: np.ndarray,
        feature_names: list[str],
        calibration_factor: float = 1.0,
        quantile_levels: tuple[float, float, float] = (0.10, 0.50, 0.90),
    ) -> None:
        self.model = model
        self.station_index = np.asarray(station_index, dtype=float)
        self.n_stations = int(self.station_index.size)
        self.feature_names = list(feature_names)
        self.calibration_factor = float(calibration_factor)
        self.quantile_levels = quantile_levels
        self._buffer: HistoryBuffer | None = None
        self._day_start_hour = 0
        self._is_weekend = False

    def reset(self, buffer: HistoryBuffer, day_start_hour: int, is_weekend: bool) -> None:
        self._buffer = buffer
        self._day_start_hour = day_start_hour
        self._is_weekend = is_weekend

    # -- feature construction -------------------------------------------------------------

    def build_features(self, issue_hour: int, horizon: int) -> np.ndarray:
        """The frozen 19-column feature vector per station, as a ``(n_stations, 19)`` array.

        Mirrors :func:`src.data.features.build_supervised_frame` exactly: recent lags are
        offsets from the *issue* time, seasonal lags are offsets from the *target* time, and
        rolling windows end at the issue time inclusive.
        """
        from src.data.features import RECENT_LAGS, ROLLING_WINDOWS, SEASONAL_LAGS

        if self._buffer is None:
            raise RuntimeError("reset() must be called before build_features()")
        buffer = self._buffer
        target_hour = issue_hour + horizon

        columns: list[np.ndarray] = []

        # Recent lags: `y_tminus_0` is the hour ending at the issue time, i.e. the last fully
        # observed hour. Offsets therefore index `issue_hour - 1 - lag`.
        for lag in RECENT_LAGS:
            columns.append(buffer.value_at(issue_hour - 1 - lag, issue_hour))

        for lag in SEASONAL_LAGS:
            columns.append(buffer.value_at(target_hour - lag, issue_hour))

        longest = max(ROLLING_WINDOWS + (24,))
        window = buffer.window_ending_at(issue_hour, longest)
        for size in ROLLING_WINDOWS:
            columns.append(window[-size:].mean(axis=0))
        columns.append(window[-24:].std(axis=0, ddof=1))

        hour_of_day = (self._day_start_hour + (target_hour - buffer.origin_index)) % 24
        columns.append(np.full(self.n_stations, np.sin(2 * np.pi * hour_of_day / 24.0)))
        columns.append(np.full(self.n_stations, np.cos(2 * np.pi * hour_of_day / 24.0)))
        dow = 5.0 if self._is_weekend else 2.0
        columns.append(np.full(self.n_stations, dow))
        columns.append(np.full(self.n_stations, 1.0 if self._is_weekend else 0.0))
        columns.append(np.zeros(self.n_stations))  # is_holiday: episode days are non-holidays
        columns.append(self.station_index)

        return np.column_stack(columns)

    def forecast(self, issue_hour: int) -> Forecast:
        import pandas as pd

        mu = np.zeros((self.n_stations, 3))
        sigma = np.zeros((self.n_stations, 3))

        for horizon in (1, 2, 3):
            features = self.build_features(issue_hour, horizon)
            frame = pd.DataFrame(features, columns=self.feature_names)
            # The fitted model applies the frozen calibration factor and the frozen sigma
            # definition itself, so the in-episode path and the offline evaluation path go
            # through exactly the same code. Recomputing sigma here would let the two drift.
            output = self.model.predict(frame)
            mu[:, horizon - 1] = np.maximum(np.asarray(output.mean, dtype=float), 0.0)
            sigma[:, horizon - 1] = np.asarray(output.sigma, dtype=float)

        return Forecast(mu=mu, sigma=sigma, issue_hour_index=issue_hour)


#: Where `scripts/persist_forecast_models.py` writes the fitted forecasters.
FORECAST_MODEL_DIR = "models/forecasting"


def load_trained_provider(network, *, model_dir=None) -> TrainedForecastProvider:
    """Build a :class:`TrainedForecastProvider` from the persisted M4 artefacts.

    Args:
        network: The modelled line; fixes station order and count.
        model_dir: Override for the artefact directory.

    Raises:
        FileNotFoundError: if the models have not been persisted. `src/forecasting/train.py`
            evaluates and discards its fitted models, so `scripts/persist_forecast_models.py`
            refits them deterministically at the selected hyperparameters and writes them here.
    """
    import json
    from pathlib import Path

    from src.common.paths import MODELS_DIR
    from src.forecasting.models import XGBQuantileForecaster

    directory = Path(model_dir) if model_dir else MODELS_DIR / "forecasting"
    metadata_path = directory / "metadata.json"
    booster_path = directory / "xgb_quantile.json"
    if not metadata_path.exists() or not booster_path.exists():
        raise FileNotFoundError(
            f"no persisted forecasting model in {directory}. Run "
            "`python scripts/persist_forecast_models.py` after the M4 full fit completes."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = XGBQuantileForecaster(
        quantiles=tuple(metadata["quantiles"]), **metadata["hyperparameters"]
    )
    import xgboost as xgb

    booster = xgb.XGBRegressor()
    booster.load_model(str(booster_path))
    model._model = booster
    model._columns = list(metadata["feature_columns"])
    model.calibration_factor = float(metadata["calibration_factor"])

    order = metadata.get("station_index", {})
    station_index = np.array(
        [order.get(str(sid), position) for position, sid in enumerate(network.station_ids)],
        dtype=float,
    )

    logger.info(
        "loaded trained forecaster from %s (calibration factor %.4f, %d features)",
        directory, model.calibration_factor, len(model._columns),
    )
    return TrainedForecastProvider(
        model=model,
        station_index=station_index,
        feature_names=list(metadata["feature_columns"]),
        calibration_factor=float(metadata["calibration_factor"]),
        quantile_levels=(0.10, 0.50, 0.90),
    )
