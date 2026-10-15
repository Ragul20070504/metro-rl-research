"""Demand-regime screening and chronological splitting (protocol section 3.1, decision D-11).

Two responsibilities, both pure functions over a daily demand series so they can be tested
without touching the network or the filesystem.

**Regime screen.** The modelled demand series spans a structural break (the pandemic and the
recovery that followed). Training a forecaster across that break inflates its error, and
therefore its predictive dispersion, for reasons that have nothing to do with short-horizon
demand uncertainty. Because that dispersion is exactly what feeds the uncertainty channel
being tested by H1, the study window is placed **wholly inside one regime** rather than
straddling the break. The screen finds the earliest date from which year-on-year growth of
the 28-day mean stays within tolerance for a sustained run.

**Chronological split.** Train / validation / test are contiguous blocks in time, separated by
embargo gaps that are discarded. Random splitting is never used: with lag features spanning up
to 168 hours, a shuffled split would place a row's own recent history on the other side of the
partition, which is textbook leakage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RegimeScreenResult:
    """Outcome of the demand-regime screen."""

    start_date: pd.Timestamp
    used_fallback: bool
    stable_run_days: int
    max_abs_yoy_in_run: float
    detail: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["start_date"] = str(self.start_date.date())
        return payload


@dataclass(frozen=True)
class SplitBoundaries:
    """Chronological split boundaries, inclusive of both endpoints."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_days: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "train_start": str(self.train_start.date()), "train_end": str(self.train_end.date()),
            "val_start": str(self.val_start.date()), "val_end": str(self.val_end.date()),
            "test_start": str(self.test_start.date()), "test_end": str(self.test_end.date()),
            "embargo_days": self.embargo_days,
        }

    def label_for(self, timestamps: pd.Series) -> pd.Series:
        """Label each timestamp ``train`` / ``val`` / ``test`` / ``embargo`` / ``outside``."""
        dates = pd.to_datetime(timestamps).dt.normalize()
        labels = pd.Series("outside", index=dates.index, dtype=object)
        labels[(dates >= self.train_start) & (dates <= self.train_end)] = "train"
        labels[(dates >= self.val_start) & (dates <= self.val_end)] = "val"
        labels[(dates >= self.test_start) & (dates <= self.test_end)] = "test"
        labels[(dates > self.train_end) & (dates < self.val_start)] = "embargo"
        labels[(dates > self.val_end) & (dates < self.test_start)] = "embargo"
        return labels

    def validate(self) -> None:
        """Assert ordering and that both embargo gaps are non-empty."""
        if not self.train_start <= self.train_end < self.val_start <= self.val_end < self.test_start <= self.test_end:
            raise ValueError(f"split boundaries are not strictly ordered: {self.to_dict()}")
        gap_one = (self.val_start - self.train_end).days - 1
        gap_two = (self.test_start - self.val_end).days - 1
        if gap_one < self.embargo_days or gap_two < self.embargo_days:
            raise ValueError(
                f"embargo gaps too small: train->val {gap_one}d, val->test {gap_two}d, "
                f"required {self.embargo_days}d"
            )


def daily_line_demand(demand: pd.DataFrame) -> pd.Series:
    """Aggregate station-hour demand to a line-level daily series indexed by date."""
    daily = (
        demand.assign(date=pd.to_datetime(demand["transit_timestamp"]).dt.normalize())
        .groupby("date", observed=True)["ridership"]
        .sum()
        .sort_index()
    )
    return daily.asfreq("D")


def screen_demand_regime(
    daily: pd.Series,
    *,
    short_window_days: int,
    yoy_lag_days: int,
    max_abs_yoy_growth: float,
    min_stable_run_days: int,
    fallback_start: str,
) -> RegimeScreenResult:
    """Find the earliest date from which the demand regime is stable.

    The statistic is the year-on-year growth of the rolling short-window mean::

        g(d) = mean(D[d-w+1 .. d]) / mean(D[d-365-w+1 .. d-365]) - 1

    A regime is 'stable from d0' when ``|g(d)| <= tolerance`` for every day in a run of at
    least ``min_stable_run_days`` starting at ``d0``. Using a *sustained run* rather than a
    single crossing prevents a transient dip through the tolerance band from being mistaken
    for recovery.

    If no qualifying run exists, the configured fallback date is returned with
    ``used_fallback=True``. That is reported as a finding, never silently.
    """
    smoothed = daily.rolling(short_window_days, min_periods=short_window_days).mean()
    previous = smoothed.shift(yoy_lag_days)
    with np.errstate(divide="ignore", invalid="ignore"):
        growth = (smoothed / previous) - 1.0
    growth = growth.replace([np.inf, -np.inf], np.nan)

    within = (growth.abs() <= max_abs_yoy_growth) & growth.notna()

    # Longest-run scan: for each qualifying start, how far does the run extend?
    best_start: pd.Timestamp | None = None
    run_length = 0
    current_start: pd.Timestamp | None = None
    current_length = 0
    for date, ok in within.items():
        if ok:
            if current_start is None:
                current_start = date
                current_length = 0
            current_length += 1
            if current_length >= min_stable_run_days and best_start is None:
                best_start = current_start
                run_length = current_length
            elif best_start is not None and current_start == best_start:
                run_length = current_length
        else:
            current_start = None
            current_length = 0

    if best_start is None:
        fallback = pd.Timestamp(fallback_start)
        longest = int(within.astype(int).groupby((~within).cumsum()).sum().max() or 0)
        logger.warning(
            "regime screen found no stable run of >= %d days (longest was %d); using fallback %s",
            min_stable_run_days, longest, fallback.date(),
        )
        return RegimeScreenResult(
            start_date=fallback,
            used_fallback=True,
            stable_run_days=longest,
            max_abs_yoy_in_run=float("nan"),
            detail=(
                f"no run of >= {min_stable_run_days} days with |YoY growth| <= "
                f"{max_abs_yoy_growth}; longest qualifying run was {longest} days. "
                f"Fell back to the configured start {fallback.date()}. This is a finding, not a "
                "silent default: conclusions inherit whatever regime instability remains."
            ),
        )

    run_slice = growth.loc[best_start:].head(run_length)
    max_abs = float(run_slice.abs().max())
    logger.info("regime screen: stable from %s (run %d days, max |YoY| %.3f)",
                best_start.date(), run_length, max_abs)
    return RegimeScreenResult(
        start_date=best_start,
        used_fallback=False,
        stable_run_days=int(run_length),
        max_abs_yoy_in_run=max_abs,
        detail=(
            f"stable from {best_start.date()}: |YoY growth of the {short_window_days}-day mean| "
            f"stayed <= {max_abs_yoy_growth} (observed max {max_abs:.3f}) for {run_length} "
            "consecutive days"
        ),
    )


def regime_diagnostics(
    daily: pd.Series,
    *,
    short_window_days: int,
    yoy_lag_days: int,
    study_start: pd.Timestamp,
    tolerances: tuple[float, ...] = (0.10, 0.12, 0.15, 0.20, 0.25),
) -> dict[str, Any]:
    """Characterise residual regime instability in the chosen study window.

    Reported whether or not the screen succeeded, and especially when it falls back. The point
    is to say *how far* from stationary the study window actually is, rather than leaving a
    bare "fallback used" flag that a reader cannot weigh.

    The ``longest_run_by_tolerance`` table shows what tolerance *would* have admitted a full
    year. It is a diagnostic, not a licence to retune: the pre-registered tolerance is fixed,
    and reading this table after the fact and loosening the rule would be precisely the
    post-hoc adjustment the protocol forbids.
    """
    smoothed = daily.rolling(short_window_days, min_periods=short_window_days).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        growth = ((smoothed / smoothed.shift(yoy_lag_days)) - 1.0).replace([np.inf, -np.inf], np.nan)

    by_year = {
        int(year): {
            "median_yoy": float(block.median()),
            "p10_yoy": float(block.quantile(0.10)),
            "p90_yoy": float(block.quantile(0.90)),
            "fraction_within_10pct": float((block.abs() <= 0.10).mean()),
        }
        for year, block in growth.dropna().groupby(growth.dropna().index.year)
    }

    in_window = growth[growth.index >= study_start].dropna()
    longest_run: dict[str, int] = {}
    for tolerance in tolerances:
        qualifying = (growth.abs() <= tolerance) & growth.notna()
        runs = qualifying.astype(int).groupby((~qualifying).cumsum()).sum()
        longest_run[f"{tolerance:.2f}"] = int(runs.max()) if len(runs) else 0

    return {
        "by_year": by_year,
        "study_window": {
            "median_yoy": float(in_window.median()) if len(in_window) else float("nan"),
            "max_abs_yoy": float(in_window.abs().max()) if len(in_window) else float("nan"),
            "fraction_within_10pct": float((in_window.abs() <= 0.10).mean()) if len(in_window) else float("nan"),
            "fraction_within_15pct": float((in_window.abs() <= 0.15).mean()) if len(in_window) else float("nan"),
            "fraction_within_20pct": float((in_window.abs() <= 0.20).mean()) if len(in_window) else float("nan"),
        },
        "longest_run_by_tolerance": longest_run,
        "mean_daily_by_year": {
            int(year): float(block.mean())
            for year, block in daily.dropna().groupby(daily.dropna().index.year)
        },
    }


def _snap_forward_to_weekday(date: pd.Timestamp, weekday: int) -> pd.Timestamp:
    """Move ``date`` forward to the next occurrence of ``weekday`` (Monday = 0)."""
    offset = (weekday - date.dayofweek) % 7
    return date + pd.Timedelta(days=offset)


def make_chronological_split(
    first_date: pd.Timestamp,
    last_date: pd.Timestamp,
    *,
    train_fraction: float,
    val_fraction: float,
    embargo_days: int,
    align_to_weekday: int | None = 0,
) -> SplitBoundaries:
    """Build contiguous train / val / test blocks with embargo gaps.

    Boundaries are snapped forward to a fixed weekday so that weekly seasonality -- the
    strongest cycle in transit demand -- is never cut mid-week, which would otherwise give the
    splits systematically different day-of-week compositions.
    """
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1 or train_fraction + val_fraction >= 1:
        raise ValueError(f"invalid split fractions: train={train_fraction}, val={val_fraction}")

    total_days = (last_date - first_date).days + 1
    usable = total_days - 2 * embargo_days
    if usable <= 0:
        raise ValueError(f"window of {total_days} days is too short for {embargo_days}-day embargoes")

    train_end = first_date + pd.Timedelta(days=int(usable * train_fraction) - 1)
    val_start = train_end + pd.Timedelta(days=embargo_days + 1)
    val_end = val_start + pd.Timedelta(days=int(usable * val_fraction) - 1)
    test_start = val_end + pd.Timedelta(days=embargo_days + 1)

    if align_to_weekday is not None:
        val_start = _snap_forward_to_weekday(val_start, align_to_weekday)
        train_end = val_start - pd.Timedelta(days=embargo_days + 1)
        test_start = _snap_forward_to_weekday(test_start, align_to_weekday)
        val_end = test_start - pd.Timedelta(days=embargo_days + 1)

    boundaries = SplitBoundaries(
        train_start=first_date, train_end=train_end,
        val_start=val_start, val_end=val_end,
        test_start=test_start, test_end=last_date,
        embargo_days=embargo_days,
    )
    boundaries.validate()
    logger.info("split: train %s..%s | val %s..%s | test %s..%s",
                boundaries.train_start.date(), boundaries.train_end.date(),
                boundaries.val_start.date(), boundaries.val_end.date(),
                boundaries.test_start.date(), boundaries.test_end.date())
    return boundaries


def population_stability_index(reference: np.ndarray, comparison: np.ndarray, bins: int = 10) -> float:
    """PSI between two samples, using quantile bins of the reference.

    Reported as a distribution-shift diagnostic between splits (protocol section 3.2 check 9).
    It is *reported, not gated*: a chronological split of a real series is expected to show
    some shift, and gating on it would amount to selecting the test set for convenience.
    """
    reference = reference[np.isfinite(reference)]
    comparison = comparison[np.isfinite(comparison)]
    if reference.size == 0 or comparison.size == 0:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    reference_share = np.histogram(reference, bins=edges)[0] / reference.size
    comparison_share = np.histogram(comparison, bins=edges)[0] / comparison.size
    epsilon = 1e-6
    reference_share = np.clip(reference_share, epsilon, None)
    comparison_share = np.clip(comparison_share, epsilon, None)
    return float(np.sum((comparison_share - reference_share) * np.log(comparison_share / reference_share)))
