"""Load stored run cells and reduce them to the unit of analysis (M14).

Protocol section 8.2: *"The 30 episodes from one trained policy are not 30 independent
observations of the method -- they share a policy."* Everything downstream of this module works
on **training-seed means over CRN-matched evaluation episodes**, and the episode-level frame is
kept only for distributional descriptions such as Cliff's delta.

Methods with no training seed -- the fixed timetable, the heuristic and both MILP variants -- are
deterministic given the environment, so their 30 evaluation episodes *are* the sample. They are
represented as a single pseudo-seed, and any comparison against a trained method is unpaired at
the seed level; :func:`paired_frame` refuses to fabricate a pairing where none exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.evaluation.runner import Cell, load_cell_metrics

logger = get_logger(__name__)

__all__ = ["load_frame", "seed_level", "paired_frame", "METRICS_OF_INTEREST"]

#: Reported for every (method, scenario). The first is the pre-registered primary endpoint.
METRICS_OF_INTEREST: tuple[str, ...] = (
    "left_behind_fraction",
    "waiting_accrued_passenger_hours",
    "waiting_time_mean_s",
    "waiting_time_p95_s",
    "energy_units",
    "cost_units",
    "unmet_demand_fraction",
    "service_level",
    "denial_rate_per_passenger",
    "inductions",
    "headway_violations",
    "total_reward",
)

#: Metrics where a larger value is better. Everything else is lower-is-better.
HIGHER_IS_BETTER = frozenset({"service_level", "total_reward"})


def load_frame(cells: Iterable[Cell]) -> pd.DataFrame:
    """Episode-level frame of every cell that has a stored result.

    Cells with no metrics are skipped and counted; a sweep that silently dropped its failures
    would report a biased subset.
    """
    rows: list[dict[str, Any]] = []
    missing = 0
    for cell in cells:
        metrics = load_cell_metrics(cell)
        if metrics is None:
            missing += 1
            continue
        row = {
            "method": cell.method,
            "scenario": cell.scenario,
            "train_seed": -1 if cell.train_seed is None else cell.train_seed,
            "eval_seed": cell.eval_seed,
            "weight_id": cell.weight_id,
            "variant": cell.variant or "main",
            "experiment_id": cell.experiment_id,
        }
        for key in METRICS_OF_INTEREST:
            row[key] = float(metrics.get(key, np.nan))
        for extra in ("mask_clip_rate", "induction_success_rate",
                      "inference_latency_mean_ms", "inference_latency_p95_ms"):
            row[extra] = float(metrics.get(extra, np.nan))
        rows.append(row)

    if missing:
        logger.warning("%d cells had no stored metrics and were skipped", missing)
    frame = pd.DataFrame(rows)
    logger.info("loaded %d episode rows across %d methods",
                len(frame), frame["method"].nunique() if len(frame) else 0)
    return frame


def seed_level(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per (method, scenario, weight, variant, training seed).

    This is the unit of analysis. Reporting the episode-level ``n`` as the sample size is
    prohibited by protocol section 8.2, and this function is how that prohibition is enforced in
    practice: nothing downstream ever sees the episode-level frame for inference.
    """
    if frame.empty:
        return frame
    keys = ["method", "scenario", "weight_id", "variant", "train_seed"]
    aggregated = frame.groupby(keys, observed=True).agg(
        {**{m: "mean" for m in METRICS_OF_INTEREST},
         "mask_clip_rate": "mean", "induction_success_rate": "mean",
         "inference_latency_mean_ms": "mean", "inference_latency_p95_ms": "mean",
         "eval_seed": "count"}
    ).rename(columns={"eval_seed": "n_episodes"}).reset_index()
    return aggregated


@dataclass
class PairedFrame:
    """Two methods' seed-level values, aligned by training seed."""

    method_a: str
    method_b: str
    train_seeds: list[int]
    values_a: np.ndarray
    values_b: np.ndarray
    episodes_a: np.ndarray
    episodes_b: np.ndarray
    paired: bool


def paired_frame(
    seeds: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    method_a: str,
    method_b: str,
    scenario: str,
    metric: str,
    weight_id: str = "w_balanced",
    variant: str = "main",
) -> PairedFrame | None:
    """Align two methods for a paired comparison, or report honestly that they cannot be paired.

    Returns ``None`` when either method has no data. Sets ``paired=False`` when the two do not
    share training seeds -- which is the case for every comparison between a trained policy and a
    deterministic baseline, since the baseline has no training seed at all. A paired test on
    fabricated pairs would be a silent error, so the flag travels with the data.
    """
    def select(method: str) -> pd.DataFrame:
        return seeds[
            (seeds["method"] == method) & (seeds["scenario"] == scenario)
            & (seeds["weight_id"] == weight_id) & (seeds["variant"] == variant)
        ].sort_values("train_seed")

    a, b = select(method_a), select(method_b)
    if a.empty or b.empty:
        return None

    shared = sorted(set(a["train_seed"]) & set(b["train_seed"]))
    is_paired = len(shared) >= 3 and -1 not in shared

    if is_paired:
        a = a[a["train_seed"].isin(shared)].sort_values("train_seed")
        b = b[b["train_seed"].isin(shared)].sort_values("train_seed")
        train_seeds = shared
    else:
        train_seeds = []

    def episode_values(method: str) -> np.ndarray:
        subset = episodes[
            (episodes["method"] == method) & (episodes["scenario"] == scenario)
            & (episodes["weight_id"] == weight_id) & (episodes["variant"] == variant)
        ]
        return subset[metric].to_numpy(dtype=float)

    return PairedFrame(
        method_a=method_a, method_b=method_b, train_seeds=train_seeds,
        values_a=a[metric].to_numpy(dtype=float), values_b=b[metric].to_numpy(dtype=float),
        episodes_a=episode_values(method_a), episodes_b=episode_values(method_b),
        paired=is_paired,
    )
