"""Mechanism analysis for the uncertainty-aware arm's underperformance (EXP-UQ-004 / H1b).

Uses **only stored artefacts** from the original confirmatory study: no episode is re-simulated
and no policy is retrained. Every quantity is aggregated from `results/runs/**/epochs.parquet`
and `results/seed_level.parquet`, both written by the registered M11 sweep.

What the design licenses. The three arms differ **only** in the content of a two-value channel --
identically zero (`ppo_point`), shuffled with the same marginal distribution (`ppo_placebo`), or
the calibrated sigma (`ppo_uq`) -- with architecture, parameter count, reward, seeds, training
days and evaluation days held fixed, and common random numbers enforced structurally. Channel
content is therefore a manipulated variable, and a difference in behaviour between arms is
attributable to it. What the design does **not** license is any claim about the internal
computation that produces the behaviour: we observe that informative sigma changes dispatch, not
how the network represents it.

Writes ``results/mechanism_analysis.json``.

    python scripts/mechanism_analysis.py
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.common.logging import get_logger, setup_logging
from src.common.paths import RESULTS_DIR

logger = get_logger(__name__)

RUNS = RESULTS_DIR / "runs" / "main"
SCENARIOS = ("normal", "surge", "disruption", "combined")
ARMS = ("ppo_point", "ppo_placebo", "ppo_uq")
TRAIN_SEEDS = tuple(range(101, 111))
EVAL_SEEDS = tuple(range(1, 31))

#: `ppo_uq` and `fixed_timetable` dispatch almost the same number of trains, which makes them the
#: one pair in the study whose service outcomes can be compared at matched dispatch volume.
VOLUME_MATCHED = ("ppo_uq", "fixed_timetable")


def _epoch_frames(method: str, scenario: str, train_seeds) -> pd.DataFrame:
    frames = []
    for train_seed in train_seeds:
        for eval_seed in EVAL_SEEDS:
            path = RUNS / scenario / method / "w_balanced" / str(train_seed) / str(eval_seed)
            path = path / "epochs.parquet"
            if not path.is_file():
                continue
            frame = pd.read_parquet(path)
            frame["train_seed"], frame["eval_seed"] = train_seed, eval_seed
            frames.append(frame)
    if not frames:
        raise SystemExit(f"no stored epoch data for {method}/{scenario}")
    return pd.concat(frames, ignore_index=True)


def _episode_summary(epochs: pd.DataFrame) -> pd.DataFrame:
    """One row per (train_seed, eval_seed): how the day's dispatching was shaped."""
    grouped = epochs.groupby(["train_seed", "eval_seed"])
    return pd.DataFrame({
        "inductions": grouped["inductions"].sum(),
        # A day's service gaps: epochs where the policy released nothing at all.
        "idle_epochs": grouped["inductions"].apply(lambda s: int((s == 0).sum())),
        # Burstiness: the same daily total can be spread evenly or clustered.
        "dispatch_sd": grouped["inductions"].std(),
        "max_queue": grouped["max_queue"].max(),
        "left_behind": grouped["left_behind"].sum(),
        "headway_violations": grouped["headway_violations"].sum(),
    }).reset_index()


def _seed_means(summary: pd.DataFrame) -> pd.DataFrame:
    """Collapse to the unit of analysis (protocol section 8.2) before any test."""
    return summary.groupby("train_seed").mean(numeric_only=True).drop(columns=["eval_seed"])


def _paired(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str) -> dict[str, Any]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    result = stats.ttest_rel(a, b)
    return {
        "method_a": label_a, "method_b": label_b, "n": int(a.size),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_difference": float(diff.mean()), "sd_difference": float(diff.std(ddof=1)),
        "t": float(result.statistic), "p_value": float(result.pvalue),
        "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue) if not np.allclose(diff, 0) else 1.0,
        "seeds_favouring_a": int((diff < 0).sum()),
    }


def main() -> int:
    setup_logging(log_file="mechanism_analysis.log")

    seed_level = pd.read_parquet(RESULTS_DIR / "seed_level.parquet")
    seed_level = seed_level[(seed_level.variant == "main")
                            & (seed_level.weight_id == "w_balanced")]

    payload: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "EXP-UQ-004",
        "hypothesis": "H1b (mechanism)",
        "family": "exploratory",
        "provenance": "stored artefacts of the original confirmatory study only; nothing re-run",
        "unit_of_analysis": "training seed (protocol section 8.2)",
        "what_the_design_licenses": (
            "Channel content is manipulated with everything else held fixed under common random "
            "numbers, so behavioural differences between arms are attributable to it. No claim "
            "is made about the network's internal representation, and none of this is evidence "
            "for or against H1, which is settled by the confirmatory test."
        ),
    }

    # --- 1. Does the channel change dispatch volume, and is it the channel or its content? ----
    volume: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        piv = seed_level[(seed_level.scenario == scenario)
                         & (seed_level.train_seed >= 101)].pivot_table(
            index="train_seed", columns="method", values="inductions")
        volume.append({
            "scenario": scenario,
            "informative_sigma_vs_zero": _paired(piv["ppo_uq"], piv["ppo_point"],
                                                 "ppo_uq", "ppo_point"),
            "uninformative_sigma_vs_zero": _paired(piv["ppo_placebo"], piv["ppo_point"],
                                                   "ppo_placebo", "ppo_point"),
        })
    payload["dispatch_volume"] = volume
    payload["dispatch_volume_reading"] = (
        "The placebo arm carries a sigma channel of identical marginal distribution with the "
        "association to forecast error destroyed. If merely having the channel changed "
        "behaviour, the placebo would move with the uncertainty arm. It does not: the placebo "
        "dispatches like the point arm, while only the calibrated sigma produces a large "
        "reduction. The dispatch collapse is attributable to the sigma being informative, not to "
        "the extra inputs."
    )

    # --- 2. At matched dispatch volume, is the timing worse too? ------------------------------
    matched: dict[str, Any] = {}
    for scenario in ("surge",):
        uq = _seed_means(_episode_summary(_epoch_frames("ppo_uq", scenario, TRAIN_SEEDS)))
        ft = _episode_summary(_epoch_frames("fixed_timetable", scenario, ["na"]))
        ft_mean = ft.mean(numeric_only=True)
        matched[scenario] = {
            "note": (
                "`fixed_timetable` is deterministic and has no training seed, so this is a "
                "descriptive comparison, not a paired test. It is included because it is the "
                "only method in the study that dispatches essentially the same number of trains "
                "as `ppo_uq`, which lets timing be compared with volume held roughly constant."
            ),
            "ppo_uq": {k: float(v) for k, v in uq.mean().items()},
            "fixed_timetable": {k: float(ft_mean[k]) for k in uq.columns if k in ft_mean},
        }
    payload["volume_matched_comparison"] = matched
    payload["volume_matched_reading"] = (
        "At almost identical dispatch volume, the uncertainty arm leaves markedly more epochs "
        "with no dispatch at all and reaches a larger peak queue than a fixed timetable. So the "
        "arm is not only releasing fewer trains: it is also clustering the releases it does "
        "make, and a non-adaptive schedule running the same number of trains allocates them "
        "better. Under-dispatching and mis-timing are both present."
    )

    # --- 3. Can dispatch volume alone account for the service gap? ----------------------------
    surge = seed_level[(seed_level.scenario == "surge") & (seed_level.train_seed >= 101)]
    rl = surge[surge.method.isin(ARMS)]
    slope, intercept, r, p, se = stats.linregress(rl["inductions"], rl["left_behind_fraction"])
    payload["volume_versus_outcome"] = {
        "pooled_regression": {
            "slope_per_induction": float(slope), "r_squared": float(r ** 2),
            "p_value": float(p), "n": int(len(rl)),
        },
        "caution": (
            "Arm and dispatch volume are near-collinear by construction -- the arm is what "
            "changes the volume -- so a regression cannot apportion the outcome gap between "
            "'fewer trains' and 'worse timing'. The volume-matched comparison above is the "
            "identification that is actually available, and it indicates both are present."
        ),
        "not_monotone_across_methods": (
            seed_level[seed_level.scenario == "surge"]
            .groupby("method")[["inductions", "left_behind_fraction"]].mean()
            .round(6).sort_values("inductions").to_dict("index")
        ),
        "not_monotone_reading": (
            "More dispatch is not simply better: the tuned heuristic achieves the best service "
            "in the study while dispatching fewer trains than `ppo_point`, `milp_static` or "
            "`milp_rolling`. Timing dominates volume across methods, which is why the "
            "uncertainty arm's clustering matters as much as its restraint."
        ),
    }

    # --- 4. What the stored artefacts cannot answer -------------------------------------------
    payload["open_questions"] = [
        {
            "question": ("The registered form of H1b -- does dispatch rate rise with forecast "
                         "dispersion after conditioning on the forecast mean? -- is an "
                         "epoch-level question."),
            "status": "NOT ANSWERABLE FROM STORED ARTEFACTS",
            "reason": ("`epochs.parquet` records actions and outcomes but not the forecast mean "
                       "or sigma the policy saw. Answering it needs the per-epoch forecast "
                       "re-derived from the frozen forecaster, which is deterministic but is "
                       "computation this analysis deliberately does not run."),
        },
        {
            "question": "Is the dispatch collapse a property of the information or of the budget?",
            "status": "NOT ANSWERABLE FROM THIS STUDY",
            "reason": ("One frozen configuration at one budget (amendment A-05). No learning "
                       "curves were stored, so convergence cannot be compared across arms from "
                       "the artefacts either. This is the question the registered follow-up "
                       "exists to answer."),
        },
    ]

    out = RESULTS_DIR / "mechanism_analysis.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s", out)

    for row in volume:
        info = row["informative_sigma_vs_zero"]
        plac = row["uninformative_sigma_vs_zero"]
        print(f"{row['scenario']:<11} informative sigma {info['mean_difference']:+6.1f} "
              f"(p={info['p_value']:.1e})   uninformative {plac['mean_difference']:+5.1f} "
              f"(p={plac['p_value']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
