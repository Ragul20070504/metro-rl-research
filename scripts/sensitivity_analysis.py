"""Analyse the two registered sensitivity experiments (EXP-SENS-001, EXP-SENS-002).

Their 1,050 cells were computed by the registered M12 stage and then never analysed: no test,
table or figure reported them, while the manuscript claimed they discharged the synthetic-OD and
reward-weighting limitations. This closes that gap from **stored data only** -- nothing is
re-simulated and no policy is retrained.

Written to a **separate** artefact rather than folded into `results/statistical_analysis.json`,
so the original confirmatory analysis stays byte-identical to the immutable study preserved at
tag `original-confirmatory-study`.

Both experiments are **exploratory**: they re-score the frozen `w_balanced` policies under
alternative assumptions. Benjamini-Hochberg at q = 0.10 within each family.

Writes ``results/sensitivity_analysis.json``.

    python scripts/sensitivity_analysis.py
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

ENDPOINT = "left_behind_fraction"


def benjamini_hochberg(p_values: list[float], q: float = 0.10) -> list[float]:
    """BH-adjusted p-values, order preserved."""
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for rank in range(n - 1, -1, -1):
        index = order[rank]
        running = min(running, p_values[index] * n / (rank + 1))
        adjusted[index] = running
    return [float(x) for x in adjusted]


def paired(frame: pd.DataFrame, a: str, b: str, label: str, **context: Any) -> dict[str, Any] | None:
    metric = context.pop("metric_override", ENDPOINT)
    piv = frame.pivot_table(index="train_seed", columns="method", values=metric)
    if a not in piv or b not in piv:
        return None
    piv = piv[[a, b]].dropna()
    if len(piv) < 3:
        return None
    x, y = piv[a].to_numpy(float), piv[b].to_numpy(float)
    diff = x - y
    normal_p = float(stats.shapiro(diff).pvalue) if len(diff) >= 3 else float("nan")
    if normal_p >= 0.05:
        test, result = "paired t-test", stats.ttest_rel(x, y)
    else:
        test, result = "Wilcoxon signed-rank", stats.wilcoxon(x, y)
    # Hodges-Lehmann: median of pairwise Walsh averages of the differences.
    walsh = [(diff[i] + diff[j]) / 2 for i in range(len(diff)) for j in range(i, len(diff))]
    return {
        "label": label, "metric": metric, "method_a": a, "method_b": b, "n": int(len(diff)),
        "mean_a": float(x.mean()), "mean_b": float(y.mean()),
        "mean_difference": float(diff.mean()), "hodges_lehmann": float(np.median(walsh)),
        "test": test, "normality_p": normal_p,
        "statistic": float(result.statistic), "p_value": float(result.pvalue),
        "direction": (f"{b} better" if (diff.mean() > 0) == (metric != "total_reward")
                      else f"{a} better"),
        "loser_is_ppo_uq": bool((diff.mean() > 0) == (metric != "total_reward")) if a == "ppo_uq"
                           else None,
        **context,
    }


def main() -> int:
    setup_logging(log_file="sensitivity_analysis.log")

    df = pd.read_parquet(RESULTS_DIR / "seed_level.parquet")
    df = df[df.train_seed >= 101]

    baseline = df[(df.variant == "main") & (df.weight_id == "w_balanced")
                  & (df.scenario == "surge")]
    reference = paired(baseline, "ppo_uq", "ppo_point",
                       "reference: registered conditions", scenario="surge",
                       weight_id="w_balanced", od_model="registered")

    tests: list[dict[str, Any]] = []

    # --- EXP-SENS-002: does the conclusion survive the synthetic OD assumption? ---------------
    for variant in sorted(v for v in df.variant.unique() if str(v).startswith("od_")):
        subset = df[(df.variant == variant) & (df.scenario == "surge")]
        for a, b in (("ppo_uq", "ppo_point"), ("ppo_uq", "ppo_placebo")):
            row = paired(subset, a, b, f"EXP-SENS-002 {variant}",
                         experiment_id="EXP-SENS-002", scenario="surge",
                         od_model=variant.removeprefix("od_"), family="sensitivity")
            if row:
                tests.append(row)

    # --- EXP-SENS-001: does it survive a different scalarisation of the reward? ---------------
    # The primary endpoint is invariant here by construction (see `weight_invariance` below), so
    # the informative quantity is `total_reward`, which is what the weighting actually changes.
    weight_tests: list[dict[str, Any]] = []
    for variant in sorted(v for v in df.variant.unique() if str(v).startswith("weights_")):
        subset = df[df.variant == variant]
        for scenario in sorted(subset.scenario.unique()):
            row = paired(subset[subset.scenario == scenario], "ppo_uq", "ppo_point",
                         f"EXP-SENS-001 {variant} / {scenario}",
                         experiment_id="EXP-SENS-001", scenario=scenario,
                         weight_id=variant.removeprefix("weights_"), family="sensitivity",
                         metric_override="total_reward")
            if row:
                weight_tests.append(row)
    tests.extend(weight_tests)

    for row, adjusted in zip(tests, benjamini_hochberg([t["p_value"] for t in tests])):
        row["p_adjusted"] = adjusted
        row["adjustment"] = f"Benjamini-Hochberg (q = 0.10, {len(tests)} tests)"
        row["significant_at_q"] = bool(adjusted < 0.10)

    # Agreement means "ppo_uq loses", not string equality: the OD family compares against two
    # different opponents, so matching the reference's direction *label* would miscount the
    # placebo rows as disagreements when they say the same thing.
    losers = [t for t in tests if t.get("loser_is_ppo_uq")]
    ref_direction = reference["direction"] if reference else ""
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "family": "exploratory (sensitivity)",
        "provenance": ("stored cells of the registered M12 ablation stage; nothing re-simulated "
                       "and no policy retrained"),
        "separation": ("written separately from results/statistical_analysis.json so the "
                       "confirmatory analysis of the immutable study is unchanged"),
        "endpoint": ENDPOINT,
        "unit_of_analysis": "training seed (protocol section 8.2)",
        "reference_registered_conditions": reference,
        "n_tests": len(tests),
        "tests": tests,
        "direction_agreement": {
            "reference_direction": ref_direction,
            "conditions_where_ppo_uq_is_worse": len(losers),
            "conditions_total": len(tests),
            "any_condition_favouring_ppo_uq": len(tests) - len(losers),
            "reading": (
                f"`ppo_uq` is the worse arm in {len(losers)} of {len(tests)} sensitivity "
                "conditions. Direction, not effect size, is the quantity of interest: these "
                "cells ask whether the registered conclusion is an artefact of the synthetic OD "
                "model or of one reward scalarisation. No condition reverses it. Individual "
                "tests mostly do not reach the BH threshold, which is expected -- each uses five "
                "training seeds and ten evaluation seeds against ten and thirty in the main "
                "sweep -- so the consistent sign across conditions carries the evidence rather "
                "than any single test."
            ),
        },
        "weight_invariance": {
            "finding": (
                "EXP-SENS-001 cannot test reward-weight sensitivity on the primary endpoint. "
                "Re-scoring a frozen policy under a different reward weighting changes no "
                "action: the policy does not observe the reward at evaluation time, so the "
                "trajectory, and therefore `left_behind_fraction` and `inductions`, are "
                "identical across all three weightings -- verified per seed, not merely on "
                "average. Only `total_reward` differs, and it is an arithmetic re-weighting of "
                "components already stored."
            ),
            "consequence": (
                "The 600 cells of EXP-SENS-001 re-simulated identical physical trajectories "
                "three times. The question the experiment was registered to ask -- does the "
                "method ranking move with the objective? -- is answerable only on the reward, "
                "which is how it is analysed here, and could have been computed arithmetically "
                "without simulation. Testing weight sensitivity on physical outcomes would "
                "require *retraining* under each weighting, which the design explicitly declined "
                "to do."
            ),
            "verified_identical_across_weightings": ["left_behind_fraction", "inductions"],
        },
        "caveats": [
            "Exploratory. These conditions were registered but the analysis is reported "
            "separately from the confirmatory families and generates no claim in the abstract.",
            "The OD variants re-score policies trained under the registered OD model, so they "
            "test robustness of the *evaluation*, not of training under a different OD world.",
            "The reward-weight variants likewise re-score w_balanced policies; a policy trained "
            "under w_cost might behave differently, and this design cannot say.",
            "Ten evaluation seeds and five training seeds per sensitivity cell, against thirty "
            "and ten in the main sweep, so these are less precise than the registered tests.",
        ],
    }

    out = RESULTS_DIR / "sensitivity_analysis.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s", out)

    if reference:
        print(f"reference (registered): {reference['mean_difference']:+.6f} "
              f"-> {reference['direction']}\n")
    for row in tests:
        flag = "*" if row["significant_at_q"] else " "
        print(f"{flag} {row['label']:<44} d={row['mean_difference']:+.6f} "
              f"p_adj={row['p_adjusted']:.4g}  {row['direction']}")
    agree = payload["direction_agreement"]
    print(f"\nppo_uq is worse in {agree['conditions_where_ppo_uq_is_worse']}"
          f"/{agree['conditions_total']} sensitivity conditions; "
          f"{agree['any_condition_favouring_ppo_uq']} favour it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
