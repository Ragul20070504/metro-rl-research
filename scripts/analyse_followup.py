"""Analyse FOLLOWUP-UQ-TRAINING exactly as its protocol pre-specifies.

**Written before the follow-up produced any result.** The statistical plan is
`docs/followup_training_budget_protocol.md` section 4; this file implements it and nothing else.
Committing the analysis before the data exist is what makes "pre-specified" checkable rather than
asserted.

Reads the registered 60,000-step point from the original study's `variant == "main"` cells --
read-only -- and every other budget from `variant == "budget_<N>"`. Writes
`results/followup_analysis.json`. The confirmatory artefact is never regenerated.

    python scripts/analyse_followup.py
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
from src.evaluation.aggregate import load_frame, seed_level
from src.evaluation.runner import discover_cells
from src.evaluation.statistics import bca_bootstrap_ci, benjamini_hochberg

from scripts.run_followup import ARMS, BUDGETS, REGISTERED_BUDGET, TRAIN_SEEDS

logger = get_logger(__name__)

ENDPOINT = "left_behind_fraction"
PRIMARY_SCENARIO = "surge"
SCENARIOS = ("normal", "surge", "disruption", "combined")


def budget_coverage(episodes: pd.DataFrame) -> dict[int, dict[str, Any]]:
    """Require every registered episode, not merely three surviving seed means."""
    expected = {(arm, seed, scenario, evaluation)
                for arm in ARMS for seed in TRAIN_SEEDS
                for scenario in SCENARIOS for evaluation in range(1, 31)}
    coverage = {}
    for budget in BUDGETS:
        frame = _budget_frame(episodes, budget)
        keys = list(frame[["method", "train_seed", "scenario", "eval_seed"]]
                    .itertuples(index=False, name=None))
        actual = set(keys)
        finite = bool(np.isfinite(frame[ENDPOINT].to_numpy(float)).all())
        coverage[budget] = {
            "expected": len(expected), "present": len(actual & expected),
            "missing": len(expected - actual), "unexpected": len(actual - expected),
            "duplicates": len(keys) - len(actual), "finite": finite,
            "complete": actual == expected and len(keys) == len(actual) and finite,
        }
    return coverage


def _budget_frame(df: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Cells for one budget. The registered point comes from `main`; the rest from `budget_<N>`."""
    variant = "main" if budget == REGISTERED_BUDGET else f"budget_{budget}"
    return df[(df.variant == variant) & (df.weight_id == "w_balanced")
              & (df.method.isin(ARMS)) & (df.train_seed.isin(TRAIN_SEEDS))]


def _paired(frame: pd.DataFrame, scenario: str) -> pd.DataFrame | None:
    piv = frame[frame.scenario == scenario].pivot_table(
        index="train_seed", columns="method", values=ENDPOINT)
    if not set(ARMS) <= set(piv.columns):
        return None
    piv = piv[list(ARMS)].dropna()
    return piv if len(piv) >= 3 else None


def _holm(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (n - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _bca_ci(diff: np.ndarray, n_boot: int = 10_000, seed: int = 20260828) -> tuple[float, float]:
    """BCa interval for the protocol's Hodges–Lehmann estimator (amendment F-02)."""
    return bca_bootstrap_ci(diff, resamples=n_boot, seed=seed)


def _per_budget(piv: pd.DataFrame, budget: int, scenario: str) -> dict[str, Any]:
    a, b = piv["ppo_uq"].to_numpy(float), piv["ppo_point"].to_numpy(float)
    diff = a - b
    normal_p = None if np.allclose(diff, diff[0]) else float(stats.shapiro(diff).pvalue)
    if np.all(diff == 0):
        test, statistic, p_value = "Wilcoxon signed-rank", 0.0, 1.0
    elif normal_p is not None and normal_p >= 0.05:
        test, result = "paired t-test", stats.ttest_rel(a, b)
        statistic, p_value = float(result.statistic), float(result.pvalue)
    else:
        test, result = "Wilcoxon signed-rank", stats.wilcoxon(a, b)
        statistic, p_value = float(result.statistic), float(result.pvalue)
    walsh = [(diff[i] + diff[j]) / 2 for i in range(diff.size) for j in range(i, diff.size)]
    lo, hi = _bca_ci(diff)
    return {
        "budget": budget, "scenario": scenario, "n": int(diff.size),
        "mean_ppo_uq": float(a.mean()), "mean_ppo_point": float(b.mean()),
        "median_ppo_uq": float(np.median(a)), "median_ppo_point": float(np.median(b)),
        "sd_ppo_uq": float(a.std(ddof=1)), "sd_ppo_point": float(b.std(ddof=1)),
        "mean_difference": float(diff.mean()), "median_difference": float(np.median(diff)),
        "sd_difference": float(diff.std(ddof=1)),
        "hodges_lehmann": float(np.median(walsh)),
        "ci_low": lo, "ci_high": hi, "ci_estimand": "Hodges-Lehmann paired location",
        "test": test, "normality_p": normal_p,
        "statistic": statistic, "p_value": p_value,
        "direction": "no difference" if diff.mean() == 0 else "ppo_point better" if diff.mean() > 0 else "ppo_uq better",
    }


def main() -> int:
    setup_logging(log_file="analyse_followup.log")

    # Read stored cells directly: a retained aggregate can predate resumed evaluations.
    episodes = load_frame(discover_cells())
    coverage = budget_coverage(episodes)
    incomplete = [budget for budget, status in coverage.items() if not status["complete"]]
    if incomplete:
        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "study": "FOLLOWUP-UQ-TRAINING", "evidence_class": "FOLLOW-UP / INCOMPLETE",
            "protocol": "docs/followup_training_budget_protocol.md",
            "grid_complete": False, "coverage": coverage, "budgets_missing": incomplete,
            "budgets_analysed": [], "per_budget": [], "scenario_breakdown": [],
            "primary_interaction": {"status": "NOT RUN", "reason": "Incomplete grid; no interim inference."},
            "registered_conclusion_unchanged": "Read results/statistical_analysis.json; original evidence is unchanged.",
        }
        (RESULTS_DIR / "followup_analysis.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        print(f"INCOMPLETE: budgets {incomplete}; no follow-up inference performed")
        return 1
    df = seed_level(episodes)

    available, missing_budgets, long_rows = [], [], []
    per_budget: list[dict[str, Any]] = []
    for budget in BUDGETS:
        frame = _budget_frame(df, budget)
        piv = _paired(frame, PRIMARY_SCENARIO)
        if piv is None:
            missing_budgets.append(budget)
            continue
        available.append(budget)
        per_budget.append(_per_budget(piv, budget, PRIMARY_SCENARIO))
        for seed, row in piv.iterrows():
            long_rows.append({"seed": int(seed), "budget": budget,
                              "log2_budget": float(np.log2(budget)),
                              "diff": float(row["ppo_uq"] - row["ppo_point"])})

    for row, adjusted in zip(per_budget, _holm([r["p_value"] for r in per_budget])):
        row["p_adjusted"] = adjusted
        row["adjustment"] = f"Holm (family of {len(per_budget)} per-budget tests)"
        row["rejects_at_0.05"] = bool(adjusted < 0.05)

    # --- primary: the budget interaction ----------------------------------------------------
    interaction: dict[str, Any] = {"status": "NOT ESTIMABLE",
                                   "reason": "fewer than three budgets have complete data"}
    if len(available) == len(BUDGETS):
        import statsmodels.formula.api as smf

        long = pd.DataFrame(long_rows)
        model = smf.mixedlm("diff ~ log2_budget", long, groups=long["seed"]).fit()
        if not model.converged:
            raise RuntimeError("Budget interaction model did not converge; inference withheld")
        slope = float(model.params["log2_budget"])
        se = float(model.bse["log2_budget"])
        p = float(model.pvalues["log2_budget"])
        if p >= 0.05:
            reading = ("No budget dependence detected. The confidence interval describes "
                       "compatible slopes; a nonsignificant test does not establish a flat "
                       "trend or budget-independence within or beyond the tested range.")
        elif slope < 0:
            reading = ("The disadvantage shrinks as the budget grows. This supports a "
                       "sample-efficiency reading and does NOT by itself establish that the "
                       "uncertainty channel is beneficial: that would need ppo_uq to win at some "
                       "budget, tested and surviving correction.")
        else:
            reading = "The disadvantage grows as the budget grows."
        interaction = {
            "status": "ESTIMATED",
            "model": "mixedlm: diff ~ log2(budget), random intercept by training seed",
            "slope_per_log2_budget": slope, "standard_error": se,
            "ci_low": slope - 1.96 * se, "ci_high": slope + 1.96 * se,
            "z": slope / se if se else float("nan"), "p_value": p,
            "n_observations": int(len(long)), "n_seeds": int(long["seed"].nunique()),
            "budgets": available,
            "reading": reading,
        }

    scenario_breakdown = [
        _per_budget(piv, budget, scenario)
        for budget in available
        for scenario in SCENARIOS
        if (piv := _paired(_budget_frame(df, budget), scenario)) is not None
    ]
    for row, adjusted in zip(scenario_breakdown, benjamini_hochberg(
            [row["p_value"] for row in scenario_breakdown])):
        row["p_adjusted"] = adjusted
        row["adjustment"] = "Benjamini-Hochberg, all budget-by-scenario contrasts, exploratory q=0.10"

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "study": "FOLLOWUP-UQ-TRAINING",
        "evidence_class": "FOLLOW-UP / EXPLORATORY",
        "protocol": "docs/followup_training_budget_protocol.md",
        "analysis_prespecified": ("this script was written and committed before the follow-up "
                                  "produced any result"),
        "endpoint": ENDPOINT, "primary_scenario": PRIMARY_SCENARIO,
        "unit_of_analysis": "training seed",
        "budgets_planned": list(BUDGETS),
        "budgets_analysed": available,
        "budgets_missing": missing_budgets,
        "grid_complete": not missing_budgets,
        "registered_conclusion_unchanged": (
            "Under the registered 60,000-step experiment H1 was NOT SUPPORTED: ppo_uq left more "
            "passengers behind than ppo_point, +0.013742, 95% BCa CI [0.006570, 0.018523], "
            "Holm-adjusted p = 0.007741. Nothing in this follow-up alters that."
        ),
        "primary_interaction": interaction,
        "per_budget": per_budget,
        "scenario_breakdown": scenario_breakdown,
        "caveats": [
            "FOLLOW-UP, never confirmatory: the study was designed after the confirmatory result "
            "was known, even though its analysis was pre-specified.",
            "The 60,000-step row reproduces the registered point estimate exactly, because it "
            "reads the same stored cells. Its confidence interval will differ slightly from the "
            "one in results/statistical_analysis.json: both estimate the Hodges-Lehmann paired "
            "location, using different resampling seeds. The original follow-up implementation "
            "instead bootstrapped the mean; amendment F-02 corrects that mismatch. The registered interval is the one to quote for H1; the "
            "interval here exists only so the 60k row is computed the same way as every other "
            "row in this grid.",
            "Underpowered for per-budget comparisons at SESOI magnitude; the realised between-"
            "seed SD in the original study was about twice the pilot estimate. The interaction "
            "test borrows strength across budgets and is the primary for that reason.",
            "Only total_timesteps varies. Nothing here speaks to a tuned configuration.",
            "The baselines are unchanged and remain descriptively better than every RL arm; this "
            "study does not test RL against them.",
        ],
    }
    if missing_budgets:
        payload["incomplete_grid_notice"] = (
            f"Budgets {missing_budgets} have no complete data and are reported as missing, not "
            "imputed or dropped. Read the interaction over the reduced range only."
        )

    out = RESULTS_DIR / "followup_analysis.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s", out)

    print(f"budgets analysed: {available}   missing: {missing_budgets}")
    for row in per_budget:
        print(f"  {row['budget']:>7} steps  n={row['n']:2d}  diff={row['mean_difference']:+.6f}"
              f"  CI [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}]"
              f"  Holm p={row['p_adjusted']:.4g}  {row['direction']}")
    if interaction["status"] == "ESTIMATED":
        print(f"\ninteraction slope {interaction['slope_per_log2_budget']:+.6f} per doubling "
              f"(p={interaction['p_value']:.4g})")
        print(f"  {interaction['reading']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
