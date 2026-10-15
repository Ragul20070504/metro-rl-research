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

from scripts.run_followup import ARMS, BUDGETS, REGISTERED_BUDGET, TRAIN_SEEDS

logger = get_logger(__name__)

ENDPOINT = "left_behind_fraction"
PRIMARY_SCENARIO = "surge"
SCENARIOS = ("normal", "surge", "disruption", "combined")


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
    """BCa interval for the mean paired difference, as the protocol specifies."""
    rng = np.random.default_rng(seed)
    n = diff.size
    observed = float(diff.mean())
    boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    proportion = float((boot < observed).mean())
    if proportion <= 0 or proportion >= 1:  # degenerate; fall back to the percentile interval
        return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    z0 = stats.norm.ppf(proportion)
    jack = np.array([np.delete(diff, i).mean() for i in range(n)])
    centred = jack.mean() - jack
    denom = 6.0 * (float((centred ** 2).sum()) ** 1.5)
    a = float((centred ** 3).sum()) / denom if denom else 0.0
    lo_z, hi_z = stats.norm.ppf(0.025), stats.norm.ppf(0.975)
    lo = stats.norm.cdf(z0 + (z0 + lo_z) / (1 - a * (z0 + lo_z)))
    hi = stats.norm.cdf(z0 + (z0 + hi_z) / (1 - a * (z0 + hi_z)))
    return float(np.percentile(boot, 100 * lo)), float(np.percentile(boot, 100 * hi))


def _per_budget(piv: pd.DataFrame, budget: int, scenario: str) -> dict[str, Any]:
    a, b = piv["ppo_uq"].to_numpy(float), piv["ppo_point"].to_numpy(float)
    diff = a - b
    normal_p = float(stats.shapiro(diff).pvalue)
    if normal_p >= 0.05:
        test, result = "paired t-test", stats.ttest_rel(a, b)
    else:
        test, result = "Wilcoxon signed-rank", stats.wilcoxon(a, b)
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
        "ci_low": lo, "ci_high": hi,
        "test": test, "normality_p": normal_p,
        "statistic": float(result.statistic), "p_value": float(result.pvalue),
        "direction": "ppo_point better" if diff.mean() > 0 else "ppo_uq better",
    }


def main() -> int:
    setup_logging(log_file="analyse_followup.log")

    df = pd.read_parquet(RESULTS_DIR / "seed_level.parquet")

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
    if len(available) >= 3:
        import statsmodels.formula.api as smf

        long = pd.DataFrame(long_rows)
        model = smf.mixedlm("diff ~ log2_budget", long, groups=long["seed"]).fit()
        slope = float(model.params["log2_budget"])
        se = float(model.bse["log2_budget"])
        p = float(model.pvalues["log2_budget"])
        if p >= 0.05:
            reading = ("No budget dependence detected. Over the range tested the disadvantage is "
                       "flat in log-budget; the confidence interval below bounds the trend the "
                       "design could have excluded. This does not establish budget-independence "
                       "beyond the largest budget run.")
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
            "one in results/statistical_analysis.json: both are BCa bootstraps, drawn with "
            "different resampling seeds. The registered interval is the one to quote for H1; the "
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
