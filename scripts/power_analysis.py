"""R-1 power analysis: the mandatory gate before the M11 sweep.

Protocol section 8.4 leaves the power analysis ``TBD@M11, frozen before the sweep``. This script
supplies it, from a pilot whose only job is to estimate the between-training-seed standard
deviation of the paired difference on the primary endpoint.

**What the pilot must not be used for**, and is not: tuning hyperparameters, selecting a model,
changing reward weights, changing scenarios, selecting seeds, or obtaining significance. Its
policies are trained on a dedicated seed block (901-904), evaluated on a dedicated evaluation
block (41-50), and **discarded**. Neither block appears anywhere in the registered sweep, so no
pilot observation can contribute to a confirmatory result.

**The SESOI is frozen before the pilot runs**, in `docs/experimental_protocol.md` amendment A-04.
Protocol section 8.4 is explicit: *"Lowering the SESOI after seeing the pilot is not an option."*

Writes ``results/power_analysis.json`` and ``docs/power_analysis.md``.
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import json
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.common.logging import get_logger
from src.common.paths import RESULTS_DIR, REPO_ROOT, ensure_dir

logger = get_logger(__name__)

#: Smallest effect size of interest on `left_behind_fraction`, frozen in amendment A-04 BEFORE
#: the pilot was run. 0.002 absolute is roughly a tenth of the fixed-timetable baseline's
#: left-behind rate under surge (measured 0.018), and about 570 passengers a day on a line
#: carrying ~285,000 -- an operationally noticeable change, and not a threshold chosen to be easy.
SESOI = 0.002

#: Family-wise alpha 0.05 over the two confirmatory tests (protocol section 7.3), Holm-corrected.
#: The smaller of the two Holm thresholds is used, which is the conservative reading.
ALPHA_PER_TEST = 0.025
TARGET_POWER = 0.80

#: Registered design (protocol section 8.1 and the experiment registry).
REGISTERED_TRAIN_SEEDS = 10
REGISTERED_EVAL_SEEDS = 30


def collect_pilot() -> dict[str, Any]:
    """Load the pilot cells and reduce them to seed-level means, as section 8.2 requires."""
    from scripts.run_experiments import PILOT_EVAL_SEEDS, PILOT_METHODS, PILOT_TRAIN_SEEDS
    from src.evaluation.runner import Cell, load_cell_metrics

    per_method: dict[str, dict[int, list[float]]] = {m: {} for m in PILOT_METHODS}
    episode_level: dict[str, list[float]] = {m: [] for m in PILOT_METHODS}
    missing = 0

    for method in PILOT_METHODS:
        for train_seed in PILOT_TRAIN_SEEDS:
            values: list[float] = []
            for eval_seed in PILOT_EVAL_SEEDS:
                metrics = load_cell_metrics(Cell(
                    method=method, scenario="surge", eval_seed=eval_seed,
                    train_seed=train_seed, variant="pilot", experiment_id="R-1-PILOT",
                ))
                if metrics is None:
                    missing += 1
                    continue
                values.append(float(metrics["left_behind_fraction"]))
            if values:
                per_method[method][train_seed] = values
                episode_level[method].extend(values)

    seeds = sorted(set(per_method[PILOT_METHODS[0]]) & set(per_method[PILOT_METHODS[1]]))
    a = np.array([np.mean(per_method[PILOT_METHODS[0]][s]) for s in seeds])
    b = np.array([np.mean(per_method[PILOT_METHODS[1]][s]) for s in seeds])

    return {
        "methods": list(PILOT_METHODS),
        "train_seeds": seeds,
        "seed_means": {PILOT_METHODS[0]: a.tolist(), PILOT_METHODS[1]: b.tolist()},
        "episode_level": episode_level,
        "missing_cells": missing,
        "n_pairs": len(seeds),
    }


def required_n(effect: float, sd: float, *, alpha: float, power: float) -> int | None:
    """Sample size for a paired test, by normal approximation with a t-correction.

    Returns ``None`` when the effect is zero or the standard deviation is degenerate, because no
    finite sample size detects a zero effect and reporting one would be meaningless.
    """
    from scipy import stats

    if not np.isfinite(effect) or not np.isfinite(sd) or effect == 0 or sd <= 0:
        return None
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) * sd / abs(effect)) ** 2
    # One iteration with the t quantile: the normal approximation understates n at small samples.
    for _ in range(6):
        df = max(1, math.ceil(n) - 1)
        t_alpha = stats.t.ppf(1 - alpha / 2, df)
        n = ((t_alpha + z_beta) * sd / abs(effect)) ** 2
    return int(math.ceil(n))


def detectable_effect(n: int, sd: float, *, alpha: float, power: float) -> float:
    from scipy import stats

    if n < 2 or not np.isfinite(sd) or sd <= 0:
        return float("nan")
    t_alpha = stats.t.ppf(1 - alpha / 2, n - 1)
    z_beta = stats.norm.ppf(power)
    return float((t_alpha + z_beta) * sd / math.sqrt(n))


def analyse_pilot() -> dict[str, Any]:
    pilot = collect_pilot()
    if pilot["n_pairs"] < 3:
        raise SystemExit(
            f"the pilot produced only {pilot['n_pairs']} paired training seeds; "
            "at least 3 are needed to estimate a between-seed standard deviation"
        )

    method_a, method_b = pilot["methods"]
    a = np.asarray(pilot["seed_means"][method_a], dtype=float)
    b = np.asarray(pilot["seed_means"][method_b], dtype=float)
    differences = a - b

    observed_effect = float(differences.mean())
    between_seed_sd = float(differences.std(ddof=1))
    endpoint_sd_a = float(a.std(ddof=1))
    endpoint_sd_b = float(b.std(ddof=1))

    n_for_observed = required_n(observed_effect, between_seed_sd,
                                alpha=ALPHA_PER_TEST, power=TARGET_POWER)
    n_for_sesoi = required_n(SESOI, between_seed_sd, alpha=ALPHA_PER_TEST, power=TARGET_POWER)
    mde = detectable_effect(REGISTERED_TRAIN_SEEDS, between_seed_sd,
                            alpha=ALPHA_PER_TEST, power=TARGET_POWER)

    adequate_for_sesoi = bool(n_for_sesoi is not None and n_for_sesoi <= REGISTERED_TRAIN_SEEDS)
    adequate_for_observed = bool(
        n_for_observed is not None and n_for_observed <= REGISTERED_TRAIN_SEEDS)

    if adequate_for_sesoi:
        decision = "PROCEED with the registered n"
        rationale = (
            f"the registered S = {REGISTERED_TRAIN_SEEDS} training seeds detect the "
            f"pre-registered SESOI of {SESOI} with power >= {TARGET_POWER:.0%}"
        )
        amended_n = REGISTERED_TRAIN_SEEDS
    elif n_for_sesoi is not None and n_for_sesoi <= 20:
        decision = "AMEND n upward"
        rationale = (
            f"the registered S = {REGISTERED_TRAIN_SEEDS} is underpowered for the SESOI; "
            f"S = {n_for_sesoi} is the smallest defensible increase supported by the "
            "power calculation"
        )
        amended_n = int(n_for_sesoi)
    else:
        decision = "PROCEED with the registered n and report as potentially underpowered"
        rationale = (
            f"detecting the SESOI would need S = {n_for_sesoi}, which is beyond the compute "
            "available. Protocol section 8.4's honest option is taken: run the registered design "
            "and state the minimum detectable effect in the paper. Lowering the SESOI is not an "
            "option."
        )
        amended_n = REGISTERED_TRAIN_SEEDS

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Estimate the between-training-seed SD of the paired difference on the primary "
            "endpoint, to supply protocol section 8.4's TBD power analysis before the sweep."
        ),
        "pilot": {
            "methods": pilot["methods"],
            "train_seeds": pilot["train_seeds"],
            "eval_seeds_per_seed": len(pilot["episode_level"][method_a]) // max(1, pilot["n_pairs"]),
            "n_pairs": pilot["n_pairs"],
            "missing_cells": pilot["missing_cells"],
            "seed_means": pilot["seed_means"],
            "isolation": (
                "pilot policies use training seeds 901-904 and evaluation seeds 41-50, both "
                "disjoint from the registered pools; the policies are discarded and no pilot "
                "observation enters a confirmatory result"
            ),
        },
        "primary_endpoint": "left_behind_fraction",
        "scenario": "surge",
        "observed_effect_size": observed_effect,
        "between_seed_sd": between_seed_sd,
        "endpoint_sd": {method_a: endpoint_sd_a, method_b: endpoint_sd_b},
        "episode_level_sd": {
            m: float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
            for m, v in pilot["episode_level"].items()
        },
        "alpha_per_test": ALPHA_PER_TEST,
        "alpha_family_wise": 0.05,
        "multiplicity": "Holm over the two confirmatory tests (protocol section 7.3)",
        "target_power": TARGET_POWER,
        "sesoi": SESOI,
        "sesoi_frozen_in": "docs/experimental_protocol.md amendment A-04, before the pilot ran",
        "registered_train_seeds": REGISTERED_TRAIN_SEEDS,
        "registered_eval_seeds": REGISTERED_EVAL_SEEDS,
        "required_n_for_sesoi": n_for_sesoi,
        "required_n_for_observed_effect": n_for_observed,
        "minimum_detectable_effect_at_registered_n": mde,
        "registered_n_adequate_for_sesoi": adequate_for_sesoi,
        "registered_n_adequate_for_observed_effect": adequate_for_observed,
        "decision": decision,
        "rationale": rationale,
        "amended_train_seeds": amended_n,
    }

    ensure_dir(RESULTS_DIR)
    (RESULTS_DIR / "power_analysis.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")
    _write_report(payload)

    logger.info("power analysis: %s (%s)", decision, rationale)
    return payload


def _write_report(payload: dict[str, Any]) -> None:
    method_a, method_b = payload["pilot"]["methods"]
    lines = [
        "# R-1 power analysis",
        "",
        "Supplies the power analysis protocol section 8.4 leaves as `TBD@M11, frozen before the "
        "sweep`. Generated by `scripts/power_analysis.py` from the pilot; every number below "
        "comes from a stored result file.",
        "",
        "## What the pilot was and was not",
        "",
        f"Pilot: `{method_a}` vs `{method_b}` under `surge`, "
        f"{payload['pilot']['n_pairs']} training seeds x "
        f"{payload['pilot']['eval_seeds_per_seed']} evaluation seeds.",
        "",
        payload["pilot"]["isolation"].capitalize() + ".",
        "",
        "It was **not** used to tune hyperparameters, select a model, change reward weights, "
        "change scenarios, select seeds, or obtain significance.",
        "",
        "## Estimates",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| primary endpoint | `{payload['primary_endpoint']}` |",
        f"| observed paired effect ({method_a} - {method_b}) | {payload['observed_effect_size']:.6f} |",
        f"| between-training-seed SD of the difference | {payload['between_seed_sd']:.6f} |",
        f"| endpoint SD, `{method_a}` | {payload['endpoint_sd'][method_a]:.6f} |",
        f"| endpoint SD, `{method_b}` | {payload['endpoint_sd'][method_b]:.6f} |",
        "",
        "## Power",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| pre-registered SESOI | {payload['sesoi']} |",
        f"| alpha per confirmatory test (Holm) | {payload['alpha_per_test']} |",
        f"| target power | {payload['target_power']:.0%} |",
        f"| registered S (training seeds) | {payload['registered_train_seeds']} |",
        f"| S required for the SESOI | {payload['required_n_for_sesoi']} |",
        f"| S required for the observed effect | {payload['required_n_for_observed_effect']} |",
        f"| minimum detectable effect at registered S | "
        f"{payload['minimum_detectable_effect_at_registered_n']:.6f} |",
        "",
        f"**Decision: {payload['decision']}.**",
        "",
        payload["rationale"],
        "",
        "## What this does not license",
        "",
        "- The SESOI was frozen before the pilot and is not revisited. Protocol section 8.4: "
        "*\"Lowering the SESOI after seeing the pilot is not an option.\"*",
        "- A power calculation says what the design can detect; it says nothing about whether an "
        "effect exists. If the confirmatory test does not reject, the TOST equivalence test "
        "against this SESOI decides between *no evidence of benefit* and *evidence of no "
        "meaningful benefit*.",
        "- The pilot's observed effect is an unbiased-but-noisy estimate from "
        f"{payload['pilot']['n_pairs']} seeds. It is used to report the required sample size, "
        "never as evidence about H1.",
        "",
    ]
    (REPO_ROOT / "docs" / "power_analysis.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    analyse_pilot()
