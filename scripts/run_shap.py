"""Run the registered SHAP analysis (M13, EXP-XAI-001/002/003).

    python scripts/run_shap.py

Writes ``results/shap_analysis.json``, ``results/shap_stability.json`` and
``docs/explainability.md``.
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import json
from datetime import datetime, timezone

import numpy as np

from src.common.logging import get_logger
from src.common.paths import MODELS_DIR, REPO_ROOT, RESULTS_DIR
from src.explainability.shap_analysis import attribution_stability, explain_policy

logger = get_logger(__name__)


def main() -> int:
    policies = sorted((MODELS_DIR / "policies").rglob("ppo_uq/*/policy.zip"))
    if not policies:
        raise SystemExit("no trained ppo_uq policy found; run the training stage first")
    seed = int(policies[0].parent.name)

    payload = explain_policy(method="ppo_uq", train_seed=seed)
    logger.info("SHAP: %d features attributed", len(payload["global_importance"]))

    available = sorted({int(p.parent.name) for p in policies})[:3]
    stability = attribution_stability(method="ppo_uq", train_seeds=tuple(available))
    (RESULTS_DIR / "shap_stability.json").write_text(
        json.dumps(stability, indent=2, default=float), encoding="utf-8")

    top = payload["global_importance"][:15]
    sigma_features = [r for r in payload["global_importance"]
                      if r["feature"].startswith("forecast_sigma")]
    sigma_share = (sum(abs(r["mean_abs_shap"]) for r in sigma_features)
                   / max(1e-12, sum(abs(r["mean_abs_shap"]) for r in payload["global_importance"])))

    lines = [
        "# Explainability (M13)",
        "",
        "**SHAP attribution is associational, not causal.** It decomposes this policy's output "
        "on this input distribution. It shows which observation channels the policy's action "
        "responds to; it does not show that changing a channel in the world would change "
        "outcomes, and it is not evidence for H1. Only the confirmatory tests answer that.",
        "",
        f"Generated {datetime.now(timezone.utc).date().isoformat()} from `ppo_uq` at training "
        f"seed {seed}, explained quantity: {payload['explained_quantity']}, "
        f"explainer: {payload['explainer']}.",
        "",
        "## Global attribution (top 15)",
        "",
        "| rank | feature | mean \\|SHAP\\| | mean SHAP |",
        "|---:|---|---:|---:|",
    ]
    for rank, row in enumerate(top, 1):
        lines.append(f"| {rank} | `{row['feature']}` | {row['mean_abs_shap']:.5g} | "
                     f"{row['mean_shap']:.5g} |")

    lines += [
        "",
        "## Attribution to the uncertainty channels",
        "",
        f"The two sigma blocks account for **{sigma_share:.1%}** of total absolute attribution.",
        "",
        "What this does and does not mean: a high share is evidence that the policy's dispatch "
        "probability *responds* to forecast dispersion. It is **not** evidence that responding "
        "improves outcomes — a policy could respond strongly to a channel and be worse for it. "
        "The `ppo_uq` vs `ppo_placebo` comparison is what separates those two possibilities, "
        "because the placebo arm responds to a channel with the same marginal distribution and "
        "no information.",
        "",
        "## Stability across training seeds (EXP-XAI-002)",
        "",
    ]
    correlations = stability.get("pairwise_spearman", [])
    if correlations:
        lines += [
            f"Mean pairwise Spearman rank correlation of global attributions: "
            f"**{stability['mean_spearman']:.3f}** across "
            f"{len(stability['train_seeds'])} training seeds.",
            "",
            "| seed A | seed B | Spearman rho |",
            "|---:|---:|---:|",
        ]
        for row in correlations:
            lines.append(f"| {row['seed_a']} | {row['seed_b']} | {row['spearman_rho']:.4f} |")
        lines += [
            "",
            stability["interpretation"].capitalize() + ".",
            "",
        ]
    else:
        lines += ["Only one training seed was available; stability could not be assessed.", ""]

    lines += [
        "## Scenario-specific attribution",
        "",
        "| scenario | top feature | mean \\|SHAP\\| |",
        "|---|---|---:|",
    ]
    for scenario, block in payload.get("per_scenario", {}).items():
        if block.get("top_features"):
            first = block["top_features"][0]
            lines.append(f"| {scenario} | `{first['feature']}` | {first['mean_abs_shap']:.5g} |")

    lines += [
        "",
        "## Caveats",
        "",
    ] + [f"- {c}" for c in payload["caveats"]] + [""]

    (REPO_ROOT / "docs" / "explainability.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"SHAP complete: sigma channels carry {sigma_share:.1%} of attribution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
