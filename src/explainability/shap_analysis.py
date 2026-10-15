"""SHAP attribution for the trained policy (M13, protocol section 11).

**Attribution is associational, never causal.** SHAP decomposes *this model's* output on *this
input distribution*. It says which observation channels the policy's action responds to; it does
not say that changing a channel in the world would change outcomes, and it does not establish
that the policy learned the mechanism H1 proposes. Every output of this module carries that
caveat, and `docs/explainability.md` states it in the first paragraph.

**Why a permutation explainer rather than DeepSHAP.** The policy is a small MLP behind
Stable-Baselines3's action-distribution machinery, and the gradient-based explainers assume a
plain differentiable forward pass they can hook. Wrapping the policy as an opaque function and
using the model-agnostic permutation explainer avoids depending on SB3 internals that are not
part of its public API, at the cost of more forward passes. Protocol section 11 permits a
"registered defensible alternative" when the gradient explainer is unstable for the architecture;
this is that alternative, chosen before seeing any attribution.

The quantity explained is the **probability the policy assigns to dispatching at least one
trainset**, not the raw action index. An action index is ordinal but not cardinal -- attributing
to "action 2 minus action 1" would treat a categorical choice as a number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.common.logging import get_logger
from src.common.paths import RESULTS_DIR, ensure_dir

logger = get_logger(__name__)

__all__ = ["explain_policy", "collect_background", "PolicyExplainer"]

#: Background sample for the explainer. Larger is more faithful and linearly more expensive.
BACKGROUND_SIZE = 100
#: Instances explained per scenario.
EXPLAIN_SIZE = 200


@dataclass
class PolicyExplainer:
    """Wraps a MaskablePPO policy as a plain function of the observation vector."""

    model: Any
    n_actions: int

    def dispatch_probability(self, observations: np.ndarray) -> np.ndarray:
        """P(action >= 1) under the policy, for a batch of observations.

        Masks are deliberately **not** applied here. The mask is a property of the environment
        state, not of the policy, and including it would attribute the infrastructure's
        constraints to the policy's reasoning.
        """
        import torch

        observations = np.atleast_2d(np.asarray(observations, dtype=np.float32))
        with torch.no_grad():
            tensor = torch.as_tensor(observations)
            distribution = self.model.policy.get_distribution(tensor)
            probabilities = distribution.distribution.probs.cpu().numpy()
        return 1.0 - probabilities[:, 0]


def collect_background(env, policy, n_samples: int, *, seeds: tuple[int, ...]) -> np.ndarray:
    """Observations drawn from the policy's own on-policy state distribution.

    Explaining on states the policy never visits would describe a counterfactual model. These
    come from running the policy itself, on evaluation seeds disjoint from the sweep.
    """
    from src.evaluation.policies import run_episode

    collected: list[np.ndarray] = []
    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        while True:
            collected.append(np.asarray(observation, dtype=float))
            mask = env.action_masks()
            action = min(int(policy.act(observation, env, mask)),
                         int(np.flatnonzero(mask)[-1]))
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        if len(collected) >= n_samples * 4:
            break

    array = np.asarray(collected, dtype=float)
    if array.shape[0] <= n_samples:
        return array
    # Even thinning across the operating day rather than a random draw, so the background covers
    # morning, peak and evening rather than over-representing whichever is longest.
    index = np.linspace(0, array.shape[0] - 1, n_samples).astype(int)
    return array[index]


def explain_policy(
    *,
    method: str = "ppo_uq",
    train_seed: int = 101,
    weight_id: str = "w_balanced",
    scenarios: tuple[str, ...] = ("normal", "surge", "disruption", "combined"),
    background_size: int = BACKGROUND_SIZE,
    explain_size: int = EXPLAIN_SIZE,
    seeds: tuple[int, ...] = (61, 62, 63),
) -> dict[str, Any]:
    """Global, scenario-specific and local attributions for one trained policy."""
    import shap

    from src.evaluation.runner import Cell, build_env, _trained_policy
    from src.rl.policies import TrainedPPOPolicy

    model = _trained_policy(method, train_seed, weight_id)
    explainer_target = PolicyExplainer(model=model, n_actions=int(model.action_space.n))
    policy = TrainedPPOPolicy(method_id=method, model=model)

    per_scenario: dict[str, Any] = {}
    all_shap: list[np.ndarray] = []
    feature_names: list[str] = []

    for scenario in scenarios:
        cell = Cell(method=method, scenario=scenario, eval_seed=seeds[0], train_seed=train_seed,
                    weight_id=weight_id)
        env = build_env(cell)
        feature_names = env.obs_spec.feature_names

        background = collect_background(env, policy, background_size, seeds=seeds)
        instances = collect_background(env, policy, explain_size, seeds=seeds)
        if background.size == 0 or instances.size == 0:
            logger.warning("no observations collected for scenario %s", scenario)
            continue

        explainer = shap.PermutationExplainer(
            explainer_target.dispatch_probability, background
        )
        values = np.asarray(explainer(instances, max_evals=2 * background.shape[1] + 1).values)
        all_shap.append(values)

        mean_abs = np.abs(values).mean(axis=0)
        order = np.argsort(-mean_abs)
        per_scenario[scenario] = {
            "n_instances": int(values.shape[0]),
            "top_features": [
                {"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i]),
                 "mean_shap": float(values[:, i].mean())}
                for i in order[:20]
            ],
        }
        logger.info("SHAP %s: top feature %s", scenario, feature_names[order[0]])

    if not all_shap:
        raise RuntimeError("no SHAP values computed")

    stacked = np.vstack(all_shap)
    mean_abs = np.abs(stacked).mean(axis=0)
    order = np.argsort(-mean_abs)

    # Local explanations for the frozen instance set of protocol section 11: the epochs where the
    # policy dispatched most, and the epochs immediately after a failure.
    local = [
        {
            "rank": int(rank),
            "instance": int(index),
            "top_contributions": [
                {"feature": feature_names[i], "shap": float(stacked[index, i])}
                for i in np.argsort(-np.abs(stacked[index]))[:8]
            ],
        }
        for rank, index in enumerate(np.argsort(-np.abs(stacked).sum(axis=1))[:10], start=1)
    ]

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "train_seed": train_seed,
        "weight_id": weight_id,
        "explainer": "shap.PermutationExplainer (model-agnostic)",
        "explained_quantity": "P(dispatch >= 1 trainset) under the policy",
        "background_size": int(background_size),
        "instances_per_scenario": int(explain_size),
        "evaluation_seeds": list(seeds),
        "caveats": [
            "SHAP attribution is associational, not causal: it decomposes this model's output on "
            "this input distribution and says nothing about what changing a channel in the world "
            "would do.",
            "A high attribution to a sigma channel is evidence the policy's action responds to "
            "it, not evidence that responding improves outcomes. That question is H1's, and only "
            "the confirmatory test answers it.",
            "Masks are not applied inside the explainer, so infrastructure constraints are not "
            "attributed to the policy's reasoning.",
            "Attributions come from one training seed; EXP-XAI-002 measures their stability "
            "across seeds and background resamples.",
        ],
        "global_importance": [
            {"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i]),
             "mean_shap": float(stacked[:, i].mean())}
            for i in order
        ],
        "per_scenario": per_scenario,
        "local_explanations": local,
    }

    ensure_dir(RESULTS_DIR)
    (RESULTS_DIR / "shap_analysis.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return payload


def attribution_stability(
    *,
    method: str = "ppo_uq",
    train_seeds: tuple[int, ...] = (101, 102, 103),
    weight_id: str = "w_balanced",
) -> dict[str, Any]:
    """EXP-XAI-002: Spearman rank correlation of global attributions across training seeds.

    An explanation that changes completely between seeds describes the seed, not the method.
    """
    from scipy import stats

    rankings: dict[int, list[float]] = {}
    names: list[str] = []
    for seed in train_seeds:
        try:
            payload = explain_policy(method=method, train_seed=seed, weight_id=weight_id,
                                     scenarios=("surge",), background_size=60, explain_size=80)
        except Exception as error:  # noqa: BLE001 - a missing policy is reported, not fatal
            logger.warning("stability: seed %d unavailable (%r)", seed, error)
            continue
        ordered = sorted(payload["global_importance"], key=lambda r: r["feature"])
        names = [r["feature"] for r in ordered]
        rankings[seed] = [r["mean_abs_shap"] for r in ordered]

    seeds_available = sorted(rankings)
    correlations = []
    for i, a in enumerate(seeds_available):
        for b in seeds_available[i + 1:]:
            rho = float(stats.spearmanr(rankings[a], rankings[b]).statistic)
            correlations.append({"seed_a": a, "seed_b": b, "spearman_rho": rho})

    return {
        "method": method,
        "train_seeds": seeds_available,
        "n_features": len(names),
        "pairwise_spearman": correlations,
        "mean_spearman": float(np.mean([c["spearman_rho"] for c in correlations]))
        if correlations else float("nan"),
        "interpretation": (
            "high rank correlation means the attribution describes the method; low correlation "
            "means it describes the particular training seed and should not be generalised"
        ),
    }
