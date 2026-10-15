"""Environment responsiveness report on the real network (M6 gate, execution directive section 12).

Runs a set of **deterministic, non-RL** policies through the environment and records how waiting,
left-behind passengers, energy and dispatch cost respond as dispatch behaviour changes. The point
is to establish, before any PPO training begins, that the environment actually responds to
decisions -- and to measure how much room a policy has to improve on a trivial one, which bears
directly on the power analysis in protocol section 8.4.

Nothing here is a research result: these are not the pre-registered baselines (`fixed_timetable`,
`heuristic_threshold`, `milp_*` arrive at M7-M9), and no comparison between methods is drawn.

Writes ``results/env_sanity_report.{md,json}``.
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
from src.common.paths import REPO_ROOT, RESULTS_DIR
from src.environment.forecast_provider import SeasonalNaiveProvider
from src.environment.metro_env import EnvConfig, MetroEnv, constant_history_sampler
from src.environment.reward import ReferenceScales
from src.simulator.demand import DemandProcess
from src.simulator.engine import SimulatorConfig
from src.simulator.network import Network

logger = get_logger(__name__)

SEEDS = (1, 2, 3, 4, 5)
SCENARIOS = ("normal", "surge")


def constant(k: int):
    def policy(env: MetroEnv, mask: np.ndarray) -> int:
        return k if mask[k] else int(np.flatnonzero(mask)[-1])
    return policy


def queue_threshold(limit: float):
    """Induct one trainset only when the line queue exceeds ``limit`` passengers."""
    def policy(env: MetroEnv, mask: np.ndarray) -> int:
        queued = sum(q.size for row in env.simulator.queues for q in row)
        want = 1 if queued > limit else 0
        return want if mask[want] else 0
    return policy


POLICIES = {
    "constant_0_no_service": constant(0),
    "constant_1": constant(1),
    "constant_2": constant(2),
    "queue_threshold_100": queue_threshold(100.0),
    "queue_threshold_400": queue_threshold(400.0),
    "queue_threshold_1000": queue_threshold(1000.0),
}

FIELDS = (
    "total_reward", "inductions", "waiting_accrued_passenger_hours",
    "waiting_time_mean_s", "waiting_time_p95_s", "left_behind_passengers",
    "left_behind_fraction", "unmet_demand_fraction", "energy_units", "cost_units",
    "service_level", "headway_violations", "trips_completed",
)


def build_env(network, process, scales, scenario: str, eval_seed: int) -> MetroEnv:
    config = SimulatorConfig(fleet_size=36, capacity_passengers=1000.0)
    return MetroEnv(
        network=network,
        demand_process=process,
        simulator_config=config,
        env_config=EnvConfig(arm="uq", scenario=scenario),
        forecast_provider=SeasonalNaiveProvider(network.n_stations),
        history_sampler=constant_history_sampler(process, config.day_start_hour),
        scales=scales,
        eval_seed=eval_seed,
    )


def run(env: MetroEnv, policy) -> dict:
    env.reset()
    while True:
        mask = env.action_masks()
        _, _, terminated, truncated, info = env.step(policy(env, mask))
        if terminated or truncated:
            return info["episode_summary"]


def main() -> None:
    network = Network.load()
    process = DemandProcess.load()
    scales = ReferenceScales.load(REPO_ROOT / "configs" / "reward_scales.json")

    results: dict[str, dict[str, dict[str, float]]] = {}
    for scenario in SCENARIOS:
        results[scenario] = {}
        for name, policy in POLICIES.items():
            per_seed = [
                run(build_env(network, process, scales, scenario, seed), policy)
                for seed in SEEDS
            ]
            results[scenario][name] = {
                field: float(np.mean([s[field] for s in per_seed])) for field in FIELDS
            }
            logger.info("%s / %s: reward %.1f", scenario, name,
                        results[scenario][name]["total_reward"])

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(SEEDS),
        "network_stations": network.n_stations,
        "reward_scales": {
            "ref_W": scales.ref_W, "ref_U": scales.ref_U,
            "ref_E": scales.ref_E, "ref_C": scales.ref_C,
            "provenance": scales.provenance,
        },
        "results": results,
        "disclaimer": (
            "Deterministic non-RL probe policies for environment validation only. These are NOT "
            "the pre-registered baselines and no result here is reportable."
        ),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "env_sanity_report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# Environment responsiveness report (M6)",
        "",
        "Deterministic, non-RL probe policies on the real network, averaged over "
        f"{len(SEEDS)} evaluation seeds. **These are not the pre-registered baselines and "
        "nothing here is a reportable result.** Their purpose is to establish that the "
        "environment responds to dispatch decisions before any PPO training begins.",
        "",
        f"Reward reference scales: ref_W={scales.ref_W:.2f} passenger-minutes/epoch, "
        f"ref_U={scales.ref_U:.2f} passengers/epoch, ref_E={scales.ref_E:.2f} energy units/epoch, "
        f"ref_C={scales.ref_C:.2f} cost units/epoch "
        f"(source policy `{scales.provenance.get('source_policy', 'unknown')}`).",
        "",
    ]
    for scenario in SCENARIOS:
        lines += [
            f"## Scenario `{scenario}`",
            "",
            "| policy | reward | inductions | wait (pass-h) | p95 wait (s) | left behind | "
            "left-behind frac | energy | cost | served |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name in POLICIES:
            row = results[scenario][name]
            lines.append(
                f"| `{name}` | {row['total_reward']:,.1f} | {row['inductions']:,.0f} | "
                f"{row['waiting_accrued_passenger_hours']:,.0f} | {row['waiting_time_p95_s']:,.0f} | "
                f"{row['left_behind_passengers']:,.0f} | {row['left_behind_fraction']:.4f} | "
                f"{row['energy_units']:,.0f} | {row['cost_units']:,.0f} | "
                f"{row['service_level']:.3f} |"
            )
        lines.append("")

    (RESULTS_DIR / "env_sanity_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
