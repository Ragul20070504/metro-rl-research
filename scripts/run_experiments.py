"""Experiment orchestration (M10-M12), staged and resumable.

Stages run in order and each is independently resumable, so an interruption costs only the cells
in flight::

    python scripts/run_experiments.py --stage prepare    # persist models, fit scaler, tune, calibrate
    python scripts/run_experiments.py --stage baselines  # non-RL methods, all scenarios
    python scripts/run_experiments.py --stage pilot      # R-1 power pilot (the gate before M11)
    python scripts/run_experiments.py --stage train      # PPO, all arms x all training seeds
    python scripts/run_experiments.py --stage sweep      # the full registered matrix
    python scripts/run_experiments.py --stage ablations  # M12

The **pilot is a gate, not a warm-up**. Protocol section 8.4 leaves the power analysis
`TBD@M11, frozen before the sweep`; the pilot supplies the between-seed standard deviation it
needs. It is used for nothing else: not for tuning, not for model selection, not for choosing
scenarios or seeds, and its policies are discarded rather than reused.
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.common.logging import get_logger
from src.common.paths import RESULTS_DIR, ensure_dir
from src.evaluation.runner import Cell, run_cells

logger = get_logger(__name__)

SCENARIOS = ("normal", "surge", "disruption", "combined")
EVAL_SEEDS = tuple(range(1, 31))

#: The registered training-seed pool (experiments/registry.yaml, `defaults.train_seeds`).
TRAIN_SEEDS = tuple(range(101, 111))


def train_seed_list(count: int) -> tuple[int, ...]:
    """The training seeds to use for a sample size of ``count`` (0 = the registered pool).

    If the R-1 power gate amends the sample size **upward**, the registered pool is extended
    contiguously rather than truncated. Slicing a fixed 10-seed tuple would silently run 10 seeds
    when the gate asked for more, which is exactly the kind of quiet design change the protocol
    forbids -- so the over-request is honoured and logged, not clipped.
    """
    if count <= 0:
        return TRAIN_SEEDS
    if count <= len(TRAIN_SEEDS):
        return TRAIN_SEEDS[:count]
    extended = tuple(range(TRAIN_SEEDS[0], TRAIN_SEEDS[0] + count))
    logger.warning(
        "the power gate asked for S = %d, above the registered pool of %d; extending the pool "
        "contiguously to seeds %d-%d (this must be covered by a recorded amendment)",
        count, len(TRAIN_SEEDS), extended[0], extended[-1],
    )
    return extended

NON_RL_METHODS = ("fixed_timetable", "heuristic_threshold", "milp_static", "milp_rolling")
RL_METHODS = ("ppo_point", "ppo_placebo", "ppo_uq", "ppo_uq_single_obj", "ppo_uq_noavail")

#: The pilot's only job is to estimate the between-training-seed SD of the primary endpoint.
PILOT_TRAIN_SEEDS = (901, 902, 903, 904)
PILOT_EVAL_SEEDS = tuple(range(41, 51))
PILOT_METHODS = ("ppo_uq", "ppo_point")

EXPERIMENT_FOR_METHOD = {
    "ppo_uq": "EXP-UQ-001", "ppo_point": "EXP-UQ-001", "ppo_placebo": "EXP-UQ-002",
    "ppo_uq_single_obj": "EXP-MO-001", "ppo_uq_noavail": "EXP-AVAIL-001",
    "fixed_timetable": "EXP-BASE-001", "heuristic_threshold": "EXP-BASE-001",
    "milp_static": "EXP-BASE-001", "milp_rolling": "EXP-BASE-005",
}


def workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


# --------------------------------------------------------------------------- prepare


def stage_prepare(args) -> None:
    """Everything the sweep depends on, in dependency order."""
    from pathlib import Path

    from src.common.paths import MODELS_DIR, REPO_ROOT

    # 1. Persist the fitted forecasters (src/forecasting/train.py evaluates and discards them).
    if not (REPO_ROOT / "models" / "forecasting" / "xgb_quantile.json").exists():
        logger.info("persisting forecasting models")
        from scripts.persist_forecast_models import main as persist

        persist()

    # 2. Freeze the observation scaler, using the trained forecaster (protocol section 6.1).
    if not (MODELS_DIR / "obs_scaler.json").exists():
        logger.info("fitting the observation scaler")
        from scripts.fit_scalers import fit_observation_scaler

        fit_observation_scaler(10, trained_forecaster=True)

    # 3. Recalibrate the MILP surrogate against the trained-forecaster environment.
    logger.info("calibrating the MILP surrogate")
    from src.baselines.milp import calibrate_surrogate

    calibrate_surrogate(trained_forecaster=True).save()

    # 4. Tune the heuristic thresholds on days disjoint from every scored pool.
    logger.info("tuning heuristic thresholds")
    from src.baselines.heuristic import tune_heuristic

    tune_heuristic(trained_forecaster=True).save()

    logger.info("prepare complete")


# --------------------------------------------------------------------------- baselines


def stage_baselines(args) -> None:
    cells = [
        Cell(method=method, scenario=scenario, eval_seed=seed,
             experiment_id=EXPERIMENT_FOR_METHOD[method])
        for method in NON_RL_METHODS
        for scenario in SCENARIOS
        for seed in EVAL_SEEDS
    ]
    run_cells(cells, workers=workers(), label="baselines")


# --------------------------------------------------------------------------- training


def _train_one(payload: tuple[str, int, str, bool]) -> str:
    from src.common.compute import single_threaded

    single_threaded()  # before torch is touched; see src/common/compute.py

    method, seed, weight_id, trained = payload
    from src.rl.train import train_policy

    return str(train_policy(method, seed, weight_id=weight_id, trained_forecaster=trained))


def _train_many(jobs: list[tuple[str, int, str, bool]], label: str) -> None:
    from src.rl.train import policy_path

    pending = [j for j in jobs if not policy_path(j[0], j[1], j[2]).exists()]
    logger.info("%s: %d policies, %d already trained, %d to train",
                label, len(jobs), len(jobs) - len(pending), len(pending))
    if not pending:
        return
    with ProcessPoolExecutor(max_workers=workers()) as pool:
        futures = {pool.submit(_train_one, job): job for job in pending}
        for index, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                future.result()
            except Exception as error:  # noqa: BLE001 - a failed policy is recorded, not fatal
                logger.error("training %s seed %s failed: %r", job[0], job[1], error)
            logger.info("%s: %d/%d", label, index, len(pending))


def stage_train(args) -> None:
    seeds = train_seed_list(args.train_seeds)
    jobs = [(method, seed, "w_balanced", True)
            for method in RL_METHODS for seed in seeds]
    _train_many(jobs, "train")


# --------------------------------------------------------------------------- pilot


def stage_pilot(args) -> None:
    """Estimate the between-seed SD of the primary endpoint. The gate before M11."""
    _train_many(
        [(method, seed, "w_balanced", True)
         for method in PILOT_METHODS for seed in PILOT_TRAIN_SEEDS],
        "pilot-train",
    )

    cells = [
        Cell(method=method, scenario="surge", eval_seed=eval_seed, train_seed=train_seed,
             variant="pilot", experiment_id="R-1-PILOT")
        for method in PILOT_METHODS
        for train_seed in PILOT_TRAIN_SEEDS
        for eval_seed in PILOT_EVAL_SEEDS
    ]
    run_cells(cells, workers=workers(), label="pilot")

    from scripts.power_analysis import analyse_pilot

    analyse_pilot()


# --------------------------------------------------------------------------- sweep


def sweep_cells(train_seeds: tuple[int, ...]) -> list[Cell]:
    """The full registered matrix: every RL arm at every training seed and scenario."""
    return [
        Cell(method=method, scenario=scenario, eval_seed=eval_seed, train_seed=train_seed,
             experiment_id=EXPERIMENT_FOR_METHOD[method])
        for method in RL_METHODS
        for scenario in SCENARIOS
        for train_seed in train_seeds
        for eval_seed in EVAL_SEEDS
    ]


def stage_sweep(args) -> None:
    from scripts.freeze_registry import require_frozen

    require_frozen("the M11 confirmatory sweep")
    run_cells(sweep_cells(train_seed_list(args.train_seeds)), workers=workers(), label="sweep")


# --------------------------------------------------------------------------- ablations


def ablation_seed_list(seeds: tuple[int, ...]) -> tuple[int, ...]:
    """Half the confirmatory pool, never fewer than three seeds."""
    return seeds[: max(3, len(seeds) // 2)]


def od_ablation_cells(seeds: tuple[int, ...]) -> list[Cell]:
    """EXP-SENS-002. The OD model is synthetic (assumption A03) and drives the primary endpoint,
    so the primary conclusion must hold under all three variants or be reported as OD-dependent.
    """
    ablation_seeds = ablation_seed_list(seeds)
    return [
        Cell(method=method, scenario="surge", eval_seed=eval_seed, train_seed=train_seed,
             variant=f"od_{od}", od_model=od, experiment_id="EXP-SENS-002")
        for od in ("uniform_downstream", "distance_decay", "cbd_attracted")
        for method in ("ppo_uq", "ppo_point", "ppo_placebo")
        for train_seed in ablation_seeds
        for eval_seed in EVAL_SEEDS[:10]
    ]


def weight_ablation_cells(seeds: tuple[int, ...]) -> list[Cell]:
    """EXP-SENS-001. Reward-weight sensitivity.

    Policies trained at ``w_balanced`` are *evaluated* under other weightings, which measures how
    the ranking moves with the objective rather than retraining five times over. The scoring
    weight therefore differs from the training weight, and ``train_weight_id`` keeps the
    checkpoint lookup pointed at the policies that actually exist.
    """
    ablation_seeds = ablation_seed_list(seeds)
    return [
        Cell(method=method, scenario=scenario, eval_seed=eval_seed, train_seed=train_seed,
             weight_id=weight_id, train_weight_id="w_balanced",
             variant=f"weights_{weight_id}", experiment_id="EXP-SENS-001")
        for weight_id in ("w_service", "w_cost", "w_unmet_heavy")
        for method in ("ppo_uq", "ppo_point")
        for scenario in ("normal", "surge")
        for train_seed in ablation_seeds
        for eval_seed in EVAL_SEEDS[:10]
    ]


def stage_ablations(args) -> None:
    """M12: reward-weight sensitivity and OD-model sensitivity.

    The uncertainty ablation (EXP-UQ-001/002), the single-objective ablation (EXP-MO-001) and the
    availability ablation (EXP-AVAIL-001) are already part of the main sweep, because those arms
    *are* registered methods. Nothing is retrained for them.
    """
    from scripts.freeze_registry import require_frozen

    require_frozen("the M12 ablations")
    seeds = train_seed_list(args.train_seeds)

    run_cells(od_ablation_cells(seeds), workers=workers(), label="ablation-od")
    run_cells(weight_ablation_cells(seeds), workers=workers(), label="ablation-weights")


# --------------------------------------------------------------------------- entry point


STAGES = {
    "prepare": stage_prepare,
    "baselines": stage_baselines,
    "pilot": stage_pilot,
    "train": stage_train,
    "sweep": stage_sweep,
    "ablations": stage_ablations,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGES) + ["all"], required=True)
    parser.add_argument("--train-seeds", type=int, default=0,
                        help="use only the first N training seeds (0 = all registered)")
    args = parser.parse_args()

    stages = list(STAGES) if args.stage == "all" else [args.stage]
    started = datetime.now(timezone.utc)
    for name in stages:
        logger.info("=== stage %s ===", name)
        STAGES[name](args)

    ensure_dir(RESULTS_DIR)
    (RESULTS_DIR / "last_run.json").write_text(json.dumps({
        "stages": stages,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
