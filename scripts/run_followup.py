"""FOLLOWUP-UQ-TRAINING — the registered training-budget grid.

Executes `docs/followup_training_budget_protocol.md`, which was written and fixed **before** this
script ran. Nothing here may be changed once execution begins; if the design must change, the
protocol is superseded by a new pre-registration first.

**This is not part of M0-M17.** The original confirmatory study is immutable and preserved at git
tag `original-confirmatory-study`. Separation is structural rather than editorial:

- policies are written to ``models/policies/budget_<N>/``, never to ``w_balanced/``;
- cells carry ``variant="budget_<N>"``, so they cannot satisfy the ``variant == "main"`` filter
  that every confirmatory and exploratory test applies;
- ``experiments/registry.yaml`` is not touched;
- the 60,000-step point **reuses the registered policies and their stored cells read-only**, so
  the original result is a datum inside the grid rather than something compared to it.

Only ``total_timesteps`` varies. Architecture, optimiser, every hyperparameter, reward, scaler,
forecaster, seeds, scenarios and evaluation days are identical to the registered study.

    python scripts/run_followup.py --stage train
    python scripts/run_followup.py --stage sweep
    python scripts/run_followup.py            # both, in order
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

from src.common.logging import get_logger, setup_logging
from src.common.paths import REPO_ROOT, RESULTS_DIR
from src.evaluation.runner import Cell, run_cells

from scripts.run_experiments import EVAL_SEEDS, SCENARIOS, workers

logger = get_logger(__name__)

EXPERIMENT_ID = "FOLLOWUP-UQ-TRAINING"

#: The grid, fixed by protocol section 3 and not extendable after seeing results.
BUDGETS: tuple[int, ...] = (15_000, 30_000, 60_000, 120_000, 240_000)

#: Trained by the registered study; reused read-only, never retrained.
REGISTERED_BUDGET = 60_000

ARMS: tuple[str, ...] = ("ppo_point", "ppo_uq")
TRAIN_SEEDS: tuple[int, ...] = tuple(range(101, 111))


def missing_inputs() -> list[str]:
    """Frozen inputs must be restored, never silently rebuilt from a newer data feed."""
    from src.rl.train import policy_path

    required = [REPO_ROOT / name for name in (
        "data/processed/network/network.json", "data/processed/network/demand_process.json",
        "models/forecasting/metadata.json", "models/forecasting/xgb_quantile.json",
        "models/obs_scaler.json", "configs/reward_scales.json",
    )]
    required.extend(path for arm in ARMS for seed in TRAIN_SEEDS
                    for path in (policy_path(arm, seed, "w_balanced"),
                                 policy_path(arm, seed, "w_balanced").with_name("metadata.json")))
    return [str(path.relative_to(REPO_ROOT)) for path in required if not path.is_file()]


def namespace(budget: int) -> str:
    """Where this budget's checkpoints live, and the variant its cells carry."""
    return "w_balanced" if budget == REGISTERED_BUDGET else f"budget_{budget}"


def variant(budget: int) -> str:
    """Cells always carry a budget variant, including the reused 60k point.

    The registered 60k cells live at ``variant='main'`` and are read from there by the analysis;
    this variant exists only so that any cell *this script* writes is unmistakably follow-up.
    """
    return f"budget_{budget}"


def _train_one(payload: tuple[str, int, int]) -> str:
    from src.common.compute import single_threaded

    single_threaded()  # before torch is touched; see src/common/compute.py

    method, train_seed, budget = payload
    from src.rl.train import PPOConfig, train_policy

    return str(train_policy(
        method, train_seed,
        weight_id="w_balanced",                       # the reward is unchanged
        config=PPOConfig(total_timesteps=budget),     # the only thing that varies
        store_as=namespace(budget),                   # keeps the registered policies untouched
        trained_forecaster=True,
    ))


def stage_train() -> None:
    from src.rl.train import policy_path

    jobs = [(method, seed, budget)
            for budget in BUDGETS if budget != REGISTERED_BUDGET
            for method in ARMS
            for seed in TRAIN_SEEDS]
    pending = [j for j in jobs
               if not policy_path(j[0], j[1], namespace(j[2])).exists()]
    logger.info("follow-up training: %d policies, %d already trained, %d to train "
                "(the %d-step point is reused from the registered study)",
                len(jobs), len(jobs) - len(pending), len(pending), REGISTERED_BUDGET)
    if not pending:
        return

    failed = []
    with ProcessPoolExecutor(max_workers=workers()) as pool:
        futures = {pool.submit(_train_one, job): job for job in pending}
        for index, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                future.result()
            except Exception as error:  # noqa: BLE001 - recorded, not fatal to the grid
                logger.error("training %s seed %s at %d steps failed: %r",
                             job[0], job[1], job[2], error)
                failed.append(job)
            logger.info("follow-up training: %d/%d", index, len(pending))
    errors = []
    for job in failed:
        try:
            _train_one(job)  # exactly one recorded retry, per protocol section 5
        except Exception as error:
            errors.append({"job": job, "error": repr(error)})
    if errors:
        (RESULTS_DIR / "followup_training_failures.json").write_text(
            json.dumps(errors, indent=2), encoding="utf-8")
        raise RuntimeError(f"{len(errors)} training jobs failed after retry; sweep not started")


def followup_cells() -> list[Cell]:
    """Every evaluation cell the grid needs, excluding the reused registered point."""
    return [
        Cell(method=method, scenario=scenario, eval_seed=eval_seed, train_seed=train_seed,
             weight_id="w_balanced",              # reward identical to the registered study
             train_weight_id=namespace(budget),   # checkpoint lookup follows the budget
             variant=variant(budget),             # structural separation from `main`
             experiment_id=EXPERIMENT_ID)
        for budget in BUDGETS if budget != REGISTERED_BUDGET
        for method in ARMS
        for train_seed in TRAIN_SEEDS
        for scenario in SCENARIOS
        for eval_seed in EVAL_SEEDS
    ]


def stage_sweep() -> None:
    cells = followup_cells()
    logger.info("follow-up sweep: %d cells across %d budgets", cells and len(cells),
                len(BUDGETS) - 1)
    run_cells(cells, workers=workers(), label="followup-sweep")
    from src.evaluation.runner import load_cell_metrics
    missing = [cell.key() for cell in cells if load_cell_metrics(cell) is None]
    if missing:
        raise RuntimeError(f"{len(missing)} evaluation cells still missing; grid is incomplete")


def _record(stages: list[str], started: str) -> None:
    (RESULTS_DIR / "followup_last_run.json").write_text(json.dumps({
        "study": EXPERIMENT_ID,
        "protocol": "docs/followup_training_budget_protocol.md",
        "budgets": list(BUDGETS),
        "arms": list(ARMS),
        "train_seeds": list(TRAIN_SEEDS),
        "stages": stages,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "separation": ("policies under models/policies/budget_<N>/; cells under "
                       "variant='budget_<N>'; the registered study is untouched"),
    }, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "sweep"), default=None,
                        help="omit to run training then the sweep, in order")
    parser.add_argument("--check", action="store_true", help="inspect prerequisites without running experiments")
    args = parser.parse_args()

    setup_logging(log_file="run_followup.log")
    started = datetime.now(timezone.utc).isoformat()
    missing = missing_inputs()
    if args.check or missing:
        print(json.dumps({"ready": not missing, "missing_inputs": missing}, indent=2))
        if missing:
            return 1
        if args.check:
            return 0

    stages = [args.stage] if args.stage else ["train", "sweep"]
    logger.info("FOLLOWUP-UQ-TRAINING: budgets %s, arms %s, seeds %d-%d, stages %s",
                BUDGETS, ARMS, TRAIN_SEEDS[0], TRAIN_SEEDS[-1], stages)
    for stage in stages:
        {"train": stage_train, "sweep": stage_sweep}[stage]()
    _record(stages, started)
    logger.info("follow-up %s complete", "+".join(stages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
