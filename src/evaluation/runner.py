"""Experiment execution (M11): resumable, parallel, and self-documenting.

One *cell* is one ``(method, scenario, train_seed, eval_seed, weight_id, variant)`` combination.
Every cell writes its own directory containing the metrics, the per-epoch record and a run
manifest carrying the full traceability chain of protocol section 9.1.

Three properties matter more than speed here:

**Resumable.** A completed cell is never recomputed and never overwritten. An interrupted sweep
resumes by skipping what already has a valid manifest, so a crash halfway through several
thousand episodes costs only the cells that were in flight.

**Failures are recorded, not dropped.** A cell that raises writes a ``FAILED.json`` with the
traceback and keeps going. A sweep that silently omitted its failures would report a biased
subset and look cleaner than it was.

**Common random numbers hold across methods by construction.** The environment streams derive
from ``eval_seed`` alone, so every method in a given ``(scenario, eval_seed)`` cell faces the
identical demand, failure and surge realisation. Nothing in this module can perturb that, because
nothing here touches the environment's entropy.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.common.logging import get_logger
from src.common.paths import MODELS_DIR, REPO_ROOT, RESULTS_DIR, ensure_dir

logger = get_logger(__name__)

__all__ = ["Cell", "cell_dir", "run_cell", "run_cells", "load_cell_metrics"]

RUNS_DIR = RESULTS_DIR / "runs"


@dataclass(frozen=True)
class Cell:
    """One evaluated experimental unit.

    Attributes:
        method: A registered ``method_id`` from protocol section 7.1.
        scenario: One of the four frozen families.
        eval_seed: Drives the environment. Shared across methods -- this is the CRN pairing key.
        train_seed: The policy's seed; ``None`` for methods that are not trained.
        weight_id: Row of the frozen reward-weight grid.
        variant: Optional sensitivity label (e.g. an OD model), kept out of the main results tree.
        experiment_id: The registry entry this cell serves, recorded in the manifest.
        od_model: Override for the OD sensitivity sweep (EXP-SENS-002).
        day_type: ``weekday`` or ``weekend``.
        train_weight_id: The weighting the policy was *trained* under, when that differs from the
            weighting it is *scored* under. Only EXP-SENS-001 separates the two: it re-scores the
            ``w_balanced`` policies against other weightings instead of retraining, so the reward
            follows ``weight_id`` while the checkpoint must still be looked up under
            ``w_balanced``. ``None`` means the two coincide, which is the case everywhere else.
    """

    method: str
    scenario: str
    eval_seed: int
    train_seed: int | None = None
    weight_id: str = "w_balanced"
    variant: str = ""
    experiment_id: str = ""
    od_model: str | None = None
    day_type: str = "weekday"
    train_weight_id: str | None = None

    def key(self) -> str:
        seed = "na" if self.train_seed is None else str(self.train_seed)
        return f"{self.variant or 'main'}/{self.scenario}/{self.method}/{self.weight_id}/{seed}/{self.eval_seed}"

    def policy_weight(self) -> str:
        """Where the checkpoint lives. Never the scoring weight unless the two coincide."""
        return self.train_weight_id or self.weight_id


def cell_dir(cell: Cell) -> Path:
    return RUNS_DIR / cell.key()


def discover_cells() -> list[Cell]:
    """Every cell with a stored result, recovered from the directory tree.

    Scanning rather than re-enumerating the intended matrix means the analysis reports what was
    actually produced. If a sweep was interrupted, the gap shows up as a smaller ``n`` in the
    tables rather than as a silently-assumed complete design.
    """
    cells: list[Cell] = []
    if not RUNS_DIR.exists():
        return cells
    for metrics in RUNS_DIR.rglob("metrics.json"):
        parts = metrics.relative_to(RUNS_DIR).parts
        if len(parts) != 7:
            logger.warning("unexpected result path depth: %s", metrics)
            continue
        variant, scenario, method, weight_id, train_seed, eval_seed, _ = parts
        cells.append(Cell(
            method=method,
            scenario=scenario,
            eval_seed=int(eval_seed),
            train_seed=None if train_seed == "na" else int(train_seed),
            weight_id=weight_id,
            variant="" if variant == "main" else variant,
        ))
    logger.info("discovered %d stored cells", len(cells))
    return cells


def load_cell_metrics(cell: Cell) -> dict[str, Any] | None:
    path = cell_dir(cell) / "metrics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------- cached heavy artefacts


@lru_cache(maxsize=1)
def _network():
    from src.simulator.network import Network

    return Network.load()


@lru_cache(maxsize=1)
def _demand_process():
    from src.simulator.demand import DemandProcess

    return DemandProcess.load()


@lru_cache(maxsize=1)
def _reward_scales():
    from src.environment.reward import ReferenceScales

    return ReferenceScales.load(REPO_ROOT / "configs" / "reward_scales.json")


@lru_cache(maxsize=1)
def _obs_scaler():
    from src.environment.observation import ObservationScaler

    path = MODELS_DIR / "obs_scaler.json"
    if path.exists():
        return ObservationScaler.load(path)
    logger.warning("no frozen observation scaler at %s; running unnormalised", path)
    return None


@lru_cache(maxsize=2)
def _forecast_provider(kind: str):
    """Cached per process. Loading the quantile model is the expensive part of a cell."""
    from src.environment.forecast_provider import SeasonalNaiveProvider, load_trained_provider

    network = _network()
    if kind == "trained":
        return load_trained_provider(network)
    return SeasonalNaiveProvider(network.n_stations)


@lru_cache(maxsize=1)
def _fixed_timetable():
    from src.baselines.fixed_timetable import FixedTimetable

    return FixedTimetable.load()


@lru_cache(maxsize=32)
def _trained_policy(method: str, train_seed: int, weight_id: str):
    from sb3_contrib import MaskablePPO

    path = MODELS_DIR / "policies" / weight_id / method / str(train_seed) / "policy.zip"
    if not path.exists():
        raise FileNotFoundError(f"no trained policy at {path}")
    return MaskablePPO.load(path, device="cpu")


# --------------------------------------------------------------------- construction


#: Which methods consume the forecast at all. The fixed timetable does not (protocol section
#: 7.1 gives it nothing beyond the clock), so it never pays the cost of loading the model.
FORECAST_FREE_METHODS = frozenset({"fixed_timetable"})

#: Method -> the sigma arm its observation carries (protocol section 6.2).
from src.environment.arms import METHOD_TO_ARM  # noqa: E402


def build_env(cell: Cell, *, trained_forecaster: bool = True):
    from src.environment.metro_env import EnvConfig, MetroEnv, constant_history_sampler
    from src.simulator.engine import SimulatorConfig

    network, process = _network(), _demand_process()
    simulator_kwargs: dict[str, Any] = dict(fleet_size=36, capacity_passengers=1000.0)
    if cell.od_model:
        simulator_kwargs["od_model"] = cell.od_model
    config = SimulatorConfig(**simulator_kwargs)

    kind = "naive" if (cell.method in FORECAST_FREE_METHODS or not trained_forecaster) else "trained"
    provider = _forecast_provider(kind)

    env_config = EnvConfig(
        arm=METHOD_TO_ARM.get(cell.method, "uq"),
        scenario=cell.scenario,
        weight_id=cell.weight_id,
        day_type=cell.day_type,
        include_availability_channel=cell.method != "ppo_uq_noavail",
        strict_scaler=False,
    )
    return MetroEnv(
        network=network,
        demand_process=process,
        simulator_config=config,
        env_config=env_config,
        forecast_provider=provider,
        history_sampler=constant_history_sampler(process, config.day_start_hour),
        scales=_reward_scales(),
        scaler=_obs_scaler(),
        train_seed=cell.train_seed if cell.train_seed is not None else 101,
        eval_seed=cell.eval_seed,
    )


def build_policy(cell: Cell, env):
    """Instantiate the method under test."""
    from src.baselines.fixed_timetable import FixedTimetablePolicy
    from src.evaluation.policies import ConstantPolicy

    method = cell.method
    if method == "fixed_timetable":
        return FixedTimetablePolicy(_fixed_timetable())
    if method == "heuristic_threshold":
        from src.baselines.heuristic import HeuristicPolicy

        return HeuristicPolicy.load()
    if method in ("milp_static", "milp_rolling"):
        from src.baselines.milp import MILPPolicy

        return MILPPolicy(mode="static" if method == "milp_static" else "rolling")
    if method.startswith("ppo"):
        from src.rl.policies import TrainedPPOPolicy

        assert cell.train_seed is not None, "a PPO cell needs a train_seed"
        return TrainedPPOPolicy(
            method_id=method,
            model=_trained_policy(method, cell.train_seed, cell.policy_weight()),
        )
    if method.startswith("constant_"):
        return ConstantPolicy(int(method.split("_")[1]), method_id=method)
    raise ValueError(f"unknown method_id {method!r}")


# --------------------------------------------------------------------- provenance


@lru_cache(maxsize=1)
def _git_state() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                         text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                                             text=True).strip())
        return commit, dirty
    except Exception:
        return "unknown", True


@lru_cache(maxsize=1)
def _hardware() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }


# --------------------------------------------------------------------- execution


def run_cell(cell: Cell, *, overwrite: bool = False, trained_forecaster: bool = True) -> dict[str, Any]:
    """Evaluate one cell, writing metrics, per-epoch records and a run manifest.

    A cell that already has metrics is returned untouched unless ``overwrite`` is set. Successful
    results are never overwritten by a rerun of the sweep.
    """
    existing = load_cell_metrics(cell)
    if existing is not None and not overwrite:
        existing["_cached"] = True
        return existing

    # Also here, not only in the pool worker: `run_cells` falls back to running cells in the
    # parent process when the pool breaks, and a cell must not depend on which path executed it.
    from src.common.compute import single_threaded

    single_threaded()

    from src.evaluation.policies import run_episode

    directory = ensure_dir(cell_dir(cell))

    # A previous attempt's diagnosis is preserved, not overwritten: a cell that succeeds only on
    # retry must stay distinguishable from one that succeeded first time, and the traceback that
    # explains the first failure is the whole point of recording it.
    attempt = 1 + len(list(directory.glob("FAILED*.json")))
    if (directory / "FAILED.json").exists():
        (directory / "FAILED.json").replace(directory / f"FAILED_attempt{attempt - 1}.json")

    started = datetime.now(timezone.utc)
    commit, dirty = _git_state()

    try:
        env = build_env(cell, trained_forecaster=trained_forecaster)
        policy = build_policy(cell, env)
        result = run_episode(env, policy, eval_seed=cell.eval_seed)
    except Exception as error:  # noqa: BLE001 - a failed cell must be recorded, not raised away
        payload = {
            "cell": asdict(cell),
            "attempt": attempt,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "failed_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit,
        }
        (directory / "FAILED.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.error("cell %s FAILED (attempt %d): %r", cell.key(), attempt, error)
        return {"failed": True, **payload}

    finished = datetime.now(timezone.utc)
    latency = np.asarray(result.inference_latency_ms, dtype=float)

    metrics = dict(result.summary)
    metrics.update({
        "method": cell.method,
        "experiment_id": cell.experiment_id,
        "variant": cell.variant,
        "attempt": attempt,
        "mask_clip_rate": result.mask_clip_rate,
        "requested_inductions": int(sum(result.requested_actions)),
        "delivered_inductions": float(result.summary.get("inductions", 0.0)),
        "induction_success_rate": (
            float(result.summary.get("inductions", 0.0)) / max(1, sum(result.requested_actions))
        ),
        "inference_latency_mean_ms": float(latency.mean()) if latency.size else float("nan"),
        "inference_latency_p95_ms": float(np.quantile(latency, 0.95)) if latency.size else float("nan"),
        "wall_clock_s": (finished - started).total_seconds(),
    })

    (directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float), encoding="utf-8")

    try:
        import pandas as pd

        pd.DataFrame(result.epochs).to_parquet(directory / "epochs.parquet", index=False)
    except Exception as error:  # noqa: BLE001 - the metrics are the deliverable; logs are a bonus
        logger.warning("could not write per-epoch parquet for %s: %r", cell.key(), error)

    manifest = {
        "experiment_id": cell.experiment_id,
        "cell": asdict(cell),
        "git_commit": commit,
        "git_dirty": dirty,
        "provisional": dirty,
        "reward_scales": _reward_scales().provenance,
        "obs_scaler": (_obs_scaler().provenance if _obs_scaler() else
                       {"status": "identity", "note": "no frozen scaler present"}),
        "forecaster": getattr(_forecast_provider(
            "naive" if cell.method in FORECAST_FREE_METHODS or not trained_forecaster else "trained"
        ), "model_id", "unknown"),
        "hardware": _hardware(),
        "start_utc": started.isoformat(),
        "end_utc": finished.isoformat(),
        "wall_clock_s": (finished - started).total_seconds(),
    }
    (directory / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    return metrics


def _worker(args: tuple[Cell, bool, bool]) -> dict[str, Any]:
    from src.common.compute import single_threaded

    single_threaded()  # before torch is touched; see src/common/compute.py

    cell, overwrite, trained = args
    return run_cell(cell, overwrite=overwrite, trained_forecaster=trained)


def run_cells(
    cells: Iterable[Cell],
    *,
    workers: int | None = None,
    overwrite: bool = False,
    trained_forecaster: bool = True,
    label: str = "sweep",
) -> list[dict[str, Any]]:
    """Execute many cells, in parallel where the machine allows.

    Cells are independent by construction -- each builds its own environment from frozen
    artefacts and its own random streams -- so process-level parallelism cannot change any
    result, only the wall clock.
    """
    cells = list(cells)
    pending = [c for c in cells if overwrite or load_cell_metrics(c) is None]
    done = len(cells) - len(pending)
    logger.info("%s: %d cells, %d already complete, %d to run", label, len(cells), done, len(pending))

    results: list[dict[str, Any]] = [load_cell_metrics(c) for c in cells if load_cell_metrics(c)]
    if not pending:
        return results

    _execute(pending, workers=workers, overwrite=overwrite,
             trained_forecaster=trained_forecaster, label=label, results=results)

    # One recorded retry, sequentially. A cell can fail for reasons that have nothing to do with
    # the cell -- a transient solver timeout, memory pressure from fifteen concurrent workers --
    # and running the survivors alone distinguishes those from a genuine defect. Cells that fail
    # twice keep both tracebacks and are reported, never dropped.
    failed = [c for c in pending if load_cell_metrics(c) is None]
    if failed:
        logger.warning("%s: %d cell(s) failed; retrying each once, sequentially", label, len(failed))
        for index, cell in enumerate(failed, 1):
            results.append(run_cell(cell, trained_forecaster=trained_forecaster))
            logger.info("%s (retry): %d/%d", label, index, len(failed))
        still = [c for c in failed if load_cell_metrics(c) is None]
        recovered = len(failed) - len(still)
        logger.warning("%s: retry recovered %d of %d; %d cell(s) remain failed and are recorded "
                       "in FAILED.json, not discarded", label, recovered, len(failed), len(still))
        _record_failures(label, still)
    return results


def _record_failures(label: str, failed: list[Cell]) -> None:
    """Collect twice-failed cells into one file, so a sweep cannot end quietly incomplete."""
    if not failed:
        return
    path = RUNS_DIR / f"FAILED_{label}.json"
    path.write_text(json.dumps({
        "label": label,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "note": "these cells failed on both the first attempt and the retry",
        "cells": [asdict(c) for c in failed],
    }, indent=2), encoding="utf-8")
    logger.error("%s: wrote %s listing %d unrecovered cell(s)", label, path, len(failed))


def _execute(
    pending: list[Cell],
    *,
    workers: int | None,
    overwrite: bool,
    trained_forecaster: bool,
    label: str,
    results: list[dict[str, Any]],
) -> None:

    workers = workers or max(1, min(len(pending), (os.cpu_count() or 2) - 1))
    if workers == 1:
        for index, cell in enumerate(pending, 1):
            results.append(run_cell(cell, overwrite=overwrite, trained_forecaster=trained_forecaster))
            if index % 10 == 0 or index == len(pending):
                logger.info("%s: %d/%d", label, index, len(pending))
        return

    payload = [(cell, overwrite, trained_forecaster) for cell in pending]
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, item): item[0] for item in payload}
            for index, future in enumerate(as_completed(futures), 1):
                cell = futures[future]
                try:
                    results.append(future.result())
                except Exception as error:  # noqa: BLE001
                    logger.error("worker for %s raised: %r", cell.key(), error)
                    results.append({"failed": True, "cell": asdict(cell), "error": repr(error)})
                if index % 25 == 0 or index == len(pending):
                    logger.info("%s: %d/%d", label, index, len(pending))
    except BrokenProcessPool as error:
        # On Windows the pool dies if the calling module re-executes on import, and an
        # unattended sweep must not lose hours to that. Falling back to sequential execution is
        # slower but produces identical results -- cells are independent by construction.
        logger.error("process pool broke (%r); falling back to sequential execution", error)
        remaining = [c for c in pending if load_cell_metrics(c) is None]
        for index, cell in enumerate(remaining, 1):
            results.append(run_cell(cell, overwrite=overwrite,
                                    trained_forecaster=trained_forecaster))
            if index % 10 == 0 or index == len(remaining):
                logger.info("%s (sequential): %d/%d", label, index, len(remaining))
