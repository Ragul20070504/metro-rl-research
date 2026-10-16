"""Isolated, resumable independent replication; see docs/replication_20261015_protocol.md.

This orchestrator never writes the original study's data, models or results. Its environment
overrides are installed before importing any module that binds repository paths.
"""
from __future__ import annotations

try:
    import _path  # noqa: F401
except ModuleNotFoundError:
    pass

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "replication" / "20261015"
PROTOCOL = ROOT / "docs" / "replication_20261015_protocol.md"
BUDGETS = (15_000, 30_000, 60_000, 120_000, 240_000)
SEEDS = tuple(range(101, 111))
SCENARIOS = ("normal", "surge", "disruption", "combined")
BASELINES = ("fixed_timetable", "heuristic_threshold", "milp_static", "milp_rolling")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    def safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None  # undefined diagnostics, never a fabricated numeric result
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(safe(data), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def status(stage: str, **detail) -> None:
    report = {"study": "REPLICATION-20261015", "stage": stage,
              "updated_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), **detail}
    write_json(ROOT / "results" / "replication_20261015_status.json", report)
    print(json.dumps(report), flush=True)


def configure() -> None:
    for key, value in {"METRORL_DATA_DIR": BASE / "data", "METRORL_MODELS_DIR": BASE / "models",
                       "METRORL_RESULTS_DIR": BASE / "results", "METRORL_LOGS_DIR": BASE / "logs"}.items():
        os.environ[key] = str(value)
        value.mkdir(parents=True, exist_ok=True)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["MPLBACKEND"] = "Agg"


def namespace(budget: int) -> str:
    return "w_balanced" if budget == 60_000 else f"budget_{budget}"


def jobs() -> list[tuple[str, int, int]]:
    return [(arm, seed, budget) for budget in BUDGETS
            for arm in ("ppo_point", "ppo_uq") for seed in SEEDS] + [
                ("ppo_placebo", seed, 60_000) for seed in SEEDS]


def checkpoint(job) -> Path:
    arm, seed, budget = job
    return BASE / "models" / "policies" / namespace(budget) / arm / str(seed) / "policy.zip"


def checkpoint_complete(job) -> bool:
    from src.rl.train import PPOConfig

    path = checkpoint(job)
    if not path.exists():
        return False
    metadata = json.loads(path.with_name("metadata.json").read_text())
    if metadata["config"] != PPOConfig(total_timesteps=job[2]).to_dict():
        raise ValueError(f"Checkpoint config mismatch: {path}")
    if metadata["method"] != job[0] or metadata["train_seed"] != job[1]:
        raise ValueError(f"Checkpoint identity mismatch: {path}")
    if not path.with_name(metadata["learning_curve"]).is_file():
        raise ValueError(f"Checkpoint has no learning curve: {path}")
    return True


def computation_files() -> list[Path]:
    return sorted(path for path in [*ROOT.glob("src/**/*.py"), *ROOT.glob("configs/*"),
                   *(ROOT / "scripts" / name for name in ("run_replication.py", "run_followup.py",
                     "run_experiments.py", "analyse_followup.py")), PROTOCOL,
                   ROOT / "requirements-linux.lock.txt"] if path.is_file())


def verify_frozen() -> dict:
    manifest = json.loads((BASE / "frozen_inputs.json").read_text())
    if digest(BASE / "computation-source.tar.gz") != manifest["source_archive_sha256"]:
        raise ValueError("Frozen computation archive is missing or changed")
    for relative, expected in manifest["input_hashes"].items():
        if digest(BASE / relative) != expected:
            raise ValueError(f"Frozen input changed: {relative}")
    for relative, expected in manifest["computation_hashes"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Frozen computation changed: {relative}")
    return manifest


def prepare() -> None:
    import numpy as np
    import pandas as pd
    from src.simulator.build_network import build_network
    from src.simulator.demand import fit_demand_process
    from src.baselines.fixed_timetable import build_fixed_timetable
    from src.forecasting.models import XGBQuantileForecaster
    from src.forecasting.metrics import evaluate_probabilistic, rmse

    if (BASE / "frozen_inputs.json").exists():
        verify_frozen()
        status("PREPARED", note="Frozen inputs verified; no refit")
        return
    status("PREPARING")
    original_raw = ROOT / "data" / "raw"
    raw_manifest = []
    for manifest_path in sorted(original_raw.glob("*/MANIFEST.json")):
        manifest = json.loads(manifest_path.read_text())
        for item in manifest["files"]:
            relative = Path(item["path"].replace("\\", "/"))
            source = original_raw / relative
            exact = source.is_file()
            if not exact:
                source = original_raw / ".recovery-quarantine" / relative
            if not source.is_file():
                raise FileNotFoundError(f"No recovered or quarantined publisher file: {relative}")
            if exact and digest(source) != item["sha256"]:
                raise ValueError(f"Recovered file no longer matches original hash: {relative}")
            target = BASE / "data" / "raw" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            raw_manifest.append({"path": str(relative), "sha256": digest(target),
                                 "original_sha256": item["sha256"], "exact_original": exact,
                                 "source_url": manifest["source_url"]})

    reference = ROOT / "data" / "processed"
    processed = BASE / "data" / "processed"
    (processed / "network").mkdir(parents=True, exist_ok=True)
    for name in ("supervised_train.parquet", "supervised_val.parquet", "demand_train.parquet"):
        shutil.copy2(reference / "recovery" / name, processed / name)
    shutil.copy2(reference / "network" / "station_order.json", processed / "network" / "station_order.json")
    for name in ("obs_scaler.json", "heuristic_threshold.json", "milp_surrogate.json"):
        shutil.copy2(ROOT / "models" / name, BASE / "models" / name)
    network = build_network()
    expected_order = json.loads((reference / "network" / "station_order.json").read_text())["station_complex_to_index"]
    if list(network.station_ids) != sorted(expected_order, key=expected_order.get):
        raise ValueError("New GTFS does not preserve the specified 24-station ordering")
    network.save()
    demand = pd.read_parquet(processed / "demand_train.parquet")
    fit_demand_process(demand, network.station_ids, split="train").save()
    build_fixed_timetable().save()

    fc = json.loads((ROOT / "results" / "forecasting_metrics.json").read_text())
    train = pd.read_parquet(processed / "supervised_train.parquet")
    val = pd.read_parquet(processed / "supervised_val.parquet")
    columns = fc["feature_columns"]
    if len(train) != fc["rows"]["train"] or len(val) != fc["rows"]["val"]:
        raise ValueError("Reconstructed train/validation row counts disagree with the frozen reference")
    status("FITTING_FORECASTER", train_rows=len(train), validation_rows=len(val))
    started = time.monotonic()
    model = XGBQuantileForecaster(**fc["selected_hyperparameters"]["xgb_quantile"])
    model.fit(train[columns], train.target.to_numpy(float))
    model.calibration_factor = fc["calibration"]["factor"]
    prediction = model.predict(val[columns])
    metrics = evaluate_probabilistic(val.target.to_numpy(float), prediction.quantiles).to_dict()
    measured = {"rmse": rmse(val.target.to_numpy(float), prediction.mean),
                "pinball_mean": metrics["pinball_mean"], "picp_80": metrics["picp_80"]}
    old = fc["metrics"]["xgb_quantile"]["val"]
    agreement = {"rmse": old["point"]["rmse"],
                 "pinball_mean": old["probabilistic"]["pinball_mean"],
                 "picp_80": old["probabilistic"]["picp_80"]}
    if not 0.76 <= measured["picp_80"] <= 0.84 or measured["pinball_mean"] >= fc["gate"]["naive_pinball_val"]:
        raise ValueError(f"Frozen forecaster failed its original validation gate: {measured}")
    directory = BASE / "models" / "forecasting"
    directory.mkdir(exist_ok=True)
    model._model.save_model(directory / "xgb_quantile.json")
    metadata = json.loads((ROOT / "models" / "forecasting" / "metadata.json").read_text())
    metadata["reference_run_git_commit"] = metadata.pop("source_run_git_commit", None)
    metadata.update(source="Independent replication refit; not a recovered original binary",
                    validation_metrics=measured, original_validation_metrics=agreement,
                    maximum_metric_disagreement=max(abs(measured[k] - agreement[k]) for k in measured),
                    refit_wall_seconds=time.monotonic() - started)
    write_json(directory / "metadata.json", metadata)
    inputs = [path for directory in (BASE / "data", BASE / "models")
              for path in directory.rglob("*") if path.is_file()]
    code = computation_files()
    archive = BASE / "computation-source.tar.gz"
    with tarfile.open(archive, "w:gz") as package:
        for path in code:
            package.add(path, arcname=str(path.relative_to(ROOT)))
    manifest = {"study": "REPLICATION-20261015", "frozen_utc": datetime.now(timezone.utc).isoformat(),
                "protocol_sha256": digest(PROTOCOL), "raw_files": raw_manifest,
                "input_hashes": {str(path.relative_to(BASE)): digest(path) for path in sorted(inputs)},
                "computation_hashes": {str(path.relative_to(ROOT)): digest(path) for path in code},
                "source_archive_sha256": digest(archive), "platform": platform.platform(),
                "python": platform.python_version(), "forecaster_validation": metadata["validation_metrics"],
                "original_metric_disagreement": metadata["maximum_metric_disagreement"]}
    write_json(BASE / "frozen_inputs.json", manifest)
    write_json(ROOT / "results" / "replication_20261015_inputs.json", manifest)
    status("PREPARED", input_files=len(inputs), original_metric_disagreement=manifest["original_metric_disagreement"])


def train_one(job) -> str:
    from src.common.compute import single_threaded
    single_threaded()
    from src.rl.train import PPOConfig, train_policy
    arm, seed, budget = job
    return str(train_policy(arm, seed, config=PPOConfig(total_timesteps=budget),
                            store_as=namespace(budget), record_learning_curve=True))


def smoke() -> None:
    """A disjoint software-check policy; no evaluation cell or hypothesis result is produced."""
    verify_frozen()
    from src.common.compute import single_threaded
    single_threaded()
    from src.rl.train import PPOConfig, train_policy
    started = time.monotonic()
    target = train_policy("ppo_point", 50_000, config=PPOConfig(total_timesteps=456),
                          store_as="software_smoke", record_learning_curve=True)
    metadata = json.loads(target.with_name("metadata.json").read_text())
    status("SMOKE_PASSED", actual_timesteps=metadata["actual_timesteps"],
           training_seconds=metadata["wall_clock_s"], total_seconds=time.monotonic() - started,
           note="Software validation only; disjoint seed, no scientific evaluation cell")


def cells():
    from src.evaluation.runner import Cell
    matrix = [Cell(method=arm, scenario=scenario, eval_seed=evaluation, train_seed=seed,
                   train_weight_id=namespace(budget), variant="" if budget == 60_000 else f"budget_{budget}",
                   experiment_id="REPLICATION-20261015")
              for arm, seed, budget in jobs() for scenario in SCENARIOS for evaluation in range(1, 31)]
    matrix.extend(Cell(method=method, scenario=scenario, eval_seed=evaluation,
                       experiment_id="REPLICATION-20261015-BASELINE")
                  for method in BASELINES for scenario in SCENARIOS for evaluation in range(1, 31))
    assert len(matrix) == 13_680
    return matrix


def run(workers: int) -> None:
    verify_frozen()
    pending = [job for job in jobs() if not checkpoint_complete(job)]
    status("TRAINING", completed=110 - len(pending), total=110, workers=workers)
    failed = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(train_one, job): job for job in pending}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception:
                failed.append({"job": futures[future], "first_error": traceback.format_exc()})
            status("TRAINING", completed=110 - len(pending) + index - len(failed),
                   failures=len(failed), total=110, workers=workers)
    unrecovered = []
    for failure in failed:
        try:
            train_one(failure["job"])
        except Exception:
            failure["retry_error"] = traceback.format_exc()
            unrecovered.append(failure)
    write_json(BASE / "results" / "training_failures.json", {"attempts": failed})
    if unrecovered or not all(checkpoint_complete(job) for job in jobs()):
        raise RuntimeError("Training incomplete after one retry; no evaluation or inference started")
    verify_frozen()
    from src.evaluation.runner import run_cells, load_cell_metrics, cell_dir
    matrix = cells()
    status("EVALUATING", completed=sum(load_cell_metrics(cell) is not None for cell in matrix), total=len(matrix))
    run_cells(matrix, workers=workers, label="independent-replication")
    missing = [cell.key() for cell in matrix if load_cell_metrics(cell) is None
               or not (cell_dir(cell) / "run_manifest.json").is_file()]
    if missing:
        raise RuntimeError(f"{len(missing)} cells lack complete results/manifests; inference withheld")
    status("EVALUATED", completed=len(matrix), total=len(matrix))


def analyze() -> None:
    import numpy as np
    import statsmodels.formula.api as smf
    from scripts.analyse_followup import budget_coverage, _budget_frame, _paired, _per_budget
    from src.evaluation.aggregate import load_frame, seed_level
    from src.evaluation.runner import discover_cells, cell_dir
    from src.evaluation.statistics import holm, benjamini_hochberg, paired_analysis
    verify_frozen()
    if not all(checkpoint_complete(job) for job in jobs()):
        raise RuntimeError("Incomplete policies; inference withheld")
    discovered = discover_cells()
    if {cell.key() for cell in discovered} != {cell.key() for cell in cells()}:
        raise RuntimeError("Evaluation matrix is incomplete or contains unexpected cells")
    if not all((cell_dir(cell) / "run_manifest.json").is_file() for cell in discovered):
        raise RuntimeError("Evaluation manifests missing; inference withheld")
    episodes = load_frame(discovered)
    if not np.isfinite(episodes.left_behind_fraction).all() or not episodes.left_behind_fraction.between(0, 1).all():
        raise RuntimeError("Invalid endpoint values; inference withheld")
    if not all(row["complete"] for row in budget_coverage(episodes).values()):
        raise RuntimeError("Budget grid incomplete; inference withheld")
    frame = seed_level(episodes)
    per_budget = []
    long = []
    for budget in BUDGETS:
        paired = _paired(_budget_frame(frame, budget), "surge")
        per_budget.append(_per_budget(paired, budget, "surge"))
        long.extend({"seed": int(seed), "log2_budget": float(np.log2(budget / 60_000)),
                     "difference": float(row.ppo_uq - row.ppo_point)} for seed, row in paired.iterrows())
    for row, p in zip(per_budget, holm([row["p_value"] for row in per_budget])):
        row["p_adjusted"] = p
    import pandas as pd
    data = pd.DataFrame(long)
    model = smf.mixedlm("difference ~ log2_budget", data, groups=data.seed).fit()
    slope, se, p = (float(series["log2_budget"]) for series in (model.params, model.bse, model.pvalues))
    if not model.converged or not np.isfinite([slope, se, p]).all() or se <= 0:
        raise RuntimeError("Interaction fit invalid or not converged; inference withheld")
    secondary = []
    main = frame[(frame.variant == "main") & (frame.scenario == "surge")]
    pivot = main[main.train_seed.isin(SEEDS)].pivot(index="train_seed", columns="method", values="left_behind_fraction")
    for comparator in ("ppo_point", "ppo_placebo"):
        secondary.append(paired_analysis(pivot.ppo_uq, pivot[comparator], label="Independent replication at 60k",
                         metric="left_behind_fraction", method_a="ppo_uq", method_b=comparator).to_dict())
    for row, adjusted in zip(secondary, holm([row["p_value"] for row in secondary])):
        row.update(p_adjusted=adjusted, family="Independent replication 60k, Holm across two contrasts")
        row.pop("cliffs_delta", None)
    scenario = [_per_budget(_paired(_budget_frame(frame, budget), family), budget, family)
                for budget in BUDGETS for family in SCENARIOS]
    for row, adjusted in zip(scenario, benjamini_hochberg([row["p_value"] for row in scenario])):
        row["p_adjusted"] = adjusted
    report = {"study": "REPLICATION-20261015", "evidence_class": "INDEPENDENT REPLICATION / EXPLORATORY",
              "generated_utc": datetime.now(timezone.utc).isoformat(), "complete": True,
              "policies": 110, "evaluation_cells": len(episodes), "primary_budget_interaction": {
                  "slope_per_budget_doubling": slope, "standard_error": se, "p_value": p,
                  "ci_low": slope - 1.96 * se, "ci_high": slope + 1.96 * se,
                  "converged": True, "n_seeds": 10, "n_observations": 50},
              "per_budget_surge_holm": per_budget, "secondary_60k_holm": secondary,
              "scenario_contrasts_bh": scenario,
              "original_study_unchanged": "Original statistical_analysis.json is separate; no replication number replaces it."}
    write_json(BASE / "results" / "analysis.json", report)
    write_json(ROOT / "results" / "replication_20261015_analysis.json", report)
    summary = ["# Independent replication and training-budget results", "",
               "**New simulation evidence; the historical study is unchanged.**", "",
               "## Design and scope", "",
               "Protocol: `docs/replication_20261015_protocol.md`. All 110 terminal policies and "
               "13,680 cells completed before inference. Ten training seeds, 30 evaluation seeds "
               "and four scenarios were retained without tuning or exclusion. All 60k policies "
               "were trained anew. This is a prospectively specified local replication, not an "
               "externally registered confirmatory study or operational validation.", "",
               "Full input/code hashes, raw cells and learning curves are retained under "
               "`replication/20261015/`. The new snapshot uses 113 exact recovered source files "
               "and 11 explicitly updated publisher files. The inherited scales, heuristic and "
               "MILP settings were not retuned. The placebo bootstraps a changing empirical pool "
               "with replacement; it is not an exact permutation or proven information-free "
               "control. Temporal/distributional effects remain confounded with information "
               "content. Original historical provenance limitations remain.", "",
               "## Primary budget interaction", "",
               f"Slope per budget doubling: **{slope:+.6f}**, 95% Wald CI "
               f"[{slope - 1.96 * se:+.6f}, {slope + 1.96 * se:+.6f}], p = **{p:.6g}** "
               "(mixed model with a training-seed random intercept; 50 observations, 10 seeds).", "",
               "No budget dependence was detected; this does not establish independence or equivalence."
               if p >= 0.05 else
               "The UQ-minus-point contrast decreases with budget." if slope < 0 else
               "The UQ-minus-point contrast increases with budget.", "",
               "## Every budget under surge", "",
               "Positive differences mean a larger left-behind fraction for uncertainty-aware PPO. "
               "BCa intervals estimate the paired Hodges–Lehmann location, not the mean difference.", "",
               "| Requested steps | n | Mean UQ | Mean point | Mean difference | HL | 95% BCa CI | Holm p (five tests) |",
               "|---:|---:|---:|---:|---:|---:|---|---:|"]
    summary.extend(f"| {row['budget']} | {row['n']} | {row['mean_ppo_uq']:.6f} | "
                   f"{row['mean_ppo_point']:.6f} | {row['mean_difference']:+.6f} | "
                   f"{row['hodges_lehmann']:+.6f} | [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] | "
                   f"{row['p_adjusted']:.6g} |" for row in per_budget)
    summary += ["", "## Separate 60k replication family", "",
                "This two-test family is distinct from the five-budget family; both adjustments "
                "are reported without selecting the smaller p-value.", "",
                "| Comparison | n | Mean difference | HL | 95% BCa CI | Holm p (two tests) |",
                "|---|---:|---:|---:|---|---:|"]
    summary.extend(f"| UQ vs {row['method_b']} | {row['n']} | {row['mean_difference']:+.6f} | "
                   f"{row['hodges_lehmann']:+.6f} | [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] | "
                   f"{row['p_adjusted']:.6g} |" for row in secondary)
    ranking = frame[frame.variant == "main"].pivot_table(
        index="method", columns="scenario", values="left_behind_fraction", aggfunc="mean")
    summary += ["", "## Descriptive baseline comparison at 60k", "",
                "Lower is better. Deterministic baselines have no training-seed replication, "
                "so this table does not supply an inferential RL-versus-baseline test.", "",
                "| Method | Normal | Surge | Disruption | Combined |", "|---|---:|---:|---:|---:|"]
    summary.extend("| " + method + " | " + " | ".join(f"{row[family]:.6f}" for family in SCENARIOS) + " |"
                   for method, row in ranking.sort_values("surge").iterrows())
    summary += ["", "## Diagnostics and limitations", "",
                "Every training run retains its Monitor learning curve and actual timestep count. "
                "Requested budgets round to complete 456-step rollouts; no checkpoint was selected "
                "from these curves. Neither a flat diagnostic curve nor a nonsignificant interaction "
                "establishes convergence, optimality or budget independence.", "",
                "The design remains underpowered for a SESOI of 0.002 at n=10. Crowding, OD, "
                "capacity and other simulator assumptions limit external validity. No physical "
                "passenger, energy, financial, deployment or general algorithmic claim follows.", "",
                "All 20 budget-by-scenario exploratory contrasts and their BH adjustments are "
                "reported in `results/replication_20261015_analysis.json`. Author details and "
                "declarations remain placeholders. This computational report requires scientific "
                "review and integration before journal submission.", ""]
    text = "\n".join(summary)
    output = ROOT / "docs" / "replication_20261015_results.md"
    output.write_text(text)
    for suffix in ("docx", "pdf"):
        command = ["pandoc", "--from=gfm", "--standalone", "-V", "geometry:margin=1in",
                   "-V", "mainfont=DejaVu Serif", "--output=" + str(output.with_suffix("." + suffix))]
        if suffix == "pdf":
            command.append("--pdf-engine=xelatex")
        subprocess.run(command, input=text, text=True, cwd=ROOT, check=True)
    status("ANALYSIS_COMPLETE", policies=110, cells=len(episodes),
           note="Results require scientific review and manuscript integration; not automatic submission readiness")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "smoke", "run", "analyze", "all"), default="all")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    multiprocessing.set_start_method("spawn", force=True)
    configure()
    from src.common.logging import setup_logging
    setup_logging(log_file=BASE / "logs" / "orchestrator.log")
    try:
        if args.stage in ("prepare", "all"):
            # Isolate XGBoost's fit process from the later spawned policy workers.
            if args.stage == "all":
                subprocess.run([os.sys.executable, __file__, "--stage", "prepare"], cwd=ROOT, check=True)
            else:
                prepare()
        if args.stage in ("smoke", "all"):
            if args.stage == "all":
                subprocess.run([os.sys.executable, __file__, "--stage", "smoke"], cwd=ROOT, check=True)
            else:
                smoke()
        if args.stage in ("run", "all"):
            run(args.workers)
        if args.stage in ("analyze", "all"):
            analyze()
    except Exception:
        status("FAILED", error=traceback.format_exc(), note="No incomplete-grid inference is valid")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
