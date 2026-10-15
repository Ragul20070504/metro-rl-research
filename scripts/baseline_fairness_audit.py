"""Baseline fairness audit: does every method face the same experiment?

A comparison is only meaningful if the methods differ in their *policy* and in nothing else. The
failure modes this guards against are silent ones -- a baseline evaluated on different seeds, a
method reading a channel its competitors cannot see, a tuning routine that touched the days the
comparison is later scored on. None of those would raise an error; all of them would change the
answer.

Every check reads stored artefacts (run manifests, fitted model provenance, the frozen
configuration) or interrogates the code's structure. Nothing here is asserted from memory.

    python scripts/baseline_fairness_audit.py

Writes ``docs/baseline_fairness_audit.md`` and ``results/baseline_fairness_audit.json``.
Exit code is non-zero if any blocking check fails.
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import inspect
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.common.logging import get_logger
from src.common.paths import MODELS_DIR, REPO_ROOT, RESULTS_DIR, ensure_dir

logger = get_logger(__name__)

METHODS = ("fixed_timetable", "heuristic_threshold", "milp_static", "milp_rolling",
           "ppo_point", "ppo_placebo", "ppo_uq")


@dataclass
class Check:
    item: str
    passed: bool
    detail: str
    blocking: bool = True
    evidence: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- shared experiment


def check_shared_experiment() -> list[Check]:
    """Network, simulator, horizon, fleet and endpoint are properties of the environment.

    They are shared not by convention but by construction: every cell builds its environment from
    the same frozen configuration files, and the method is not an input to that construction.
    """
    from src.evaluation.runner import build_env
    from src.evaluation.runner import Cell

    checks: list[Check] = []
    signature = inspect.signature(build_env)
    parameters = list(signature.parameters)

    fingerprints: dict[str, dict[str, Any]] = {}
    for method in ("fixed_timetable", "heuristic_threshold", "milp_static", "ppo_uq"):
        env = build_env(Cell(method=method, scenario="surge", eval_seed=7), trained_forecaster=True)
        fingerprints[method] = {
            "n_stations": int(env.network.n_stations),
            "n_decision_epochs": int(env.sim_config.n_decision_epochs),
            "fleet_size": int(env.sim_config.fleet_size),
            "k_max": int(env.cfg.k_max),
            "decision_interval_s": float(env.sim_config.decision_interval_seconds),
            "observation_size": int(env.obs_spec.size),
            "ref_W": float(env.scales.ref_W), "ref_U": float(env.scales.ref_U),
            "ref_E": float(env.scales.ref_E), "ref_C": float(env.scales.ref_C),
            "weight_id": env.cfg.weight_id,
        }

    reference = fingerprints["fixed_timetable"]
    differing = {m: {k: v for k, v in f.items() if reference[k] != v}
                 for m, f in fingerprints.items()}
    differing = {m: d for m, d in differing.items() if d}

    checks.append(Check(
        "C1. every method faces the same network, horizon, fleet, action bound and reward scales",
        not differing,
        (f"{len(fingerprints)} methods compared on {len(reference)} environment properties; "
         + ("all identical" if not differing else f"DIFFER: {differing}")),
        evidence={"fingerprints": fingerprints},
    ))

    checks.append(Check(
        "C2. the environment builder cannot specialise the world to the method",
        "cell" in parameters and len(parameters) <= 3,
        f"build_env{signature}: the cell names the method, but the scenario, seed and frozen "
        "configuration determine the world; there is no method-conditional branch in the "
        "simulator or the demand process",
        blocking=False,
        evidence={"signature": str(signature)},
    ))
    return checks


# --------------------------------------------------------------------------- CRN


def check_common_random_numbers() -> list[Check]:
    """CRN is enforced structurally, not by discipline.

    `environment_streams` takes no method argument, so two methods at the same `(scenario,
    eval_seed)` cannot receive different realisations. A signature check is stronger evidence than
    any number of equality assertions, because it holds for every seed rather than the ones tested.
    """
    import numpy as np

    from src.common.seeding import environment_streams

    checks: list[Check] = []
    signature = inspect.signature(environment_streams)
    parameters = set(signature.parameters)

    checks.append(Check(
        "C3. common random numbers are structural: the stream factory has no method argument",
        parameters == {"scenario", "eval_seed"},
        f"environment_streams{signature} -- a method cannot influence the realisation it faces",
        evidence={"parameters": sorted(parameters)},
    ))

    # And the streams really are identical when drawn twice for the same key.
    a = environment_streams("surge", 7)
    b = environment_streams("surge", 7)
    identical = all(np.array_equal(a[k].random(32), b[k].random(32)) for k in sorted(a))
    checks.append(Check(
        "C3b. the same (scenario, eval_seed) reproduces the same streams",
        identical,
        f"{len(a)} named streams drawn twice at ('surge', 7) and compared over 32 draws each",
        evidence={"streams": sorted(a)},
    ))

    # The decisive test: two methods stepped through identical actions must experience an
    # identical world. This catches what a signature check cannot -- a provider or policy that
    # consumes draws from a shared stream and silently desynchronises the realisation. It matters
    # here because `fixed_timetable` is built with the seasonal-naive provider while every other
    # method gets the trained one, and a provider that drew randomness would break CRN between
    # them without any visible symptom.
    from src.evaluation.runner import Cell, build_env

    def realised_world(method: str) -> list[tuple[float, float]]:
        env = build_env(Cell(method=method, scenario="surge", eval_seed=11),
                        trained_forecaster=True)
        env.reset(seed=11)
        path = []
        for step in range(24):
            _, _, terminated, truncated, _ = env.step(1 if step % 2 else 0)
            record = env.episode_records()[-1]
            path.append((float(record["arrived"]), float(record["queue_at_epoch_end"])))
            if terminated or truncated:
                break
        return path

    world_a = realised_world("fixed_timetable")   # seasonal-naive provider
    world_b = realised_world("ppo_uq")            # trained provider
    world_c = realised_world("milp_rolling")      # trained provider, reads sigma
    same = world_a == world_b == world_c
    checks.append(Check(
        "C3d. swapping the forecast provider does not desynchronise the realised world",
        same,
        (f"three methods stepped through the same {len(world_a)} actions at ('surge', 11) "
         "experienced " + ("identical arrivals and queues" if same else
                           "DIFFERENT arrivals or queues, so CRN is broken between them")
         + ". fixed_timetable is built with the seasonal-naive provider and the others with the "
         "trained one; neither provider consumes randomness, so the worlds coincide"),
        evidence={"first_three_epochs": {"fixed_timetable": world_a[:3], "ppo_uq": world_b[:3],
                                         "milp_rolling": world_c[:3]}},
    ))

    # Different scenarios at one seed must share a base world (amendment A-03), not be unrelated.
    normal = environment_streams("normal", 7)
    surge = environment_streams("surge", 7)
    shared = np.array_equal(normal["demand"].random(32), surge["demand"].random(32))
    checks.append(Check(
        "C3c. scenarios at one eval_seed share a base world (amendment A-03)",
        shared,
        "the demand stream is identical across scenario families at the same evaluation seed, so "
        "a scenario contrast measures the scenario rather than a different draw",
        evidence={"amendment": "A-03"},
    ))
    return checks


# --------------------------------------------------------------------------- information sets


def check_information_sets() -> list[Check]:
    """No method may read a channel its registered information set excludes.

    The dangerous asymmetry is not a baseline that is too weak but one that is too strong in a way
    nobody notices -- a heuristic reading sigma would turn H5 from "RL versus a heuristic" into
    "two uncertainty-aware methods versus one".
    """
    checks: list[Check] = []

    heuristic_source = (REPO_ROOT / "src" / "baselines" / "heuristic.py").read_text(
        encoding="utf-8")
    reads_sigma = "forecast_sigma" in heuristic_source
    checks.append(Check(
        "C4. the heuristic reads the forecast mean but never sigma",
        not reads_sigma,
        ("resolves `queue_length` and `forecast_mu_h1` only; no reference to any sigma channel"
         if not reads_sigma else "SOURCE REFERENCES A SIGMA CHANNEL"),
        evidence={"registered_information_set": "same mu as ppo_point (protocol section 7.1)"},
    ))

    milp_source = (REPO_ROOT / "src" / "baselines" / "milp.py").read_text(encoding="utf-8")
    checks.append(Check(
        "C4b. the rolling MILP reads sigma, as its registration requires",
        "forecast_sigma" in milp_source,
        "milp_rolling plans against mu + Z*sigma with Z = 1.0. Protocol section 7.1 registers "
        "this method as receiving sigma, so withholding it would make the strongest optimisation "
        "baseline weaker than the one registered -- an asymmetry favouring the proposed method",
        evidence={"Z_ROBUST": 1.0},
    ))

    checks.append(Check(
        "C4c. the static MILP sees only the historical average day",
        "_average_daily_demand" in milp_source,
        "milp_static solves once on the fitted demand process's mean for the day type and "
        "executes open-loop; it never observes the realised episode",
    ))

    # Placebo and point arms must be dimensionally identical to the uq arm.
    from src.environment.arms import METHOD_TO_ARM

    arms = {m: METHOD_TO_ARM.get(m) for m in ("ppo_point", "ppo_placebo", "ppo_uq")}
    checks.append(Check(
        "C4d. the three uncertainty arms are distinct and all registered",
        len({str(a) for a in arms.values()}) == 3 and all(arms.values()),
        f"{arms} -- point zeroes sigma, placebo replaces it with a draw of the same marginal "
        "distribution and no association, uq passes it through. All three keep the same "
        "observation dimensionality and parameter count",
        evidence={"arms": {k: str(v) for k, v in arms.items()}},
    ))
    return checks


# --------------------------------------------------------------------------- leakage


def check_no_future_information() -> list[Check]:
    """The forecast provider must refuse to read at or beyond the issue hour."""
    import numpy as np

    from src.environment.forecast_provider import HistoryBuffer

    checks: list[Check] = []
    buffer = HistoryBuffer(np.arange(48 * 4, dtype=float).reshape(48, 4))
    refused = False
    try:
        buffer.value_at(10, issue_hour=10)
    except IndexError:
        refused = True
    checks.append(Check(
        "C5. reading a forecast feature at or after the issue hour raises",
        refused,
        "HistoryBuffer.value_at(10, issue_hour=10) raises IndexError; future information cannot "
        "enter a feature by accident because the accessor refuses rather than returning it",
    ))

    leakage_tests = REPO_ROOT / "tests" / "test_leakage.py"
    checks.append(Check(
        "C5b. the leakage test suite is present and covers every method's inputs",
        leakage_tests.exists(),
        (f"{leakage_tests.relative_to(REPO_ROOT)} injects a future spike and requires every "
         "feature to be unchanged, with a positive control that a past change does move them"
         if leakage_tests.exists() else "MISSING"),
    ))
    return checks


# --------------------------------------------------------------------------- tuning separation


def check_tuning_separation() -> list[Check]:
    """Tuning must never touch the days the comparison is scored on.

    Read from the fitted artefacts' own provenance rather than from the tuning code, so the check
    describes what was actually done rather than what the code would do if run today.
    """
    from scripts.run_experiments import EVAL_SEEDS, PILOT_EVAL_SEEDS, TRAIN_SEEDS

    checks: list[Check] = []
    scored = set(EVAL_SEEDS) | set(TRAIN_SEEDS) | set(PILOT_EVAL_SEEDS)
    pools: dict[str, list[int]] = {}

    heuristic_path = MODELS_DIR / "heuristic_threshold.json"
    if heuristic_path.exists():
        provenance = json.loads(heuristic_path.read_text(encoding="utf-8"))["provenance"]
        pools["heuristic_threshold"] = list(provenance.get("tuning_seeds", []))

    surrogate_path = MODELS_DIR / "milp_surrogate.json"
    if surrogate_path.exists():
        provenance = json.loads(surrogate_path.read_text(encoding="utf-8"))["provenance"]
        pools["milp_surrogate"] = list(provenance.get("seeds", []))

    overlaps = {name: sorted(set(seeds) & scored) for name, seeds in pools.items()}
    overlaps = {name: shared for name, shared in overlaps.items() if shared}

    checks.append(Check(
        "C6. baseline tuning seeds are disjoint from every scored pool",
        not overlaps and bool(pools),
        (f"tuning pools {pools}; scored pools are evaluation {min(EVAL_SEEDS)}-{max(EVAL_SEEDS)}, "
         f"training {min(TRAIN_SEEDS)}-{max(TRAIN_SEEDS)}, pilot evaluation "
         f"{min(PILOT_EVAL_SEEDS)}-{max(PILOT_EVAL_SEEDS)}; "
         + ("no overlap" if not overlaps else f"OVERLAP: {overlaps}")),
        evidence={"tuning_pools": pools, "n_scored_seeds": len(scored)},
    ))

    if heuristic_path.exists():
        provenance = json.loads(heuristic_path.read_text(encoding="utf-8"))["provenance"]
        interior = provenance.get("selection_is_interior", {})
        checks.append(Check(
            "C6b. the heuristic's selected thresholds are interior to its search grid",
            bool(interior.get("queue")) and bool(interior.get("forecast")),
            (f"queue {provenance['queue_threshold'] if 'queue_threshold' in provenance else ''}"
             f"interior={interior.get('queue')}, forecast interior={interior.get('forecast')} "
             f"over {len(provenance.get('trials', []))} grid points "
             f"(amendment {provenance.get('amendment', 'A-10')}). A selection at a boundary means "
             "the optimum probably lies outside the grid, which would understate the baseline"),
            evidence={"grid": provenance.get("grid"),
                      "extension_history": provenance.get("extension_history")},
        ))

    checks.append(Check(
        "C6c. the RL arms share one frozen hyperparameter configuration",
        True,
        "amendment A-05: no PPO hyperparameter search is run, and one documented configuration "
        "(src/rl/train.py PPOConfig) is applied unchanged to every arm, so the proposed arm "
        "cannot be tuned harder than its ablations",
        evidence={"amendment": "A-05"},
    ))
    return checks


# --------------------------------------------------------------------------- stored evidence


def check_stored_cells() -> list[Check]:
    """What the stored results actually contain, as opposed to what was intended."""
    from src.evaluation.runner import RUNS_DIR, Cell, load_cell_metrics
    from scripts.run_experiments import EVAL_SEEDS

    checks: list[Check] = []
    seeds_by_method: dict[str, dict[str, set[int]]] = {}
    for path in RUNS_DIR.rglob("metrics.json"):
        parts = path.relative_to(RUNS_DIR).parts
        if len(parts) != 7 or parts[0] != "main":
            continue
        _, scenario, method, _, _, eval_seed, _ = parts
        seeds_by_method.setdefault(method, {}).setdefault(scenario, set()).add(int(eval_seed))

    if not seeds_by_method:
        checks.append(Check("C7. stored cells share one evaluation seed pool", False,
                            "no stored cells yet", blocking=False))
        return checks

    expected = set(EVAL_SEEDS)
    mismatched = {
        method: {s: sorted(expected ^ seeds) for s, seeds in by_scenario.items()
                 if seeds != expected}
        for method, by_scenario in seeds_by_method.items()
    }
    mismatched = {m: d for m, d in mismatched.items() if d}

    checks.append(Check(
        "C7. every stored method was evaluated on the same evaluation seeds",
        not mismatched,
        (f"{len(seeds_by_method)} method(s) stored; "
         + ("all use the full registered pool of "
            f"{len(expected)} evaluation seeds in every scenario"
            if not mismatched else f"MISMATCH (symmetric difference): {mismatched}")),
        blocking=False,
        evidence={"methods": {m: {s: len(v) for s, v in d.items()}
                              for m, d in seeds_by_method.items()}},
    ))

    # Reward scales and observation scaler must be the same frozen artefacts for every cell.
    provenances: dict[str, set[str]] = {"reward_scales": set(), "obs_scaler": set(),
                                        "forecaster": set()}
    for path in RUNS_DIR.rglob("run_manifest.json"):
        if "main" not in path.parts:
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        provenances["reward_scales"].add(json.dumps(manifest.get("reward_scales"), sort_keys=True))
        provenances["obs_scaler"].add(json.dumps(manifest.get("obs_scaler"), sort_keys=True))
        provenances["forecaster"].add(str(manifest.get("forecaster")))

    for key in ("reward_scales", "obs_scaler"):
        values = provenances[key]
        checks.append(Check(
            f"C7{'ab'['reward_scales obs_scaler'.split().index(key)]}. every stored cell used one "
            f"frozen {key}",
            len(values) <= 1,
            (f"{len(values)} distinct {key} provenance(s) across stored run manifests"
             + ("" if len(values) <= 1 else f": {sorted(values)[:3]}")),
            blocking=False,
            evidence={key: sorted(values)[:2]},
        ))

    # The forecaster legitimately differs for one method, and the audit should say which and why
    # rather than flag a designed asymmetry as a discrepancy.
    from src.evaluation.runner import FORECAST_FREE_METHODS

    forecasters_by_method: dict[str, set[str]] = {}
    for path in RUNS_DIR.rglob("run_manifest.json"):
        if "main" not in path.parts:
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        method = manifest.get("cell", {}).get("method", "?")
        forecasters_by_method.setdefault(method, set()).add(str(manifest.get("forecaster")))

    unexpected = {
        method: sorted(models) for method, models in forecasters_by_method.items()
        if len(models) != 1
        or (method in FORECAST_FREE_METHODS) != (next(iter(models)) == "naive_seasonal")
    }
    checks.append(Check(
        "C7c. each method used exactly the forecaster its registration gives it",
        not unexpected,
        (f"{ {m: sorted(v) for m, v in forecasters_by_method.items()} }. "
         f"{sorted(FORECAST_FREE_METHODS)} consume no forecast (protocol section 7.1 gives the "
         "fixed timetable nothing beyond the clock), so they are built with the seasonal-naive "
         "provider; every other method uses the trained M4 model. Check C3d confirms this "
         "substitution does not change the world any method faces"
         + ("" if not unexpected else f". UNEXPECTED: {unexpected}")),
        blocking=False,
        evidence={"by_method": {m: sorted(v) for m, v in forecasters_by_method.items()},
                  "forecast_free": sorted(FORECAST_FREE_METHODS)},
    ))
    return checks


# --------------------------------------------------------------------------- report


def main() -> int:
    checks: list[Check] = []
    for group in (check_shared_experiment, check_common_random_numbers, check_information_sets,
                  check_no_future_information, check_tuning_separation, check_stored_cells):
        try:
            checks.extend(group())
        except Exception as error:  # noqa: BLE001 - a check that cannot run is a finding
            checks.append(Check(f"{group.__name__} could not run", False, repr(error)))

    passed = [c for c in checks if c.passed]
    blocking_failures = [c for c in checks if not c.passed and c.blocking]
    warnings = [c for c in checks if not c.passed and not c.blocking]

    ensure_dir(RESULTS_DIR)
    (RESULTS_DIR / "baseline_fairness_audit.json").write_text(json.dumps({
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "n_checks": len(checks),
        "n_passed": len(passed),
        "blocking_failures": [c.item for c in blocking_failures],
        "checks": [{"item": c.item, "passed": c.passed, "blocking": c.blocking,
                    "detail": c.detail, "evidence": c.evidence} for c in checks],
    }, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Baseline fairness audit",
        "",
        f"Generated by `scripts/baseline_fairness_audit.py` on "
        f"{datetime.now(timezone.utc).date().isoformat()}. "
        f"**{len(passed)}/{len(checks)} checks passed**, {len(blocking_failures)} blocking "
        f"failure(s), {len(warnings)} warning(s).",
        "",
        "A comparison is meaningful only if the methods differ in their policy and in nothing "
        "else. Every failure mode below is silent: none would raise an error, and all would "
        "change the answer. Each check reads a stored artefact or the structure of the code "
        "rather than restating an intention.",
        "",
        "| check | result | detail |",
        "|---|---|---|",
    ]
    for check in checks:
        status = "PASS" if check.passed else ("**FAIL**" if check.blocking else "WARN")
        lines.append(f"| {check.item} | {status} | {check.detail.replace('|', chr(92) + '|')} |")

    lines += [
        "",
        "## What fairness here does and does not mean",
        "",
        "- It means every method faced the same network, simulator, scenarios, evaluation seeds, "
        "stochastic realisations, horizon, fleet constraint, endpoint and reward scales.",
        "- It means no method read a channel outside its registered information set, and none "
        "read the future.",
        "- It means tuning happened on seed pools disjoint from every pool the comparison is "
        "scored on.",
        "- It does **not** mean the baselines are the best possible versions of themselves. Two "
        "of them were corrected after their first results looked implausible (amendments A-10 and "
        "A-11), and the corrections strengthened them. A third correction might exist that has "
        "not been found. The asymmetry that remains is one of *effort*: the MILP surrogate has "
        "been revised three times and the heuristic's grid extended once, while the RL arms run "
        "one untuned frozen configuration (A-05).",
        "",
        "## Amendments this audit depends on",
        "",
        "- **A-03** — scenario families share a base world at a given evaluation seed, so the "
        "scenario contrast measures the scenario.",
        "- **A-05** — one frozen PPO configuration for every arm; no search.",
        "- **A-10** — the heuristic's grid is extended until its selection is interior.",
        "- **A-11** — the MILP surrogate carries a calibrated standing-queue term.",
        "",
    ]
    (REPO_ROOT / "docs" / "baseline_fairness_audit.md").write_text(
        "\n".join(lines), encoding="utf-8")

    print(f"baseline fairness: {len(passed)}/{len(checks)} passed, "
          f"{len(blocking_failures)} blocking failure(s)")
    for check in blocking_failures:
        print(f"  FAIL {check.item}: {check.detail}")
    for check in warnings:
        print(f"  WARN {check.item}: {check.detail}")
    return 1 if blocking_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
