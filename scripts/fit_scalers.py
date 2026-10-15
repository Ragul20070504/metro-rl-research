"""Fit and freeze the reward reference scales and the observation scaler (M7).

Protocol section 6.4 names the **fixed-timetable baseline** as the reference policy for the reward
scales, and section 6.1 requires the observation scaler to be fitted on `train`-period days and
never recomputed. Both are fitted here, once, from the same source, and every arm then uses the
identical frozen constants.

**Two artefacts, two schedules.** The fixed timetable ignores the observation entirely (protocol
section 7.1 gives it nothing beyond the clock), so its trajectory -- and therefore the reward
components it produces -- does not depend on which forecaster is running. The reward scales can
therefore be frozen before M4 finishes. The *observation* scaler obviously cannot: its statistics
for the forecast channels depend on the forecaster, so it is fitted with the trained
`xgb_quantile` provider once M4 lands.

    python scripts/fit_scalers.py --what reward
    python scripts/fit_scalers.py --what obs        # after M4

**Seeds.** Fitting uses seeds 1001+, disjoint from both the frozen training pool (101-110) and the
frozen evaluation pool (1-30). Fitting a component of the objective on the days the comparison is
later scored on would be a subtle form of test-set tuning.
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import argparse
import json
import subprocess
from datetime import datetime, timezone

import numpy as np

from src.baselines.fixed_timetable import FixedTimetable, FixedTimetablePolicy
from src.common.logging import get_logger
from src.common.paths import MODELS_DIR, REPO_ROOT, ensure_dir
from src.environment.metro_env import EnvConfig, MetroEnv, constant_history_sampler
from src.environment.observation import ObservationScaler
from src.environment.reward import ReferenceScales
from src.evaluation.policies import run_episode
from src.simulator.demand import DemandProcess
from src.simulator.engine import SimulatorConfig
from src.simulator.network import Network

logger = get_logger(__name__)

REWARD_SCALES = REPO_ROOT / "configs" / "reward_scales.json"
OBS_SCALER = MODELS_DIR / "obs_scaler.json"

#: Disjoint from the frozen training (101-110) and evaluation (1-30) pools.
FIT_SEEDS = tuple(range(1001, 1021))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def build_env(*, trained_forecaster: bool, arm: str = "uq") -> MetroEnv:
    """The environment the scalers are fitted in.

    Args:
        trained_forecaster: Use the M4 `xgb_quantile` provider. Required for the observation
            scaler; irrelevant for the reward scales, whose trajectory is forecaster-independent
            because the fixed timetable never reads the observation.
        arm: Fitted on the ``uq`` arm so the sigma channels carry their true scale. The same
            constants are then applied to every arm -- see the decision note in
            `docs/decisions.md` D-15.
    """
    network = Network.load()
    process = DemandProcess.load()
    config = SimulatorConfig(fleet_size=36, capacity_passengers=1000.0)

    if trained_forecaster:
        from src.environment.forecast_provider import load_trained_provider

        provider = load_trained_provider(network)
    else:
        from src.environment.forecast_provider import SeasonalNaiveProvider

        provider = SeasonalNaiveProvider(network.n_stations)

    return MetroEnv(
        network=network,
        demand_process=process,
        simulator_config=config,
        env_config=EnvConfig(arm=arm, scenario="normal", strict_scaler=False),
        forecast_provider=provider,
        history_sampler=constant_history_sampler(process, config.day_start_hour),
        scales=ReferenceScales.unit(),
    )


def fit_reward_scales(n_days: int) -> ReferenceScales:
    env = build_env(trained_forecaster=False)
    policy = FixedTimetablePolicy(FixedTimetable.load())

    per_epoch: list[tuple[float, float, float, float]] = []
    for seed in FIT_SEEDS[:n_days]:
        result = run_episode(env, policy, eval_seed=seed)
        for record in result.epochs:
            per_epoch.append((record["W_minutes"], record["U_passengers"],
                              record["E_units"], record["C_units"]))

    means = np.asarray(per_epoch, dtype=float).mean(axis=0)
    fallbacks: dict[str, str] = {}
    values = {}
    for name, mean in zip(("ref_W", "ref_U", "ref_E", "ref_C"), means):
        if mean > 1e-9:
            values[name] = float(mean)
        else:
            values[name] = 1.0
            fallbacks[name] = (
                "the reference policy produced a zero mean for this term; the scale is set to 1.0 "
                "so the term stays in physical units rather than being divided by zero"
            )
            logger.warning("%s was zero under the fixed timetable; using 1.0", name)

    scales = ReferenceScales(**values, provenance={
        "source_policy": "fixed_timetable",
        "provisional": False,
        "note": "Fitted from the fixed-timetable baseline, per protocol section 6.4.",
        "timetable_provenance": FixedTimetable.load().provenance["source"],
        "episodes": n_days,
        "seeds": f"{FIT_SEEDS[0]}..{FIT_SEEDS[n_days - 1]} (disjoint from both frozen seed pools)",
        "scenario": "normal",
        "epochs_averaged": len(per_epoch),
        "W_definition": "passenger-minutes accrued per epoch (queue integral, amendment A-02)",
        "U_definition": "distinct passengers left behind per epoch (amendment A-01)",
        "fallbacks": fallbacks,
        "forecaster_independent": (
            "the fixed timetable never reads the observation, so these scales do not depend on "
            "which forecaster supplies mu and sigma"
        ),
        "git_commit": git_commit(),
        "fitted_utc": datetime.now(timezone.utc).isoformat(),
    })
    scales.save(REWARD_SCALES)
    logger.info("reward scales: W=%.3f U=%.3f E=%.3f C=%.3f",
                scales.ref_W, scales.ref_U, scales.ref_E, scales.ref_C)
    return scales


def fit_observation_scaler(n_days: int, *, trained_forecaster: bool = True) -> ObservationScaler:
    env = build_env(trained_forecaster=trained_forecaster)
    policy = FixedTimetablePolicy(FixedTimetable.load())

    samples: list[np.ndarray] = []
    for seed in FIT_SEEDS[:n_days]:
        env.reset(seed=seed)
        policy.reset(env)
        while True:
            mask = env.action_masks()
            want = min(policy.act(np.zeros(1), env, mask), int(np.flatnonzero(mask)[-1]))
            samples.append(env.raw_observation())
            _, _, terminated, truncated, _ = env.step(want)
            if terminated or truncated:
                break

    scaler = ObservationScaler.fit(np.asarray(samples), env.obs_spec, {
        "source_policy": "fixed_timetable",
        "forecaster": getattr(env.forecast_provider, "model_id", "unknown"),
        "arm_fitted_on": env.cfg.arm,
        "episodes": n_days,
        "seeds": f"{FIT_SEEDS[0]}..{FIT_SEEDS[n_days - 1]} (disjoint from both frozen seed pools)",
        "n_samples": len(samples),
        "note": (
            "One frozen scaler for every arm (protocol section 6.1). Fitted on the uq arm so the "
            "sigma channels carry their true scale; the identical affine map is then applied to "
            "point and placebo. See docs/decisions.md D-15."
        ),
        "git_commit": git_commit(),
        "fitted_utc": datetime.now(timezone.utc).isoformat(),
    })
    ensure_dir(OBS_SCALER.parent)
    scaler.save(OBS_SCALER)
    logger.info("observation scaler fitted on %d samples -> %s", len(samples), OBS_SCALER)
    return scaler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--what", choices=("reward", "obs", "both"), default="both")
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--allow-untrained-forecaster", action="store_true",
                        help="fit the observation scaler with the seasonal-naive provider; "
                             "for smoke use only, never for a reportable run")
    args = parser.parse_args()

    if args.what in ("reward", "both"):
        scales = fit_reward_scales(args.days)
        print(f"reward scales -> {REWARD_SCALES}")
        print(f"  ref_W={scales.ref_W:.4f} ref_U={scales.ref_U:.4f} "
              f"ref_E={scales.ref_E:.4f} ref_C={scales.ref_C:.4f}")

    if args.what in ("obs", "both"):
        scaler = fit_observation_scaler(
            args.days, trained_forecaster=not args.allow_untrained_forecaster)
        print(f"observation scaler -> {OBS_SCALER} ({scaler.centre.size} channels)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
