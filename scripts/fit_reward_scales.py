"""Fit the frozen reward reference scales (protocol section 6.4).

The four reward terms have incommensurable units, so alpha..delta only read as priorities once
each term is divided by a fixed reference scale. Those scales are computed **once**, from a
reference policy over `train`-period days, stored in ``configs/reward_scales.json``, and never
recomputed -- recomputing them per arm or per scenario would let the objective drift between the
things being compared.

Protocol section 6.4 specifies the **fixed-timetable baseline** as the reference policy. That
baseline is an M7 deliverable, so this script takes ``--policy`` and records which policy
produced the file. Until M7 lands, run it with the default constant-action policy and treat the
output as provisional; the provenance block says so in the file itself, and reportable runs
should regenerate it from ``fixed_timetable``.

Usage::

    python scripts/fit_reward_scales.py --days 20 --policy constant_1
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import argparse
import subprocess
from datetime import datetime, timezone

import numpy as np

from src.common.logging import get_logger
from src.common.paths import REPO_ROOT
from src.environment.forecast_provider import SeasonalNaiveProvider
from src.environment.metro_env import EnvConfig, MetroEnv, constant_history_sampler
from src.environment.reward import ReferenceScales
from src.simulator.demand import DemandProcess
from src.simulator.engine import SimulatorConfig
from src.simulator.network import Network

logger = get_logger(__name__)

OUTPUT = REPO_ROOT / "configs" / "reward_scales.json"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:  # pragma: no cover - provenance must never block the fit
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=20, help="reference episodes to average over")
    parser.add_argument("--policy", default="constant_1",
                        help="constant_<k> today; fixed_timetable once M7 lands")
    parser.add_argument("--fleet-size", type=int, default=36)
    parser.add_argument("--capacity", type=float, default=1000.0)
    args = parser.parse_args()

    if not args.policy.startswith("constant_"):
        raise SystemExit(
            f"policy {args.policy!r} is not implemented here yet. The fixed-timetable reference "
            "policy arrives with M7; until then use constant_<k> and accept a provisional file."
        )
    action = int(args.policy.split("_")[1])

    network = Network.load()
    process = DemandProcess.load()
    sim_config = SimulatorConfig(fleet_size=args.fleet_size, capacity_passengers=args.capacity)
    env_config = EnvConfig(arm="uq", scenario="normal", strict_scaler=False)

    env = MetroEnv(
        network=network,
        demand_process=process,
        simulator_config=sim_config,
        env_config=env_config,
        forecast_provider=SeasonalNaiveProvider(network.n_stations),
        history_sampler=constant_history_sampler(process, sim_config.day_start_hour),
        scales=ReferenceScales.unit(),
    )

    per_epoch: list[tuple[float, float, float, float]] = []
    # Seeds 1001+ deliberately sit outside both the training pool (101..110) and the evaluation
    # pool (1..30) frozen in protocol section 8.1. Fitting the reference scales on evaluation
    # seeds would tune a component of the objective on the days the comparison is scored.
    for offset in range(args.days):
        env.reset(seed=1001 + offset)
        while True:
            mask = env.action_masks()
            chosen = action if mask[action] else 0
            _, _, terminated, truncated, info = env.step(chosen)
            component = info["components"]
            per_epoch.append(
                (
                    component["W_minutes"],
                    component["U_passengers"],
                    component["E_units"],
                    component["C_units"],
                )
            )
            if terminated or truncated:
                break

    values = np.asarray(per_epoch, dtype=float)
    means = values.mean(axis=0)

    # A reference scale of zero would make its term infinite or silently drop it. That happens
    # for U on a well-provisioned line where almost nobody is ever left behind, so fall back to
    # the smallest strictly positive scale that keeps the term meaningful, and say so.
    fallbacks: dict[str, str] = {}
    names = ("ref_W", "ref_U", "ref_E", "ref_C")
    scales = {}
    for name, value in zip(names, means):
        if value > 1e-9:
            scales[name] = float(value)
        else:
            scales[name] = 1.0
            fallbacks[name] = (
                "reference policy produced a zero mean for this term; scale set to 1.0 so the "
                "term stays in physical units rather than being divided by zero"
            )
            logger.warning("%s was zero under the reference policy; using 1.0", name)

    reference = ReferenceScales(
        **scales,
        provenance={
            "source_policy": args.policy,
            "provisional": args.policy != "fixed_timetable",
            "note": (
                "Protocol section 6.4 specifies the fixed-timetable baseline as the reference "
                "policy. That baseline is an M7 deliverable; regenerate this file with "
                "--policy fixed_timetable before any reportable run."
                if args.policy != "fixed_timetable"
                else "Fitted from the fixed-timetable baseline, per protocol section 6.4."
            ),
            "episodes": args.days,
            "seeds": f"1001..{1000 + args.days} (disjoint from both frozen seed pools)",
            "scenario": "normal",
            "epochs_averaged": int(values.shape[0]),
            "fleet_size": args.fleet_size,
            "capacity_passengers": args.capacity,
            "W_definition": "passenger-minutes accrued per epoch (queue integral, A-02)",
            "U_definition": "distinct passengers left behind per epoch (A-01)",
            "fallbacks": fallbacks,
            "git_commit": git_commit(),
            "fitted_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    path = reference.save(OUTPUT)
    logger.info("wrote %s", path)
    print(f"ref_W={reference.ref_W:.4f}  ref_U={reference.ref_U:.4f}  "
          f"ref_E={reference.ref_E:.4f}  ref_C={reference.ref_C:.4f}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
