"""Compare simulator behaviour against real published statistics (M5 exit gate).

Research brief section 11: do not merely build a simulator and call it a digital twin. Every
quantity below is compared with a real published figure where one exists, and where none
exists that is stated plainly rather than glossed.

Each row gets one of three verdicts:

``validated``
    A real reference exists and the simulated value falls inside the stated tolerance.
``partially validated``
    A real reference exists but constrains the quantity only indirectly or in aggregate.
``not validatable``
    No public data constrains this quantity. Reported as such, never quietly assumed.

Run::

    python -m src.simulator.validate --config configs/simulator.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.common.logging import get_logger, setup_logging
from src.common.manifest import git_info
from src.common.paths import RAW_DIR, RESULTS_DIR, ensure_dir
from src.common.seeding import environment_streams
from src.data.gtfs import line_stop_times, scheduled_headways
from src.simulator.demand import DemandProcess
from src.simulator.disruptions import estimate_failure_rate_per_km, generate_scenario
from src.simulator.engine import MetroSimulator, SimulatorConfig
from src.simulator.network import Network

logger = get_logger(__name__)


@dataclass
class ValidationRow:
    """One simulator-versus-reality comparison."""

    quantity: str
    real_value: float | None
    real_source: str
    simulated_value: float | None
    units: str
    verdict: str
    tolerance: str = ""
    note: str = ""

    @property
    def absolute_error(self) -> float | None:
        if self.real_value is None or self.simulated_value is None:
            return None
        return abs(self.simulated_value - self.real_value)

    @property
    def relative_error(self) -> float | None:
        if self.real_value in (None, 0) or self.simulated_value is None:
            return None
        return abs(self.simulated_value - self.real_value) / abs(self.real_value)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["absolute_error"] = self.absolute_error
        payload["relative_error"] = self.relative_error
        return payload


def _config_to_simulator(config: Any, failure_rate: float) -> SimulatorConfig:
    return SimulatorConfig(
        internal_step_seconds=float(config.get("time.internal_step_seconds")),
        decision_interval_seconds=float(config.get("time.decision_interval_seconds")),
        day_start_hour=int(config.get("time.day_start_hour")),
        day_end_hour=int(config.get("time.day_end_hour")),
        dwell_base_seconds=float(config.get("dwell.base_seconds")),
        dwell_per_boarder_seconds=float(config.get("dwell.per_boarder_seconds")),
        dwell_max_seconds=float(config.get("dwell.max_seconds")),
        capacity_passengers=float(config.get("capacity.passengers_per_train")),
        fleet_size=int(config.get("fleet.size")),
        initial_maintenance_fraction=float(config.get("fleet.initial_maintenance_fraction")),
        max_induct_per_decision=int(config.get("fleet.max_induct_per_decision")),
        intra_hour_profile=str(config.get("demand.intra_hour_profile")),
        od_model=str(config.get("demand.od_model")),
        od_decay_km=float(config.get("demand.od_decay_km")),
        failures_enabled=bool(config.get("failures.enabled")),
        failure_rate_per_km=failure_rate,
        energy_units_per_train_km=float(config.get("energy.units_per_train_km")),
        energy_units_per_stop=float(config.get("energy.units_per_stop")),
        cost_units_per_induction=float(config.get("cost.units_per_induction")),
        cost_units_per_train_hour=float(config.get("cost.units_per_train_hour")),
    )


def run_reference_episodes(
    network: Network, process: DemandProcess, sim_config: SimulatorConfig, n_seeds: int = 5
) -> dict[str, Any]:
    """Run the fixed-timetable-like reference policy used for validation.

    Validation uses a service level matched to the published timetable rather than a learned
    policy: the question is whether the *simulator* reproduces reality, not whether a policy
    performs well.
    """
    real_trips = _real_weekday_trips_per_direction()
    epochs = sim_config.n_decision_epochs
    # Induct at the rate that reproduces the published daily trip count.
    target_rate = real_trips / epochs

    summaries: list[dict[str, Any]] = []
    for seed in range(1, n_seeds + 1):
        streams = environment_streams("normal", seed)
        spec = generate_scenario("normal", streams, day_seconds=sim_config.day_seconds,
                                 fleet_size=sim_config.fleet_size)
        sim = MetroSimulator(network, process, sim_config)
        sim.reset(spec, streams)
        carry = 0.0
        while not sim.done:
            carry += target_rate
            action = int(carry)
            carry -= action
            sim.step_epoch(min(action, sim_config.max_induct_per_decision))
        summaries.append(sim.episode_summary())

    return {
        key: float(np.nanmean([s[key] for s in summaries]))
        for key in summaries[0]
        if isinstance(summaries[0][key], (int, float))
    }


def _real_weekday_trips_per_direction() -> float:
    """Published weekday trips per direction, from the GTFS timetable."""
    stop_times = line_stop_times("L")
    weekday = stop_times[stop_times["service_id"].astype(str).str.contains("Weekday", case=False, na=False)]
    per_direction = weekday.groupby("direction_id")["trip_id"].nunique()
    return float(per_direction.min())


def _real_running_time_minutes() -> tuple[float, float]:
    """Published scheduled and actual end-to-end running times, in minutes."""
    frame = pd.read_parquet(RAW_DIR / "mta_running_times" / "data.parquet")
    for column in ("average_actual_runtime", "average_scheduled_runtime", "number_of_stops"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    full = frame[frame["number_of_stops"] >= 22]
    return float(full["average_scheduled_runtime"].median()), float(full["average_actual_runtime"].median())


def _real_additional_platform_time() -> float:
    """Published Additional Platform Time in minutes (excess wait vs schedule)."""
    frame = pd.read_parquet(RAW_DIR / "mta_cjfm" / "data.parquet")
    frame["additional_platform_time"] = pd.to_numeric(frame["additional_platform_time"], errors="coerce")
    frame["month"] = pd.to_datetime(frame["month"])
    recent = frame[frame["month"] >= frame["month"].max() - pd.DateOffset(years=2)]
    return float(recent["additional_platform_time"].median())


def _real_service_delivered() -> float:
    """Published fraction of scheduled trains actually run, recomputed from the counts."""
    frame = pd.read_parquet(RAW_DIR / "mta_service_delivered" / "data.parquet")
    for column in ("num_actual_trains", "num_sched_trains"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["month"] = pd.to_datetime(frame["month"])
    recent = frame[frame["month"] >= frame["month"].max() - pd.DateOffset(years=2)]
    # The published `service_delivered` field is mis-scaled by 1/100; recompute it.
    return float((recent["num_actual_trains"] / recent["num_sched_trains"]).median())


def build_validation(config_path: str) -> tuple[list[ValidationRow], dict[str, Any]]:
    """Run the comparisons and return the rows plus supporting context."""
    config = load_config(config_path, allow_unresolved=True, sweep_tag="sensitivity_sweep")
    network = Network.load()
    process = DemandProcess.load()

    failure_calibration = estimate_failure_rate_per_km(
        str(config.get("failures.car_class")),
        int(config.get("failures.cars_per_train")),
        int(config.get("failures.mdbf_lookback_months")),
    )
    sim_config = _config_to_simulator(config, failure_calibration["train_failures_per_km"])
    simulated = run_reference_episodes(network, process, sim_config)

    rows: list[ValidationRow] = []

    # --- 1. end-to-end running time -------------------------------------------------------
    real_scheduled, real_actual = _real_running_time_minutes()
    sim_runtime = network.end_to_end_seconds(0) / 60.0
    rows.append(ValidationRow(
        quantity="End-to-end running time (one direction)",
        real_value=real_scheduled, real_source="mta_running_times (sp9g-mzjh), median scheduled",
        simulated_value=sim_runtime, units="minutes",
        verdict="validated" if abs(sim_runtime - real_scheduled) / real_scheduled <= 0.10 else "mismatch",
        tolerance="10%",
        note=f"published actual runtime median {real_actual:.1f} min",
    ))

    # --- 2. daily service level ------------------------------------------------------------
    real_trips = _real_weekday_trips_per_direction()
    sim_trips = simulated["trips_completed"]
    rows.append(ValidationRow(
        quantity="Weekday round trips per day",
        real_value=real_trips, real_source="mta_gtfs_static, distinct weekday trip ids per direction",
        simulated_value=sim_trips, units="trips",
        verdict="validated" if abs(sim_trips - real_trips) / real_trips <= 0.15 else "mismatch",
        tolerance="15%",
        note="reference policy inducts at the rate that reproduces the published trip count",
    ))

    # --- 3. minimum headway -----------------------------------------------------------------
    headways = scheduled_headways("L")
    weekday = headways[headways["service_id"].astype(str).str.contains("Weekday", case=False, na=False)]
    real_min_headway = float(weekday["headway_min_s"].min())
    rows.append(ValidationRow(
        quantity="Minimum headway",
        real_value=real_min_headway, real_source="mta_gtfs_static, measured at each direction terminus",
        simulated_value=float(network.min_headway_seconds), units="seconds",
        verdict="validated",
        tolerance="exact",
        note="the simulator's constraint is set from this measurement, so agreement is by construction",
    ))

    # --- 4. daily demand volume --------------------------------------------------------------
    real_daily = float(process.mean["weekday"].sum())
    rows.append(ValidationRow(
        quantity="Weekday boardings per day",
        real_value=real_daily, real_source="mta_hourly_ridership (fitted on the training split)",
        simulated_value=simulated["passengers_arrived"], units="passengers",
        verdict="validated" if abs(simulated["passengers_arrived"] - real_daily) / real_daily <= 0.10 else "mismatch",
        tolerance="10%",
        note="demand process is fitted to observed ridership, so this checks the sampler, not the level",
    ))

    # --- 5. excess platform wait ---------------------------------------------------------------
    real_apt = _real_additional_platform_time()
    sim_wait = simulated["waiting_time_mean_s"] / 60.0
    rows.append(ValidationRow(
        quantity="Mean passenger platform wait",
        real_value=real_apt, real_source="mta_cjfm (r7qk-6tcy), Additional Platform Time",
        simulated_value=sim_wait, units="minutes",
        verdict="partially validated",
        tolerance="not directly comparable",
        note=(
            "Additional Platform Time measures EXCESS wait relative to schedule, whereas the "
            "simulator reports TOTAL wait. The published figure is therefore a lower bound on "
            "the simulated quantity, not a target. Reported for order-of-magnitude only."
        ),
    ))

    # --- 6. service delivered -------------------------------------------------------------------
    real_delivered = _real_service_delivered()
    sim_delivered = simulated["trips_completed"] / max(simulated["inductions"], 1.0)
    rows.append(ValidationRow(
        quantity="Fraction of dispatched trains completing their trip",
        real_value=real_delivered, real_source="mta_service_delivered (32ch-sei3), recomputed from counts",
        simulated_value=sim_delivered, units="fraction",
        verdict="partially validated",
        tolerance="qualitative",
        note=(
            "the published metric counts trains run against trains scheduled; the simulated "
            "analogue counts completed round trips against inductions, which is close but not "
            "identical in definition"
        ),
    ))

    # --- 7. failure frequency ---------------------------------------------------------------------
    implied_per_day = (
        failure_calibration["train_failures_per_km"] * simulated["train_km"]
    )
    # Honest accounting: a rate this low cannot be resolved by a handful of episodes. Over
    # `n_validation_seeds` days the expected count is small, so observing zero is entirely
    # consistent with a correctly calibrated rate -- and equally consistent with a rate that is
    # wrong by a factor of two. A wide tolerance would let this "pass" while proving nothing,
    # so it is reported as partially validated with the Poisson probability made explicit.
    n_days = context_n_seeds = 5
    expected_total = implied_per_day * n_days
    p_zero = float(np.exp(-expected_total))
    rows.append(ValidationRow(
        quantity="Train failures per operating day",
        real_value=implied_per_day, real_source="mta_mdbf (e2qc-xgxs), car class "
                                                f"{failure_calibration['car_class']}",
        simulated_value=simulated["failures"], units="failures/day",
        verdict="partially validated",
        tolerance="not resolvable at this sample size",
        note=(
            f"MDBF {failure_calibration['mdbf_car_miles']:,.0f} car-miles over "
            f"{failure_calibration['months_used']:.0f} months implies {implied_per_day:.3f} "
            f"failures/day, i.e. {expected_total:.2f} expected across the {n_days} validation "
            f"episodes (P[observe zero] = {p_zero:.2f}). The hazard is set FROM this rate, so "
            "the comparison checks the implementation, not the rate; and at this sample size it "
            "could not distinguish a correct rate from one wrong by a factor of two. The "
            "aggregate rate is calibrated to real data; the within-day timing structure remains "
            "synthetic (assumption A13)."
        ),
    ))

    # --- rows with no public reference --------------------------------------------------------
    for quantity, note in (
        ("Train capacity / peak occupancy",
         "no public occupancy data exists for this system; capacity is assumption A09 and is swept"),
        ("Station dwell-time distribution",
         "GTFS reports arrival == departure at intermediate stops, so it carries no dwell "
         "information at all; assumption A08, swept"),
        ("Passenger waiting behaviour (balking, patience)",
         "no public data; infinite patience is assumed by default and a finite-patience "
         "variant is swept (assumption A06)"),
        ("Origin-destination distribution",
         "the demand source publishes entries only; the OD model is fabricated (assumption "
         "A03) and three variants are swept in EXP-SENS-002"),
        ("Per-train trajectories and delays",
         "GTFS-RT has no public historical archive; only aggregate realised statistics exist"),
        ("Energy consumption in physical units",
         "no citable kWh/train-km figure sourced (assumption A15); energy is reported in "
         "train-km-equivalent units and never in kWh"),
        ("Operating cost in currency",
         "no citable cost figure sourced (assumption A16); cost is reported in "
         "induction-equivalent units and never in currency"),
    ):
        rows.append(ValidationRow(
            quantity=quantity, real_value=None, real_source="none available",
            simulated_value=None, units="-", verdict="not validatable", note=note,
        ))

    context = {
        "failure_calibration": failure_calibration,
        "simulated_reference": simulated,
        "sim_config": sim_config.__dict__,
    }
    return rows, context


def write_report(rows: list[ValidationRow], context: dict[str, Any]) -> None:
    """Render ``docs/simulator_validation.md`` and its machine-readable twin."""
    ensure_dir(RESULTS_DIR)
    info = git_info()

    counts = {
        verdict: sum(1 for r in rows if r.verdict == verdict)
        for verdict in ("validated", "partially validated", "not validatable", "mismatch")
    }

    lines = [
        "# Simulator Validation",
        "",
        "Generated by `python -m src.simulator.validate`. Every number is computed from stored "
        "data; none is typed by hand.",
        "",
        f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Git commit: `{info.commit}`" + ("  **(dirty tree)**" if info.dirty else ""),
        "",
        "## Verdict summary",
        "",
        f"| validated | partially validated | not validatable | mismatch |",
        "|---|---|---|---|",
        f"| {counts['validated']} | {counts['partially validated']} | "
        f"{counts['not validatable']} | {counts['mismatch']} |",
        "",
        "**This is a calibrated simulator, not an asset-coupled digital twin** (manuscript "
        "correction MC-04). No conclusion drawn from it is a claim about real-world "
        "performance.",
        "",
        "## Comparisons against real published statistics",
        "",
        "| Quantity | Real | Simulated | Units | Rel. error | Verdict | Source |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row.verdict == "not validatable":
            continue
        real = f"{row.real_value:,.2f}" if row.real_value is not None else "—"
        sim = f"{row.simulated_value:,.2f}" if row.simulated_value is not None else "—"
        rel = f"{row.relative_error:.1%}" if row.relative_error is not None else "—"
        lines.append(
            f"| {row.quantity} | {real} | {sim} | {row.units} | {rel} | "
            f"**{row.verdict}** | {row.real_source} |"
        )

    lines += ["", "### Notes on the comparisons", ""]
    for row in rows:
        if row.verdict != "not validatable" and row.note:
            lines.append(f"- **{row.quantity}** — {row.note}")

    lines += [
        "",
        "## Quantities that CANNOT be validated",
        "",
        "Research brief section 11 requires these to be stated explicitly rather than passed "
        "over. Each is a limitation of the study, not an oversight.",
        "",
        "| Quantity | Why no validation is possible |",
        "|---|---|",
    ]
    for row in rows:
        if row.verdict == "not validatable":
            lines.append(f"| {row.quantity} | {row.note} |")

    lines += [
        "",
        "**The two most consequential unvalidatable quantities — peak occupancy and the "
        "origin-destination distribution — both feed the pre-registered primary endpoint.** "
        "That is why the OD sweep (`EXP-SENS-002`) and the capacity sweep are mandatory rather "
        "than optional, and why any primary conclusion must either hold across the sweep or be "
        "reported as parameter-dependent.",
        "",
        "## Failure-rate calibration",
        "",
    ]
    calibration = context["failure_calibration"]
    for key, value in calibration.items():
        formatted = f"{value:,.6g}" if isinstance(value, (int, float)) else value
        lines.append(f"- `{key}`: {formatted}")
    lines += [
        "",
        "The aggregate failure *rate* is calibrated against real published MDBF. The within-day "
        "*timing* structure (a constant hazard per train-km) is synthetic — MDBF constrains the "
        "rate only, and cannot constrain when within a day a failure occurs.",
        "",
    ]

    (RESULTS_DIR / "simulator_validation.json").write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": info.commit,
                "counts": counts,
                "rows": [r.to_dict() for r in rows],
                "context": context,
            },
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    from src.common.paths import DOCS_DIR
    (DOCS_DIR / "simulator_validation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/simulator.yaml")
    args = parser.parse_args()

    setup_logging(level=logging.INFO, log_file="simulator_validate.log")
    rows, context = build_validation(args.config)
    write_report(rows, context)

    for row in rows:
        real = f"{row.real_value:,.2f}" if row.real_value is not None else "—"
        sim = f"{row.simulated_value:,.2f}" if row.simulated_value is not None else "—"
        print(f"{row.verdict:<22} {row.quantity:<48} real={real:>12} sim={sim:>12}")

    mismatches = [r for r in rows if r.verdict == "mismatch"]
    print(f"\nmismatches: {len(mismatches)}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
