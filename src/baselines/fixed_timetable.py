"""Fixed-timetable baseline (M7).

Protocol section 7.1: *"Deterministic induction schedule reproducing the GTFS scheduled service
pattern; information available to it: none beyond the clock."*

The schedule is **extracted from the real published timetable**, not invented: it counts the
scheduled departures at the direction-0 origin terminus in each 5-minute decision epoch of a
weekday service day, and inducts that many trainsets. Since a trainset in this simulator returns
to the depot only after a full round trip, one induction corresponds to one terminal departure,
so the mapping is direct.

This baseline is load-bearing well beyond its own row in the results table. Protocol section 6.4
names it as the reference policy for the reward scales, and section 6.1 for the observation
scaler. Both are fitted from it once and frozen (`scripts/fit_scalers.py`), which is why M7 must
land before any RL training.

**It does not receive the forecast.** Its `act` ignores the observation entirely, which is what
"none beyond the clock" means and what makes it the honest floor for H4.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.common.logging import get_logger
from src.common.paths import PROCESSED_DIR, ensure_dir
from src.environment.metro_env import MetroEnv

logger = get_logger(__name__)

__all__ = ["FixedTimetable", "FixedTimetablePolicy", "build_fixed_timetable", "TIMETABLE_FILE"]

TIMETABLE_FILE = PROCESSED_DIR / "network" / "fixed_timetable.json"


@dataclass
class FixedTimetable:
    """Scheduled inductions per decision epoch, per day type.

    Attributes:
        inductions: ``inductions[day_type]`` is a list of length ``n_epochs``.
        day_start_hour: Operating-day start the epochs are indexed from.
        decision_interval_s: Epoch length.
        provenance: Which GTFS service ids and reference station produced each day type.
    """

    inductions: dict[str, list[int]]
    day_start_hour: int
    decision_interval_s: float
    provenance: dict[str, Any]

    def at(self, day_type: str, epoch: int) -> int:
        schedule = self.inductions.get(day_type) or self.inductions["weekday"]
        return int(schedule[epoch]) if 0 <= epoch < len(schedule) else 0

    def total(self, day_type: str = "weekday") -> int:
        return int(sum(self.inductions.get(day_type, [])))

    def save(self, path: Path = TIMETABLE_FILE) -> Path:
        ensure_dir(path.parent)
        path.write_text(json.dumps({
            "inductions": self.inductions,
            "day_start_hour": self.day_start_hour,
            "decision_interval_s": self.decision_interval_s,
            "provenance": self.provenance,
        }, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path = TIMETABLE_FILE) -> "FixedTimetable":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            inductions={k: list(map(int, v)) for k, v in payload["inductions"].items()},
            day_start_hour=int(payload["day_start_hour"]),
            decision_interval_s=float(payload["decision_interval_s"]),
            provenance=payload.get("provenance", {}),
        )


def _service_ids_by_day_type(tables: dict) -> dict[str, list[str]]:
    """Split GTFS service ids into weekday and weekend by their calendar rows."""
    calendar = tables["calendar.txt"]
    weekday_cols = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    weekend_cols = ["saturday", "sunday"]
    out: dict[str, list[str]] = {"weekday": [], "weekend": []}
    for _, row in calendar.iterrows():
        runs_weekday = any(int(row.get(c, 0)) == 1 for c in weekday_cols)
        runs_weekend = any(int(row.get(c, 0)) == 1 for c in weekend_cols)
        if runs_weekday and not runs_weekend:
            out["weekday"].append(str(row["service_id"]))
        elif runs_weekend and not runs_weekday:
            out["weekend"].append(str(row["service_id"]))
    return out


def build_fixed_timetable(
    *,
    route_short_name: str = "L",
    day_start_hour: int = 5,
    day_end_hour: int = 24,
    decision_interval_s: float = 300.0,
    k_max: int = 2,
) -> FixedTimetable:
    """Extract the induction schedule from the published GTFS timetable.

    Departures at the direction-0 origin terminus are binned into decision epochs. Where the real
    timetable schedules more departures in one 5-minute window than ``k_max``, the surplus is
    **carried forward** to the next epoch rather than dropped: dropping it would silently make the
    baseline run less service than the real line does, which would flatter every method compared
    against it.
    """
    from src.data.gtfs import dominant_patterns, line_stop_times, load_gtfs_tables

    tables = load_gtfs_tables()
    stop_times = line_stop_times(route_short_name)
    patterns = dominant_patterns(route_short_name)
    by_day_type = _service_ids_by_day_type(tables)

    n_epochs = int(round((day_end_hour - day_start_hour) * 3600.0 / decision_interval_s))

    # GTFS `direction_id` arrives as an Arrow-backed *string* column, and `dominant_patterns`
    # keys its result the same way. Comparing against the integer 0 matches nothing and silently
    # produced an all-zero timetable -- a baseline that runs no service at all, which would have
    # made every other method look excellent.
    direction_key = "0" if "0" in patterns else next(iter(patterns))
    reference = patterns[direction_key].parent_stations[0]
    direction_mask = stop_times["direction_id"].astype(str) == "0"

    inductions: dict[str, list[int]] = {}
    detail: dict[str, Any] = {}

    for day_type, service_ids in by_day_type.items():
        at_reference = stop_times[
            direction_mask
            & (stop_times["parent_station"].astype(str) == str(reference))
            & (stop_times["service_id"].astype(str).isin(service_ids))
        ]
        departures = at_reference["departure_s"].dropna().to_numpy(dtype=float)
        if departures.size == 0:
            logger.warning("no direction-0 departures for day type %s; using zeros", day_type)
            inductions[day_type] = [0] * n_epochs
            continue

        # One service day's worth: if several service ids cover the same day type, average them
        # rather than summing, or the baseline would run several times the real service.
        n_services = max(1, len(set(at_reference["service_id"].astype(str))))

        counts = np.zeros(n_epochs, dtype=float)
        origin_s = day_start_hour * 3600.0
        for departure in departures:
            index = int((departure - origin_s) // decision_interval_s)
            if 0 <= index < n_epochs:
                counts[index] += 1.0
        counts /= n_services

        schedule: list[int] = []
        carry = 0.0
        for value in counts:
            want = value + carry
            take = int(min(k_max, np.floor(want + 1e-9)))
            schedule.append(take)
            carry = max(0.0, want - take)
        inductions[day_type] = schedule

        detail[day_type] = {
            "service_ids": sorted(set(at_reference["service_id"].astype(str))),
            "n_services_averaged": n_services,
            "raw_departures": int(departures.size),
            "scheduled_inductions": int(sum(schedule)),
            "departures_in_operating_day": float(counts.sum()),
            "carry_forward_remaining": float(carry),
        }
        logger.info(
            "fixed timetable %s: %d inductions over %d epochs (%.1f scheduled departures)",
            day_type, sum(schedule), n_epochs, counts.sum(),
        )

    if "weekday" not in inductions:
        raise RuntimeError("no weekday service found in the GTFS calendar")
    inductions.setdefault("weekend", inductions["weekday"])

    return FixedTimetable(
        inductions=inductions,
        day_start_hour=day_start_hour,
        decision_interval_s=decision_interval_s,
        provenance={
            "source": "real_data:mta_gtfs_static",
            "route_short_name": route_short_name,
            "reference_station": reference,
            "method": (
                "scheduled direction-0 departures at the origin terminus, binned into decision "
                "epochs; surplus above k_max carried forward to the next epoch"
            ),
            "k_max": k_max,
            "by_day_type": detail,
        },
    )


class FixedTimetablePolicy:
    """Deterministic schedule replay. Observes nothing but the clock."""

    method_id = "fixed_timetable"

    def __init__(self, timetable: FixedTimetable | None = None) -> None:
        self.timetable = timetable or FixedTimetable.load()
        self._day_type = "weekday"

    def reset(self, env: MetroEnv) -> None:
        self._day_type = env.cfg.day_type

    def act(self, observation: np.ndarray, env: MetroEnv, mask: np.ndarray) -> int:
        # `observation` is deliberately unused: protocol section 7.1 gives this method no
        # information beyond the clock, and reading the forecast here would quietly turn the
        # honest floor for H4 into a forecast-driven method.
        return self.timetable.at(self._day_type, env.simulator.epoch)
