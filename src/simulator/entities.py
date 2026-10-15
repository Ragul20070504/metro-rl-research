"""Trains and station queues (M5).

Two design choices worth stating, because both affect what the metrics mean:

**Queues are FIFO cohorts, not just counters.** Each cohort records when its passengers
arrived, so boarding yields an exact waiting time per passenger group. A running integral of
queue length would give the mean cheaply but could not produce the median and 95th percentile
the protocol requires (section 7.4), and percentile waiting time is precisely where a
surge or disruption shows up.

**Passenger counts are floats.** Arrivals are drawn as integer Poisson counts -- that is the
genuinely stochastic part. The split of boarders across destinations is then applied as a
deterministic proportional allocation rather than a multinomial draw, because the
origin-destination distribution is a *synthetic* assumption (A03): sampling from it would add
Monte-Carlo noise to a distribution we already do not trust, without adding information.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class TrainStatus(IntEnum):
    """Lifecycle of a trainset over one operating day."""

    DEPOT = 0        # available for induction
    DWELL = 1        # stopped at a station, boarding and alighting
    RUNNING = 2      # traversing a segment
    TURNAROUND = 3   # at a terminus, reversing
    FAILED = 4       # removed by a failure event; cannot return to service today
    MAINTENANCE = 5  # unavailable from the start of the day


@dataclass
class Train:
    """One trainset.

    Attributes:
        train_id: Stable identifier.
        status: Current lifecycle state.
        station: Index of the station the train is at, or the one it most recently left.
        direction: Travel direction, 0 forward or 1 backward.
        timer_s: Seconds remaining in the current activity.
        load: Passengers on board, indexed by destination station.
        distance_km: Cumulative distance travelled today; drives energy and failure hazard.
        stops_made: Cumulative station stops; drives the stop component of energy.
        service_seconds: Cumulative time in service; drives the fleet-cost term.
        frailty: Multiplier on this train's failure hazard, drawn at episode start from a
            distribution with mean exactly 1 so the fleet-aggregate rate stays equal to the
            value calibrated against published MDBF. It is the *latent health state* that the
            synthetic availability signal (assumption A14, decision D-04) is a noisy readout
            of. Default 1.0 leaves the M5 failure process exactly as validated.
    """

    train_id: int
    n_stations: int
    status: TrainStatus = TrainStatus.DEPOT
    station: int = 0
    direction: int = 0
    timer_s: float = 0.0
    load: np.ndarray = field(default=None)  # type: ignore[assignment]
    distance_km: float = 0.0
    stops_made: int = 0
    service_seconds: float = 0.0
    trips_completed: int = 0
    frailty: float = 1.0

    def __post_init__(self) -> None:
        if self.load is None:
            self.load = np.zeros(self.n_stations, dtype=float)

    @property
    def occupancy(self) -> float:
        """Total passengers on board."""
        return float(self.load.sum())

    @property
    def in_service(self) -> bool:
        return self.status in (TrainStatus.DWELL, TrainStatus.RUNNING, TrainStatus.TURNAROUND)

    @property
    def available_for_induction(self) -> bool:
        return self.status == TrainStatus.DEPOT

    def reset(self) -> None:
        """Return to the depot, empty."""
        self.status = TrainStatus.DEPOT
        self.station = 0
        self.direction = 0
        self.timer_s = 0.0
        self.load[:] = 0.0
        self.distance_km = 0.0
        self.stops_made = 0
        self.service_seconds = 0.0
        self.trips_completed = 0
        # Back to the calibrated homogeneous hazard. The environment re-draws frailty after
        # reset when the availability channel is on, so a stale value can never carry over.
        self.frailty = 1.0


@dataclass
class _Cohort:
    """A group of passengers who arrived at the same simulation step.

    ``left_behind`` marks a cohort that has already watched at least one train depart without
    it. Tracking it per cohort is what makes a *bounded* left-behind fraction possible: raw
    denial counts double-count a passenger denied by several successive trains, so
    ``denials / arrivals`` is a rate that can exceed 1 rather than a fraction.
    """

    arrival_time_s: float
    count: float
    left_behind: bool = False


class StationQueue:
    """FIFO passenger queue for one (station, direction) platform.

    Boarding consumes cohorts from the front, so the waiting time returned is exactly the
    time the boarded passengers spent on the platform.
    """

    __slots__ = ("_cohorts", "_total", "arrived", "boarded", "denied", "left_behind_passengers")

    def __init__(self) -> None:
        self._cohorts: deque[_Cohort] = deque()
        self._total: float = 0.0
        self.arrived: float = 0.0
        self.boarded: float = 0.0
        #: Total denial *events* (a passenger denied by three trains contributes three).
        self.denied: float = 0.0
        #: Unique passengers denied at least once. Bounded by ``arrived``.
        self.left_behind_passengers: float = 0.0

    def __len__(self) -> int:
        return len(self._cohorts)

    @property
    def size(self) -> float:
        """Passengers currently waiting."""
        return self._total

    def add(self, count: float, now_s: float) -> None:
        """Add newly arrived passengers."""
        if count <= 0:
            return
        if self._cohorts and self._cohorts[-1].arrival_time_s == now_s:
            self._cohorts[-1].count += count
        else:
            self._cohorts.append(_Cohort(arrival_time_s=now_s, count=count))
        self._total += count
        self.arrived += count

    def board(self, capacity: float, now_s: float) -> tuple[float, list[tuple[float, float]]]:
        """Remove up to ``capacity`` passengers, oldest first.

        Returns:
            ``(boarded, waits)`` where ``waits`` is a list of ``(wait_seconds, count)`` pairs
            for the passengers who boarded. Returning weighted pairs rather than expanded
            per-passenger values keeps percentile computation exact without materialising
            hundreds of thousands of scalars.
        """
        if capacity <= 0 or self._total <= 0:
            return 0.0, []

        remaining = float(capacity)
        boarded = 0.0
        waits: list[tuple[float, float]] = []

        while remaining > 1e-12 and self._cohorts:
            cohort = self._cohorts[0]
            take = min(cohort.count, remaining)
            wait = now_s - cohort.arrival_time_s
            waits.append((wait, take))
            boarded += take
            remaining -= take
            cohort.count -= take
            if cohort.count <= 1e-12:
                self._cohorts.popleft()

        self._total -= boarded
        self.boarded += boarded
        return boarded, waits

    def record_denied(self, count: float) -> float:
        """Record a train departing while ``count`` passengers remain on the platform.

        Increments the denial *event* count, and separately counts how many of those
        passengers are being left behind for the **first** time. Returns the newly
        left-behind total so the caller can aggregate it.
        """
        if count <= 0:
            return 0.0
        self.denied += count

        newly_left_behind = 0.0
        for cohort in self._cohorts:
            if not cohort.left_behind:
                cohort.left_behind = True
                newly_left_behind += cohort.count
        self.left_behind_passengers += newly_left_behind
        return newly_left_behind

    def oldest_wait_s(self, now_s: float) -> float:
        """How long the longest-waiting passenger has been queuing."""
        return (now_s - self._cohorts[0].arrival_time_s) if self._cohorts else 0.0

    def residual_waits(self, now_s: float) -> list[tuple[float, float]]:
        """Waiting times of passengers who never boarded, at end of episode."""
        return [(now_s - c.arrival_time_s, c.count) for c in self._cohorts]

    def clear(self) -> None:
        self._cohorts.clear()
        self._total = 0.0
        self.arrived = 0.0
        self.boarded = 0.0
        self.denied = 0.0
        self.left_behind_passengers = 0.0


def weighted_quantile(pairs: list[tuple[float, float]], quantile: float) -> float:
    """Quantile of a weighted sample given as ``(value, weight)`` pairs.

    Used for the median and 95th-percentile waiting times, where passengers are grouped into
    cohorts rather than represented individually.
    """
    if not pairs:
        return float("nan")
    values = np.array([p[0] for p in pairs], dtype=float)
    weights = np.array([p[1] for p in pairs], dtype=float)
    keep = weights > 0
    if not keep.any():
        return float("nan")
    values, weights = values[keep], weights[keep]

    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    total = cumulative[-1]
    if total <= 0:
        return float("nan")
    # Position each cohort at the midpoint of the probability mass it occupies.
    positions = (cumulative - 0.5 * weights) / total
    return float(np.interp(quantile, positions, values))


def weighted_mean(pairs: list[tuple[float, float]]) -> float:
    """Mean of a weighted sample given as ``(value, weight)`` pairs."""
    if not pairs:
        return float("nan")
    values = np.array([p[0] for p in pairs], dtype=float)
    weights = np.array([p[1] for p in pairs], dtype=float)
    total = weights.sum()
    return float((values * weights).sum() / total) if total > 0 else float("nan")
