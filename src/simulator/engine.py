"""Discrete-time metro simulator (M5).

A calibrated digital-twin-style environment, **not** an asset-coupled digital twin: there is no
live telemetry link to a real system (manuscript correction MC-04).

Time advances in fixed internal steps (30 s by default) with induction decisions taken every
decision interval (300 s). A discrete-time formulation was chosen over discrete-event because
the invariants that matter here -- passenger conservation, capacity never exceeded, headway
never violated -- are far easier to assert at every step than to reason about between events,
and those assertions are what make the simulator trustworthy.

Train lifecycle::

    DEPOT --induct--> DWELL(terminal) --> RUNNING --> DWELL --> ... --> TURNAROUND
      ^                                                                     |
      |                                                                     v
      +------------------- DWELL(origin terminal) <-- ... <-- RUNNING <-----+

A train therefore performs one full round trip per induction and then returns to the depot,
where it becomes available again. That makes induction a genuinely repeated decision with a
real service-versus-cost trade-off, rather than a one-way ratchet.

Safety constraints are enforced structurally, not by reward penalties (manuscript section 3.4):
a train cannot depart within the minimum headway of the previous departure, and cannot enter
an occupied platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.common.logging import get_logger
from src.simulator.demand import DemandProcess, intra_hour_weights
from src.simulator.disruptions import ScenarioSpec
from src.simulator.entities import (
    StationQueue,
    Train,
    TrainStatus,
    weighted_mean,
    weighted_quantile,
)
from src.simulator.network import DIRECTIONS, Network
from src.simulator.od import build_od_matrix, destination_weights, direction_split

logger = get_logger(__name__)


@dataclass
class SimulatorConfig:
    """Frozen simulator parameters. Every field's provenance lives in ``configs/simulator.yaml``."""

    internal_step_seconds: float = 30.0
    decision_interval_seconds: float = 300.0
    day_start_hour: int = 5
    day_end_hour: int = 24
    dwell_base_seconds: float = 30.0
    dwell_per_boarder_seconds: float = 0.08
    dwell_max_seconds: float = 120.0
    capacity_passengers: float = 1000.0
    fleet_size: int = 30
    initial_maintenance_fraction: float = 0.10
    max_induct_per_decision: int = 2
    intra_hour_profile: str = "uniform"
    od_model: str = "distance_decay"
    od_decay_km: float = 4.0
    passenger_patience_seconds: float | None = None
    failures_enabled: bool = True
    failure_rate_per_km: float = 4.0e-5
    #: Replace the Poisson arrival draw with its rounded expectation. Off for every experiment;
    #: it exists so `tests/test_env_hand_calculation.py` can compare the simulator against a
    #: trajectory worked out by hand. A stochastic arrival stream cannot be hand-checked, and
    #: "the code runs" is not the same test as "the code computes the expected number".
    deterministic_arrivals: bool = False
    energy_units_per_train_km: float = 1.0
    energy_units_per_stop: float = 0.35
    cost_units_per_induction: float = 1.0
    cost_units_per_train_hour: float = 0.5

    @property
    def steps_per_decision(self) -> int:
        return int(round(self.decision_interval_seconds / self.internal_step_seconds))

    @property
    def day_seconds(self) -> float:
        return (self.day_end_hour - self.day_start_hour) * 3600.0

    @property
    def n_decision_epochs(self) -> int:
        return int(round(self.day_seconds / self.decision_interval_seconds))


@dataclass
class EpochMetrics:
    """Per-decision-epoch outcomes. Every reward component is recoverable from these."""

    waiting_time_mean_s: float = float("nan")
    #: Passenger-seconds of platform waiting **accrued during this epoch**, by everyone on a
    #: platform, whether or not they were served. This is the queue integral, and it is the only
    #: waiting measure that is well defined when nobody boards: a mean over boarders is NaN when
    #: the policy runs no service at all, which would make doing nothing look costless.
    waiting_accrued_passenger_s: float = 0.0
    #: Distinct passengers who spent any time on a platform during this epoch: those queuing at
    #: its start plus those who arrived during it. The denominator for a per-passenger mean.
    waiting_population: float = 0.0
    #: Passengers still queuing at the end of the epoch, across the whole line.
    queue_at_epoch_end: float = 0.0
    boarded: float = 0.0
    arrived: float = 0.0
    denied: float = 0.0
    left_behind: float = 0.0
    energy_units: float = 0.0
    cost_units: float = 0.0
    inductions: int = 0
    trains_in_service: int = 0
    headway_violations: int = 0
    platform_holds: int = 0
    failures: int = 0
    total_queue: float = 0.0
    max_queue: float = 0.0
    mean_occupancy_ratio: float = 0.0
    max_occupancy_ratio: float = 0.0
    wait_samples: list[tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = {k: v for k, v in self.__dict__.items() if k != "wait_samples"}
        return payload


class MetroSimulator:
    """Discrete-time simulator for one operating day on the modelled line."""

    def __init__(
        self,
        network: Network,
        demand_process: DemandProcess,
        config: SimulatorConfig,
    ) -> None:
        self.network = network
        self.demand_process = demand_process
        self.config = config

        self.n_stations = network.n_stations
        self.od = build_od_matrix(
            network.segment_km, config.od_model, decay_km=config.od_decay_km
        )
        self.direction_share = direction_split(self.od)
        self._destination_cache: dict[tuple[int, int], np.ndarray] = {
            (station, direction): destination_weights(self.od, station, direction)
            for station in range(self.n_stations)
            for direction in DIRECTIONS
        }

        steps_per_hour = int(round(3600.0 / config.internal_step_seconds))
        self._hour_weights = intra_hour_weights(steps_per_hour, config.intra_hour_profile)
        self._steps_per_hour = steps_per_hour

        self.trains: list[Train] = [
            Train(train_id=i, n_stations=self.n_stations) for i in range(config.fleet_size)
        ]
        self.queues: list[list[StationQueue]] = []
        self._rng: dict[str, np.random.Generator] = {}
        self.scenario: ScenarioSpec | None = None

        self.time_s = 0.0
        self.epoch = 0
        self._hourly_intensity: np.ndarray = np.zeros((24, self.n_stations))
        self._last_departure_s: np.ndarray = np.zeros((self.n_stations, 2))
        self._prev_departure_s: np.ndarray = np.zeros((self.n_stations, 2))
        self._platform_occupant: list[list[int | None]] = []
        self._all_wait_samples: list[tuple[float, float]] = []
        self._totals: dict[str, float] = {}
        self._last_induction_s: float = float("-inf")

    # ------------------------------------------------------------------ lifecycle

    def reset(self, scenario: ScenarioSpec, rng_streams: dict[str, np.random.Generator]) -> None:
        """Start a fresh operating day.

        Args:
            scenario: The realised scenario instance for this episode.
            rng_streams: Environment streams keyed by name. These depend only on
                ``(scenario, eval_seed)``, never on the method under evaluation, which is what
                makes the common-random-numbers guarantee hold.
        """
        self.scenario = scenario
        self._rng = rng_streams
        self.time_s = 0.0
        self.epoch = 0

        for train in self.trains:
            train.reset()

        # Start-of-day maintenance holds, drawn from the dedicated stream so that changing the
        # failure process cannot shift which trains start unavailable.
        n_maintenance = int(round(self.config.initial_maintenance_fraction * len(self.trains)))
        if n_maintenance > 0:
            held = rng_streams["maintenance"].choice(len(self.trains), size=n_maintenance, replace=False)
            for index in held:
                self.trains[int(index)].status = TrainStatus.MAINTENANCE

        self.queues = [[StationQueue() for _ in DIRECTIONS] for _ in range(self.n_stations)]
        self._last_departure_s = np.full((self.n_stations, 2), -1e9)
        self._prev_departure_s = np.full((self.n_stations, 2), -1e9)
        self._platform_occupant = [[None, None] for _ in range(self.n_stations)]
        self._all_wait_samples = []
        self._last_induction_s = float("-inf")
        self._totals = {
            "arrived": 0.0, "boarded": 0.0, "alighted": 0.0, "denied": 0.0, "left_behind": 0.0,
            "energy_units": 0.0, "cost_units": 0.0,
            "inductions": 0.0, "failures": 0.0,
            "headway_violations": 0.0, "platform_holds": 0.0,
            "train_km": 0.0, "train_stops": 0.0, "train_seconds": 0.0,
            "waiting_accrued_passenger_s": 0.0,
        }

        self._hourly_intensity = self.demand_process.hourly_intensity(
            scenario.day_type, rng_streams["demand"]
        )
        self._pending_failures = list(scenario.forced_failures)

    # ------------------------------------------------------------------ stepping

    def step_epoch(self, n_induct: int) -> EpochMetrics:
        """Advance one decision interval, inducting ``n_induct`` trains at its start."""
        metrics = EpochMetrics()

        # Inductions are *scheduled across* the decision interval, not fired instantaneously
        # at its start. Firing them together is physically wrong -- the terminal platform can
        # hold one train, so the second induction is always refused -- and it made actions 1
        # and 2 behaviourally identical, collapsing the action space to a binary choice. The
        # simulator now attempts a pending induction on each internal step, which is how a
        # real terminal dispatches: successive departures separated by the minimum headway.
        self._pending_inductions = int(np.clip(n_induct, 0, self.config.max_induct_per_decision))

        metrics.waiting_population = float(sum(q.size for row in self.queues for q in row))

        for _ in range(self.config.steps_per_decision):
            self._internal_step(metrics)

        in_service = [t for t in self.trains if t.in_service]
        metrics.trains_in_service = len(in_service)
        metrics.total_queue = float(sum(q.size for row in self.queues for q in row))
        metrics.queue_at_epoch_end = metrics.total_queue
        metrics.waiting_population += metrics.arrived
        metrics.max_queue = float(max((q.size for row in self.queues for q in row), default=0.0))
        if in_service:
            ratios = [t.occupancy / self.config.capacity_passengers for t in in_service]
            metrics.mean_occupancy_ratio = float(np.mean(ratios))
            metrics.max_occupancy_ratio = float(np.max(ratios))
        metrics.waiting_time_mean_s = weighted_mean(metrics.wait_samples)

        self.epoch += 1
        return metrics

    def _internal_step(self, metrics: EpochMetrics) -> None:
        """One fine-grained integration step."""
        dt = self.config.internal_step_seconds

        self._generate_arrivals(metrics)

        # Queue integral, taken after arrivals and before any boarding, so a passenger who
        # arrives and boards within the same step still accrues one step of waiting rather than
        # none. Rectangular integration at a 30 s step; the induced error is bounded by
        # half a step per passenger and is identical across every method, so it cannot favour
        # one policy over another.
        queued = sum(q.size for row in self.queues for q in row)
        metrics.waiting_accrued_passenger_s += queued * dt
        self._totals["waiting_accrued_passenger_s"] += queued * dt

        self._apply_forced_failures(metrics)
        self._try_pending_induction(metrics)
        self._advance_trains(metrics, dt)

        # Fleet cost accrues per train-second in service, so a policy that inducts a train and
        # leaves it running pays for the whole round trip.
        in_service = sum(1 for t in self.trains if t.in_service)
        service_cost = in_service * dt / 3600.0 * self.config.cost_units_per_train_hour
        metrics.cost_units += service_cost
        self._totals["cost_units"] += service_cost
        self._totals["train_seconds"] += in_service * dt
        for train in self.trains:
            if train.in_service:
                train.service_seconds += dt

        self.time_s += dt

    # ------------------------------------------------------------------ mechanisms

    def _generate_arrivals(self, metrics: EpochMetrics) -> None:
        """Realise passenger arrivals for this step (assumption A02 for the within-hour shape)."""
        elapsed = self.time_s
        hour_of_day = int(self.config.day_start_hour + elapsed // 3600) % 24
        step_in_hour = int((elapsed % 3600) // self.config.internal_step_seconds)
        weight = self._hour_weights[min(step_in_hour, self._steps_per_hour - 1)]
        scale = self.scenario.scale_at(elapsed) if self.scenario else 1.0

        rates = self._hourly_intensity[hour_of_day] * weight * scale
        if not np.any(rates > 0):
            return
        rates = np.clip(rates, 0.0, None)
        if self.config.deterministic_arrivals:
            counts = np.rint(rates).astype(int)
        else:
            counts = self._rng["demand"].poisson(rates)

        for station, count in enumerate(counts):
            if count <= 0:
                continue
            forward_share = self.direction_share[station]
            forward = float(count) * forward_share
            backward = float(count) - forward
            if forward > 0:
                self.queues[station][0].add(forward, self.time_s)
            if backward > 0:
                self.queues[station][1].add(backward, self.time_s)
            metrics.arrived += float(count)
            self._totals["arrived"] += float(count)

    def _try_pending_induction(self, metrics: EpochMetrics) -> None:
        """Attempt one scheduled induction, if the constraints currently allow it.

        Called every internal step. An induction that is not feasible now is retried on the
        next step and simply expires at the end of the decision interval if the terminal never
        frees up -- which is the correct behaviour: the policy asked for service the
        infrastructure could not deliver, and the shortfall shows up in the metrics rather
        than being silently granted.
        """
        if self._pending_inductions <= 0:
            return

        terminal = self.network.terminal_for(0)
        if not self._can_induct_now(terminal):
            return

        train = next((t for t in self.trains if t.available_for_induction), None)
        if train is None:
            self._pending_inductions = 0
            return

        train.status = TrainStatus.DWELL
        train.station = terminal
        train.direction = 0
        self._platform_occupant[terminal][0] = train.train_id
        # A newly inducted train must board the terminal's waiting passengers. Omitting this
        # stranded the entire origin terminus for the whole day.
        train.timer_s = self._service_station(train, metrics)

        self._pending_inductions -= 1
        self._last_induction_s = self.time_s
        metrics.inductions += 1
        cost = self.config.cost_units_per_induction
        metrics.cost_units += cost
        self._totals["cost_units"] += cost
        self._totals["inductions"] += 1

    def _can_induct_now(self, terminal: int) -> bool:
        """Feasibility of one more induction at this instant."""
        if not any(t.available_for_induction for t in self.trains):
            return False
        if self._platform_occupant[terminal][0] is not None:
            return False
        since_last = self.time_s - self._last_departure_s[terminal][0]
        return since_last >= self.network.min_headway_seconds

    def feasible_inductions(self) -> int:
        """How many inductions are feasible right now; drives the action mask."""
        terminal = self.network.terminal_for(0)
        available = sum(1 for t in self.trains if t.available_for_induction)
        if available == 0 or self._platform_occupant[terminal][0] is not None:
            return 0
        since_last = self.time_s - self._last_departure_s[terminal][0]
        if since_last < self.network.min_headway_seconds:
            return 0
        # Within one decision interval the platform can clear at most
        # interval / min_headway times.
        by_headway = int(self.config.decision_interval_seconds // self.network.min_headway_seconds)
        return int(min(available, self.config.max_induct_per_decision, max(1, by_headway)))

    def _apply_forced_failures(self, metrics: EpochMetrics) -> None:
        """Remove trains at scenario-scheduled disruption times."""
        while self._pending_failures and self._pending_failures[0][0] <= self.time_s:
            _, n_trains = self._pending_failures.pop(0)
            for _ in range(n_trains):
                candidates = [t for t in self.trains if t.in_service]
                if not candidates:
                    break
                index = int(self._rng["failures"].integers(len(candidates)))
                self._fail_train(candidates[index], metrics)

    def _fail_train(self, train: Train, metrics: EpochMetrics) -> None:
        """Take a train out of service; its passengers return to the platform.

        Returning them to the platform rather than deleting them preserves passenger
        conservation, which ``tests/test_simulator.py`` asserts. Deleting them would quietly
        improve every disruption metric.
        """
        stranded = train.occupancy
        if stranded > 0:
            queue = self.queues[train.station][train.direction]
            queue.add(stranded, self.time_s)
            # These passengers already arrived once; re-adding inflates `arrived`, so correct
            # it back out to keep the conservation identity exact.
            queue.arrived -= stranded
        if self._platform_occupant[train.station][train.direction] == train.train_id:
            self._platform_occupant[train.station][train.direction] = None
        train.load[:] = 0.0
        train.status = TrainStatus.FAILED
        metrics.failures += 1
        self._totals["failures"] += 1

    def _advance_trains(self, metrics: EpochMetrics, dt: float) -> None:
        """Advance every in-service train by one step."""
        for train in self.trains:
            if train.status in (TrainStatus.DEPOT, TrainStatus.FAILED, TrainStatus.MAINTENANCE):
                continue

            # A stopped train keeps its doors open: passengers arriving while it dwells -- or
            # while it is held for headway -- board it rather than waiting for the next train.
            # Without this a held train sits full-of-nobody while the platform fills behind it,
            # which manufactured convoy behaviour and inflated 95th-percentile waits far beyond
            # anything the headway could explain.
            if train.status == TrainStatus.DWELL:
                self._board_waiting(train, metrics)

            train.timer_s -= dt
            if train.timer_s > 1e-9:
                continue

            if train.status == TrainStatus.DWELL:
                self._handle_dwell_expiry(train, metrics)
            elif train.status == TrainStatus.RUNNING:
                self._handle_arrival(train, metrics)
            elif train.status == TrainStatus.TURNAROUND:
                self._handle_turnaround_expiry(train, metrics)

    def _handle_dwell_expiry(self, train: Train, metrics: EpochMetrics) -> None:
        """A dwelling train has finished boarding: depart, hold, or turn around."""
        station, direction = train.station, train.direction
        is_terminus = self.network.next_station(station, direction) is None

        if is_terminus:
            if direction == 0:
                train.status = TrainStatus.TURNAROUND
                train.timer_s = self.network.turnaround_seconds
                self._platform_occupant[station][direction] = None
            else:
                # Round trip complete: back to the depot and available again.
                train.trips_completed += 1
                self._platform_occupant[station][direction] = None
                train.status = TrainStatus.DEPOT
                train.timer_s = 0.0
            return

        since_last = self.time_s - self._last_departure_s[station][direction]
        if since_last < self.network.min_headway_seconds:
            # Held for headway. This is the constraint doing its job, not a violation.
            train.timer_s = self.config.internal_step_seconds
            metrics.platform_holds += 1
            self._totals["platform_holds"] += 1
            return

        self._record_denied_on_departure(train, metrics)
        # Keep the previous departure too, so the realised headway between consecutive
        # departures is observable. `_last_departure_s` alone gives only "time since the last
        # one", which is not a headway.
        self._prev_departure_s[station][direction] = self._last_departure_s[station][direction]
        self._last_departure_s[station][direction] = self.time_s
        self._platform_occupant[station][direction] = None
        train.status = TrainStatus.RUNNING
        train.timer_s = self.network.segment_time(station, direction)

    def _handle_arrival(self, train: Train, metrics: EpochMetrics) -> None:
        """A running train reaches the next station."""
        station, direction = train.station, train.direction
        nxt = self.network.next_station(station, direction)
        if nxt is None:  # defensive; a running train always has a next station
            train.status = TrainStatus.DEPOT
            return

        if self._platform_occupant[nxt][direction] is not None:
            # Platform busy: wait in the segment. Real congestion, and the reason short
            # headways stop paying off.
            train.timer_s = self.config.internal_step_seconds
            metrics.platform_holds += 1
            self._totals["platform_holds"] += 1
            return

        distance = self.network.segment_distance(station, direction)
        train.distance_km += distance
        train.stops_made += 1
        self._totals["train_km"] += distance
        self._totals["train_stops"] += 1

        energy = (
            distance * self.config.energy_units_per_train_km
            + self.config.energy_units_per_stop
        )
        metrics.energy_units += energy
        self._totals["energy_units"] += energy

        self._maybe_fail_on_distance(train, distance, metrics)
        if train.status == TrainStatus.FAILED:
            return

        train.station = nxt
        train.status = TrainStatus.DWELL
        self._platform_occupant[nxt][direction] = train.train_id
        train.timer_s = self._service_station(train, metrics)

    def _handle_turnaround_expiry(self, train: Train, metrics: EpochMetrics) -> None:
        """Reverse direction at the far terminus and board the return-direction queue.

        Boarding here is essential: the far terminus is the origin of every direction-1 trip,
        so skipping it strands that station's entire demand for the day.
        """
        train.direction = 1
        train.status = TrainStatus.DWELL
        self._platform_occupant[train.station][1] = train.train_id
        train.timer_s = self._service_station(train, metrics)

    def _service_station(self, train: Train, metrics: EpochMetrics) -> float:
        """Alight, board, and return the resulting dwell time."""
        station, direction = train.station, train.direction

        alighting = float(train.load[station])
        if alighting > 0:
            train.load[station] = 0.0
            # Tracked explicitly rather than derived from boardings. A passenger stranded by a
            # train failure returns to the platform and boards again, so `boarded` counts them
            # twice while `arrived` counts them once; only a direct alighting count makes the
            # conservation identity exact.
            self._totals["alighted"] += alighting

        boarded = self._board_waiting(train, metrics)

        dwell = (
            self.config.dwell_base_seconds
            + self.config.dwell_per_boarder_seconds * (boarded + alighting)
        )
        return float(min(dwell, self.config.dwell_max_seconds))

    def _board_waiting(self, train: Train, metrics: EpochMetrics) -> float:
        """Board whoever is waiting, up to remaining capacity. Returns the number boarded.

        Called on arrival and again on every step the train remains stopped, so a dwelling or
        held train keeps accepting passengers. Denial is *not* recorded here -- being unable to
        board this instant is not the same as being left behind, and that distinction is only
        resolved when the train actually departs.
        """
        station, direction = train.station, train.direction
        queue = self.queues[station][direction]
        capacity_left = max(0.0, self.config.capacity_passengers - train.occupancy)
        if capacity_left <= 0 or queue.size <= 0:
            return 0.0

        boarded, waits = queue.board(capacity_left, self.time_s)
        if boarded > 0:
            train.load += boarded * self._destination_cache[(station, direction)]
            metrics.wait_samples.extend(waits)
            self._all_wait_samples.extend(waits)
            metrics.boarded += boarded
            self._totals["boarded"] += boarded
        return boarded

    def _record_denied_on_departure(self, train: Train, metrics: EpochMetrics) -> None:
        """Anyone still queued as the train pulls out was genuinely left behind.

        Two quantities are recorded, and the distinction matters for the primary endpoint:
        the number of denial *events* (a passenger passed over by three trains contributes
        three) and the number of *distinct passengers* left behind at least once. Only the
        latter is a fraction of demand; the former is a rate that can exceed 1.
        """
        queue = self.queues[train.station][train.direction]
        remaining = queue.size
        if remaining > 0:
            newly_left_behind = queue.record_denied(remaining)
            metrics.denied += remaining
            metrics.left_behind += newly_left_behind
            self._totals["denied"] += remaining
            self._totals["left_behind"] += newly_left_behind

    def _maybe_fail_on_distance(self, train: Train, distance_km: float, metrics: EpochMetrics) -> None:
        """Apply the calibrated per-train-km failure hazard."""
        if not self.config.failures_enabled or self.config.failure_rate_per_km <= 0:
            return
        # `frailty` is 1.0 unless the environment's availability channel has drawn per-train
        # health states. It is drawn with mean exactly 1, so the fleet-aggregate hazard remains
        # the rate calibrated against published MDBF in M5.
        rate = self.config.failure_rate_per_km * train.frailty
        probability = 1.0 - np.exp(-rate * distance_km)
        if self._rng["failures"].random() < probability:
            self._fail_train(train, metrics)

    # ------------------------------------------------------------------ reporting

    @property
    def done(self) -> bool:
        return self.epoch >= self.config.n_decision_epochs

    def min_observed_headway_s(self) -> float:
        """Smallest realised gap between consecutive departures anywhere on the line today.

        Returns the operating-day length when fewer than two departures have occurred at every
        platform, which is the correct "no headway pressure yet" reading rather than a
        misleading zero.
        """
        gaps = self._last_departure_s - self._prev_departure_s
        valid = gaps[(self._prev_departure_s > -1e8) & (self._last_departure_s > -1e8)]
        if valid.size == 0:
            return float(self.config.day_seconds)
        return float(valid.min())

    def last_induction_s(self) -> float:
        """Simulation time of the most recent induction, or ``-inf`` if none yet."""
        return self._last_induction_s

    def residual_queue(self) -> float:
        """Passengers still waiting at end of day: the unmet-demand endpoint."""
        return float(sum(q.size for row in self.queues for q in row))

    def episode_summary(self) -> dict[str, Any]:
        """Full end-of-episode metrics.

        Every reward component is reported separately and in its own units, per research brief
        section 14: the evaluator must be able to recover waiting time, unmet demand, energy
        and cost from any experiment, never only the scalar reward.
        """
        residual_waits = [
            pair for row in self.queues for q in row for pair in q.residual_waits(self.time_s)
        ]
        served = self._all_wait_samples
        all_waits = served + residual_waits

        arrived = self._totals["arrived"]
        residual = self.residual_queue()

        return {
            # passenger service
            "waiting_time_mean_s": weighted_mean(served),
            "waiting_time_median_s": weighted_quantile(served, 0.50),
            "waiting_time_p95_s": weighted_quantile(served, 0.95),
            "waiting_time_mean_incl_unserved_s": weighted_mean(all_waits),
            # The queue integral over the whole day. Unlike the mean over boarders, this is
            # defined even when a policy runs no service, and it is the quantity a transit
            # operator actually minimises.
            "waiting_accrued_passenger_hours": self._totals["waiting_accrued_passenger_s"] / 3600.0,
            "passengers_arrived": arrived,
            "passengers_boarded": self._totals["boarded"],
            "passengers_alighted": self._totals["alighted"],
            "passengers_onboard_at_end": float(sum(t.load.sum() for t in self.trains)),
            # PRIMARY ENDPOINT (protocol amendment A-01): distinct passengers who watched at
            # least one full train depart without them, as a fraction of arrivals. This is the
            # standard transit reading of "unmet demand" and is bounded in [0, 1].
            "left_behind_passengers": self._totals["left_behind"],
            "left_behind_fraction": (self._totals["left_behind"] / arrived) if arrived > 0 else 0.0,
            # Secondary: denial EVENTS. A passenger passed over by three trains contributes
            # three, so this is a rate that may exceed 1 and is NOT a fraction of demand.
            "denied_boardings": self._totals["denied"],
            "denial_rate_per_passenger": (self._totals["denied"] / arrived) if arrived > 0 else 0.0,
            # Secondary: passengers never served by end of day.
            "unmet_demand": residual,
            "unmet_demand_fraction": (residual / arrived) if arrived > 0 else 0.0,
            "service_level": (self._totals["boarded"] / arrived) if arrived > 0 else 0.0,
            # operations
            "inductions": self._totals["inductions"],
            "trips_completed": float(sum(t.trips_completed for t in self.trains)),
            "headway_violations": self._totals["headway_violations"],
            "platform_holds": self._totals["platform_holds"],
            "failures": self._totals["failures"],
            "fleet_utilisation": float(
                sum(1 for t in self.trains if t.status != TrainStatus.DEPOT) / len(self.trains)
            ),
            "train_km": self._totals["train_km"],
            "train_stops": self._totals["train_stops"],
            "train_hours": self._totals["train_seconds"] / 3600.0,
            # energy and cost, in the units declared by the config
            "energy_units": self._totals["energy_units"],
            "cost_units": self._totals["cost_units"],
            # provenance of the endpoint definitions
            "units_note": (
                "energy is in train-km-equivalent units and cost in induction-equivalent "
                "units; neither is kWh nor currency, because no citable conversion has been "
                "sourced (assumptions A15/A16)"
            ),
        }

