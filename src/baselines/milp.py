"""MILP dispatch baselines (M9): static open-loop and rolling-horizon.

Protocol section 7.1 registers two:

* ``milp_static`` — solved once per day type over *average historical* demand, executed open-loop.
* ``milp_rolling`` — re-solved every epoch over the forecast horizon, receiving exactly the same
  mu and sigma-hat the RL arms see.

**The MILP is not deliberately weakened.** That is an explicit fairness rule in protocol section
7.1, and it is the rule most often broken in RL-versus-optimisation papers: a strawman MILP makes
any learned policy look good. Three things follow from taking it seriously here.

*The surrogate is calibrated from the simulator, not guessed.* A MILP optimising a made-up
queueing model would lose for reasons that have nothing to do with optimisation. The two
coefficients that matter — passengers served per in-service trainset per epoch, and round-trip
duration in epochs — are measured by running the simulator (:func:`calibrate_surrogate`) and
stored with their provenance.

*Solver status and optimality gap are reported for every instance.* An instance that hits the
time limit is reported as such, never silently treated as optimal.

*The rolling variant actually uses sigma.* Registering it as receiving sigma-hat and then feeding
it only mu would make it weaker than the registered method.

**Known approximation, stated rather than buried.** The simulator is a discrete-event queueing
model with per-station FIFO cohorts, finite train capacity and platform contention. The MILP
optimises a line-aggregate fluid queue. It therefore plans against a *proxy* for the true
objective — the queue integral is represented exactly, but left-behind passengers can only be
approximated by residual queue. Any advantage the RL arms show over the MILP is partly an
advantage over this abstraction gap, and that must be said in the results rather than claimed as
pure optimisation superiority.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.common.logging import get_logger
from src.common.paths import MODELS_DIR, ensure_dir
from src.environment.metro_env import MetroEnv
from src.environment.reward import WEIGHT_GRID

logger = get_logger(__name__)

__all__ = ["SurrogateParameters", "MILPPolicy", "solve_dispatch", "calibrate_surrogate",
           "SURROGATE_FILE"]

SURROGATE_FILE = MODELS_DIR / "milp_surrogate.json"

#: Robustness margin applied to the rolling MILP's demand estimate: it plans against
#: ``mu + Z_ROBUST * sigma``. Frozen at one standard deviation — the least arbitrary non-zero
#: choice, and non-zero because protocol section 7.1 registers this method as receiving sigma.
Z_ROBUST = 1.0

#: Rolling-horizon lookahead, in decision epochs. Three hours matches the forecaster's frozen
#: h in {1, 2, 3} horizon, so the MILP never plans on demand it has no forecast for.
ROLLING_HORIZON_EPOCHS = 36

#: Per-instance solve budget. Recorded in the results with the achieved status and gap.
TIME_LIMIT_S = 1.0


@dataclass
class SurrogateParameters:
    """Coefficients of the fluid queueing surrogate, measured from the simulator.

    Attributes:
        served_per_train_epoch: Service capacity per in-service trainset per decision epoch.
        round_trip_epochs: Epochs a trainset is occupied by one induction.
        energy_per_induction: Energy units attributable to one round trip.
        cost_per_induction: Dispatch cost of one induction.
        cost_per_train_epoch: Fleet cost of one trainset for one epoch in service.
        fleet_size: Trainsets available.
        wait_minutes_per_arrival: ``wait_minutes_per_arrival[k]`` is the passenger-minutes each
            arriving passenger accrues in an epoch where ``k`` trainsets are inducted. This is
            what a pure fluid queue misses: on a line with spare capacity the queue clears every
            epoch, so a capacity-only model concludes that service frequency is free. In reality
            passengers wait for the *next train*, and that wait is set by frequency. Because the
            action space has only three values, the dependence is represented exactly with three
            fitted coefficients rather than approximated.
        standing_queue_per_arrival: ``standing_queue_per_arrival[n]`` is the queue, per arriving
            passenger, that persists when ``n`` trainsets are in service (amendment A-11). A fluid
            queue drives ``q`` to zero whenever capacity exceeds demand, but passengers arrive
            continuously while trains arrive discretely, so a queue always stands between train
            arrivals and its size is set by the headway. Without this the surrogate predicted a
            mean queue of 0.5 where the simulator realised 1,018, and the MILP faced almost no
            penalty for under-dispatching.
        provenance: How each coefficient was obtained.
    """

    served_per_train_epoch: float
    round_trip_epochs: int
    energy_per_induction: float
    cost_per_induction: float
    cost_per_train_epoch: float
    fleet_size: int
    wait_minutes_per_arrival: list[float] = field(default_factory=lambda: [5.0, 2.5, 1.25])
    standing_queue_per_arrival: list[float] = field(default_factory=list)
    #: ``{window_epochs: most inductions the environment ever delivered in a window that long}``.
    #: The instantaneous bound is ``max_induct_per_decision``, but the terminal platform and the
    #: minimum headway make that unsustainable: a plan asking for two every interval is asking for
    #: something the environment cannot deliver, and 48% of it is clipped away. Measured from the
    #: always-request-two probe, so it is the loosest bound consistent with observed behaviour.
    max_inductions_per_window: dict[str, int] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def standing_queue_table(self) -> list[float]:
        """Per-``n`` table, padded to the fleet size. Empty means the term is disabled."""
        if not self.standing_queue_per_arrival:
            return []
        table = list(self.standing_queue_per_arrival)
        while len(table) <= self.fleet_size:
            table.append(table[-1])
        return table[: self.fleet_size + 1]

    def save(self, path: Path = SURROGATE_FILE) -> Path:
        ensure_dir(path.parent)
        payload = {k: v for k, v in self.__dict__.items()}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path = SURROGATE_FILE) -> "SurrogateParameters":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)


def calibrate_surrogate(
    *,
    seeds: tuple[int, ...] = tuple(range(3001, 3005)),
    trained_forecaster: bool = False,
) -> SurrogateParameters:
    """Measure the surrogate's coefficients by running the fixed timetable in the simulator.

    Calibrating rather than guessing is what keeps the MILP a real baseline. Seeds 3001+ are
    disjoint from every frozen pool, so no coefficient is fitted on an evaluation day.
    """
    from src.baselines.fixed_timetable import FixedTimetable, FixedTimetablePolicy
    from src.evaluation.policies import ConstantPolicy, run_episode
    from src.evaluation.runner import Cell, build_env

    cell = Cell(method="fixed_timetable", scenario="normal", eval_seed=0)
    env = build_env(cell, trained_forecaster=trained_forecaster)

    energy_per: list[float] = []
    inductions: list[float] = []
    train_epochs: list[float] = []
    #: (queue at epoch start, arrivals, trainsets in service, queue at epoch end) per epoch.
    trajectory: list[tuple[float, float, float, float]] = []
    #: (action, arrivals, queue at epoch start, accrued passenger-minutes) per epoch.
    waiting: list[tuple[int, float, float, float]] = []
    #: As `trajectory`, but pooled over every probe, for the standing-queue fit (A-11).
    standing: list[tuple[float, float, float, float]] = []
    #: Executed action sequences from the always-two probe, for the dispatch-capacity fit (A-13).
    saturating: list[np.ndarray] = []

    # Calibrating on the fixed timetable alone would give no variation in the action, so the
    # frequency-dependent waiting term could not be identified. The constant policies supply
    # that variation; all four are deterministic, so the calibration is reproducible.
    probes = {
        "fixed_timetable": FixedTimetablePolicy(FixedTimetable.load()),
        "constant_0": ConstantPolicy(0),
        "constant_1": ConstantPolicy(1),
        "constant_2": ConstantPolicy(2),
    }

    for name, policy in probes.items():
        for seed in seeds:
            result = run_episode(env, policy, eval_seed=seed)
            summary = result.summary
            frame = result.epochs
            in_service = np.array([r["trains_in_service"] for r in frame], dtype=float)
            arrivals = np.array([r["arrived"] for r in frame], dtype=float)
            queue_end = np.array([r["queue_at_epoch_end"] for r in frame], dtype=float)
            queue_start = np.concatenate([[0.0], queue_end[:-1]])
            accrued = np.array([r["W_minutes"] for r in frame], dtype=float)
            actions = np.array(result.executed_actions, dtype=int)

            waiting.extend(zip(actions, arrivals, queue_start, accrued))
            # Every probe contributes to the standing-queue fit: the term is a function of
            # trainsets in service, and the fixed timetable alone spans too narrow a range of n
            # to identify it.
            standing.extend(zip(queue_start, arrivals, in_service, queue_end))
            if name == "constant_2":
                # The saturating probe: what the environment delivers when asked for the maximum
                # in every interval. That is the physical dispatch ceiling.
                saturating.append(actions.copy())

            if name == "fixed_timetable":
                trajectory.extend(zip(queue_start, arrivals, in_service, queue_end))
                n_inductions = float(summary["inductions"])
                if n_inductions > 0:
                    energy_per.append(float(summary["energy_units"]) / n_inductions)
                inductions.append(n_inductions)
                train_epochs.append(float(in_service.sum()))

    served, kappa_error = _fit_service_rate(trajectory)
    wait_coefficients, wait_error = _fit_wait_coefficients(waiting, k_max=2)
    standing_queue, standing_provenance = _fit_standing_queue(
        standing, fleet_size=int(env.sim_config.fleet_size))
    dispatch_caps, dispatch_provenance = _fit_dispatch_capacity(
        saturating, k_max=int(env.cfg.k_max))

    # A trainset occupied for `train_epochs / inductions` epochs per induction: that is the
    # round trip as the simulator actually realises it, including platform holds and turnaround,
    # not the idealised sum of segment times.
    round_trip = float(np.mean(np.array(train_epochs) / np.maximum(np.array(inductions), 1.0)))

    parameters = SurrogateParameters(
        served_per_train_epoch=served,
        round_trip_epochs=max(1, int(round(round_trip))),
        energy_per_induction=float(np.mean(energy_per)),
        cost_per_induction=float(env.sim_config.cost_units_per_induction),
        cost_per_train_epoch=float(
            env.sim_config.cost_units_per_train_hour
            * env.sim_config.decision_interval_seconds / 3600.0
        ),
        fleet_size=int(env.sim_config.fleet_size),
        wait_minutes_per_arrival=wait_coefficients,
        standing_queue_per_arrival=standing_queue,
        max_inductions_per_window=dispatch_caps,
        provenance={
            "method": (
                "measured by running the fixed-timetable baseline in the simulator; the MILP "
                "surrogate is calibrated, not assumed, so that any RL advantage is not an "
                "artefact of a strawman optimisation model"
            ),
            "seeds": list(seeds),
            "seed_pool_note": "3001+ is disjoint from the frozen evaluation and training pools",
            "episodes": len(seeds),
            "observed_round_trip_epochs": round_trip,
            "service_rate_fit": {
                "objective": "least squares on the simulator's realised queue path",
                "rms_queue_error_passengers": kappa_error,
                "note": (
                    "kappa is fitted so the fluid recursion q_t = max(0, q_{t-1} + d_t - "
                    "kappa*n_t) reproduces the queue the simulator actually realises. The "
                    "obvious alternative -- average realised boardings per trainset per epoch -- "
                    "is demand-limited, not capacity-limited, on a line with spare capacity: it "
                    "measured 71 passengers/train/epoch, which told the MILP that service and "
                    "demand were exactly matched and that extra trainsets bought nothing. The "
                    "static plan then dispatched 137 trains instead of 228 and left 17.6 % of "
                    "passengers behind. That would have been a strawman baseline, which "
                    "protocol section 7.1 forbids."
                ),
            },
            "wait_coefficient_fit": {
                "objective": (
                    "least squares of accrued passenger-minutes on (arrivals x action-specific "
                    "coefficient) + (queue at epoch start x epoch length)"
                ),
                "probe_policies": list(probes),
                "rms_error_passenger_minutes": wait_error,
                "coefficients_minutes_per_arrival": wait_coefficients,
                "note": (
                    "A pure fluid queue prices only capacity. On a line with spare capacity it "
                    "therefore concludes that frequency is free, dispatches the minimum, and "
                    "strands a quarter of demand -- which is exactly the strawman baseline "
                    "protocol section 7.1 forbids. These three coefficients price the wait for "
                    "the *next train*. Because the action space has only three values the "
                    "dependence is exact, not approximated."
                ),
            },
            "dispatch_capacity_fit": {**dispatch_provenance, "amendment": "A-13"},
            "standing_queue_fit": {
                **standing_provenance,
                "amendment": "A-11",
                "probe_policies": list(probes),
                "why": (
                    "Without this term the surrogate predicted a mean queue of 0.5 passengers "
                    "under its own plan while the simulator realised 1,018 -- a factor of 1,856. "
                    "The MILP's objective was then dominated by induction and service cost, so "
                    "under-dispatching looked nearly free: it planned 164 trains against the "
                    "fixed timetable's 228 and stranded three times as many passengers. Scoring "
                    "both schedules under the surrogate and under the simulator ranked them "
                    "oppositely, which localised the fault to this missing term."
                ),
            },
            "calibrated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    logger.info(
        "MILP surrogate: %.1f passengers/train/epoch, round trip %d epochs, "
        "%.2f energy units/induction",
        parameters.served_per_train_epoch, parameters.round_trip_epochs,
        parameters.energy_per_induction,
    )
    return parameters


def _fit_service_rate(
    trajectory: list[tuple[float, float, float, float]],
    *,
    low: float = 1.0,
    high: float = 400.0,
    steps: int = 800,
) -> tuple[float, float]:
    """Fit the per-trainset service rate to the simulator's realised queue path.

    Minimises the squared error of the one-step fluid recursion
    ``q_hat = max(0, q_prev + arrivals - kappa * n)`` against the queue the simulator actually
    produced, over a dense grid. One free parameter over a bounded range does not need anything
    cleverer, and a grid cannot get stuck in the kink introduced by the ``max``.

    Returns:
        ``(kappa, rms_error_in_passengers)``.
    """
    if not trajectory:
        return 100.0, float("nan")
    data = np.asarray(trajectory, dtype=float)
    q_prev, arrivals, n_service, q_true = data.T

    grid = np.linspace(low, high, steps)
    errors = np.empty_like(grid)
    for index, kappa in enumerate(grid):
        predicted = np.maximum(0.0, q_prev + arrivals - kappa * n_service)
        errors[index] = np.mean((predicted - q_true) ** 2)

    best = int(np.argmin(errors))
    return float(grid[best]), float(np.sqrt(errors[best]))


def _fit_wait_coefficients(
    observations: list[tuple[int, float, float, float]],
    *,
    k_max: int,
    epoch_minutes: float = 5.0,
) -> tuple[list[float], float]:
    """Fit per-action waiting coefficients against the simulator's realised queue integral.

    Model: ``accrued_t = w[x_t] * arrivals_t + epoch_minutes * queue_start_t``. The second term
    is fixed, not fitted -- passengers already queuing at the start of an epoch accrue the whole
    epoch, which is arithmetic rather than a modelling choice. Only the ``k_max + 1`` waiting
    coefficients are free, and each is fitted from the epochs that actually used that action, so
    they are identified separately rather than being three views of one average.

    Returns:
        ``(coefficients, rms_error_in_passenger_minutes)``.
    """
    if not observations:
        return [epoch_minutes] * (k_max + 1), float("nan")

    data = np.asarray(observations, dtype=float)
    actions, arrivals, queue_start, accrued = data.T
    residual = accrued - epoch_minutes * queue_start

    coefficients: list[float] = []
    squared_errors: list[np.ndarray] = []
    for k in range(k_max + 1):
        selected = actions == k
        if not selected.any() or arrivals[selected].sum() <= 0:
            # No evidence for this action; fall back to the headway-free upper bound of a full
            # epoch of waiting rather than inventing a number that looks measured.
            coefficients.append(epoch_minutes)
            continue
        # Least squares through the origin: w = sum(a*r) / sum(a*a).
        a, r = arrivals[selected], residual[selected]
        w = float(np.dot(a, r) / max(np.dot(a, a), 1e-9))
        w = float(np.clip(w, 0.0, epoch_minutes * 20.0))
        coefficients.append(w)
        squared_errors.append((r - w * a) ** 2)

    error = float(np.sqrt(np.mean(np.concatenate(squared_errors)))) if squared_errors else float("nan")

    # No monotone clamp (amendment A-12). An earlier version forced w to be non-increasing in k,
    # reasoning that "waiting must not increase with more service, or the MILP would learn to
    # withhold trains". That conflates two different quantities. More service *over time* does
    # reduce waiting, and the queue recursion plus the standing-queue floor price that correctly.
    # More service *at one instant* -- two trainsets in a single decision interval -- bunches
    # departures, and the simulator says bunching increases waiting: the unclamped fit gives
    # w2 = 0.680 against w1 = 0.633, and the always-two probe strands five times as many
    # passengers as the always-one probe for the same number of trains. Clamping overwrote that
    # measurement with an assumption and left the MILP believing a second trainset was free
    # capacity, which is why it burst and underperformed a fixed timetable.
    return coefficients, error


#: Window lengths, in decision epochs, over which the sustainable induction rate is measured.
DISPATCH_RATE_WINDOWS: tuple[int, ...] = (2, 4, 8, 12, 20, 36)


def _fit_dispatch_capacity(
    executed: list[np.ndarray], *, k_max: int
) -> tuple[dict[str, int], dict[str, Any]]:
    """The most inductions the environment ever delivered over each window length.

    `feasible_inductions` returns zero while the terminal platform is occupied or the minimum
    headway has not elapsed, so the instantaneous bound `max_induct_per_decision` is not
    sustainable. Without this the MILP plans two inductions in nearly every interval, has half its
    plan clipped away, and executes a bursty schedule -- and bursting is materially worse than
    even spacing: the always-two probe strands five times as many passengers as the always-one
    probe for the same number of trains.

    Taken as the **maximum** over probe episodes, so the bound is the loosest one consistent with
    what the environment has actually been observed to deliver. It can therefore never make the
    baseline weaker than physics allows.
    """
    if not executed:
        return {}, {"status": "no probe episodes; the dispatch-rate constraint is disabled"}

    caps: dict[str, int] = {}
    for window in DISPATCH_RATE_WINDOWS:
        best = 0
        for actions in executed:
            if len(actions) < window:
                continue
            sums = np.convolve(actions, np.ones(window, dtype=int), mode="valid")
            best = max(best, int(sums.max()))
        if best and best < window * k_max:  # a non-binding cap is not worth a constraint
            caps[str(window)] = best

    provenance = {
        "model": "sum of inductions over any window of W epochs <= observed maximum",
        "source": "always-request-two probe; the maximum over probe episodes",
        "windows": {w: {"cap": caps[w], "per_epoch": caps[w] / int(w)} for w in caps},
        "why": (
            "the terminal platform and the minimum headway make max_induct_per_decision "
            "unsustainable; a plan that assumes otherwise is infeasible and is clipped, and the "
            "clipped result is a bursty schedule rather than the planned one"
        ),
        "note": (
            "this is a feasibility envelope measured from the environment, in the same category "
            "as the fleet-size constraint -- not a performance hint. It is the observed maximum, "
            "so it cannot bind tighter than what the environment has delivered"
        ),
    }
    return caps, provenance


#: Bucket edges for the standing-queue fit, in trainsets in service. Coarse where the probes give
#: few epochs and finer where they give many; the fit reports the epoch count behind each bucket
#: so a thinly-supported one is visible rather than implied.
STANDING_QUEUE_BUCKETS: tuple[int, ...] = (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 10_000)


def _fit_standing_queue(
    observations: list[tuple[float, float, float, float]],
    *,
    fleet_size: int,
) -> tuple[list[float], dict[str, Any]]:
    """Fit the queue that persists per arriving passenger, as a function of trainsets in service.

    Model: ``queue_end_t >= phi[n_t] * arrivals_t``, with ``phi`` fitted by least squares through
    the origin within each bucket of ``n``, then made non-increasing in ``n``.

    Epochs with no trainset in service are excluded: with no service the queue diverges without
    bound, so those epochs describe the absence of a system rather than its behaviour, and
    including them would let one degenerate probe dominate the fit.

    A functional form was considered and rejected: a ``c/n`` fit explained R^2 = 0.07 of the
    variance. The bucketed table is the resolution the probe data actually supports.

    Args:
        observations: ``(queue_start, arrivals, trains_in_service, queue_end)`` per epoch.
        fleet_size: Upper bound of the returned table.

    Returns:
        ``(table indexed by n, provenance)``.
    """
    if not observations:
        return [], {"status": "no observations; the standing-queue term is disabled"}

    data = np.asarray(observations, dtype=float)
    _, arrivals, in_service, queue_end = data.T
    served = in_service >= 1
    if not served.any():
        return [], {"status": "no epoch had a trainset in service"}

    arrivals, in_service, queue_end = arrivals[served], in_service[served], queue_end[served]

    fitted: list[tuple[int, int, float, int]] = []  # (lo, hi, phi, n_epochs)
    for lo, hi in zip(STANDING_QUEUE_BUCKETS[:-1], STANDING_QUEUE_BUCKETS[1:]):
        selected = (in_service >= lo) & (in_service < hi)
        a = arrivals[selected]
        if selected.sum() < 5 or float(np.dot(a, a)) <= 0:
            continue  # too thin to fit; the neighbouring buckets carry it
        phi = float(np.dot(a, queue_end[selected]) / np.dot(a, a))
        fitted.append((lo, hi, max(0.0, phi), int(selected.sum())))

    if not fitted:
        return [], {"status": "no bucket had enough epochs to fit"}

    # More service must not predict a larger standing queue, or the MILP would learn to withhold
    # trains -- the same guard the waiting coefficients carry, and for the same reason.
    for index in range(1, len(fitted)):
        lo, hi, phi, count = fitted[index]
        fitted[index] = (lo, hi, min(phi, fitted[index - 1][2]), count)

    table: list[float] = []
    for n in range(fleet_size + 1):
        if n < fitted[0][0]:
            table.append(fitted[0][2])  # below the first fitted bucket: use its value
            continue
        value = fitted[-1][2]
        for lo, hi, phi, _ in fitted:
            if lo <= n < hi:
                value = phi
                break
        table.append(value)

    predicted = np.array([table[min(int(n), fleet_size)] for n in in_service]) * arrivals
    rms = float(np.sqrt(np.mean((queue_end - predicted) ** 2)))

    provenance = {
        "model": "queue_end >= phi[trains_in_service] * arrivals",
        "objective": "least squares through the origin within each bucket of trains in service",
        "rms_error_passengers": rms,
        "excluded": "epochs with no trainset in service; the queue diverges there",
        "monotone": "phi is made non-increasing in n, so more service can never predict a larger queue",
        "rejected_alternative": (
            "phi = c / n was fitted and rejected: it explained R^2 = 0.07 of the variance, so a "
            "bucketed table at the resolution the data supports was used instead"
        ),
        "buckets": [
            {"trains_in_service": f"[{lo},{hi if hi < 10_000 else fleet_size}]",
             "phi_per_arrival": phi, "epochs": count}
            for lo, hi, phi, count in fitted
        ],
    }
    return table, provenance


@dataclass
class SolveReport:
    """Solver outcome for one instance. Reported, never discarded."""

    status: str
    objective: float
    best_bound: float
    gap: float
    wall_time_s: float
    hit_time_limit: bool


def solve_dispatch(
    demand: np.ndarray,
    surrogate: SurrogateParameters,
    weights,
    scales,
    *,
    k_max: int = 2,
    initial_queue: float = 0.0,
    initial_in_service: np.ndarray | None = None,
    time_limit_s: float = TIME_LIMIT_S,
) -> tuple[np.ndarray, SolveReport]:
    """Solve the induction schedule over a horizon.

    The model, in the notation of protocol section 6.4::

        minimise  sum_t [ alpha*q_t/ref_W' + beta*q_t/ref_U' + gamma*E*x_t/ref_E
                          + delta*(c_i*x_t + c_e*n_t)/ref_C ]

        q_t  >=  q_{t-1} + d_t - kappa * n_t        (fluid queue, never negative)
        q_t  >=  0
        n_t  =   carried-over in-service trainsets + sum_{tau in window} x_tau
        n_t  <=  fleet_size
        x_t  in  {0, ..., k_max}

    ``q_t`` carries both the waiting term (it *is* the queue integral, up to the epoch length)
    and the unmet-demand term (residual queue proxies left-behind passengers). Representing the
    left-behind count exactly would need per-train capacity accounting the fluid model does not
    have; the approximation is stated in the module docstring and in the results.

    Returns:
        The induction sequence and a :class:`SolveReport` carrying status and optimality gap.
    """
    from ortools.sat.python import cp_model

    horizon = len(demand)
    model = cp_model.CpModel()

    # CP-SAT is integral, so passengers are rounded to whole people -- which they are -- and the
    # objective coefficients are scaled to integers below.
    queue_ceiling = int(max(1.0, float(np.sum(np.maximum(demand, 0.0))) + initial_queue) + 1)

    x = [model.NewIntVar(0, k_max, f"x{t}") for t in range(horizon)]
    q = [model.NewIntVar(0, queue_ceiling, f"q{t}") for t in range(horizon)]

    # One-hot indicators for the action, so the frequency-dependent waiting term is represented
    # exactly. This is only tractable because the action space has three values.
    b = [[model.NewBoolVar(f"b{t}_{k}") for k in range(k_max + 1)] for t in range(horizon)]
    for t in range(horizon):
        model.AddExactlyOne(b[t])
        model.Add(x[t] == sum(k * b[t][k] for k in range(k_max + 1)))

    carried = (
        np.zeros(horizon, dtype=int) if initial_in_service is None
        else np.asarray(initial_in_service, dtype=int)[:horizon]
    )
    if carried.size < horizon:
        carried = np.pad(carried, (0, horizon - carried.size))

    window = surrogate.round_trip_epochs
    kappa = max(1, int(round(surrogate.served_per_train_epoch)))

    in_service: list[Any] = []
    for t in range(horizon):
        contributions = [x[tau] for tau in range(max(0, t - window + 1), t + 1)]
        n_t = model.NewIntVar(0, surrogate.fleet_size, f"n{t}")
        model.Add(n_t == sum(contributions) + int(carried[t]))
        in_service.append(n_t)

    # Sustainable dispatch capacity (amendment A-13). `k_max` bounds one interval, but the
    # terminal platform and the minimum headway make that rate unsustainable over time. Without
    # these cuts the model plans two inductions in nearly every interval, half the plan is clipped
    # away by the action mask, and what executes is a bursty schedule -- which strands five times
    # as many passengers as an evenly spaced one delivering the same number of trains.
    for window_text, cap in sorted(surrogate.max_inductions_per_window.items(), key=lambda kv: int(kv[0])):
        window_length = int(window_text)
        if window_length > horizon:
            continue
        for start in range(horizon - window_length + 1):
            model.Add(sum(x[start:start + window_length]) <= int(cap))

    previous = int(round(initial_queue))
    for t in range(horizon):
        arrivals = int(round(max(0.0, float(demand[t]))))
        # q_t >= q_{t-1} + d_t - kappa * n_t, with q_t >= 0 already in the domain. Minimisation
        # drives it to equality, so the inequality is exact at the optimum.
        if t == 0:
            model.Add(q[t] >= previous + arrivals - kappa * in_service[t])
        else:
            model.Add(q[t] >= q[t - 1] + arrivals - kappa * in_service[t])

    # Standing-queue floor (amendment A-11). Passengers arrive continuously and trains arrive
    # discretely, so a queue stands between train arrivals whose size is set by the headway --
    # that is, by trainsets in service. The fluid recursion above has no representation of it and
    # drove q to zero whenever capacity exceeded demand, which is why the surrogate predicted a
    # mean queue of 0.5 where the simulator realised 1,018. `AddElement` indexes the calibrated
    # table by n_t exactly, at full resolution and with one constraint per epoch.
    standing_table = surrogate.standing_queue_table()
    if standing_table:
        for t in range(horizon):
            arrivals = max(0.0, float(demand[t]))
            floor_values = [int(round(phi * arrivals)) for phi in standing_table]
            if max(floor_values) <= 0:
                continue
            floor = model.NewIntVar(0, max(floor_values), f"qfloor{t}")
            model.AddElement(in_service[t], floor_values, floor)
            model.Add(q[t] >= floor)

    # Objective coefficients, scaled to integers. The reference scales come from the frozen
    # reward configuration so the MILP optimises the same trade-off the RL arms are scored on.
    epoch_minutes = 5.0
    SCALE = 1000
    w_queue = (
        weights.alpha * epoch_minutes / scales.ref_W + weights.beta / scales.ref_U
    )
    w_induct = (
        weights.gamma * surrogate.energy_per_induction / scales.ref_E
        + weights.delta * surrogate.cost_per_induction / scales.ref_C
    )
    w_service = weights.delta * surrogate.cost_per_train_epoch / scales.ref_C

    wait_coefficients = list(surrogate.wait_minutes_per_arrival)
    while len(wait_coefficients) <= k_max:
        wait_coefficients.append(wait_coefficients[-1])

    terms = []
    terms += [int(round(w_queue * SCALE)) * q[t] for t in range(horizon)]
    terms += [int(round(w_induct * SCALE)) * x[t] for t in range(horizon)]
    terms += [int(round(w_service * SCALE)) * in_service[t] for t in range(horizon)]
    # The wait-for-the-next-train term: arrivals in epoch t each accrue `wait_coefficients[k]`
    # minutes when k trainsets are inducted. Without it the model prices capacity only, and on a
    # line with spare capacity concludes that frequency is free.
    for t in range(horizon):
        arrivals = max(0.0, float(demand[t]))
        for k in range(k_max + 1):
            coefficient = weights.alpha * wait_coefficients[k] * arrivals / scales.ref_W
            terms.append(int(round(coefficient * SCALE)) * b[t][k])

    model.Minimize(sum(terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 1  # deterministic; parallelism is at the cell level
    status = solver.Solve(model)

    names = {
        cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE", cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }
    status_name = names.get(status, str(status))

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        schedule = np.array([solver.Value(var) for var in x], dtype=int)
        objective = float(solver.ObjectiveValue())
        bound = float(solver.BestObjectiveBound())
        gap = abs(objective - bound) / max(1.0, abs(objective))
    else:
        # No feasible schedule found inside the budget. Falling back to "induct nothing" would
        # be a silent, catastrophic choice, so fall back to steady minimal service and record
        # the status so the instance is visibly not an optimisation result.
        schedule = np.ones(horizon, dtype=int)
        objective = float("nan")
        bound = float("nan")
        gap = float("nan")
        logger.warning("MILP instance returned %s; falling back to constant service", status_name)

    report = SolveReport(
        status=status_name,
        objective=objective,
        best_bound=bound,
        gap=gap,
        wall_time_s=float(solver.WallTime()),
        hit_time_limit=bool(solver.WallTime() >= time_limit_s * 0.99
                            and status != cp_model.OPTIMAL),
    )
    return schedule, report


class MILPPolicy:
    """Static open-loop or rolling-horizon MILP dispatch."""

    def __init__(self, mode: str = "static", surrogate: SurrogateParameters | None = None) -> None:
        if mode not in ("static", "rolling"):
            raise ValueError(f"mode must be 'static' or 'rolling', got {mode!r}")
        self.mode = mode
        self.method_id = f"milp_{mode}"
        self.surrogate = surrogate or SurrogateParameters.load()
        self.reports: list[SolveReport] = []
        self._plan: np.ndarray | None = None
        self._mu_index: slice | None = None
        self._sigma_index: slice | None = None
        self._queue_index: slice | None = None
        self._epochs_per_hour = 12

    # ------------------------------------------------------------------ policy interface

    def reset(self, env: MetroEnv) -> None:
        self._mu_index = env.obs_spec.slice_of("forecast_mu_h1")
        self._sigma_index = env.obs_spec.slice_of("forecast_sigma_h1")
        self._queue_index = env.obs_spec.slice_of("queue_length")
        self._epochs_per_hour = env.epochs_per_hour
        self.reports = []
        self._plan = None

        if self.mode == "static":
            # Solved once, on *average historical* demand, and then executed open-loop --
            # exactly the manuscript's section 6 static plan. It never sees the realised day.
            demand = self._average_daily_demand(env)
            weights = WEIGHT_GRID[env.cfg.weight_id]
            self._plan, report = solve_dispatch(
                demand, self.surrogate, weights, env.scales,
                k_max=env.cfg.k_max, time_limit_s=TIME_LIMIT_S * 10,
            )
            self.reports.append(report)

    def act(self, observation: np.ndarray, env: MetroEnv, mask: np.ndarray) -> int:
        epoch = env.simulator.epoch
        if self.mode == "static":
            assert self._plan is not None
            return int(self._plan[epoch]) if epoch < len(self._plan) else 0
        return self._rolling_act(env, epoch)

    # ------------------------------------------------------------------ internals

    def _average_daily_demand(self, env: MetroEnv) -> np.ndarray:
        """Line-aggregate expected arrivals per epoch, from the fitted demand process.

        This is the *historical average* day for the day type — the only information protocol
        section 7.1 grants ``milp_static``. It is read from the fitted process's mean, not from
        the realised episode.
        """
        mean = env.demand_process.mean[env.cfg.day_type]  # (24, n_stations)
        per_hour = mean.sum(axis=1)
        epochs = env.sim_config.n_decision_epochs
        per_epoch = np.zeros(epochs)
        for epoch in range(epochs):
            hour = (env.sim_config.day_start_hour + epoch // self._epochs_per_hour) % 24
            per_epoch[epoch] = per_hour[hour] / self._epochs_per_hour
        return per_epoch

    def _carried_in_service(self, env: MetroEnv, epoch: int, horizon: int) -> np.ndarray:
        """Trainsets already in service at each epoch of the horizon, from past inductions.

        A trainset inducted at epoch ``e`` occupies the fleet through ``e + round_trip - 1``, so at
        a future epoch ``f`` the inductions still outstanding are those with
        ``f - round_trip + 1 <= e <= epoch - 1``. That is exactly computable from the episode's own
        action record, and `solve_dispatch` counts future inductions separately through its
        rolling window, so the two are complementary rather than overlapping.

        The previous version set only ``carried[0]`` and left the rest zero, which told the solver
        the entire fleet freed up after a single epoch. It therefore planned schedules the fleet
        could not execute: 46% of its requested inductions were clipped by the action mask, and the
        resulting dispatches were bunched rather than evenly spaced.
        """
        window = self.surrogate.round_trip_epochs
        records = env.episode_records()
        actions = np.zeros(max(epoch, 0), dtype=int)
        for record in records[:epoch]:
            index = int(record["epoch"])
            if 0 <= index < len(actions):
                actions[index] = int(record["action"])

        carried = np.zeros(horizon, dtype=int)
        for t in range(horizon):
            first = epoch + t - window + 1
            low, high = max(0, first), epoch  # past epochs only; future ones are the x variables
            if low < high:
                carried[t] = int(actions[low:high].sum())
        return np.minimum(carried, self.surrogate.fleet_size)

    def _rolling_act(self, env: MetroEnv, epoch: int) -> int:
        raw = env.raw_observation()
        mu_hour = float(raw[self._mu_index].sum())
        sigma_hour = float(np.sqrt((raw[self._sigma_index] ** 2).sum()))
        queue = float(raw[self._queue_index].sum())

        # The registered information set for this method is (mu, sigma-hat) -- the same one the
        # RL arms get. Planning against mu + Z*sigma is how a scheduler uses a dispersion
        # estimate; feeding it mu alone would make it weaker than the method registered.
        per_epoch = max(0.0, (mu_hour + Z_ROBUST * sigma_hour)) / self._epochs_per_hour
        horizon = min(ROLLING_HORIZON_EPOCHS, env.sim_config.n_decision_epochs - epoch)
        if horizon <= 0:
            return 0
        demand = np.full(horizon, per_epoch)

        in_service = self._carried_in_service(env, epoch, horizon)

        schedule, report = solve_dispatch(
            demand, self.surrogate, WEIGHT_GRID[env.cfg.weight_id], env.scales,
            k_max=env.cfg.k_max, initial_queue=queue, initial_in_service=in_service,
        )
        self.reports.append(report)
        return int(schedule[0])

    # ------------------------------------------------------------------ reporting

    def solver_summary(self) -> dict[str, Any]:
        """Per-instance solver statistics, required by protocol section 7.1."""
        if not self.reports:
            return {}
        times = np.array([r.wall_time_s for r in self.reports])
        gaps = np.array([r.gap for r in self.reports], dtype=float)
        statuses: dict[str, int] = {}
        for report in self.reports:
            statuses[report.status] = statuses.get(report.status, 0) + 1
        return {
            "instances": len(self.reports),
            "status_counts": statuses,
            "hit_time_limit": int(sum(r.hit_time_limit for r in self.reports)),
            "solve_time_mean_s": float(times.mean()),
            "solve_time_p95_s": float(np.quantile(times, 0.95)),
            "optimality_gap_mean": float(np.nanmean(gaps)) if np.isfinite(gaps).any() else float("nan"),
            "optimality_gap_max": float(np.nanmax(gaps)) if np.isfinite(gaps).any() else float("nan"),
            "time_limit_s": TIME_LIMIT_S,
            "robustness_margin_z": Z_ROBUST if self.mode == "rolling" else None,
        }
