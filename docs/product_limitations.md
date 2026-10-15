# Product limitations — Metro Intelligent Dispatch

Stated plainly, because a decision-support prototype that overstates itself is worse than none.

## What this system is not

- **Not connected to any real metro control system.** It reads no live feed and issues no command.
- **Not autonomous.** It recommends; an operator decides. There is no execution path to
  infrastructure, and none is stubbed out waiting to be enabled.
- **Not safety certified.** No safety case, hazard analysis or standards conformance exists.
- **Not deployment-ready.** It is a prototype for demonstration and further development.
- **Not a scheduling optimiser.** It does not claim optimality, and the completed study found the
  learned strategies were beaten by a tuned threshold heuristic.

## Simulation-based, and what that costs

Every operational number is produced by a simulator, not measured on a railway. Of the fourteen
simulator quantities checked against published statistics, **four were validated, three partially,
and seven could not be validated at all** (`results/simulator_validation.json`).

Four of the unvalidated quantities are direct inputs to the endpoint the product displays most
prominently — train capacity and peak occupancy, dwell-time distribution, passenger balking and
patience, and the origin–destination distribution. The validated quantities constrain the
*supply* side; the endpoint measures the *crowding* side.

**Consequence:** the *ordering* of strategies within the simulator is meaningful. The *magnitudes*
are simulator-internal and must not be read as predictions about a physical system.

## Research results shown here

Read from the frozen study and unmodified. Their own limitations apply, and are reported on the
Statistics and Research Results pages:

- The study is **underpowered for its own smallest effect size of interest**; every null result is
  inconclusive rather than evidence of equivalence.
- The four-seed pilot underestimated the between-seed standard deviation by roughly a factor of
  two, so the registered minimum detectable effect is optimistic.
- One frozen, untuned PPO configuration was used for every arm. The study cannot separate *the
  uncertainty channel is unhelpful* from *this optimiser could not exploit it at this budget*.
- Baselines are deterministic and carry no training seed, so **no inferential test between a
  learned strategy and a baseline is defined**. Those comparisons are differences of means.
- The origin–destination model is synthetic.
- One line, one operator, one simulator, one architecture. No external-validity claim.

## The follow-up is incomplete

The training-budget follow-up that would resolve the largest open question was stopped at 11 of 80
policies, asymmetrically across arms, with no evaluation cells. It is displayed as
**FOLLOW-UP / INCOMPLETE** and reports no result, because none exists: not even the lowest budget
supports a paired comparison.

## Interface limitations

- One line, fixed topology. No multi-line or network-wide coordination.
- No multi-user state, authentication, audit log or operator sign-off record.
- No live data ingestion; scenarios are simulated from stored artefacts.
- Recommendations come from policies trained under the research configuration and inherit its
  limitations.
- Per-station forecast detail is shown only where the session exposes it; otherwise the page says
  **NOT AVAILABLE** rather than interpolating.

## Before any operational use

1. Validate the simulator's crowding components against operator data.
2. Establish a safety case and human-factors evaluation.
3. Resolve the training-budget question through the registered follow-up.
4. Evaluate against a tuned heuristic baseline on real operations — the study's own result makes
   this the first thing to check, not the last.
5. Add authentication, audit logging and operator sign-off.
