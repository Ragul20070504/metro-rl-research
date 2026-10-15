# Product user guide — Metro Intelligent Dispatch

## Launch

```bash
streamlit run product/app.py --server.port 8502
```

Open **http://localhost:8502**. The research dashboard runs separately on port 8501; both can run
at once. If a port refuses to bind, an earlier Streamlit still holds it — stop that first.

## Sidebar

**Panel demo mode** presents nine pages in presentation order with a progress bar. Leave it off
for free navigation by section.

**Operating selection** (operational pages only):

| Control | Meaning |
|---|---|
| Strategy | Which dispatch policy or baseline drives recommendations |
| Scenario | Normal, demand surge, train disruption, or combined stress |
| Policy instance | Which trained policy — each training seed is a separate policy |
| Simulation day | The seed; the same seed reproduces the same day exactly |
| Step / Run / Reset | Advance one epoch, advance several, or restart the day |

The sidebar footer shows **Research artefacts: read-only, unchanged**, re-checked on every render.

## Pages

**Operations**

- **Home** — network status, forecast, uncertainty, availability, capacity pressure, and the
  ten-step operator workflow.
- **Network** — per-station queues and trains in service. Schematic: station order is real,
  geometry is not.
- **Forecast** — mean, dispersion and 80% bounds from the frozen forecaster, plus model quality.
  Point-forecast strategies show **NOT AVAILABLE** for uncertainty by design.
- **Dispatch** — the recommendation, why, and expected effect. Operator approval required.
- **Scenario simulator** — run the day forward and compare before/after.

**Analysis**

- **Strategy comparison** — all strategies on any metric and scenario, from the completed study.
- **Mechanism** — how uncertainty affected dispatch, and the convex service-loss relationship.
- **Policy inspection** — feature attribution and the live decision trace.

**Research**

- **Research results** — the registered H1 result, effect size, interval, adjusted p, placebo.
- **Statistics** — registered design, power, and the exploratory family.
- **Follow-up** — the incomplete training-budget study. Reports no result, because none exists.
- **Reproducibility** — provenance, audit status, immutability, launch commands.

**Info** — **About**: what the system does and does not do.

## Reading the badges

Every panel showing a number carries one:

| Badge | Meaning |
|---|---|
| `PRODUCT DEMONSTRATION` | Live simulation. One simulated draw, not a research result |
| `CONFIRMATORY` | Registered research result, fixed before the data were seen |
| `EXPLORATORY` | Hypothesis-generating research result |
| `DESCRIPTIVE` | No hypothesis test defined for this quantity |
| `FOLLOW-UP / INCOMPLETE` | Stopped before any comparison was possible — not evidence |

Research and live values are never mixed within a panel.

## Demonstration scenarios

| Demo | Strategy | Scenario | Day | Shows |
|---|---|---|---|---|
| 1 Normal operation | `ppo_point` | Normal | 7 | Baseline behaviour, light queues |
| 2 Demand surge | `ppo_point` | Demand surge | 7 | Capacity pressure rising |
| 3 Train disruption | `ppo_uq` | Train disruption | 12 | Reduced availability |
| 4 Combined stress | `ppo_uq` | Combined stress | 3 | Surge and disruption together |

Comparing demos 2 and 4 on the same strategy shows the uncertainty arm's conservative dispatch.

## If something is missing

The interface shows **NOT AVAILABLE** naming what is absent, and never substitutes a value. If a
page fails, the error names the page; other pages remain usable.

## What this system will not do

It does not control trains, is not safety certified, is not autonomous, and is simulation-based.
See `docs/product_limitations.md`.
