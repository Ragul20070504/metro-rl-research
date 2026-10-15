# M17 — Interactive research dashboard

A browser application that runs the **actual** validated research components and shows what they
do: the fitted M4 forecaster, the M5-calibrated simulator, the M6 RL environment, the tuned
baselines and the trained policies.

It is a viewer. It does not retrain, retune or recompute anything, and it cannot alter a
confirmatory result.

---

## 1. Installation

The research environment (`requirements.txt`, `requirements.lock.txt`) is left **untouched** — it
records the environment that produced the experimental results, and adding a visualisation
framework to that record would mean the environment named in the paper is not the one the numbers
came from. The dashboard's dependencies are separate:

```
.venv/Scripts/python.exe -m pip freeze > constraints.txt
.venv/Scripts/python.exe -m pip install -r requirements-demo.in --constraint constraints.txt
```

The `--constraint` is the point: pip may only **add** packages, never upgrade one the research
pipeline depends on. Verified on 2026-08-27 — no pre-existing package changed version, and
`scripts/check_environment.py` still passes all 10 checks afterwards.

## 2. Launch

```
streamlit run app.py
```

or, equivalently, without relying on `streamlit` being on `PATH`:

```
.venv/Scripts/python.exe -m streamlit run app.py
```

## 3. Browser address

```
http://localhost:8501
```

`.streamlit/config.toml` binds the server to the loopback interface only. Streamlit otherwise
listens on every interface and advertises an external URL, which would put an unauthenticated view
of research artefacts on the local network. Usage telemetry is off.

## 4. Scenarios

| Selection | Scenario id | What it is |
|---|---|---|
| Normal | `normal` | The fitted demand process for the day type, no injected disruption. |
| Peak / Surge | `surge` | A demand surge. The primary endpoint is measured here. |
| Train Disruption | `disruption` | Trainset failures removing capacity during the day. |
| Combined | `combined` | Surge and disruption together. |

Scenario families **share a base world** at a given evaluation seed (amendment A-03), so switching
scenario at a fixed seed changes the scenario rather than drawing an unrelated day. Before that
amendment they were unrelated realisations, and `surge` scored *better* than `normal` in 3 of 5
seeds — the defect that motivated it.

## 5. Policies

| Selection | Method id | Requires |
|---|---|---|
| Fixed Timetable | `fixed_timetable` | `data/processed/network/fixed_timetable.json` |
| Tuned Heuristic | `heuristic_threshold` | `models/heuristic_threshold.json` |
| MILP (static plan) | `milp_static` | `models/milp_surrogate.json` |
| MILP (rolling horizon) | `milp_rolling` | `models/milp_surrogate.json` |
| Point-Forecast PPO | `ppo_point` | a trained policy under `models/policies/` |
| Uncertainty-Aware PPO | `ppo_uq` | a trained policy under `models/policies/` |
| Placebo-Sigma PPO | `ppo_placebo` | a trained policy under `models/policies/` |

**A policy whose artefact is missing is listed as unavailable, with the reason, and cannot be
selected.** It is never silently replaced by a fallback: a dashboard that substituted a different
policy would be showing one method's behaviour under another's name.

## 6. Controls

| Control | Effect |
|---|---|
| **Step** | Advance exactly one decision epoch (5 simulated minutes). |
| **Run** | Advance by the slider's number of epochs, stopping at the end of the day. |
| **Reset** | Restart the episode from the same seed, reproducing it exactly. |
| **Evaluation seed** | Which day is simulated. The same seed always gives the same day. |
| **Training seed** | For PPO only: which trained policy to demonstrate. Each training seed is a separate policy, and the paper treats the training seed as the unit of analysis. |

## 7. What the panels show

**Network.** Stations with their real MTA identifiers and names, current queue and longest current
wait; trains in service with status, position, direction and occupancy. All read from the
simulator's own state each epoch.

**Demand forecast.** The forecast mean μ per station and, for arms that carry it, σ and the
calibrated 80% interval `μ ± 1.2816σ`. These are taken from the **unscaled** observation, so the
numbers are passengers rather than standardised values — this is literally what the policy is
looking at. For the point-forecast arm the σ channel is zeroed, and the panel says so rather than
displaying a dispersion the policy cannot see.

*Realised demand* is drawn separately and only for epochs already stepped. It is an outcome, never
an input: the policy had already acted before it existed.

**Decision.** The action the policy chose, what it requested, which actions were feasible, and the
action mask. Constraints are enforced structurally by masking rather than by reward penalties
(protocol §3.4, §6.3), so an infeasible induction is refused rather than priced. When a request is
clipped the panel says so — a method limited by infrastructure is distinguishable from one limited
by its own logic.

**Outcomes.** `left_behind_fraction` (the pre-registered primary endpoint, amendment A-01),
cumulative reward, mean and p95 waiting, distinct left-behind passengers, inductions, energy and
cost. Energy and cost are in **equivalent units, not physical ones** — assumptions A15 and A16 are
unresolved, so both are embargoed from headline claims.

**Reward decomposition.** The four components `W`, `U`, `E`, `C` for the last epoch, read from the
environment's own record. The dashboard does not recompute the reward; `tests/test_demo.py` asserts
the displayed components equal the environment's to floating-point tolerance.

**Traceability.** Git commit and dirty flag, policy, scenario, both seeds, reward weighting,
forecast-model provenance including its calibration factor and refit agreement, network size,
horizon and fleet. Enough to reproduce the run.

## 8. Interpretation

The numbers are real, and they are **one episode**.

A single interactive run is one draw from a distribution the paper characterises over many seeds
using the registered statistical tests. Watching `ppo_uq` beat `heuristic_threshold` on seed 7
tells you what happened on seed 7. It is not evidence about H1, and the interface says so on every
page.

Specifically, do not read from this dashboard:

- that one method is better than another — that is what the confirmatory analysis is for;
- any notion of statistical significance — a single run has none;
- energy or cost in physical units — see A15/A16 above;
- a causal reading of anything.

## 9. Demonstration versus confirmatory results

| | Confirmatory | Demonstration |
|---|---|---|
| Written to | `results/runs/main/…` | `results/demo/` |
| Cell variant | `main` | `demo` |
| Reaches the analysis | yes | **no** |
| Seeds | registered pools | any seed the user picks |

The separation is structural, not conventional. Every paired test in `scripts/analyse.py` reads
`variant == "main"` (the default of `paired_frame`), and demonstration sessions are constructed
with `variant="demo"`, so a demo run cannot enter a confirmatory comparison even if its seed
collided with a registered one. `tests/test_demo.py` asserts that a demo run writes nothing under
`results/runs/`, and that `src/demo/session.py` never references the confirmatory runner.

## 10. Troubleshooting

**"No policy artefacts found."** The research pipeline has not produced any models yet. Run at
least `python scripts/run_experiments.py --stage prepare`.

**A PPO policy is greyed out.** Training has not run. The RL arms appear once
`models/policies/w_balanced/<method>/<seed>/policy.zip` exists.

**"Found invalid character in key name" on launch.** `.streamlit/config.toml` was written with a
UTF-8 byte-order mark — PowerShell's `Out-File -Encoding utf8` does this on Windows PowerShell 5.1.
Rewrite the file without a BOM.

**Streamlit prompts for an email address.** `.streamlit/credentials.toml` is missing; it exists to
stop that prompt blocking an unattended start.

**The app is slow when you press Run.** Each epoch is a full simulator step across 24 stations and
36 trainsets, and `milp_rolling` re-solves a CP-SAT instance every epoch (~0.3 s). Use Step, or a
smaller epoch count, for the MILP policies.

**Port 8501 is in use.** `streamlit run app.py --server.port 8502`.
