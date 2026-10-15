# metro-rl-research

Reproducible research prototype for the manuscript *"An Uncertainty-Aware Adaptive Multi-Objective
Reinforcement Learning Framework for Real-Time Metro Train Induction and Scheduling"*.

> ## Project status: **M2 — environment and foundation built; no science run yet**
>
> **This repository currently contains ZERO experimental results.**
>
> No model has been trained, no simulation has been run, no dataset has been downloaded. Every
> quantitative statement in `docs/` is an ASSUMPTION or a HYPOTHESIS, never an EXPERIMENTAL RESULT.
> The manuscript itself states that empirical results are in progress and none are reported.
>
> Done so far: research audit (M0), frozen pre-registration (M1), all ten open decisions resolved
> and recorded (M1b), Python 3.11 environment with a functionally verified stack and the
> `src/common/` foundation (M2). See [docs/progress_log.md](docs/progress_log.md) for what was
> *actually* completed, including a defect found and fixed.
>
> The prototype may be called complete only when all thirteen conditions in §37 of the research
> brief hold — including that a fresh clone reproduces the headline numbers.

## What this project is trying to find out

> Does explicitly representing passenger-demand forecast uncertainty inside an adaptive RL
> scheduling policy improve robustness and service performance under stochastic demand and
> disruptions, compared with equivalent policies using point forecasts?

The experiment is designed so that it can **reject** that hypothesis. A null result is a finding and
will be reported as one, with an equivalence test against a pre-registered smallest effect size of
interest. The uncertainty-aware arm is not tuned harder than its ablation.

The design includes a **placebo arm** (`ppo_placebo`) whose uncertainty channel is shuffled — same
dimensionality, same marginal distribution, no information. If the true-σ arm does not beat the
placebo arm, the apparent benefit of "uncertainty awareness" is an artefact of observation size, and
that is what the paper will say.

## Evidence discipline

Documentation labels every statement as **FACT**, **OBSERVATION**, **ASSUMPTION**, **HYPOTHESIS**, or
**EXPERIMENTAL RESULT**. These are never mixed. See `docs/research_audit.md` for the convention.

- `results/` is append-only; writing to an existing run path raises.
- Every run stores a manifest tying it to a git commit, a config hash, dataset hashes, seeds, package
  versions and hardware.
- Every figure and table is regenerated from `results/` by code. No number is ever typed into a plot.
- `experiments/registry.yaml` is the pre-registration and is frozen before the expensive sweep.

## Read these first

| Document | What it is |
|---|---|
| [docs/research_audit.md](docs/research_audit.md) | RQs, hypotheses, novelty audit, assumption audit, data-availability audit, threats to validity, publication risk |
| [docs/experimental_protocol.md](docs/experimental_protocol.md) | **The pre-registration.** Datasets, splits, forecasting, simulator, RL environment, reward, methods, scenarios, statistics — frozen |
| [docs/manuscript_corrections.md](docs/manuscript_corrections.md) | Where the manuscript must change, and why. Five items are blocking |
| [docs/data_sources.md](docs/data_sources.md) | Every dataset, its URL, its role, and what it does **not** contain |
| [docs/risk_register.md](docs/risk_register.md) | Scientific and engineering risks, with mitigations and owners |
| [docs/milestone_plan.md](docs/milestone_plan.md) | M0–M16 with exit gates |
| [docs/dependency_plan.md](docs/dependency_plan.md) | Every dependency justified; what was deliberately left out |
| [docs/repository_structure.md](docs/repository_structure.md) | Layout, test map, structural rules |

## The five corrections that block implementation

Recorded in full in `docs/manuscript_corrections.md`; summarised here because they change the science:

1. **MC-06 (critical).** The manuscript generates simulated passenger arrivals from the demand
   *forecast model*. That is circular: the forecaster becomes correct by construction and forecast
   uncertainty becomes a parameter we chose, so the central hypothesis cannot be tested. Ground truth
   must come from a process fitted to observed historical ridership instead.
2. **MC-08.** No public per-train availability data exists for the modelled fleet. A classifier scored
   on labels from our own synthetic process is a software test, not evidence.
3. **MC-05.** A static MILP from average demand is close to a straw man; a rolling-horizon MILP with
   the same forecasts the agent sees is the fair strong baseline.
4. **MC-09.** With common random numbers the design is *paired*, so Wilcoxon rank-sum is the wrong
   test; and 30 evaluation episodes from one trained policy are not 30 independent observations.
5. **MC-01.** "First to jointly address…" is an unverifiable universal negative and weak novelty.

## Planned reproduction commands (not yet implemented)

```
python -m src.data.prepare                 --config configs/data.yaml
python -m src.forecasting.train            --config configs/forecasting.yaml
python -m src.simulator.validate           --config configs/simulator.yaml
python -m src.rl.train                     --config configs/ppo.yaml --arm uq --seed 101
python -m experiments.run_all              --config configs/evaluation.yaml
python -m src.evaluation.analyze           --config configs/evaluation.yaml
python -m src.explainability.shap_analysis --config configs/evaluation.yaml
python -m src.visualization.figures        --config configs/evaluation.yaml
```

## Environment

**Python 3.11** (3.11.9 verified). The system interpreter on the development machine is 3.14, which is
ahead of assured wheel support for PyTorch / Stable-Baselines3 / SHAP / OR-Tools, so the project uses
its own 3.11 virtual environment and leaves the system interpreter alone.

```bash
# Windows, using the Python Install Manager
py install 3.11
%LOCALAPPDATA%\Python\pythoncore-3.11-64\python.exe -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt   # hash-pinned lockfile

# any platform, if you already have 3.11
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Then verify the stack **functionally** — importing a package proves it installed, not that it works:

```bash
.venv/Scripts/python.exe scripts/check_environment.py   # expect: all 10 checks passed
.venv/Scripts/python.exe -m pytest                      # expect: 38 passed
```

`check_environment.py` asserts, among other things, that MaskablePPO actually refuses a masked
action and that CP-SAT reports solver status and wall time — two things the protocol depends on and
that a plain import would not catch.

## Data attribution

Primary data is published by the **Metropolitan Transportation Authority** via NY State Open Data
(<https://www.mta.info/open-data>). The conditional rolling-stock sensor dataset **MetroPT-3**
(Metro do Porto, UCI ML Repository #791, DOI 10.24432/C5VW3R) is **CC BY 4.0**. Licence text as
published at download time is stored alongside each dataset in `data/raw/<dataset_id>/LICENSE.txt`.

Raw data bytes are git-ignored; manifests and hashes are committed, so a fresh clone can verify it
re-downloaded identical bytes.
