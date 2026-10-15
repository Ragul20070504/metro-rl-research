# FOLLOWUP-UQ-TRAINING — status

## FOLLOW-UP / INCOMPLETE

**Stopped 2026-08-28 12:27 IST by instruction, before any comparison was possible.**

This study is **not** part of M0–M17 and contributes **nothing** to the confirmatory result. The
original conclusion is unchanged and unchangeable by anything here:

> **Under the registered 60,000-step experiment, H1 was NOT SUPPORTED.** `ppo_uq` left more
> passengers behind than `ppo_point` under surge: +0.013742, 95% BCa CI [0.006570, 0.018523],
> Holm-adjusted p = 0.007741.

## What was completed

| item | planned | completed |
|---|---:|---:|
| Training budgets | 4 new (15k, 30k, 120k, 240k) + 60k reused | **1 partially** (15k) |
| Policies | 80 | **11** |
| Evaluation cells | 9,600 | **0** |
| Analysis | interaction test over 5 budgets | **none** |

Completed policies, all at the 15,000-step budget:

- `ppo_point` seeds 101–110 (10 policies)
- `ppo_uq` seed 101 (1 policy)

Stored under `models/policies/budget_15000/`. Nothing has been deleted.

## Why no result can be reported from this

Not merely incomplete — **structurally unusable**. The 15,000-step budget has ten `ppo_point`
policies against one `ppo_uq` policy, so not even the lowest budget supports a paired comparison,
which requires both arms at the same seed. The sweep stage never began, so no evaluation cell
exists for any follow-up policy.

There is therefore no follow-up number to report, to mix in, or to misread. Reporting anything
from an asymmetric 11-of-80 sample would be exactly the selective reporting the protocol forbids.

## Separation, verified mechanically

`scripts/separation_audit.py` (14/14 pass) confirms:

- every artefact of the registered study is **byte-identical** to git tag
  `original-confirmatory-study` (commit `c9b92e49`) — registry, policies, all 7,610 cells,
  statistical analysis and protocol;
- follow-up policies live under `models/policies/budget_15000/`, never under `w_balanced/`;
- no follow-up experiment id appears in any confirmatory or exploratory family;
- no follow-up episode is counted in the registered analysis.

## What the follow-up was for, and what remains open

The unresolved question it was designed to answer is unchanged and is now reported as an explicit
limitation rather than as a result:

> Amendment A-05 froze one PPO configuration at 60,000 timesteps for every arm. The study
> therefore cannot separate *the uncertainty channel is unhelpful* from *this optimiser could not
> exploit it at this budget*. Supporting the second reading: no learned arm beats the tuned
> threshold heuristic in any scenario, and one single-objective training seed diverged by two
> orders of magnitude.

The protocol that would settle it is complete and fixed at
`docs/followup_training_budget_protocol.md`, and its analysis is pre-committed at
`scripts/analyse_followup.py` — written before any follow-up data existed. Both are retained so
the study can be run properly by anyone with the compute, with no further design freedom.

**It must be run to completion or not at all.** Resuming it partially and reporting whichever
budgets happened to finish would reintroduce the selective-reporting problem the pre-registration
exists to prevent.
