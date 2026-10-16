# Independent replication and training-budget study — 2026-10-15

**Prospectively specified before any policy training or evaluation for this study.**
This is a new replication, not a continuation using the lost original checkpoints. It does
not alter the original protocol, registry, results, H1 verdict or incomplete follow-up record.
Its chronology is documented locally; no external registration or publication is claimed.

## Why a separate replication

The original binary policies, network/demand artifacts and historical source objects are
unavailable, and origin has no releases or tags. Exact-hash recovery restored 113 raw files,
including the full original training/validation periods. Eleven files have changed upstream,
including GTFS and recent ridership. Replacing those silently would invalidate a claim to
reuse the historical 60k anchor. This design instead freezes an explicit new input snapshot.

## Fixed questions and design

Primary question: does the paired uncertainty-aware minus point-forecast service difference
depend on training budget? Secondary question: is the original three-arm ordering reproduced
at the requested 60,000-step budget? No configuration is selected to make uncertainty win.

| Element | Fixed value |
|---|---|
| Budget grid | 15,000; 30,000; 60,000; 120,000; 240,000 requested steps |
| Grid arms | `ppo_point`, `ppo_uq` |
| Additional placebo | `ppo_placebo` at 60,000 steps only |
| Training seeds | 101–110 for every arm/budget |
| Evaluation seeds | 1–30, matched across methods |
| Scenarios | normal, surge, disruption, combined |
| Primary outcome | `left_behind_fraction` under surge |
| Baselines | fixed timetable, previously tuned threshold heuristic, static and rolling MILP |
| Policies | 110: 100 grid policies plus 10 placebo policies |
| Evaluation cells | 13,680: 12,000 grid + 1,200 placebo + 480 baseline |
| Unit of inference | training seed; episode means are not independent training observations |

Every 60k policy is trained anew. The learner, reward, masking, seed rotation, scaler and
hyperparameters use the imported implementation and original frozen numeric settings.
Only budget and uncertainty-channel content vary within the two-arm grid. The inherited
observation scaler, reward scales, heuristic thresholds and MILP surrogate are copied without
retuning; their historical provenance limitations remain disclosed. Requested budgets round
up to full PPO rollouts; actual timesteps and episode-return learning curves are retained.

## New frozen inputs

Use the exact recovered monthly partitions where their original hashes match, and explicitly
record the current quarantined publisher bytes for the eleven changed files. Preserve original
manifests. Train and validation features use the original split boundaries and original station
index mapping; row counts must match the original manifests before fitting. Fit the quantile
forecaster once at the previously selected hyperparameters and original calibration factor,
without another search. Compare validation metrics to the recorded values and report agreement
or disagreement; agreement of aggregate metrics does not prove identical prediction vectors.

Build the network and fixed timetable from the newly frozen GTFS/station files. Require the
same ordered 24 station IDs. Fit ground-truth demand only on the recovered training split.
Test-period ridership is not used to fit demand, the forecaster, scaler, thresholds or policies.
The simulation comparisons evaluate synthetic worlds fitted to that training demand; they
are not operational validation or direct held-out real-world deployment tests.

Before policy computation, save all raw/derived input hashes, the exact protocol hash, package
lock hash, a source-code archive/hash, platform information and compute limits. Future tasks
must resume against those same bytes, not reread mutable upstream feeds. All new models/results
live under `replication/20261015/`; original `results/runs/` is never written.

## Analysis fixed before policy computation

1. Primary budget interaction: per seed and budget, compute the surge mean difference
   `ppo_uq - ppo_point`; fit `difference ~ log2(budget) + (1 | training_seed)` with statsmodels
   MixedLM. Report slope, Wald p and 95% CI. No inference from a nonconverged/invalid fit.
   A nonsignificant slope does not establish budget independence or equivalence.
2. Five per-budget surge contrasts: the original rule (paired t if Shapiro–Wilk p ≥ .05,
   otherwise Wilcoxon), Holm adjustment across all five, Hodges–Lehmann paired estimates and
   BCa intervals with 10,000 resamples. Report every budget, seed and direction.
3. Separate secondary two-test family at 60k: `ppo_uq` versus `ppo_point` and versus
   `ppo_placebo`, Holm-adjusted together. These are replication results, not replacements for
   the original family. The point contrast also appears in the budget family; disclose both
   multiplicity contexts rather than selecting the smaller adjusted p.
4. Budget-by-scenario contrasts are exploratory, BH-adjusted across all 20 conditions.
   Baseline rankings are descriptive; do not inflate n by treating episodes as training seeds.
5. Report all learning curves and actual step counts as diagnostics; use terminal policies
   only. No early stopping, budget/seed exclusion, checkpoint selection, configuration search
   or grid extension after observing results. The design remains underpowered for the
   original SESOI of 0.002 at n=10; nulls are not evidence of equivalence.

No interim comparison or significance test is run. All 13,680 cells and 110 policies must be
complete before inference. A failed job is retried once and its failure retained; a second
failure stops the pipeline. Resource interruption leaves a resumable incomplete study, not a
result. Every scientific output must be labeled independent replication/exploratory relative
to the historical study, regardless of direction. No model can be retuned after seeing results.

## Execution and reporting

Use the CPU hash-pinned environment, one numeric thread per policy worker and at most four
workers (current CPU quota is four cores). Preparation/fit runs separately from worker pools.
Measure a short software smoke run before estimating completion time; do not substitute its
outcomes for study data. Progress reports show counts/errors, not partial hypothesis results.

Run `python scripts/run_replication.py --stage prepare`, then `--stage run`; the orchestrator
resumes existing verified work. Run `--stage analyze` only after completeness passes. The
original negative result is preserved and must appear alongside the replication finding in
any integrated manuscript. Author declarations, external archival deposition, venue compliance
and scientific review remain separate submission prerequisites.

## Implementation clarification before policy computation

The inherited placebo is a bootstrap from a rolling pool of up to 50,000 current/past
dispersion scalars, sampled with replacement across stations and epochs. It is not an exact
permutation and does not preserve a finite-sample marginal exactly. Current values enter the
pool before sampling, so some regime information may remain. The pool persists across
training episodes; separate evaluation environments start separate pools. Temporal dependence
and station-specific association differ as well as information content. Retain this existing
algorithm to replicate the imported design, disclose the limitation, and interpret the
secondary comparison as a contrast of channel constructions. No clean isolation of information
quality or within-policy causal response is claimed. This clarification changes no budgets,
seeds, tuning, endpoint, analysis family, or numerical implementation.
