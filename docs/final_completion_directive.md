# Final post-pipeline completion directive

Issued 2026-08-27 by the principal investigator, to execute **only after the autonomous pipeline
reaches `pipeline / COMPLETE`** in `results/pipeline_progress.json` and the machine is idle.

Recorded here so that the instruction survives a lost session context. Nothing in this file
authorises any change to the registered experiment.

## Trigger

`scripts/finish_pipeline.py` writes `{"step": "pipeline", "status": "COMPLETE"}` after its last
stage (`review`, M16). M17 -- the Streamlit dashboard -- is launched separately and is part of the
work below, not part of the trigger.

Do not begin while training, the sweep, the ablations or SHAP are running: the audit must not
compete with the experiment for CPU.

## Standing prohibitions (carried from the frozen protocol)

Never, under any circumstance arising in this work:

- change a confirmatory result to improve H1;
- tune any model after seeing confirmatory results;
- change the registered sample size, primary endpoint, reward, scenarios, PPO configuration or
  seeds;
- delete an inconvenient result or fabricate a missing one;
- alter a statistic by hand, or type an experimental number that did not come from a stored
  artefact;
- present pilot results as confirmatory evidence;
- claim causality from SHAP, or claim significance the registered analysis does not support.

If a genuine scientific validity problem would require changing the research question or the
registered confirmatory protocol, **stop and report it** rather than changing it.

Fix only genuine implementation or reporting errors that leave the registered experiment intact;
rerun only the affected downstream analysis; preserve superseded artefacts and document the
correction.

## Phases

1. **Verify completion** -- M11-M17 complete; registered experiments present at expected counts;
   no missing experiment; no FAILED or partial artefacts; statistical analysis, figures, tables,
   SHAP outputs, reproducibility audit and peer-review assessment all exist; dashboard launches.
   Resume only an incomplete stage, through the existing resumable pipeline. Never rerun a
   successful experiment.
2. **Scientific integrity audit** (read-only) -- data leakage; train/test contamination; pilot/
   confirmatory seed contamination; scenario pairing; endpoint direction; test choice; Holm
   correction; confidence intervals; SESOI/TOST interpretation; pilot results used as
   confirmatory; hand-entered numbers; fabricated or missing results; inconsistent denominators;
   missing seeds, configurations or model artefacts; stale result files reaching the final
   analysis. Cross-check against the frozen registry.
3. **Final H1 result** from stored confirmatory results: `ppo_uq` vs `ppo_point` on
   `left_behind_fraction` -- n, means, medians, SD, effect estimate and direction, CI, BCa CI,
   Holm-adjusted p, SESOI 0.002, TOST, MDE 0.0059, statistical and practical conclusions, and a
   verdict of exactly `SUPPORTED`, `NOT SUPPORTED` or `INCONCLUSIVE`. A negative or underpowered
   result is reported as such.
4. `docs/final_results.md` -- complete scientific summary, every number from a stored artefact.
5. `docs/manuscript_draft.md` -- 21-section journal draft. Discloses the R-1 underpowering, SESOI,
   MDE, amendments A-01 onward, baseline corrections A-10-A-13, MILP surrogate revisions, the
   limitations of `left_behind_fraction`, and the simulator quantities that could not be
   externally validated.
6. `docs/figure_table_traceability.md` -- figure/table -> source data -> generation script ->
   experiment IDs; no pilot data in confirmatory figures; no typed values.
7. `docs/reproduction_guide.md` -- environment, data, tests, models, policies, evaluation,
   statistics, figures, dashboard; versions, commit, configs, seeds, artefact locations, expected
   counts. No secrets; no paths private to this machine.
8. **Dashboard check** -- `streamlit run app.py`, smoke test only; verify no fabricated numbers are
   displayed and that no interaction can modify the registered experiment.
9. `docs/final_peer_review.md` -- hostile-reviewer assessment, every issue classified CRITICAL /
   MAJOR / MINOR / NONE. Do not invent problems for the appearance of rigour.
10. `docs/FINAL_STATUS.md` -- M0-M17 status table, totals (tests, policies, cells, experiments,
    failures), primary result, H1 verdict, limitations, reproducibility status, dashboard status,
    commit.

Then: run the final tests, verify a clean git state, commit only legitimate final documentation
and code changes, and report the final H1 result.
