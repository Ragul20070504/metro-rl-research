# Independent replication execution

The independent study is **running, not complete**. It does not replace the historical
study, resolve its missing original artifacts, or establish submission readiness.

## Verified preparation

The single frozen-hyperparameter forecaster fit completed in 1,587.46 seconds. On the
recovered validation split, RMSE is 137.603075, mean pinball loss 15.248582, and central-80%
interval coverage 0.800776. It passed the original coverage/pinball gate without retuning.
These differ from the recorded historical metrics (maximum absolute metric disagreement
0.949790); neither identical predictions nor recovery of the historical binary is claimed.

All 136 input-file hashes and the archived computational source were verified after policy
training started. `results/replication_20261015_inputs.json` identifies the frozen bytes.
The local input archive is `dist/replication-20261015-inputs.tar.gz`, with every frozen input
hash verified after compression. Its inventory/checksum is in
`results/replication_20261015_archive.json`. This is a local artifact, not a public deposit.

The frozen forecaster metadata copied historical reference fields, including old commit/time
and zero-difference verification records. Those fields do **not** describe this new refit.
The explicit field-level clarification is
`results/replication_20261015_provenance_note.json`; the frozen bytes were preserved.

## Active computation

A disjoint seed-50,000 software smoke completed 456 actual steps, with 18.994 seconds of
training (26.035 seconds including setup). Four one-thread workers then started the fixed
110-policy design. Requested budgets round to 9,922,560 total actual training steps.
The smoke-rate extrapolation is approximately **28.7 hours for training alone**, excluding
evaluation, startup and parallel contention. This is an initial estimate, not a completion
promise. No policy outcomes or interim hypothesis comparisons were inspected.

Run `.venv/bin/python scripts/replication_status.py` for live completed-policy counts,
active episode-step counts and stored evaluation-cell counts. A last-update timestamp can
predate ongoing work: Monitor files update between policy completions.

The study supervisor is `/workspace/research-recovery/continue_replication.py`, with log
`/workspace/research-recovery/replication-supervisor.log`. It runs the complete evaluation
matrix and fixed analysis automatically after training. The finalization watcher is
`/workspace/research-recovery/finalize_replication.py`, with log
`/workspace/research-recovery/replication-finalization.log`. It waits for complete analysis,
then renders every learning curve, exports the separate replication report, creates and
verifies a complete local computational archive, and refreshes the review package.
`results/replication_20261015_finalization_status.json` records that work separately.
None of these future outputs is claimed to exist yet.

## Resuming safely

Use the existing checkout. Do not start a second copy while these jobs are active. After a
runtime interruption, first inspect processes and status; processes do not survive environment
publication. With no existing job, `scripts/run_replication.py --stage all --workers 4` resumes
against the existing frozen inputs. Do not edit computational sources or the fixed protocol;
the runner checks their hashes. Retain the ignored `replication/20261015/` directory as well
as tracked reports. Git alone does not preserve the models and simulation records.

After a resumed analysis completes, run `scripts/render_replication_diagnostics.py` for the
complete report with learning curves. It refuses absent/incomplete analysis and does not fit,
test hypotheses, choose checkpoints or plot partial training outcomes.

## Interpretation still requiring review

The placebo is a rolling empirical-pool bootstrap with replacement, not an exact permutation
or a proven information-free control. Online forecast history uses policy-dependent served
boardings; offline calibration does not establish online calibration under that feedback.
These are inherited limitations, not corrected numerical algorithms. The separately generated
replication report must be scientifically reviewed and integrated with the historical negative
result before submission. Personal details remain placeholders by user instruction; author
confirmation, journal compliance and public archival availability remain outstanding.
