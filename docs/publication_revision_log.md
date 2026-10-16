# Publication revision log

## 2026-10-15 — autonomous revision authorized

The user authorized continued research, experiments and necessary changes. Personal author
details and declarations remain placeholders by request. No journal submission, external
message, public deposit or claim of acceptance was made.

### Evidence boundary

Work began at `e053dea4180cec859f7241c1ce529e0d7235a8ac`, an initial import with no earlier
Git objects or original-study tag. Original result cells, the frozen registry, original
protocol, research locks and primary statistical analysis are preserved. A current audit
fingerprints original cell JSON and recomputes both primary comparisons.

Recalculation from 7,610 stored cells (6,480 main episodes) reproduces the reported means,
paired tests, Holm-adjusted p-values, Hodges–Lehmann estimates and BCa intervals. This checks
arithmetic, not historical experiment execution or external validity.

### Corrections

- Added a current publication audit based on actual files, matrix completeness, cell
  manifests, missing binaries and data-file hashes. The README now points here instead of
  the stale “no science run yet” statement. Historical PASS reports are not fresh validation.
- Corrected incomplete-grid inference, stale aggregates, the follow-up confidence-interval
  estimand, missing exploratory multiplicity adjustment and swallowed training failures.
  Protocol amendment F-02 records the corrections before any new follow-up run.
- Added episode-return curves and actual-timestep metadata for future budget runs.
- Added a separate Linux CPU hash lock without changing the original research locks.
- Removed the manuscript title's unverified pre-registration claim and disclosed unavailable
  history, tag and binaries. Preserved the negative result and baseline superiority. Separated
  episode-level dispatch associations from an unperformed within-policy causal mechanism test.
- Corrected figure/table references and replaced incomplete custom document conversion with
  installed Pandoc/XeLaTeX. Historical figures are labelled rather than called newly regenerated.
- Replaced unconditional manuscript-audit PASS statements with explicit unresolved checks.

### Blockers and next work

124 raw files listed in original manifests are absent, and 69 recorded policy binaries are
absent. The follow-up prerequisite check finds 23 missing files, including simulator inputs,
the forecaster and twenty original 60k checkpoints. Restore an exact archive before executing
the existing follow-up; do not substitute today's mutable GTFS feed.

The network proxy denied research/publisher retrieval with HTTP CONNECT 403. Required domains
were saved to the environment draft; that does not apply them to the running machine.
Reference and current journal-guideline verification remain pending access. Twelve references
still require bibliographic field verification.

Next: recover the original artifacts/history; validate them against stored evaluations using
separate outputs; execute the complete fixed budget grid with retained curves; analyze only
when complete; update claims from the actual outcome; verify references and venue requirements;
complete declarations and archival availability. No partial follow-up result may be promoted.

### Verification and review package

The full suite after the pipeline repairs completed with **518 passed, 33 failed, 2 skipped**.
Comparison with the initial failure list found **zero new failures** and **nine resolved
failures**. After correcting the forecasting plot/table, all **six focused regression tests**
passed. The dependency functional checks passed all ten again, and the revised installation
script completed successfully. Exact outcomes and scope are in
`results/publication_validation.json`; the full suite predates the final plotting correction,
which was verified by its focused test and document rendering.

The revised manuscript has a 200-word abstract. Pandoc/XeLaTeX produced a 22-page PDF and a
valid DOCX containing ten images. Figure 3 and Table 2 now read the nested model/split metrics;
their defective earlier versions are retained under `results/superseded/pre_publication_20261015/`.
Figure 7 is withheld from the revised paper and replaced by an explanation; its historical
file is preserved. The clipped method-ranking table was split into two readable tables.
The theory discussion no longer infers a structural convex loss or optimal dispatch direction
from a binned arrival/loss association. The title/abstract, primary tables and corrected forecast
figure were inspected visually. The original research files passed a Git diff check unchanged.

An exact-SHA Git fetch of the historical final-status commit
`886dc84205a34e7e7e7354db5db6fe0e410a3ec6` was rejected by origin with
`upload-pack: not our ref`; no historical source was recovered. No new training-budget study,
reference verification, public deposit or journal submission was completed.

## Source recovery and independent replication — 2026-10-15

This section supersedes the earlier network and recovery status above. Crossref, arXiv,
GitHub API and NY Open Data became reachable. All 18 references were checked against
publisher-deposited or official preprint metadata, with source records retained in
`results/reference_verification.json`. Several publication years, venues, author lists and
DOIs were corrected. Publisher author-guideline pages remain inaccessible through the proxy;
current journal compliance is not claimed. GitHub reports no releases or tags.

Exact-hash retrieval recovered 113 of 124 original raw files. Eleven mutable publisher
files differ and remain quarantined; the old manifests were not rewritten. Recovery at
frozen split boundaries restored 1,126,584 training rows and 254,016 validation rows.
The original test partition is not fully recovered. Reports are retained in
`results/raw_recovery_report.json` and `results/training_input_recovery.json`.

The original policies remain unavailable. Rather than invent replacements for historical
anchors, a new, isolated replication was specified before new policy computation in
`docs/replication_20261015_protocol.md`: 110 fresh policies and 13,680 evaluation cells,
including fresh 60k point/UQ/placebo anchors and all five budget levels. It uses explicitly
updated publisher files only within its own snapshot. The runner archives computational
source and input hashes, requires full matrix completion, retains actual timesteps and
learning curves, and computes no interim hypothesis tests. Original results are unchanged.

The default forecast loader now respects the configured model directory, allowing the
replication's new forecaster to remain isolated. Recovery, matrix, multiplicity, degenerate
contrast and model-loading checks passed: **61 passed, 1 skipped** across the five focused
suites. The skip requires an absent historical forecasting binary. This does not replace the
earlier full-suite result. The software smoke and scientific replication have not yet
completed; consult `scripts/replication_status.py` for live progress.

The revised manuscript was regenerated with the corrected bibliography and author–year
citations. Personal metadata remains placeholders by explicit user instruction. No public
archival deposit, external registration, journal submission or completed replication result
is claimed.

The full suite was rerun after recovery and the isolation fixes: **524 passed, 33 failed,
2 skipped** in 152.32 seconds. The failing test identifiers match the previously diagnosed
historical-artifact failures exactly; no new failure was introduced. The supplementary
fairness table was repaired to display actual check names and recorded outcomes, with its
historical timestamp rather than implying fresh execution.

A subsequent implementation review found that the placebo is a rolling-pool bootstrap,
not an exact permutation. Sampling with replacement cannot preserve a finite-sample marginal
exactly, and current/past pool composition can retain demand-regime information. The manuscript,
cover letter, highlights and Figures 1/5 now describe the actual construction and limit
attribution to information quality. The inherited numerical algorithm was retained; only its
incorrect comments were corrected. The independent protocol clarification was written before
policy computation, without changing seeds, budgets, endpoints, tuning or analysis families.
Earlier figure versions are retained under results/superseded/pre_placebo_description_20261015/.

A missing-binary reporting bug produced a negative policy count. It now derives the 50
main-study policy identities from stored seed-level cells and reports available binaries
separately (zero). A rendered-manuscript regression guards this distinction and the placebo
disclosure. The final focused check passed all 12 tests in 7.96 seconds. The supplementary
commands now use the Linux lock and do not instruct users to overwrite original statistics.

Preparation and the disjoint software smoke have now completed. The refit passed the fixed
validation gate but differs from the original stored metrics; it is not labeled a recovered
binary. All 136 inputs and computational hashes were verified. Historical fields copied into
frozen forecasting metadata are explicitly disclaimed in a separate provenance note without
changing frozen bytes. Four workers are training the fixed 110-policy grid. The initial
smoke-rate estimate is 28.7 training hours, excluding evaluation/contention. Full-grid analysis,
learning-curve rendering and local archival packaging are supervised automatically. They are
not complete yet. See docs/replication_execution.md for logs, resumption and interpretation.

The framework schematic now separates demand generation from policy forecasts. The manuscript
also discloses that online histories contain served boardings, which can depend on previous
actions and capacity; offline calibration does not prove online coverage. No scientific
algorithm or frozen computational source was changed after the input/code freeze.

## Repository cleanup — 2026-10-16

Retired nine superseded paper/review generators and one-off pipeline drivers (4,721 lines).
Their unchanged originals remain at the initial Git commit and in a checksum-verified local
pre-cleanup backup. No current workflow/test imports those modules. Sixteen earlier handover,
status, manuscript and simulated-review documents are now under docs/archive; historical text
was preserved, with only Markdown navigation targets adjusted. The dashboard's existing
reproduction-guide path now provides current instructions. New documentation and command
indexes identify the supported workflow; the repository map now describes actual components.

The current paper uses one shared Pandoc export for TeX, DOCX and PDF. Removed unused candidate
titles and the now-unused path import. Native math and image exports passed a focused
regression check. The paper rebuilt as 22 pages with ten DOCX images. All 14 focused tests
passed; the full suite completed with **527 passed, 33 failed, 2 skipped**, with exactly the
same historical-artifact failures as before cleanup. Navigation checks found no broken links
in the 24 checked documents. The original protected results/protocol/locks are unchanged,
and all 69 frozen computation files plus 136 inputs still match the running replication.
The training supervisor and finalization watcher remain active. Full details and source
checksums are recorded in results/repository_cleanup.json.
