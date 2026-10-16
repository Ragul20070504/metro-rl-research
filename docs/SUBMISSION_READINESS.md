# Submission readiness

Checked 2026-10-16T13:53:37.732071+00:00.

**NOT READY FOR SUBMISSION**

This is a current-checkout audit, not an external peer review or proof of historical experiment execution.
Stored-results reanalysis is distinct from reproducing simulations with the original models and data.

| Check | Status | Evidence |
|---|---|---|
| Stored evidence fingerprint | PASS | Fingerprint of original cell JSON; not proof of historical authenticity. |
| Original evaluation matrix | PASS | 6480/6480 main cells present; 0 missing, 0 unexpected. |
| EXP-UQ-001 stored-cell reanalysis | PASS | Recomputed from stored cell metrics; no experiment rerun. |
| EXP-UQ-002 stored-cell reanalysis | PASS | Recomputed from stored cell metrics; no experiment rerun. |
| Historical code provenance | BLOCKED | 7609 dirty manifests; 559 unknown commits; 2 referenced commits unavailable. |
| Original study tag | BLOCKED | original-confirmatory-study is absent locally and was not advertised by origin. |
| Original raw-data bytes | BLOCKED | 11 missing files; 0 checksum mismatches. |
| Frozen follow-up inputs | BLOCKED | 23 required input files missing. Do not replace the historical GTFS snapshot with today's feed. |
| Recorded trained-policy binaries | BLOCKED | 69 metadata records lack their policy.zip. |
| Follow-up grid | BLOCKED | No new inferential result may be reported until every budget is complete. |
| Bibliography | PASS | 0 references still require authoritative verification. |
| Independent registration record | LIMITATION | An internal protocol exists, but its chronology cannot be independently established. The revision removes the externally verifiable pre-registration claim. |
| Placebo construction | LIMITATION | The inherited placebo bootstraps a rolling pool with replacement; exact marginal matching and complete information removal are not established. The manuscript now limits attribution accordingly. |
| Author declarations | BLOCKED | Affiliations, corresponding-author details, CRediT roles, funding and conflicts remain author-confirmation placeholders. |
| Journal requirements and archival deposit | BLOCKED | Current author guidelines and a public artifact archive/DOI have not been verified; no submission was made. |

## Independent replication

A separately specified replication is authorized under `docs/replication_20261015_protocol.md`. It trains a fresh 60k anchor and does not claim to recover the historical policies. Its input/code snapshot and results live under `replication/20261015/`.

Current phase: **TRAINING**; 96/110 policies and 0/13,680 evaluation cells.

Run `python scripts/replication_status.py` for live counts. No new result is available until the entire grid completes and passes analysis checks.

## Recovery and completion order

1. Complete the independent replication with `python scripts/run_replication.py --stage all --workers 4` only if no replication job is already active. Its new snapshot and fresh anchors are separate from the historical study.
2. Review the full-grid analysis, all learning curves, failures and actual timesteps; integrate every specified comparison with the unchanged original result. No incomplete-grid inference is valid.
3. Continue historical archive recovery where possible. Only exact original inputs/checkpoints permit the original held-fixed follow-up (`scripts/run_followup.py --check`). Do not overwrite original manifests or replace the lost anchor with a fresh model.
4. Retain the historical provenance limitation. Until chronology is independently verified, describe the original protocol as internal rather than externally verified pre-registration. A transparently scoped retrospective study does not require retroactive registration.
5. All 18 references are verified against authoritative metadata. Current journal guidelines, author placeholders and public archival availability remain to be completed before submission.

The original H1 conclusion is unchanged. Any future result is follow-up/exploratory and must be reported alongside it.
