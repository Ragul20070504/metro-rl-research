# Historical archive

These files describe previous project states. Their completion claims, reviewer opinions
and commands are historical records, not current operating instructions. Use the
[current documentation](../README.md) and [script guide](../../scripts/README.md).

## Superseded documents

| Original location | Archived document |
|---|---|
| `HANDOVER.md` | [HANDOVER.md](HANDOVER.md) |
| `RESUME.md` | [RESUME.md](RESUME.md) |
| `docs/FINAL_STATUS.md` | [FINAL_STATUS.md](FINAL_STATUS.md) |
| `docs/execution_status.md` | [execution_status.md](execution_status.md) |
| `docs/final_completion_directive.md` | [final_completion_directive.md](final_completion_directive.md) |
| `docs/final_peer_review.md` | [final_peer_review.md](final_peer_review.md) |
| `docs/final_results.md` | [final_results.md](final_results.md) |
| `docs/manuscript_draft.md` | [manuscript_draft.md](manuscript_draft.md) |
| `docs/manuscript_revision_plan.md` | [manuscript_revision_plan.md](manuscript_revision_plan.md) |
| `docs/milestone_status.md` | [milestone_status.md](milestone_status.md) |
| `docs/paper_peer_review.md` | [paper_peer_review.md](paper_peer_review.md) |
| `docs/paper_journal_fit.md` | [paper_journal_fit.md](paper_journal_fit.md) |
| `docs/pre_submission_peer_review.md` | [pre_submission_peer_review.md](pre_submission_peer_review.md) |
| `docs/q1_q2_readiness_audit.md` | [q1_q2_readiness_audit.md](q1_q2_readiness_audit.md) |
| `docs/reproduction_guide.md` | [reproduction_guide.md](reproduction_guide.md) |
| `docs/figure_table_traceability.md` | [figure_table_traceability.md](figure_table_traceability.md) |

Markdown navigation links were adjusted for this directory; scientific text and recorded
results were retained. The supported `docs/reproduction_guide.md` path now points to current
instructions because the dashboard links to it.

## Retired scripts

The following duplicate reporting tools and one-off drivers had no callers in the current
workflow or tests. Their original source is retained at Git commit
`e053dea4180cec859f7241c1ce529e0d7235a8ac`:

- `scripts/write_paper.py`
- `scripts/write_manuscript.py`
- `scripts/write_final_documents.py`
- `scripts/write_peer_review.py`
- `scripts/q1_q2_audit.py`
- `scripts/build_paper_review.py`
- `scripts/peer_review.py`
- `scripts/finish_pipeline.py`
- `scripts/resume_after_train.py`

For example, `git show e053dea4180cec859f7241c1ce529e0d7235a8ac:scripts/write_paper.py`
reads the original source without running it. The current manuscript has one generator:
`scripts/build_manuscript.py`. The current replication uses `scripts/run_replication.py`.

A checksum-verified copy of the pre-cleanup files is also retained locally at
`/workspace/research-cleanup/pre-cleanup-20261016.zip`. No data, trained model,
stored evaluation result, registered protocol or original dependency lock was removed.
