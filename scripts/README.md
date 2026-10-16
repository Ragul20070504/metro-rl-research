# Current commands

Run commands from the repository root with `.venv/bin/python`.

| Task | Command |
|---|---|
| Check installed dependencies | `scripts/check_environment.py` |
| Inspect publication evidence | `scripts/publication_audit.py` |
| Inspect the live replication | `scripts/replication_status.py` |
| Resume the fixed replication | `scripts/run_replication.py --stage all --workers 4` |
| Build the current paper, PDF, DOCX and audits | `scripts/build_manuscript.py` |
| Render completed replication learning curves | `scripts/render_replication_diagnostics.py` |

Do not start a second replication while its existing workers are active. Read
[execution instructions](../docs/replication_execution.md) before resuming. The publication
audit deliberately exits 1 while submission prerequisites remain unresolved. Diagnostic
rendering requires the complete replication analysis.

`run_experiments.py` and `run_followup.py` implement the historical study and held-fixed
follow-up. They require the original artifacts; fresh replication models are separate.
Other scripts provide individual data, fitting, analysis, validation and figure stages.
`make_paper_assets.py` requires its original source artifacts to regenerate the entire set;
the paper builder can use existing figure files.

The superseded paper writers, generated "peer reviews" and one-off pipeline drivers were
retired. Their historical versions remain in Git; see [the archive index](../docs/archive/README.md).
