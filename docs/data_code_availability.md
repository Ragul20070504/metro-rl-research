# Data and code availability

## Code

All source code, experiment configuration and analysis scripts are version-controlled in the project repository at commit `4dec2d835e8cac46fe39549b074c9622ccacb85a`, with the completed study preserved at the immutable tag `original-confirmatory-study`.

The repository is **not currently deposited in a public archival repository.** An accurate statement for submission is therefore:

> Code and computational artefacts will be deposited in an archival repository > (for example Zenodo, with a DOI) upon acceptance, and are available to editors and > reviewers on request during review.

Do not state that materials are already publicly available unless and until a deposit has been made.

## Data

| Category | Availability |
|---|---|
| Source ridership records | Public transit ridership data; dataset manifests and licence files are recorded in the repository. Redistribution is governed by the source licence — `[AUTHORS TO CONFIRM LICENCE TERMS BEFORE DEPOSIT]` |
| Derived features and splits | Reproducible from the source data with the included preparation scripts |
| Simulated outcomes | Regenerable from the committed code and stored model artefacts |
| Stored results | 7,610 evaluation cells, statistical analysis, figures and tables are version-controlled |
| Trained policies | Checkpoint metadata is version-controlled; binary checkpoints are large and would be included in an archival deposit |

## Reproduction

```bash
python -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.lock.txt
.venv/bin/python scripts/check_environment.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/analyse.py
```

The environment is hash-pinned. Reproducibility audit status: PASS = 23, FAIL = 0, WARN = 1, SKIPPED = 0.

**Caveat.** Most stored evaluation cells were produced from a working tree containing uncommitted changes and are marked `provisional` in their manifests; the recorded commit does not fully determine the code that produced them. Re-running from the committed state is required for commit-anchored results.
