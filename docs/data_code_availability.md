# Data and code availability

**Revision draft: complete experiment reproduction is currently blocked.**

The current code import is at `e053dea4180cec859f7241c1ce529e0d7235a8ac`. Original experiment commits and the `original-confirmatory-study` tag are absent from this checkout. A public archival deposit and DOI have not been verified. No promise of artifact availability is substituted for an actual deposit.

Stored cell metrics and manifests permit independent reanalysis of the reported two-test family. They do not establish that historical simulations can be rerun.

| Material | Current evidence |
|---|---|
| Historical code provenance | 7609 dirty manifests; 559 unknown commits; 2 referenced commits unavailable. |
| Original raw-data bytes | 11 missing files; 0 checksum mismatches. |
| Frozen follow-up inputs | 23 required input files missing. Do not replace the historical GTFS snapshot with today's feed. |
| Recorded trained-policy binaries | 69 metadata records lack their policy.zip. |

Raw-data retrieval must reproduce the SHA-256 hashes in the original manifests. The GTFS endpoint is mutable; a newer feed is not an equivalent historical input. Source redistribution rights remain an author confirmation item.

On Linux, install with the separate CPU hash lock `requirements-linux.lock.txt` using uv's `--require-hashes --torch-backend cpu` options. The original Windows lock remains unchanged. Run `python scripts/publication_audit.py` to inspect current blockers; it exits nonzero until they are resolved.

The historical reproducibility and separation reports describe a different machine state. Their PASS counts must not be quoted as fresh validation. See `docs/SUBMISSION_READINESS.md` for the current audit.
