# Execution Status

Verified starting point for the implementation and experimentation phase (M3–M16).
**Nothing in M0–M3 was regenerated.** This document records what was inspected and confirmed.

## Repository state at phase start

| Item | Value |
|---|---|
| Branch | `development` |
| Commit | `8d90b2a` — *M3 precursor: verify every data source by probing it; correct three M0 audit errors* |
| Working tree | clean |
| Test suite | **38 passed**, 0 failed |
| Interpreter | CPython 3.11.9, `.venv/` |
| Stack verification | `scripts/check_environment.py` — 10/10 functional checks |

## M0–M3 artefacts verified present

| Artefact | Status |
|---|---|
| `docs/research_audit.md` | present (42.5 KB) — RQs, hypotheses, novelty, 21 assumptions, threats, publication risk |
| `docs/experimental_protocol.md` | present (40.0 KB) — **frozen pre-registration**; treated as immutable |
| `docs/decisions.md` | present (14.3 KB) — D-01…D-11 |
| `docs/data_sources.md` | present (15.8 KB) — authoritative verified data plan |
| `docs/progress_log.md` | present |
| `experiments/registry.yaml` | present — pre-registration skeleton, `frozen: false` |
| `results/source_probe.json` | present — machine-readable evidence, 11 usable sources |
| `requirements.txt`, `environment.yml` | present, pinned to the verified environment |

## Existing code (carried forward, not rewritten)

| Module | Lines | Role |
|---|---|---|
| `src/common/paths.py` | 69 | repo-relative paths; canonical `results/` layout |
| `src/common/seeding.py` | 94 | **common-random-numbers guarantee** — env streams take no method argument |
| `src/common/config.py` | 177 | provenance enforcement; `unresolved` blocked outside a tagged sweep |
| `src/common/manifest.py` | 199 | traceability chain; append-only guard on `results/` |
| `src/common/logging.py` | 54 | structured logging |
| `scripts/check_environment.py` | 232 | functional stack verification |
| `scripts/probe_sources.py` | 269 | repeatable data-source verification |

`src/{data,forecasting,train_health,simulator,environment,rl,baselines,evaluation,explainability,visualization}`
are **empty** — this is the work of M3–M15.

## Known residual from M2

`requirements.lock.txt` (hash-pinned) is absent: `pip-compile --generate-hashes` downloads every
distribution to hash it and did not complete within the available window. **Non-blocking** —
`requirements.txt` is already fully pinned to the environment that passed the functional checks and
the test suite, and `environment.yml` mirrors it. Regeneration is retried in the background; if it
does not complete, the reproducibility audit records exact-version pinning without hash
verification, rather than claiming a guarantee that does not exist.

## Progress since phase start

| Milestone | Commit | Outcome |
|---|---|---|
| M3 data ingestion | `bbc2088` | 1,344,528 in-regime station-hours; 10/10 quality checks; 0 blocking failures |
| M4 forecasting | `a987216` | implemented, smoke gate passed; **full fit still running** |
| M5 digital twin | `bbc2088` | validated against published statistics; 0 mismatches |
| A-01 / D-12 | `976dd80` | primary endpoint operationalised as `left_behind_fraction` |
| M6 RL environment | `02109e9` | **complete** — see `docs/m6_scientific_gate.md` |

Test suite: **38 → 301 passing**, 0 failing.

Three amendments to the frozen pre-registration, all made **before any policy existed** and all
recorded with the measurement that forced them: **A-01/D-12** (primary endpoint),
**A-02/D-13** (reward waiting term — the previous reading made "run no service" reward-optimal),
**A-03/D-14** (scenario random-stream keying — `surge` was scoring better than `normal`).

`requirements.lock.txt` was subsequently produced by `scripts/generate_lockfile.py`, which reads
SHA-256 digests from the PyPI JSON API rather than downloading every distribution: 55/55 packages
hashed and verified. The M2 residual below is therefore closed.

## Starting point for this phase

**M3 — data ingestion.** The data *sources* were verified during the M3 precursor; the *pipeline*
does not exist yet. Implementation begins at `src/data/`.

## Standing constraints for M3–M16

- The manuscript, the frozen pre-registration, the eleven decisions, the research questions, the
  hypotheses and the evaluation methodology are **immutable** unless a genuine scientific
  contradiction is found — in which case work stops and the conflict is reported before any change.
- The objective is to **test** the hypotheses, not to support them. A null or negative result, or a
  baseline beating the proposed method, is a valid outcome and is preserved.
- The uncertainty-aware arm receives no tuning advantage over its ablations.
- No fabricated data, results, significance, or citations. No claim of real-world validation from
  simulation-only evidence.
