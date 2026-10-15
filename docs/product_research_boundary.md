# Research / product boundary

The completed study is immutable. The product reads it and never writes it. This document records
how that is enforced rather than merely intended.

## The boundary

| | Research | Product |
|---|---|---|
| Code | `src/`, `scripts/`, `app.py` | `product/` |
| Entry point | `app.py` (port 8501) | `product/app.py` (port 8502) |
| Artefacts | `results/`, `models/`, `experiments/` | reads them; writes nothing |
| Tests | `tests/` | `product/tests/` |
| Status | frozen, tagged `original-confirmatory-study` | under development |

The only change made outside `product/` during product development was extending `testpaths` in
`pyproject.toml` so both suites run in one command. No research file, artefact, configuration or
entry point was modified.

## Enforcement

**1. No write path.** No module under `product/` opens a research path for writing.
`SessionService` wraps the demonstration session but never calls its `save()`.

**2. Runtime hashing.** `assert_read_only()` fingerprints the protected artefacts at first use and
re-checks on every page render. The sidebar reports the result live, so a violation appears during
a demonstration rather than in a later audit.

**3. Tests.** `TestResearchIntegrity` hashes the protected set before and after exercising the
repository, and statically scans `product/` for write calls aimed at `results/` or `models/` —
which catches the realistic failure mode: a convenience `save()` added later.

**4. Dual-application check.** `scripts/check_both_apps.py` launches both applications, confirms
both serve simultaneously, and confirms the protected artefacts hash identically afterwards.

**5. Immutability audit.** `scripts/separation_audit.py` compares every registered artefact
against the git tag and reports any difference in recorded results.

## Evidence separation in the interface

Research and live values appear on the same screens, so the distinction is carried in code rather
than left to the reader. Every panel declares one evidence class, and `SimulationOutcome` and
`DispatchRecommendation` carry `evidence = "PRODUCT DEMONSTRATION"` as a field — a live value
cannot be passed around and later rendered as research.

Two classes are structurally excluded from ever appearing as a result:

- **PILOT / DISCARDED** — pilot policies are filtered out in `src/demo/session.py`, so the product
  cannot offer one even by accident.
- **FOLLOW-UP / INCOMPLETE** — the follow-up page reports counts and an explicit statement that no
  result exists. `follow_up_status()` hard-codes `usable = False`; there is no code path that
  turns partial follow-up data into a displayed comparison.

## What the product may and may not do

**May:** read stored results, run the validated simulator forward, load trained policies, produce
recommendations, display research findings with their evidence class.

**May not:** write to `results/` or `models/`, retrain or fine-tune, modify the registry, alter
the research dashboard, present demonstration values as research, present pilot or incomplete
follow-up data as evidence, or claim a result the study did not produce.

## The conclusion the product must carry

**H1 = NOT SUPPORTED.** Supplying the controller with calibrated forecast uncertainty produced
worse passenger service than a point forecast. The product presents this finding; it does not
soften, contradict or work around it.
