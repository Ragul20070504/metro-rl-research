# Product architecture — Metro Intelligent Dispatch

A decision-support prototype built on the validated artefacts of the completed M0–M17 research
study. It is a **separate application** from the research dashboard and shares no state with it.

## Two applications, deliberately separate

| | Research dashboard | Product prototype |
|---|---|---|
| Entry point | `app.py` | `product/app.py` |
| Port | 8501 | 8502 |
| Audience | reviewers, examiners | operators, demonstration panels |
| Purpose | present the completed study | operate a simulated line |
| Status | completed research deliverable — **not modified by product work** | prototype |

Both run at the same time. `scripts/check_both_apps.py` verifies that, and verifies the research
artefacts are byte-identical afterwards.

The research dashboard was not restructured, re-pointed or wrapped. `product/` is additive: the
only change outside it was extending `testpaths` in `pyproject.toml` so both suites run in one
command.

## Layers

```
                    OPERATOR
                        │
                 STREAMLIT UI  (product/app.py, product/ui/)
                        │
                 SERVICE LAYER  (product/services/)
        ┌───────────────┼────────────────┐
        │               │                │
  SessionService   Repository      presentation
  (live sim,       (read-only      helpers
   recommend,       research
   step)            artefacts)
        │               │
        ▼               ▼
  src/ research    results/ , models/
  environment      OPENED READ-ONLY
```

The UI never touches a research file directly. Every research number arrives through
`ResearchResultsRepository`, and every live number through `SessionService`.

## Modules

| Module | Responsibility |
|---|---|
| `product/app.py` | Shell: navigation, panel-demo route, read-only assertion, error containment |
| `product/ui/pages.py` | The twelve pages |
| `product/ui/components.py` | Evidence badges, metric rows, guards, "NOT AVAILABLE" |
| `product/services/research_repository.py` | Read-only access to frozen study artefacts |
| `product/services/operations.py` | Live episode, operating state, recommendation, simulation |
| `product/tests/test_product.py` | Product suite, including the research-integrity checks |

## The read-only guarantee

Three independent mechanisms, because one is a claim and three are a property:

1. **No write path exists.** No module under `product/` opens a `results/` or `models/` path for
   writing. `SessionService` never calls the underlying session's `save()`.
2. **Hashing at runtime.** `assert_read_only()` fingerprints the protected artefacts at first use
   and re-checks them on every page render. The sidebar shows the result live, so a violation
   surfaces during a demonstration rather than in a later audit.
3. **Tests.** `TestResearchIntegrity` hashes the protected set before and after exercising the
   repository, and statically scans `product/` for write calls aimed at research paths — which
   catches the realistic failure mode: someone later adding a convenience `save()`.

## Evidence separation

Every panel that shows a number declares where it came from:

| Badge | Meaning |
|---|---|
| `CONFIRMATORY` | Registered research result, fixed before the data were seen |
| `EXPLORATORY` | Hypothesis-generating research result |
| `DESCRIPTIVE` | No hypothesis test defined for the quantity |
| `FOLLOW-UP / INCOMPLETE` | Stopped before any comparison was possible — **not evidence** |
| `PILOT / DISCARDED` | Discarded by protocol — never a final result |
| `PRODUCT DEMONSTRATION` | Live simulation, one simulated draw |

The badge is attached to the panel in code, not left to the reader. `SimulationOutcome` and
`DispatchRecommendation` both carry `evidence = PRODUCT DEMONSTRATION` as a field, so a live value
cannot be passed around and later displayed as research.

Discarded pilot policies are excluded upstream in `src/demo/session.py`, so the product cannot
offer one even by accident.

## Data flow for a dispatch recommendation

1. Operator selects strategy, scenario, policy instance and simulation day.
2. `SessionService` builds the validated environment and loads the trained policy — read-only.
3. `operating_state()` reads queues, availability, clock and the frozen forecaster's output.
4. `recommend()` asks the policy for an action, applying the **same mask, clip and truncation as
   a research step**, so the recommendation cannot differ from what the environment would execute.
5. The UI renders the action, the observable reasons behind it, and the current cumulative metrics.
6. Nothing is written.

## Caching

`@st.cache_data` on artefact reads, `lru_cache` on the repository and the seed-level table, and
one live session per selection held in session state. No training, no re-fitting, no research
recomputation is reachable from the UI.

## Error handling

Missing artefacts raise `ResearchArtefactMissing` and render as **NOT AVAILABLE** naming the file.
Page-level failures are contained so one broken panel does not take down a demonstration. **No
substitute or placeholder value is ever displayed.**
