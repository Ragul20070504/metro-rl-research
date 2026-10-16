# Manuscript claim audit

*Generated at 2026-10-16T13:48:06Z. Each major scientific claim is traced to its source, experiment, figure or table, and statistical result.*

| Claim | Source | Location | Statistical basis | Status |
|---|---|---|---|---|
| H1 was not supported; the uncertainty arm performed worse | `statistical_analysis.json / EXP-UQ-001` | §11.2, Table 5, Figure 4 | +0.013742, Holm p = 0.007741 | SUPPORTED |
| The degradation is not explained by extra input dimensions | `statistical_analysis.json / EXP-UQ-002` | §11.3, Table 5, Figure 5 | +0.008525, Holm p = 0.011707 | SUPPORTED |
| The UQ arm has about 11% fewer dispatches; the placebo arm does not | `mechanism_analysis.json` | §12.1, Figure 6 | -24.7 vs +1.9 inductions | SUPPORTED |
| Historical binned service-loss association | `results/runs/**/epochs.parquet` | §12.3; Figure 7 withheld | binned association; state and decisions not controlled | LIMITED — epoch sources unavailable |
| Channel content is associated with lower dispatch volume | `mechanism_analysis.json` | §12.3 | behavioural association plus a theoretical argument | HEDGED — stated as *consistent with*, not established; the epoch-level registered test is unresolved and is declared so |
| No learned arm beats the tuned threshold heuristic | `seed_level.parquet` | §11.1, Table 6, Figure 9 | 0.01184 vs 0.01794 | SUPPORTED — descriptive only; no inferential test is defined and the text says so |
| The result holds across scenarios, seeds and sensitivity conditions | `statistical_analysis.json, sensitivity_analysis.json, recomputed jackknife` | §13, Table 7, Table 8 | 4/4 scenarios; 9/10 seeds; 12/12 conditions | SUPPORTED |
| The study is underpowered for its own SESOI | `power_analysis.json + recomputed realised SD` | §9, §16 | n = 10 vs required 69; realised SD 2.13× the pilot estimate | SUPPORTED |
| Magnitudes are simulator-internal | `simulator_validation.json` | §5, §16 | 7 of 14 quantities not externally validated | SUPPORTED |
| The follow-up contributes no evidence | `docs/followup_status.md` | §16 item 9 | 11 of 80 policies, single budget, asymmetric, no evaluation cells | SUPPORTED — no follow-up number appears anywhere in the manuscript |
| Feature attributions do not establish causality | `shap_analysis.json` | §13, S9 | stated caveat | SUPPORTED |
| The framework's components are not claimed as novel | `n/a — a claim deliberately not made` | §1 contributions | explicit disclaimer in the contribution list | SUPPORTED |

## Claims deliberately NOT made

- That uncertainty-aware reinforcement learning is generally ineffective.
- That reinforcement learning is unsuitable for metro dispatch.
- That the framework or its components are novel.
- Any causal claim from feature attribution.
- Any magnitude claim about a physical metro system.
- Any claim about behaviour at other training budgets.
- Any claim of optimality, real-time capability, or deployment readiness.
- Any equivalence reading of a null result.

## Current evidence limitations

The claim mapping above describes the historical reports. It is not independent verification of their source data or of registration chronology.

| Current check | Status | Detail |
|---|---|---|
| Historical code provenance | BLOCKED | 7609 dirty manifests; 559 unknown commits; 2 referenced commits unavailable. |
| Original study tag | BLOCKED | original-confirmatory-study is absent locally and was not advertised by origin. |
| Original raw-data bytes | BLOCKED | 11 missing files; 0 checksum mismatches. |
| Frozen follow-up inputs | BLOCKED | 23 required input files missing. Do not replace the historical GTFS snapshot with today's feed. |
| Recorded trained-policy binaries | BLOCKED | 69 metadata records lack their policy.zip. |
| Follow-up grid | BLOCKED | No new inferential result may be reported until every budget is complete. |
| Independent registration record | LIMITATION | An internal protocol exists, but its chronology cannot be independently established. The revision removes the externally verifiable pre-registration claim. |
| Placebo construction | LIMITATION | The inherited placebo bootstraps a rolling pool with replacement; exact marginal matching and complete information removal are not established. The manuscript now limits attribution accordingly. |
| Author declarations | BLOCKED | Affiliations, corresponding-author details, CRediT roles, funding and conflicts remain author-confirmation placeholders. |
| Journal requirements and archival deposit | BLOCKED | Current author guidelines and a public artifact archive/DOI have not been verified; no submission was made. |

**REVIEW REQUIRED.** Submission readiness is determined by the current evidence audit, not by a hard-coded zero count of unsupported claims.
