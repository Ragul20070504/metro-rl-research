# Manuscript claim audit

*Generated at 2026-08-28T09:57:27Z. Each major scientific claim is traced to its source, experiment, figure or table, and statistical result.*

| Claim | Source | Location | Statistical basis | Status |
|---|---|---|---|---|
| H1 was not supported; the uncertainty arm performed worse | `statistical_analysis.json / EXP-UQ-001` | §11.2, Table 5, Figure 5 | +0.013742, Holm p = 0.007741 | SUPPORTED |
| The degradation is not explained by extra input dimensions | `statistical_analysis.json / EXP-UQ-002` | §11.3, Table 6, Figure 6 | +0.008525, Holm p = 0.011707 | SUPPORTED |
| Calibrated uncertainty reduces dispatch ~11%; placebo does not | `mechanism_analysis.json` | §12.1, Table 9, Figure 7 | -24.7 vs +1.9 inductions | SUPPORTED |
| Service loss is convex once capacity binds | `results/runs/**/epochs.parquet` | §12.3, Figure 8 | measured relationship between arrivals and left-behind passengers | SUPPORTED |
| The controller learned a conservative response in the wrong direction | `mechanism_analysis.json + convex-loss measurement` | §12.3 | behavioural association plus a theoretical argument | HEDGED — stated as *consistent with*, not established; the epoch-level registered test is unresolved and is declared so |
| No learned arm beats the tuned threshold heuristic | `seed_level.parquet` | §11.1, Table 4, Figure 10 | 0.01184 vs 0.01794 | SUPPORTED — descriptive only; no inferential test is defined and the text says so |
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

## Flags

| Category | Count | Detail |
|---|---:|---|
| UNSUPPORTED CLAIM | 0 | — |
| OVERCLAIM | 0 | The mechanism interpretation is hedged in §12.3 rather than asserted |
| MISSING CITATION | 0 | — |
| MISSING SOURCE | 0 | Every claim traces to a stored artefact |
| AMBIGUOUS LANGUAGE | 1 | 'approximately 11% fewer dispatches' is a rounded derived quantity; the exact value is 10.8% and both the underlying induction counts and the p-value are reported |

## Result

**PASS.**
