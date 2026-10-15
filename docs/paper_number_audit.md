# Manuscript number audit

*Generated at 2026-08-28T09:57:27Z. Every decimal number in `docs/paper_final.md` is compared against the stored artefacts.*

## Method

The manuscript is rendered by a script that reads `results/` directly, so no experimental value is typed into prose. This audit is a second, independent check: it extracts every decimal literal from the rendered text and matches it against values recomputed from the artefacts.

## Key values verified

| Quantity | Artefact value | Source |
|---|---:|---|
| Primary mean, `ppo_uq` | 0.031681 | statistical_analysis.json |
| Primary mean, `ppo_point` | 0.017939 | statistical_analysis.json |
| Primary difference | +0.013742 | statistical_analysis.json |
| Hodges–Lehmann | 0.013588 | statistical_analysis.json |
| 95% BCa CI | [0.006570, 0.018523] | statistical_analysis.json |
| Holm-adjusted p | 0.007741 | statistical_analysis.json |
| Placebo difference | +0.008525 | statistical_analysis.json |
| Placebo Holm-adjusted p | 0.011707 | statistical_analysis.json |
| SESOI | 0.002 | power_analysis.json |
| Registered n | 10 | power_analysis.json |
| Required n for SESOI | 69 | power_analysis.json |
| Pilot SD | 0.005298 | power_analysis.json |
| Realised SD | 0.011269 | seed_level.parquet (recomputed) |
| Wilcoxon p | 0.005859 | seed_level.parquet (recomputed) |
| Worst jackknife Holm p | 0.02043 | recomputed |
| Evaluation cells | 7,610 | results/runs |
| Policy checkpoints | 58 | models/policies |
| Dispatch reduction (surge) | 10.8% | mechanism_analysis.json |

## Unmatched decimal literals

Every decimal literal in the manuscript was extracted after removing URLs, arXiv identifiers, DOIs, scientific-notation mantissas and section cross-references, then matched against values recomputed from the artefacts. The literals below did not match a registered experimental quantity; each is identified individually:

| Literal | What it is |
|---|---|
| `-10.0` | part of the OS version string `Windows-10-10.0.26200` |
| `8.2` | a cross-reference to protocol section 8.2, not a measurement |

**Unidentified literals: 0.** Every residual literal is accounted for.

## Result

**PASS.** Every experimental value in the manuscript is rendered directly from a stored artefact. No value was typed by hand, and none conflicts with the stored analysis.
