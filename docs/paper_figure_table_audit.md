# Figure and table audit

*Generated at 2026-10-16T13:48:06Z.*

The historical manifest attributes these assets to `scripts/make_paper_assets.py`. This audit checks current file availability, not whether every source computation can be repeated or every interpretation is valid.

## Figures

| Asset | Source data | Experiment | Interpretation | Provenance file |
|---|---|---|---|---|
| `fig01_research_framework` | *schematic — no data* | — | Schematic of the study design | present |
| `fig02_twin_architecture` | `results/simulator_validation.json` | §5 | Environment structure and validation status | present |
| `fig03_forecasting` | `results/forecasting_metrics.json` | §4.3 | Forecast accuracy and calibration | present |
| `fig04_primary_uq_vs_point` | `results/seed_level.parquet`, `results/statistical_analysis.json` | §11.2 | Primary comparison, paired by seed | present |
| `fig05_placebo` | `results/seed_level.parquet` | §11.3 | Three-arm distribution under surge | present |
| `fig06_mechanism` | `results/mechanism_analysis.json` | §12.1 | Dispatch response to informative vs placebo σ | present |
| `fig07_convex_service_loss` | `results/runs/main/surge/**/epochs.parquet` | §12.3 | Historical binned association — withheld | present |
| `fig08_scenario_robustness` | `results/seed_level.parquet` | §13 | Ordering across four scenarios | present |
| `fig09_baselines` | `results/seed_level.parquet` | §11.1 | Full method ranking | present |
| `fig10_sensitivity` | `results/sensitivity_analysis.json` | §13 | OD sensitivity conditions | present |
| `fig11_shap` | `results/shap_analysis.json` | §13, S9 | Feature attribution | present |

## Tables

| Asset | Source data | Caption |
|---|---|---|
| `table01_dataset` | `results/reproducibility_audit.json` | Dataset characteristics, splits and provenance |
| `table02_forecasting` | `results/forecasting_metrics.json` | Forecasting performance |
| `table03_twin_validation` | `results/simulator_validation.json` | Digital twin validation against published statistics |
| `table04_protocol` | `results/statistical_analysis.json`, `results/power_analysis.json` | Registered experimental protocol |
| `table05_primary_statistics` | `results/statistical_analysis.json` | Confirmatory results (Holm-corrected, alpha 0.05) |
| `table06_baselines` | `results/seed_level.parquet` | All methods, all scenarios. Baselines are deterministic (n=1); no paired test against them is defined under the seed-level unit of analysis |
| `table07_sensitivity` | `results/sensitivity_analysis.json` | Sensitivity conditions (exploratory, Benjamini-Hochberg q = 0.10) |
| `table08_ablations` | `results/statistical_analysis.json` | Registered arm ablations (exploratory) |
| `table09_robustness` | `results/seed_level.parquet`, `results/power_analysis.json`, `results/sensitivity_analysis.json` | Robustness of the primary result |
| `table10_limitations` | `results/power_analysis.json`, `results/simulator_validation.json`, `results/seed_level.parquet`, `docs/followup_status.md` | Limitations and practical implications |

## Missing files or source patterns

- `results/runs/main/surge/**/epochs.parquet`

## Result

**REVIEW REQUIRED.** Missing source files prevent full asset regeneration.
