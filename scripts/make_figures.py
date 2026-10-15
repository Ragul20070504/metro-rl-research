"""Generate every publication figure and table from stored results (M15).

Nothing here computes an experimental number; it only reads and renders. Each figure writes a
`.source.json` companion naming the files it came from, so the requirement that no experimental
value is hand-entered is machine-checkable rather than promised.

    python scripts/make_figures.py
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.common.paths import REPO_ROOT, RESULTS_DIR, ensure_dir
from src.visualization.figures import (
    FIGURES_DIR,
    METHOD_LABELS,
    METHOD_ORDER,
    TABLES_DIR,
    figure_ablations,
    figure_forecasting,
    figure_metric_by_method,
    figure_primary_endpoint,
    figure_robustness,
    figure_scenario_comparison,
    figure_shap,
    figure_simulator_validation,
)

logger = get_logger(__name__)


def write_tables(seeds: pd.DataFrame, analysis: dict) -> list[str]:
    """Markdown and CSV tables, generated from the same stored frames as the figures."""
    ensure_dir(TABLES_DIR)
    written: list[str] = []

    main = seeds[seeds["variant"] == "main"]
    if not main.empty:
        # Table 1: the headline metric table.
        metrics = ["left_behind_fraction", "waiting_accrued_passenger_hours",
                   "waiting_time_p95_s", "energy_units", "cost_units", "total_reward"]
        summary = main.groupby(["method", "scenario"], observed=True)[metrics].agg(["mean", "std", "count"])
        summary.to_csv(TABLES_DIR / "table01_main_metrics.csv")
        written.append("table01_main_metrics.csv")

        lines = ["# Table 1 — main metrics by method and scenario", "",
                 "Mean over training seeds (SD in brackets, n = training seeds).", "",
                 "| method | scenario | " + " | ".join(metrics) + " |",
                 "|---|---|" + "---:|" * len(metrics)]
        for (method, scenario), row in summary.iterrows():
            cells = []
            for metric in metrics:
                mean, sd, count = row[(metric, "mean")], row[(metric, "std")], row[(metric, "count")]
                cells.append(f"{mean:.4g} ({sd:.2g}, n={int(count)})" if np.isfinite(mean) else "—")
            lines.append(f"| `{method}` | {scenario} | " + " | ".join(cells) + " |")
        (TABLES_DIR / "table01_main_metrics.md").write_text("\n".join(lines), encoding="utf-8")
        written.append("table01_main_metrics.md")

    # Table 2: the confirmatory family, verbatim from the analysis file.
    confirmatory = analysis.get("confirmatory", [])
    if confirmatory:
        lines = ["# Table 2 — confirmatory family (protocol section 7.3)", "",
                 "Holm-corrected at family-wise alpha = 0.05.", "",
                 "| experiment | comparison | n | test | statistic | p | Holm p | "
                 "Hodges–Lehmann | 95% BCa CI | rejects |",
                 "|---|---|---:|---|---:|---:|---:|---:|---|---|"]
        for record in confirmatory:
            if record.get("status") != "RUN":
                lines.append(f"| {record['experiment_id']} | "
                             f"{record['method_a']} vs {record['method_b']} | — | NOT RUN | — | "
                             f"— | — | — | — | {record['reason']} |")
                continue
            lines.append(
                f"| {record['experiment_id']} | `{record['method_a']}` vs "
                f"`{record['method_b']}` | {record['n']} | {record['test']} | "
                f"{record['statistic']:.4g} | {record['p_value']:.4g} | "
                f"{record['p_adjusted']:.4g} | {record['hodges_lehmann']:.5g} | "
                f"[{record['ci_low']:.5g}, {record['ci_high']:.5g}] | "
                f"{'yes' if record['rejects_at_family_alpha'] else 'no'} |"
            )
        (TABLES_DIR / "table02_confirmatory.md").write_text("\n".join(lines), encoding="utf-8")
        written.append("table02_confirmatory.md")

    return written


def main() -> int:
    ensure_dir(FIGURES_DIR)
    seed_path = RESULTS_DIR / "seed_level.parquet"
    analysis_path = RESULTS_DIR / "statistical_analysis.json"

    seeds = pd.read_parquet(seed_path) if seed_path.exists() else pd.DataFrame()
    analysis = json.loads(analysis_path.read_text(encoding="utf-8")) if analysis_path.exists() else {}

    produced: list[str] = []
    if not seeds.empty:
        produced.append(str(figure_primary_endpoint(seeds)))
        produced.append(str(figure_scenario_comparison(seeds)))
        produced.append(str(figure_metric_by_method(
            seeds, "waiting_accrued_passenger_hours", "Passenger-hours of waiting",
            "Total passenger waiting accrued over the operating day", "fig03_waiting_time")))
        produced.append(str(figure_metric_by_method(
            seeds, "waiting_time_p95_s", "95th percentile wait (s)",
            "Tail waiting time", "fig04_p95_waiting_time")))
        produced.append(str(figure_metric_by_method(
            seeds, "energy_units", "Energy (train-km-equivalent units)",
            "Energy — units are unresolved (assumption A15), not kWh", "fig05_energy")))
        produced.append(str(figure_metric_by_method(
            seeds, "cost_units", "Cost (induction-equivalent units)",
            "Operating cost — units are unresolved (assumption A16), not currency",
            "fig06_cost")))
    else:
        logger.warning("no seed-level results; sweep figures will be placeholders")

    if analysis:
        produced.append(str(figure_robustness(analysis)))
        produced.append(str(figure_ablations(analysis)))

    produced.append(str(figure_forecasting()))
    produced.append(str(figure_shap()))
    produced.append(str(figure_simulator_validation()))

    tables = write_tables(seeds, analysis) if not seeds.empty or analysis else []

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "figures": [str(p) for p in produced],
        "tables": tables,
        "policy": (
            "every figure and table is generated from stored result files; no experimental "
            "number is hand-entered. Each figure has a .source.json companion naming its inputs."
        ),
    }
    (RESULTS_DIR / "figures_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"{len(produced)} figures, {len(tables)} tables -> {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
