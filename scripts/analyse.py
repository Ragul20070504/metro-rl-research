"""Statistical analysis of the stored sweep (M14).

Reads only what the sweep actually produced (`results/runs/`), reduces it to the unit of analysis
protocol section 8.2 requires, and runs the frozen tests of section 8.3.

**The confirmatory family is exactly two tests** (section 7.3): `ppo_uq` vs `ppo_point` and
`ppo_uq` vs `ppo_placebo`, both on `left_behind_fraction` under `surge` at `w_balanced`,
Holm-corrected at family-wise alpha = 0.05. Everything else is exploratory, Benjamini-Hochberg
controlled at q = 0.10, and labelled exploratory in every table.

**Nothing here decides what to report.** The comparison list is fixed by the registry, the tests
are fixed by the protocol, and the SESOI was frozen in amendment A-04 before the pilot ran. The
script's only judgement is arithmetic.

    python scripts/analyse.py

Writes `results/statistical_analysis.json` and `docs/results.md`.
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.common.paths import REPO_ROOT, RESULTS_DIR, ensure_dir
from src.evaluation.aggregate import (
    HIGHER_IS_BETTER,
    METRICS_OF_INTEREST,
    load_frame,
    paired_frame,
    seed_level,
)
from src.evaluation.runner import discover_cells
from src.evaluation.statistics import (
    benjamini_hochberg,
    describe,
    holm,
    paired_analysis,
)

logger = get_logger(__name__)

PRIMARY_METRIC = "left_behind_fraction"
PRIMARY_SCENARIO = "surge"
SCENARIOS = ("normal", "surge", "disruption", "combined")

#: Protocol section 7.3. Exactly two, and they are fixed here rather than chosen from results.
CONFIRMATORY = (
    ("ppo_uq", "ppo_point", "EXP-UQ-001", "H1 (primary)"),
    ("ppo_uq", "ppo_placebo", "EXP-UQ-002", "H1c (co-primary, placebo control)"),
)

EXPLORATORY_PAIRS = (
    ("ppo_uq", "ppo_point", "EXP-UQ-003", "H2"),
    ("ppo_uq", "ppo_placebo", "EXP-UQ-003", "H2"),
    ("ppo_uq", "ppo_uq_single_obj", "EXP-MO-001", "H3"),
    ("ppo_uq", "fixed_timetable", "EXP-BASE-001", "H4"),
    ("ppo_uq", "heuristic_threshold", "EXP-BASE-001", "H5"),
    ("ppo_uq", "milp_static", "EXP-BASE-001", "H6"),
    ("ppo_uq", "milp_rolling", "EXP-BASE-005", "H6b"),
    ("ppo_uq", "ppo_uq_noavail", "EXP-AVAIL-001", "H7"),
)


def _sesoi() -> float:
    from scripts.power_analysis import SESOI

    return SESOI


def run_confirmatory(seeds: pd.DataFrame, episodes: pd.DataFrame) -> list[dict[str, Any]]:
    results = []
    for method_a, method_b, experiment_id, hypothesis in CONFIRMATORY:
        pair = paired_frame(seeds, episodes, method_a=method_a, method_b=method_b,
                            scenario=PRIMARY_SCENARIO, metric=PRIMARY_METRIC)
        if pair is None or not pair.paired:
            results.append({
                "experiment_id": experiment_id, "hypothesis": hypothesis,
                "method_a": method_a, "method_b": method_b,
                "status": "NOT RUN",
                "reason": ("no stored results" if pair is None else
                           "the two arms share no training seeds, so the registered paired "
                           "test cannot be run"),
            })
            continue
        analysis = paired_analysis(
            pair.values_a, pair.values_b,
            label=f"{experiment_id} {hypothesis}", metric=PRIMARY_METRIC,
            method_a=method_a, method_b=method_b, family="confirmatory",
            lower_is_better=True, sesoi=_sesoi(),
            episode_a=pair.episodes_a, episode_b=pair.episodes_b,
        )
        record = analysis.to_dict()
        record.update({"experiment_id": experiment_id, "hypothesis": hypothesis,
                       "scenario": PRIMARY_SCENARIO, "status": "RUN",
                       "sentence": analysis.sentence()})
        results.append(record)

    live = [r for r in results if r.get("status") == "RUN"]
    if live:
        adjusted = holm([r["p_value"] for r in live])
        for record, value in zip(live, adjusted):
            record["p_adjusted"] = float(value)
            record["adjustment"] = "Holm (family-wise alpha = 0.05, 2 tests)"
            record["rejects_at_family_alpha"] = bool(value < 0.05)
    return results


def run_exploratory(seeds: pd.DataFrame, episodes: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for method_a, method_b, experiment_id, hypothesis in EXPLORATORY_PAIRS:
        for scenario in SCENARIOS:
            for metric in (PRIMARY_METRIC, "waiting_accrued_passenger_hours",
                           "waiting_time_p95_s", "energy_units", "cost_units", "total_reward"):
                pair = paired_frame(seeds, episodes, method_a=method_a, method_b=method_b,
                                    scenario=scenario, metric=metric)
                if pair is None or not pair.paired:
                    continue
                analysis = paired_analysis(
                    pair.values_a, pair.values_b,
                    label=f"{experiment_id} {hypothesis} {scenario}", metric=metric,
                    method_a=method_a, method_b=method_b, family="exploratory",
                    lower_is_better=metric not in HIGHER_IS_BETTER,
                    episode_a=pair.episodes_a, episode_b=pair.episodes_b,
                )
                record = analysis.to_dict()
                record.update({"experiment_id": experiment_id, "hypothesis": hypothesis,
                               "scenario": scenario, "sentence": analysis.sentence()})
                records.append(record)

    if records:
        adjusted = benjamini_hochberg([r["p_value"] for r in records], q=0.10)
        for record, value in zip(records, adjusted):
            record["p_adjusted"] = float(value)
            record["adjustment"] = "Benjamini-Hochberg FDR (q = 0.10)"
            record["passes_fdr"] = bool(value < 0.10)
    return records


def unpaired_comparisons(seeds: pd.DataFrame) -> list[dict[str, Any]]:
    """Descriptive contrasts against the deterministic baselines.

    A fixed timetable, a heuristic and a MILP have no training seed, so there is nothing to pair a
    trained policy against. Reporting a paired test here would fabricate the pairing; these are
    reported as descriptive differences of means, explicitly labelled.
    """
    out: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for metric in (PRIMARY_METRIC, "waiting_accrued_passenger_hours", "total_reward"):
            for baseline in ("fixed_timetable", "heuristic_threshold",
                             "milp_static", "milp_rolling"):
                base = seeds[(seeds["method"] == baseline) & (seeds["scenario"] == scenario)
                             & (seeds["variant"] == "main")]
                rl = seeds[(seeds["method"] == "ppo_uq") & (seeds["scenario"] == scenario)
                           & (seeds["variant"] == "main")]
                if base.empty or rl.empty:
                    continue
                out.append({
                    "comparison": f"ppo_uq vs {baseline}",
                    "scenario": scenario,
                    "metric": metric,
                    "kind": "descriptive (unpaired)",
                    "note": (
                        f"{baseline} is deterministic and has no training seed, so no paired "
                        "test is defined; this is a difference of means, not a hypothesis test"
                    ),
                    "mean_ppo_uq": float(rl[metric].mean()),
                    f"mean_{baseline}": float(base[metric].mean()),
                    "difference": float(rl[metric].mean() - base[metric].mean()),
                    "n_ppo_uq_seeds": int(len(rl)),
                    "n_baseline_seeds": int(len(base)),
                })
    return out


def descriptives(seeds: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    main = seeds[seeds["variant"] == "main"]
    for (method, scenario), block in main.groupby(["method", "scenario"], observed=True):
        for metric in METRICS_OF_INTEREST:
            summary = describe(block[metric].tolist())
            rows.append({"method": method, "scenario": scenario, "metric": metric,
                         **summary.to_dict()})
    return rows


def degradation(seeds: pd.DataFrame) -> list[dict[str, Any]]:
    """Protocol section 7.4: a stressed metric divided by the same metric under `normal`.

    Valid only because amendment A-03 removed the scenario name from the random-stream entropy;
    before that fix the two scenarios were unrelated realisations and the ratio mixed the effect
    with realisation noise.
    """
    rows: list[dict[str, Any]] = []
    main = seeds[seeds["variant"] == "main"]
    for method, block in main.groupby("method", observed=True):
        baseline = block[block["scenario"] == "normal"]
        if baseline.empty:
            continue
        for scenario in ("surge", "disruption", "combined"):
            stressed = block[block["scenario"] == scenario]
            if stressed.empty:
                continue
            for metric in (PRIMARY_METRIC, "waiting_accrued_passenger_hours", "waiting_time_p95_s"):
                base = float(baseline[metric].mean())
                value = float(stressed[metric].mean())
                rows.append({
                    "method": method, "scenario": scenario, "metric": metric,
                    "normal": base, "stressed": value,
                    "degradation_ratio": (value / base) if base > 0 else float("nan"),
                })
    return rows


def _unrecovered_cells() -> list[dict[str, Any]]:
    """Cells that failed twice **and are still missing**, read back from the runner's manifests.

    A sweep that ended incomplete must say so in the analysis, not only in a log nobody reads.

    The stored manifest is a record of what failed *at the time*, so it is filtered against the
    results tree before being reported: once a diagnosed failure is repaired and the cells are
    rerun, they are no longer missing, and reporting them as unrecovered would understate the
    completed design. The manifests themselves are left in place as the audit trail.
    """
    from src.evaluation.runner import RUNS_DIR, Cell, load_cell_metrics

    records: list[dict[str, Any]] = []
    for path in sorted(RUNS_DIR.glob("FAILED_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("could not read %s", path)
            continue
        label = payload.get("label", path.stem)
        recorded = payload.get("cells", [])
        still_missing = []
        for entry in recorded:
            fields = {k: v for k, v in entry.items() if k in Cell.__dataclass_fields__}
            if load_cell_metrics(Cell(**fields)) is None:
                still_missing.append(entry)
        recovered = len(recorded) - len(still_missing)
        if recovered:
            logger.info("%s: %d of %d previously failed cells have since been rerun",
                        label, recovered, len(recorded))
        if still_missing:
            records.append({"label": label, "n": len(still_missing),
                            "recovered_since": recovered, "cells": still_missing})
    if records:
        logger.warning("%d stage(s) still have unrecovered cells", len(records))
    return records


def main() -> int:
    cells = discover_cells()
    if not cells:
        raise SystemExit("no stored results under results/runs -- run the sweep first")

    episodes = load_frame(cells)
    seeds = seed_level(episodes)

    # The confirmatory and exploratory tests read `variant == "main"` only (`paired_frame`'s
    # default), so pilot and sensitivity cells cannot reach them. The counts are reported split
    # by variant because a single total would overstate the evidence behind the headline tests.
    by_variant = episodes["variant"].value_counts().to_dict() if not episodes.empty else {}

    payload: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "primary_endpoint": PRIMARY_METRIC,
        "primary_scenario": PRIMARY_SCENARIO,
        "unit_of_analysis": "training seed (protocol section 8.2); episode-level n is never reported as the sample size",
        "sesoi": _sesoi(),
        "n_episode_rows": int(len(episodes)),
        "n_episode_rows_main": int((episodes["variant"] == "main").sum()) if not episodes.empty else 0,
        "n_episode_rows_by_variant": {str(k): int(v) for k, v in by_variant.items()},
        "n_seed_rows": int(len(seeds)),
        "n_seed_rows_main": int((seeds["variant"] == "main").sum()) if not seeds.empty else 0,
        "variant_scope_of_tests": (
            "every paired test reads variant == 'main'; pilot cells (variant 'pilot', training "
            "seeds 901-904) and sensitivity cells (variant 'od_*', 'weights_*') are structurally "
            "excluded from the confirmatory and exploratory families"
        ),
        "unrecovered_cells": _unrecovered_cells(),
        "methods": sorted(episodes["method"].unique().tolist()),
        "confirmatory": run_confirmatory(seeds, episodes),
        "exploratory": run_exploratory(seeds, episodes),
        "unpaired_descriptive": unpaired_comparisons(seeds),
        "descriptives": descriptives(seeds),
        "degradation": degradation(seeds),
    }

    ensure_dir(RESULTS_DIR)
    (RESULTS_DIR / "statistical_analysis.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")
    episodes.to_parquet(RESULTS_DIR / "episodes.parquet", index=False)
    seeds.to_parquet(RESULTS_DIR / "seed_level.parquet", index=False)

    _write_report(payload, seeds)
    print(f"analysed {len(episodes)} episodes -> results/statistical_analysis.json")
    for record in payload["confirmatory"]:
        if record.get("status") == "RUN":
            print(f"  {record['experiment_id']}: {record['sentence']}")
            print(f"    Holm-adjusted p = {record['p_adjusted']:.4g}, "
                  f"rejects = {record['rejects_at_family_alpha']}")
        else:
            print(f"  {record['experiment_id']}: NOT RUN ({record['reason']})")
    return 0


def _write_report(payload: dict[str, Any], seeds: pd.DataFrame) -> None:
    lines = [
        "# Experimental results (M14)",
        "",
        f"Generated by `scripts/analyse.py` from {payload['n_episode_rows']:,} stored episodes, "
        f"of which {payload['n_episode_rows_main']:,} are registered (`variant == main`) and "
        f"reduce to {payload['n_seed_rows_main']:,} training-seed observations. Every number is "
        "read from a result file; none is typed by hand.",
        "",
        f"**Unit of analysis: {payload['unit_of_analysis']}.**",
        "",
        payload["variant_scope_of_tests"].capitalize() + ".",
        "",
    ]

    if payload["unrecovered_cells"]:
        total = sum(r["n"] for r in payload["unrecovered_cells"])
        lines += [
            f"> **Incomplete run.** {total} cell(s) failed on both their first attempt and their "
            "retry and are absent from every table below: "
            + ", ".join(f"{r['n']} in `{r['label']}`" for r in payload["unrecovered_cells"])
            + ". They are listed in `results/runs/FAILED_*.json`. The sample sizes reported here "
            "are what was actually produced, not what was intended.",
            "",
        ]

    lines += [
        "## Confirmatory family (protocol section 7.3)",
        "",
        "Exactly two tests, Holm-corrected at family-wise alpha = 0.05, both on "
        f"`{payload['primary_endpoint']}` under `{payload['primary_scenario']}` at `w_balanced`.",
        "",
    ]
    for record in payload["confirmatory"]:
        if record.get("status") != "RUN":
            lines += [f"### {record['experiment_id']} — {record['hypothesis']}", "",
                      f"**NOT RUN.** {record['reason']}", ""]
            continue
        verdict = "REJECTS the null" if record["rejects_at_family_alpha"] else "does not reject"
        lines += [
            f"### {record['experiment_id']} — {record['hypothesis']}",
            "",
            f"`{record['method_a']}` vs `{record['method_b']}`, n = {record['n']} training seeds.",
            "",
            "| quantity | value |",
            "|---|---:|",
            f"| mean `{record['method_a']}` | {record['mean_a']:.6f} |",
            f"| mean `{record['method_b']}` | {record['mean_b']:.6f} |",
            f"| mean paired difference | {record['mean_difference']:.6f} |",
            f"| test | {record['test']} |",
            f"| statistic | {record['statistic']:.6g} |",
            f"| exact p | {record['p_value']:.6g} |",
            f"| Holm-adjusted p | {record['p_adjusted']:.6g} |",
            f"| Hodges-Lehmann difference | {record['hodges_lehmann']:.6f} |",
            f"| 95% BCa CI | [{record['ci_low']:.6f}, {record['ci_high']:.6f}] |",
            f"| rank-biserial correlation | {record['rank_biserial']:.4f} |",
            f"| Cliff's delta (episode level) | {record['cliffs_delta']:.4f} |",
            f"| Shapiro-Wilk p on differences | {record['normality_p']:.4g} |",
            "",
            f"**Verdict: {verdict}** at family-wise alpha = 0.05. Direction: {record['direction']}.",
            "",
        ]
        if record.get("equivalence", {}).get("applicable"):
            eq = record["equivalence"]
            lines += [
                f"TOST equivalence against the pre-registered SESOI of {eq['sesoi']}: "
                f"p = {eq['p_value']:.4g}. {eq['interpretation'].capitalize()}.",
                "",
            ]

    lines += [
        "## Descriptives by method and scenario",
        "",
        f"Primary endpoint `{payload['primary_endpoint']}`, mean over training seeds.",
        "",
    ]
    main = seeds[seeds["variant"] == "main"]
    if not main.empty:
        pivot = main.pivot_table(index="method", columns="scenario",
                                 values=payload["primary_endpoint"], aggfunc="mean")
        lines.append("| method | " + " | ".join(pivot.columns) + " |")
        lines.append("|---|" + "---:|" * len(pivot.columns))
        for method, row in pivot.iterrows():
            values = " | ".join(f"{v:.5f}" if np.isfinite(v) else "—" for v in row)
            lines.append(f"| `{method}` | {values} |")
        lines.append("")

    exploratory = payload["exploratory"]
    if exploratory:
        passing = [r for r in exploratory if r.get("passes_fdr")]
        lines += [
            "## Exploratory comparisons",
            "",
            f"{len(exploratory)} comparisons, Benjamini-Hochberg FDR at q = 0.10; "
            f"{len(passing)} pass. **All are exploratory** and none supports a confirmatory claim.",
            "",
            "| experiment | hypothesis | scenario | metric | difference | p | BH-adjusted p | direction |",
            "|---|---|---|---|---:|---:|---:|---|",
        ]
        for record in sorted(exploratory, key=lambda r: r["p_adjusted"])[:40]:
            lines.append(
                f"| {record['experiment_id']} | {record['hypothesis']} | {record['scenario']} | "
                f"`{record['metric']}` | {record['mean_difference']:.5g} | "
                f"{record['p_value']:.4g} | {record['p_adjusted']:.4g} | {record['direction']} |"
            )
        lines.append("")

    if payload["unpaired_descriptive"]:
        lines += [
            "## Contrasts against deterministic baselines (descriptive, unpaired)",
            "",
            "A fixed timetable, a heuristic and a MILP have no training seed, so there is nothing "
            "to pair a trained policy against. These are differences of means, **not hypothesis "
            "tests**, and are labelled as such wherever they appear.",
            "",
            "| comparison | scenario | metric | ppo_uq | baseline | difference |",
            "|---|---|---|---:|---:|---:|",
        ]
        for record in payload["unpaired_descriptive"]:
            baseline_key = next(k for k in record if k.startswith("mean_") and k != "mean_ppo_uq")
            lines.append(
                f"| {record['comparison']} | {record['scenario']} | `{record['metric']}` | "
                f"{record['mean_ppo_uq']:.5g} | {record[baseline_key]:.5g} | "
                f"{record['difference']:.5g} |"
            )
        lines.append("")

    (REPO_ROOT / "docs" / "results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
