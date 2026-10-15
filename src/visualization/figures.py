"""Publication figures and tables (M15), generated entirely from stored result files.

**No experimental number is ever typed into a figure.** Every value is read from
`results/*.json` or `results/*.parquet`, and each figure writes a companion `.source.json`
recording which files it came from and which rows it used. That companion file is what makes the
traceability requirement checkable rather than asserted: `scripts/verify_figure_provenance.py`
walks it.

Figures are deliberately plain. A research figure's job is to let a reviewer read the numbers and
the uncertainty, not to look designed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.common.logging import get_logger
from src.common.paths import REPO_ROOT, RESULTS_DIR, ensure_dir

logger = get_logger(__name__)

FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

#: Consistent method ordering and labels across every figure.
METHOD_ORDER = (
    "fixed_timetable", "heuristic_threshold", "milp_static", "milp_rolling",
    "ppo_point", "ppo_placebo", "ppo_uq", "ppo_uq_single_obj", "ppo_uq_noavail",
)
METHOD_LABELS = {
    "fixed_timetable": "Fixed timetable",
    "heuristic_threshold": "Heuristic",
    "milp_static": "MILP (static)",
    "milp_rolling": "MILP (rolling)",
    "ppo_point": "PPO-Point",
    "ppo_placebo": "PPO-Placebo",
    "ppo_uq": "PPO-UQ",
    "ppo_uq_single_obj": "PPO-UQ (single-obj.)",
    "ppo_uq_noavail": "PPO-UQ (no avail.)",
}
SCENARIO_ORDER = ("normal", "surge", "disruption", "combined")


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
        "axes.spines.right": False, "figure.autolayout": True,
    })
    return plt


def _record_source(name: str, sources: Sequence[str], detail: dict[str, Any]) -> None:
    """Write the provenance companion for one figure."""
    ensure_dir(FIGURES_DIR)
    (FIGURES_DIR / f"{name}.source.json").write_text(json.dumps({
        "figure": name,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": list(sources),
        "detail": detail,
        "note": "every value in this figure is read from the listed files; none is hand-entered",
    }, indent=2, default=float), encoding="utf-8")


def _save(fig, name: str) -> Path:
    ensure_dir(FIGURES_DIR)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def _present(frame, column: str = "method") -> list[str]:
    available = set(frame[column].unique())
    ordered = [m for m in METHOD_ORDER if m in available]
    return ordered + sorted(available - set(ordered))


# --------------------------------------------------------------------------- figures


def figure_primary_endpoint(seeds, name: str = "fig01_primary_endpoint") -> Path:
    """Figure 1: the pre-registered primary endpoint, by method, under `surge`."""
    plt = _style()
    data = seeds[(seeds["scenario"] == "surge") & (seeds["variant"] == "main")]
    methods = _present(data)
    values = [data[data["method"] == m]["left_behind_fraction"].to_numpy() for m in methods]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    positions = np.arange(len(methods))
    for index, series in enumerate(values):
        if series.size == 0:
            continue
        ax.scatter(np.full(series.size, index) + np.linspace(-0.12, 0.12, series.size),
                   series, s=14, alpha=0.65, zorder=3, color="#4C72B0")
        ax.hlines(series.mean(), index - 0.28, index + 0.28,
                  color="#C44E52", lw=2, zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=30, ha="right")
    ax.set_ylabel("Left-behind fraction")
    ax.set_title("Primary endpoint under demand surge\n"
                 "one point per training seed; bar = mean", fontsize=9)
    _record_source(name, ["results/seed_level.parquet"],
                   {"scenario": "surge", "metric": "left_behind_fraction",
                    "methods": methods,
                    "n_per_method": {m: int(v.size) for m, v in zip(methods, values)}})
    return _save(fig, name)


def figure_scenario_comparison(seeds, name: str = "fig02_scenario_comparison") -> Path:
    """Figure 2: primary endpoint across all four scenario families."""
    plt = _style()
    data = seeds[seeds["variant"] == "main"]
    methods = _present(data)
    scenarios = [s for s in SCENARIO_ORDER if s in set(data["scenario"])]

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    width = 0.8 / max(1, len(methods))
    for index, method in enumerate(methods):
        means, errors = [], []
        for scenario in scenarios:
            block = data[(data["method"] == method) & (data["scenario"] == scenario)]
            series = block["left_behind_fraction"].to_numpy()
            means.append(series.mean() if series.size else np.nan)
            errors.append(series.std(ddof=1) / np.sqrt(series.size)
                          if series.size > 1 else 0.0)
        ax.bar(np.arange(len(scenarios)) + index * width - 0.4 + width / 2,
               means, width, yerr=errors, capsize=2,
               label=METHOD_LABELS.get(method, method))
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Left-behind fraction")
    ax.set_title("Primary endpoint by scenario (mean ± SE over training seeds)", fontsize=9)
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    _record_source(name, ["results/seed_level.parquet"],
                   {"metric": "left_behind_fraction", "methods": methods,
                    "scenarios": scenarios})
    return _save(fig, name)


def figure_metric_by_method(seeds, metric: str, ylabel: str, title: str, name: str) -> Path:
    """Figures 3-6: waiting, p95 waiting, energy, cost."""
    plt = _style()
    data = seeds[seeds["variant"] == "main"]
    methods = _present(data)
    scenarios = [s for s in SCENARIO_ORDER if s in set(data["scenario"])]

    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    width = 0.8 / max(1, len(methods))
    for index, method in enumerate(methods):
        means = []
        errors = []
        for scenario in scenarios:
            series = data[(data["method"] == method)
                          & (data["scenario"] == scenario)][metric].to_numpy()
            means.append(series.mean() if series.size else np.nan)
            errors.append(series.std(ddof=1) / np.sqrt(series.size) if series.size > 1 else 0.0)
        ax.bar(np.arange(len(scenarios)) + index * width - 0.4 + width / 2,
               means, width, yerr=errors, capsize=2,
               label=METHOD_LABELS.get(method, method))
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels(scenarios)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, ncol=3)
    _record_source(name, ["results/seed_level.parquet"],
                   {"metric": metric, "methods": methods, "scenarios": scenarios})
    return _save(fig, name)


def figure_robustness(analysis: dict[str, Any], name: str = "fig07_robustness") -> Path:
    """Figure 7: degradation ratio -- stressed metric divided by the same metric under `normal`."""
    plt = _style()
    rows = [r for r in analysis.get("degradation", [])
            if r["metric"] == "left_behind_fraction"]
    if not rows:
        return _empty_figure(name, "No degradation data stored")

    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    scenarios = ["surge", "disruption", "combined"]

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    width = 0.8 / max(1, len(methods))
    for index, method in enumerate(methods):
        values = []
        for scenario in scenarios:
            match = [r["degradation_ratio"] for r in rows
                     if r["method"] == method and r["scenario"] == scenario]
            values.append(match[0] if match else np.nan)
        ax.bar(np.arange(len(scenarios)) + index * width - 0.4 + width / 2,
               values, width, label=METHOD_LABELS.get(method, method))
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Degradation ratio vs `normal`")
    ax.set_title("Robustness: left-behind fraction relative to an undisrupted day\n"
                 "(valid only because amendment A-03 matched scenarios on eval_seed)",
                 fontsize=9)
    ax.legend(fontsize=7, ncol=3)
    _record_source(name, ["results/statistical_analysis.json"],
                   {"metric": "left_behind_fraction", "methods": methods,
                    "scenarios": scenarios})
    return _save(fig, name)


def figure_ablations(analysis: dict[str, Any], name: str = "fig08_ablations") -> Path:
    """Figure 8: the registered ablation contrasts with their confidence intervals."""
    plt = _style()
    records = [r for r in analysis.get("confirmatory", []) if r.get("status") == "RUN"]
    records += [r for r in analysis.get("exploratory", [])
                if r["metric"] == "left_behind_fraction" and r["scenario"] == "surge"]
    if not records:
        return _empty_figure(name, "No ablation comparisons stored")

    labels = [f"{r['method_a']} vs {r['method_b']}" for r in records]
    effects = [r["hodges_lehmann"] for r in records]
    low = [r["hodges_lehmann"] - r["ci_low"] for r in records]
    high = [r["ci_high"] - r["hodges_lehmann"] for r in records]

    fig, ax = plt.subplots(figsize=(7.0, 0.42 * len(records) + 1.6))
    positions = np.arange(len(records))
    colours = ["#C44E52" if r["family"] == "confirmatory" else "#4C72B0" for r in records]
    ax.errorbar(effects, positions, xerr=[low, high], fmt="o", ms=5,
                capsize=3, lw=1.2, ecolor="#888888", ls="none")
    for position, effect, colour in zip(positions, effects, colours):
        ax.plot(effect, position, "o", ms=5, color=colour)
    ax.axvline(0.0, color="black", lw=0.8, ls="--")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Hodges–Lehmann paired difference in left-behind fraction\n"
                  "(negative favours the first method)")
    ax.set_title("Ablation contrasts with 95% BCa intervals\n"
                 "red = confirmatory (Holm), blue = exploratory (BH FDR)", fontsize=9)
    ax.invert_yaxis()
    _record_source(name, ["results/statistical_analysis.json"],
                   {"n_comparisons": len(records), "labels": labels})
    return _save(fig, name)


def figure_forecasting(name: str = "fig09_forecasting") -> Path:
    """Figure 9: forecast calibration and accuracy, from the M4 metrics manifest."""
    plt = _style()
    path = RESULTS_DIR / "forecasting_metrics.json"
    if not path.exists():
        return _empty_figure(name, "No forecasting metrics stored")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    metrics = manifest.get("metrics", {})

    models = [m for m in ("naive_seasonal", "xgb_point", "xgb_quantile") if m in metrics]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))

    mase = [metrics[m].get("test", {}).get("point", {}).get("mase", np.nan) for m in models]
    axes[0].bar(models, mase, color="#4C72B0")
    axes[0].axhline(1.0, color="black", lw=0.8, ls="--")
    axes[0].set_ylabel("MASE (test)")
    axes[0].set_title("Point accuracy\n(1.0 = seasonal naive)", fontsize=9)
    axes[0].tick_params(axis="x", rotation=20)

    gate = manifest.get("gate", {})
    picp = [metrics[m].get("test", {}).get("probabilistic", {}).get("picp_80", np.nan)
            for m in models]
    axes[1].bar(models, picp, color="#55A868")
    axes[1].axhline(0.80, color="black", lw=0.8, ls="--")
    band = gate.get("coverage_band", [0.76, 0.84])
    axes[1].axhspan(band[0], band[1], color="grey", alpha=0.18)
    axes[1].set_ylabel("Empirical 80% coverage (test)")
    axes[1].set_title("Calibration\n(shaded = pre-registered band)", fontsize=9)
    axes[1].tick_params(axis="x", rotation=20)

    _record_source(name, ["results/forecasting_metrics.json"],
                   {"models": models, "mase": mase, "picp_80": picp, "gate": gate})
    return _save(fig, name)


def figure_simulator_validation(name: str = "fig11_simulator_validation") -> Path:
    """Figure 11: simulated against real published statistics."""
    plt = _style()
    path = RESULTS_DIR / "simulator_validation.json"
    if not path.exists():
        return _empty_figure(name, "No simulator validation stored")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        r for r in payload.get("rows", [])
        if isinstance(r, dict) and r.get("relative_error") is not None
    ]
    if not rows:
        return _empty_figure(name, "No comparable validation rows")

    labels = [r.get("quantity", "?") for r in rows]
    relative = [float(r["relative_error"]) * 100.0 for r in rows]
    verdicts = [r.get("verdict", "") for r in rows]
    palette = {"validated": "#55A868", "partially validated": "#DD8452", "mismatch": "#C44E52"}
    colours = [palette.get(v, "#888888") for v in verdicts]

    fig, ax = plt.subplots(figsize=(7.4, 0.4 * len(rows) + 1.8))
    ax.barh(np.arange(len(rows)), relative, color=colours)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Absolute relative error vs published statistic (%)")
    ax.set_title(
        "Simulator validation against real published data\n"
        f"green = validated, orange = partially validated; "
        f"{payload.get('counts', {}).get('not validatable', 0)} further quantities "
        "cannot be validated at all",
        fontsize=9,
    )
    ax.invert_yaxis()
    _record_source(name, ["results/simulator_validation.json"],
                   {"rows": labels, "relative_error_percent": relative,
                    "verdicts": verdicts, "counts": payload.get("counts", {})})
    return _save(fig, name)


def figure_shap(name: str = "fig10_shap") -> Path:
    """Figure 10: SHAP global attributions, from the stored explainability result."""
    plt = _style()
    path = RESULTS_DIR / "shap_analysis.json"
    if not path.exists():
        return _empty_figure(name, "No SHAP analysis stored")
    payload = json.loads(path.read_text(encoding="utf-8"))
    globals_ = payload.get("global_importance", [])
    if not globals_:
        return _empty_figure(name, "SHAP analysis contains no global importances")

    top = sorted(globals_, key=lambda r: -abs(r["mean_abs_shap"]))[:20]
    fig, ax = plt.subplots(figsize=(7.0, 0.34 * len(top) + 1.6))
    ax.barh(np.arange(len(top)), [r["mean_abs_shap"] for r in top], color="#8172B2")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels([r["feature"] for r in top], fontsize=7)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global feature attribution for PPO-UQ\n"
                 "attribution is associational, not causal", fontsize=9)
    ax.invert_yaxis()
    _record_source(name, ["results/shap_analysis.json"],
                   {"n_features_shown": len(top)})
    return _save(fig, name)


def _empty_figure(name: str, message: str) -> Path:
    """A placeholder that says what is missing rather than inventing content."""
    plt = _style()
    fig, ax = plt.subplots(figsize=(6.0, 2.0))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10)
    ax.axis("off")
    _record_source(name, [], {"status": "no data", "message": message})
    logger.warning("figure %s: %s", name, message)
    return _save(fig, name)
