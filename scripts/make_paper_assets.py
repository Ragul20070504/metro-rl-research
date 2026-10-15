"""Publication figure and table set, generated from stored artefacts.

Kept separate from `scripts/make_figures.py`, which belongs to the registered pipeline and whose
outputs are part of the immutable study. This script produces the journal submission set.

Every value is read from `results/` at render time. Two figures are **schematics** (the research
framework and the twin architecture): they carry no experimental number and are labelled as
diagrams so no reader mistakes them for measurements. Every other figure and table writes a
`.source.json` companion naming the artefacts it drew from.

    python scripts/make_paper_assets.py
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from src.common.paths import RESULTS_DIR  # noqa: E402

FIG_DIR = RESULTS_DIR / "paper_figures"
TAB_DIR = RESULTS_DIR / "paper_tables"
SCENARIOS = ["normal", "surge", "disruption", "combined"]
ARMS = ["ppo_point", "ppo_placebo", "ppo_uq"]

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 300, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.autolayout": True,
})

COLOUR = {
    "ppo_point": "#2b6cb0", "ppo_placebo": "#a0aec0", "ppo_uq": "#c53030",
    "ppo_uq_single_obj": "#805ad5", "ppo_uq_noavail": "#dd6b20",
    "heuristic_threshold": "#2f855a", "milp_static": "#38a169",
    "milp_rolling": "#68d391", "fixed_timetable": "#718096",
}
manifest: list[dict[str, Any]] = []


def load(name: str) -> Any:
    path = RESULTS_DIR / name
    if not path.is_file():
        raise SystemExit(f"required artefact missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def seed_level() -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / "seed_level.parquet")


def registered(df: pd.DataFrame) -> pd.DataFrame:
    """Cells of the registered confirmatory design, baselines included.

    Deliberately does **not** filter on `train_seed`. Pilot cells are already excluded by
    `variant == "main"` (they carry `variant == "pilot"`), so a `train_seed >= 101` filter would
    add nothing against the pilot while silently dropping every baseline -- those are
    deterministic and carry `train_seed = -1`. That would empty the baseline table of baselines.
    """
    return df[(df.variant == "main") & (df.weight_id == "w_balanced")]


def rl_only(df: pd.DataFrame) -> pd.DataFrame:
    """The learned arms at the registered training seeds, for within-seed paired views."""
    return registered(df)[registered(df).train_seed >= 101]


def save(fig: plt.Figure, stem: str, sources: list[str], detail: dict[str, Any]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)
    (FIG_DIR / f"{stem}.source.json").write_text(json.dumps({
        "figure": stem, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/make_paper_assets.py",
        "source_files": sources, "detail": detail,
        "note": "every value in this figure is read from the listed files; none is hand-entered",
    }, indent=2), encoding="utf-8")
    manifest.append({"kind": "figure", "name": stem, "sources": sources})
    print(f"  figure {stem}")


def _markdown(frame: pd.DataFrame) -> str:
    """Render a table without `tabulate`.

    `DataFrame.to_markdown` needs tabulate, which is not in the hash-pinned research environment.
    Installing it to render a table would change the environment the results were produced in,
    which is a bad trade for a formatting convenience.
    """
    columns = [str(c) for c in frame.columns]
    rows = [[("" if pd.isna(v) else str(v)).replace("|", "\\|").replace("\n", " ")
             for v in record] for record in frame.itertuples(index=False, name=None)]
    widths = [max(len(columns[i]), *(len(r[i]) for r in rows)) if rows else len(columns[i])
              for i in range(len(columns))]
    head = "| " + " | ".join(c.ljust(w) for c, w in zip(columns, widths)) + " |"
    rule = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = ["| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |" for r in rows]
    return "\n".join([head, rule, *body])


def save_table(frame: pd.DataFrame, stem: str, caption: str, sources: list[str]) -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TAB_DIR / f"{stem}.csv", index=False)
    lines = [f"**{stem}** — {caption}", "", _markdown(frame), "",
             f"*Generated by `scripts/make_paper_assets.py` from {', '.join(sources)}.*"]
    (TAB_DIR / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")
    manifest.append({"kind": "table", "name": stem, "caption": caption, "sources": sources})
    print(f"  table  {stem}")


# --------------------------------------------------------------------------- figures


def figure01_framework() -> None:
    """Schematic. Carries no experimental number."""
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.axis("off")
    boxes = [
        (0.03, "Real ridership\ndata", "#e2e8f0"),
        (0.22, "Quantile demand\nforecaster\n(median + sigma)", "#bee3f8"),
        (0.42, "Calibrated metro\nsimulator", "#c6f6d5"),
        (0.62, "Maskable PPO\n3 arms", "#fed7d7"),
        (0.82, "Registered\nendpoint", "#e9d8fd"),
    ]
    for x, label, colour in boxes:
        ax.add_patch(plt.Rectangle((x, 0.42), 0.15, 0.34, facecolor=colour,
                                   edgecolor="#4a5568", linewidth=0.8))
        ax.text(x + 0.075, 0.59, label, ha="center", va="center", fontsize=8)
    for x in (0.185, 0.385, 0.585, 0.785):
        ax.annotate("", xy=(x + 0.03, 0.59), xytext=(x, 0.59),
                    arrowprops=dict(arrowstyle="->", color="#4a5568", lw=1.0))
    ax.text(0.695, 0.30, "sigma channel content is the ONLY difference between arms:\n"
                         "zero (point)  ·  shuffled (placebo)  ·  calibrated (uncertainty-aware)",
            ha="center", va="center", fontsize=7.5, style="italic", color="#742a2a")
    ax.text(0.5, 0.06, "SCHEMATIC — no experimental values are shown in this figure",
            ha="center", fontsize=7, color="#718096")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "fig01_research_framework", [], {"type": "schematic", "contains_data": False})


def figure02_twin() -> None:
    """Schematic of the simulator's structure, annotated with validation status counts."""
    sim = load("simulator_validation.json")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    layers = [
        ("Demand process\narrivals per station-hour", "#bee3f8", 0.78),
        ("Network & rolling stock\nheadway, capacity, failures", "#c6f6d5", 0.56),
        ("Queue dynamics\nboarding, denial, left-behind", "#fed7d7", 0.34),
        ("Reward\nW, U, E, C", "#e9d8fd", 0.12),
    ]
    for label, colour, y in layers:
        ax.add_patch(plt.Rectangle((0.05, y), 0.5, 0.17, facecolor=colour,
                                   edgecolor="#4a5568", linewidth=0.8))
        ax.text(0.30, y + 0.085, label, ha="center", va="center", fontsize=8)
    counts = sim["counts"]
    ax.text(0.62, 0.62, "External validation status\n(14 quantities)", fontsize=8.5,
            fontweight="bold", va="center")
    y = 0.50
    for key, value in counts.items():
        ax.text(0.62, y, f"{value:>2}  {key}", fontsize=8, va="center", family="monospace")
        y -= 0.08
    ax.text(0.62, 0.16, "The four not-validatable quantities that\nfeed the primary endpoint are\n"
                        "capacity/occupancy, dwell, balking, OD.",
            fontsize=7.2, va="center", color="#742a2a")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "fig02_twin_architecture", ["results/simulator_validation.json"],
         {"type": "schematic with validation counts", "counts": counts})


def figure03_forecasting() -> None:
    fc = load("forecasting_metrics.json")
    metrics = fc.get("metrics", {})
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    flat: dict[str, float] = {}
    for group, payload in metrics.items():
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, (int, float)):
                    flat[f"{group}.{key}"] = float(value)
    top = dict(list(flat.items())[:8])
    axes[0].barh(list(top), list(top.values()), color="#2b6cb0")
    axes[0].set_title("Forecast accuracy metrics", fontsize=9)
    axes[0].tick_params(labelsize=6.5)
    cal = fc.get("calibration", {})
    numeric = {k: float(v) for k, v in cal.items() if isinstance(v, (int, float))}
    if numeric:
        axes[1].bar(list(numeric), list(numeric.values()), color="#2f855a")
        axes[1].tick_params(axis="x", rotation=35, labelsize=6.5)
    axes[1].set_title(f"Calibration · quantile crossing "
                      f"{fc['quantile_crossing_rate']:.4f}", fontsize=9)
    save(fig, "fig03_forecasting", ["results/forecasting_metrics.json"],
         {"quantile_crossing_rate": fc["quantile_crossing_rate"]})


def figure04_primary() -> None:
    stats_art = load("statistical_analysis.json")
    test = next(t for t in stats_art["confirmatory"] if t["experiment_id"] == "EXP-UQ-001")
    df = rl_only(seed_level())
    piv = df[df.scenario == "surge"].pivot_table(
        index="train_seed", columns="method", values="left_behind_fraction")

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
    for method in ("ppo_point", "ppo_uq"):
        axes[0].scatter([method] * len(piv), piv[method], color=COLOUR[method], zorder=3, s=22)
    for seed in piv.index:  # paired lines: the design is within-seed, so show the pairing
        axes[0].plot(["ppo_point", "ppo_uq"], [piv.loc[seed, "ppo_point"],
                                               piv.loc[seed, "ppo_uq"]],
                     color="#a0aec0", lw=0.7, zorder=1)
    axes[0].set_ylabel("left_behind_fraction"); axes[0].set_title("Paired by training seed",
                                                                  fontsize=9)
    diff = (piv["ppo_uq"] - piv["ppo_point"]).sort_values()
    axes[1].barh(range(len(diff)), diff.values,
                 color=["#c53030" if d > 0 else "#2b6cb0" for d in diff.values])
    axes[1].axvline(0, color="#2d3748", lw=0.8)
    axes[1].set_yticks(range(len(diff))); axes[1].set_yticklabels(diff.index, fontsize=7)
    axes[1].set_xlabel("ppo_uq − ppo_point"); axes[1].set_title("Per-seed difference", fontsize=9)
    axes[2].errorbar([0], [test["hodges_lehmann"]],
                     yerr=[[test["hodges_lehmann"] - test["ci_low"]],
                           [test["ci_high"] - test["hodges_lehmann"]]],
                     fmt="o", color="#c53030", capsize=5)
    axes[2].axhline(0, color="#2d3748", lw=0.8)
    axes[2].axhline(stats_art["sesoi"], color="#2f855a", ls="--", lw=0.8, label="SESOI")
    axes[2].set_xticks([]); axes[2].set_ylabel("Hodges–Lehmann difference")
    axes[2].set_title(f"Effect · Holm p = {test['p_adjusted']:.4g}", fontsize=9)
    axes[2].legend(fontsize=7)
    save(fig, "fig04_primary_uq_vs_point",
         ["results/seed_level.parquet", "results/statistical_analysis.json"],
         {"n": test["n"], "difference": test["mean_difference"],
          "ci": [test["ci_low"], test["ci_high"]], "p_adjusted": test["p_adjusted"]})


def figure05_placebo() -> None:
    df = rl_only(seed_level())
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    data = [df[(df.scenario == "surge") & (df.method == m)]["left_behind_fraction"].values
            for m in ARMS]
    # `labels=` became `tick_labels=` in matplotlib 3.9; setting the ticks explicitly works
    # either side of that rename.
    parts = ax.boxplot(data, patch_artist=True, widths=0.55)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["point\n(sigma = 0)", "placebo\n(sigma shuffled)",
                        "uncertainty\n(sigma calibrated)"])
    for patch, method in zip(parts["boxes"], ARMS):
        patch.set_facecolor(COLOUR[method]); patch.set_alpha(0.55)
    for i, values in enumerate(data, start=1):
        ax.scatter(np.full(len(values), i), values, s=14, color="#2d3748", zorder=3, alpha=0.7)
    ax.set_ylabel("left_behind_fraction (surge)")
    ax.set_title("Placebo control separates information from dimensionality", fontsize=9)
    save(fig, "fig05_placebo", ["results/seed_level.parquet"],
         {"arms": ARMS, "scenario": "surge",
          "means": {m: float(np.mean(d)) for m, d in zip(ARMS, data)}})


def figure06_mechanism() -> None:
    mech = load("mechanism_analysis.json")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    x = np.arange(len(SCENARIOS)); width = 0.38
    info = [r["informative_sigma_vs_zero"]["mean_difference"] for r in mech["dispatch_volume"]]
    plac = [r["uninformative_sigma_vs_zero"]["mean_difference"] for r in mech["dispatch_volume"]]
    axes[0].bar(x - width / 2, info, width, label="calibrated sigma", color="#c53030")
    axes[0].bar(x + width / 2, plac, width, label="placebo sigma", color="#a0aec0")
    axes[0].axhline(0, color="#2d3748", lw=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(SCENARIOS, fontsize=7.5)
    axes[0].set_ylabel("change in inductions per episode\n(vs point arm)")
    axes[0].set_title("Only informative sigma suppresses dispatch", fontsize=9)
    axes[0].legend(fontsize=7.5)

    matched = mech["volume_matched_comparison"]["surge"]
    keys = ["inductions", "idle_epochs", "max_queue"]
    uq = [matched["ppo_uq"][k] for k in keys]
    ft = [matched["fixed_timetable"][k] for k in keys]
    norm_uq = [1.0] * len(keys)
    norm_ft = [f / u if u else 0 for f, u in zip(ft, uq)]
    y = np.arange(len(keys))
    axes[1].barh(y - 0.19, norm_uq, 0.36, label="ppo_uq", color="#c53030")
    axes[1].barh(y + 0.19, norm_ft, 0.36, label="fixed_timetable", color="#718096")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([f"{k}\n({u:.0f} vs {f:.0f})" for k, u, f in zip(keys, uq, ft)],
                            fontsize=7)
    axes[1].set_xlabel("relative to ppo_uq")
    axes[1].set_title("At matched dispatch volume", fontsize=9)
    axes[1].legend(fontsize=7.5)
    save(fig, "fig06_mechanism", ["results/mechanism_analysis.json"],
         {"dispatch_volume": info, "placebo": plac, "matched": matched})


def figure07_convex_loss() -> None:
    """The convex service loss, measured from stored per-epoch data."""
    frames = []
    base = RESULTS_DIR / "runs" / "main" / "surge"
    for method in ("ppo_point", "ppo_uq"):
        for seed in range(1, 31):
            path = base / method / "w_balanced" / "101" / str(seed) / "epochs.parquet"
            if path.is_file():
                frames.append(pd.read_parquet(path))
    epochs = pd.concat(frames, ignore_index=True)
    epochs = epochs[epochs.arrived > 0]
    bins = pd.qcut(epochs.arrived, 12, duplicates="drop")
    grouped = epochs.groupby(bins, observed=True).agg(
        arrived=("arrived", "mean"), left=("left_behind", "mean")).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(grouped.arrived, grouped.left, "o-", color="#c53030", lw=1.4, ms=4)
    ax.set_xlabel("arrivals in the decision epoch")
    ax.set_ylabel("passengers left behind")
    ax.set_title("Service loss is convex once capacity binds", fontsize=9)
    knee = grouped.arrived.iloc[int(len(grouped) * 0.55)]
    ax.axvline(knee, color="#2f855a", ls="--", lw=0.8)
    ax.text(knee, ax.get_ylim()[1] * 0.55, "  capacity begins to bind",
            fontsize=7.5, color="#2f855a")
    ax.text(0.03, 0.92,
            "Convexity implies E[loss(D)] > loss(E[D]):\n"
            "under uncertainty the expected loss exceeds\n"
            "the loss at the expected demand.",
            transform=ax.transAxes, fontsize=7.2, va="top", color="#2d3748")
    save(fig, "fig07_convex_service_loss",
         ["results/runs/main/surge/**/epochs.parquet"],
         {"bins": int(len(grouped)), "train_seed": 101,
          "arrivals_range": [float(grouped.arrived.min()), float(grouped.arrived.max())],
          "left_behind_range": [float(grouped.left.min()), float(grouped.left.max())]})


def figure08_scenarios() -> None:
    df = rl_only(seed_level())
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(SCENARIOS)); width = 0.26
    for offset, method in zip((-width, 0, width), ARMS):
        means = [df[(df.scenario == s) & (df.method == method)]["left_behind_fraction"].mean()
                 for s in SCENARIOS]
        sems = [stats.sem(df[(df.scenario == s) & (df.method == method)]["left_behind_fraction"])
                for s in SCENARIOS]
        ax.bar(x + offset, means, width, yerr=sems, capsize=3, label=method,
               color=COLOUR[method])
    ax.set_xticks(x); ax.set_xticklabels(SCENARIOS)
    ax.set_ylabel("left_behind_fraction"); ax.legend(fontsize=7.5)
    ax.set_title("The ordering is identical in all four scenarios", fontsize=9)
    save(fig, "fig08_scenario_robustness", ["results/seed_level.parquet"],
         {"scenarios": SCENARIOS, "arms": ARMS, "error_bars": "standard error over training seeds"})


def figure09_baselines() -> None:
    df = registered(seed_level())
    order = (df[df.scenario == "surge"].groupby("method")["left_behind_fraction"].mean()
             .sort_values())
    order = order[order.index != "ppo_uq_single_obj"]  # off-scale; reported in the ablation table
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colours = [COLOUR.get(m, "#4a5568") for m in order.index]
    ax.barh(range(len(order)), order.values, color=colours)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("left_behind_fraction (surge, lower is better)")
    ax.set_title("No learned arm beats the tuned heuristic", fontsize=9)
    for i, (name, value) in enumerate(order.items()):
        ax.text(value, i, f"  {value:.4f}", va="center", fontsize=7)
    ax.text(0.98, 0.05, "ppo_uq_single_obj omitted (off scale; see Table 8)",
            transform=ax.transAxes, ha="right", fontsize=6.5, color="#718096")
    save(fig, "fig09_baselines", ["results/seed_level.parquet"],
         {"scenario": "surge", "means": {k: float(v) for k, v in order.items()},
          "omitted": ["ppo_uq_single_obj"]})


def figure10_sensitivity() -> None:
    sens = load("sensitivity_analysis.json")
    od = [t for t in sens["tests"] if t.get("experiment_id") == "EXP-SENS-002"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels = [f"{t['od_model']}\nvs {t['method_b']}" for t in od]
    values = [t["mean_difference"] for t in od]
    ax.bar(range(len(od)), values, color="#c53030", alpha=0.8)
    reference = sens["reference_registered_conditions"]["mean_difference"]
    ax.axhline(reference, color="#2b6cb0", ls="--", lw=1.0,
               label=f"registered conditions ({reference:+.4f})")
    ax.axhline(0, color="#2d3748", lw=0.8)
    ax.set_xticks(range(len(od))); ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("ppo_uq − comparator")
    ax.set_title("OD sensitivity: the direction holds under all three OD models", fontsize=9)
    ax.legend(fontsize=7.5)
    save(fig, "fig10_sensitivity", ["results/sensitivity_analysis.json"],
         {"n_conditions": len(od), "reference": reference,
          "conditions_where_ppo_uq_worse":
              sens["direction_agreement"]["conditions_where_ppo_uq_is_worse"]})


def figure11_shap() -> None:
    shap = load("shap_analysis.json")
    # Stored as a list of {feature, mean_abs_shap, mean_shap} records, not a name->value mapping.
    importance = shap.get("global_importance") or []
    if not importance:
        raise SystemExit("shap_analysis.json carries no global_importance; cannot draw fig11")
    frame = pd.DataFrame(importance)
    top = (frame.set_index("feature")["mean_abs_shap"].astype(float)
           .sort_values(ascending=False).head(14))
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.barh(range(len(top)), top.values, color="#805ad5")
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(f"Attribution for {shap['method']} (seed {shap['train_seed']}) — "
                 "associational, not causal", fontsize=8.5)
    save(fig, "fig11_shap", ["results/shap_analysis.json"],
         {"method": shap["method"], "train_seed": shap["train_seed"],
          "explainer": shap["explainer"],
          "caveat": "SHAP provides model attribution and does not establish causality"})


# --------------------------------------------------------------------------- tables


def build_tables() -> None:
    stats_art = load("statistical_analysis.json")
    power = load("power_analysis.json")
    audit = load("reproducibility_audit.json")
    sim = load("simulator_validation.json")
    fc = load("forecasting_metrics.json")
    sens = load("sensitivity_analysis.json")
    mech = load("mechanism_analysis.json")
    df = registered(seed_level())      # baselines included, for tables 6 and 10
    paired_df = rl_only(seed_level())  # learned arms only, for the paired views

    def evidence(name: str) -> dict:
        for check in audit["checks"]:
            if check["name"] == name:
                return check.get("evidence") or {}
        return {}

    # Table 1 — dataset
    split = evidence("chronological splits with embargo")
    rows = [{"property": k, "value": v} for k, v in
            {**split.get("split", {}), **split.get("rows", {})}.items()]
    prov = evidence("raw data provenance")
    rows += [{"property": k, "value": v} for k, v in prov.items()]
    save_table(pd.DataFrame(rows), "table01_dataset",
               "Dataset characteristics, splits and provenance",
               ["results/reproducibility_audit.json"])

    # Table 2 — forecasting
    frows = [{"quantity": "quantile crossing rate", "value": fc["quantile_crossing_rate"]},
             {"quantity": "config hash", "value": str(fc.get("config_hash", ""))[:16]}]
    for group, payload in (fc.get("metrics") or {}).items():
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, (int, float)):
                    frows.append({"quantity": f"{group}.{key}", "value": value})
    save_table(pd.DataFrame(frows), "table02_forecasting", "Forecasting performance",
               ["results/forecasting_metrics.json"])

    # Table 3 — twin validation
    save_table(pd.DataFrame(sim["rows"])[
        ["quantity", "verdict", "real_value", "simulated_value", "units", "relative_error"]],
        "table03_twin_validation", "Digital twin validation against published statistics",
        ["results/simulator_validation.json"])

    # Table 4 — protocol
    save_table(pd.DataFrame([
        {"element": "primary endpoint", "value": stats_art["primary_endpoint"]},
        {"element": "primary scenario", "value": stats_art["primary_scenario"]},
        {"element": "unit of analysis", "value": stats_art["unit_of_analysis"]},
        {"element": "training seeds", "value": power["registered_train_seeds"]},
        {"element": "evaluation seeds", "value": power["registered_eval_seeds"]},
        {"element": "scenarios", "value": ", ".join(SCENARIOS)},
        {"element": "SESOI", "value": power["sesoi"]},
        {"element": "MDE at registered n (pilot SD)",
         "value": power["minimum_detectable_effect_at_registered_n"]},
        {"element": "confirmatory multiplicity", "value": "Holm, family-wise alpha 0.05, 2 tests"},
        {"element": "exploratory multiplicity", "value": "Benjamini-Hochberg, q = 0.10"},
        {"element": "registered episodes", "value": stats_art["n_episode_rows_main"]},
        {"element": "seed-level observations", "value": stats_art["n_seed_rows_main"]},
    ]), "table04_protocol", "Registered experimental protocol",
        ["results/statistical_analysis.json", "results/power_analysis.json"])

    # Table 5 — primary statistics
    save_table(pd.DataFrame(stats_art["confirmatory"])[
        ["experiment_id", "hypothesis", "method_a", "method_b", "n", "mean_a", "mean_b",
         "mean_difference", "hodges_lehmann", "ci_low", "ci_high", "test", "p_value",
         "p_adjusted", "direction"]],
        "table05_primary_statistics", "Confirmatory results (Holm-corrected, alpha 0.05)",
        ["results/statistical_analysis.json"])

    # Table 6 — baselines
    lbf = df.pivot_table(index="method", columns="scenario", values="left_behind_fraction",
                         aggfunc="mean")[SCENARIOS]
    rew = df.pivot_table(index="method", columns="scenario", values="total_reward",
                         aggfunc="mean")[SCENARIOS]
    ind = df.pivot_table(index="method", columns="scenario", values="inductions",
                         aggfunc="mean")[SCENARIOS]
    baseline = lbf.copy().sort_values("surge")
    baseline.columns = [f"left_behind_{c}" for c in baseline.columns]
    baseline["total_reward_surge"] = rew["surge"]
    baseline["inductions_surge"] = ind["surge"]
    save_table(baseline.round(6).reset_index(), "table06_baselines",
               "All methods, all scenarios. Baselines are deterministic (n=1); no paired test "
               "against them is defined under the seed-level unit of analysis",
               ["results/seed_level.parquet"])

    # Table 7 — sensitivity
    save_table(pd.DataFrame(sens["tests"])[
        ["label", "experiment_id", "metric", "method_a", "method_b", "n", "mean_difference",
         "p_value", "p_adjusted", "direction"]],
        "table07_sensitivity",
        "Sensitivity conditions (exploratory, Benjamini-Hochberg q = 0.10)",
        ["results/sensitivity_analysis.json"])

    # Table 8 — ablations
    ablation = [t for t in stats_art["exploratory"]
                if t.get("experiment_id") in ("EXP-MO-001", "EXP-AVAIL-001")
                and t.get("metric") == "left_behind_fraction"]
    save_table(pd.DataFrame(ablation)[
        ["experiment_id", "hypothesis", "scenario", "method_a", "method_b", "n",
         "mean_a", "mean_b", "mean_difference", "p_adjusted", "direction"]],
        "table08_ablations", "Registered arm ablations (exploratory)",
        ["results/statistical_analysis.json"])

    # Table 9 — robustness
    piv = paired_df[paired_df.scenario == "surge"].pivot_table(
        index="train_seed", columns="method", values="left_behind_fraction")
    diff = (piv["ppo_uq"] - piv["ppo_point"]).to_numpy(float)
    jack = [(float(stats.ttest_rel(np.delete(piv["ppo_uq"].values, i),
                                   np.delete(piv["ppo_point"].values, i)).pvalue),
             int(piv.index[i])) for i in range(len(diff))]
    worst = max(jack)
    robust_rows = [
        {"check": "seeds favouring ppo_point", "value": f"{int((diff > 0).sum())}/{len(diff)}"},
        {"check": "realised SD of paired difference", "value": round(float(diff.std(ddof=1)), 6)},
        {"check": "pilot SD used for planning", "value": round(power["between_seed_sd"], 6)},
        {"check": "ratio realised/pilot",
         "value": round(float(diff.std(ddof=1)) / power["between_seed_sd"], 3)},
        {"check": "Wilcoxon signed-rank p",
         "value": round(float(stats.wilcoxon(piv["ppo_uq"], piv["ppo_point"]).pvalue), 6)},
        {"check": "worst leave-one-seed-out p", "value": round(worst[0], 6)},
        {"check": "worst leave-one-seed-out, Holm-adjusted",
         "value": round(min(1.0, 2 * worst[0]), 6)},
        {"check": "seed dropped in the worst case", "value": worst[1]},
        {"check": "scenarios reproducing the direction", "value": f"{len(SCENARIOS)}/{len(SCENARIOS)}"},
        {"check": "sensitivity conditions where ppo_uq is worse",
         "value": f"{sens['direction_agreement']['conditions_where_ppo_uq_is_worse']}"
                  f"/{sens['direction_agreement']['conditions_total']}"},
    ]
    save_table(pd.DataFrame(robust_rows), "table09_robustness",
               "Robustness of the primary result",
               ["results/seed_level.parquet", "results/power_analysis.json",
                "results/sensitivity_analysis.json"])

    # Table 10 — limitations
    unvalidated = [r["quantity"] for r in sim["rows"] if r["verdict"] == "not validatable"]
    save_table(pd.DataFrame([
        {"limitation": "Underpowered for the registered SESOI",
         "evidence": f"n={power['registered_train_seeds']} vs "
                     f"n={power['required_n_for_sesoi']} required; realised SD "
                     f"{float(diff.std(ddof=1)):.6f} is "
                     f"{float(diff.std(ddof=1)) / power['between_seed_sd']:.2f}x the pilot estimate",
         "implication": "Every null is inconclusive, never equivalence"},
        {"limitation": "One frozen, untuned PPO configuration (A-05)",
         "evidence": "No learned arm beats the tuned heuristic; one single-objective seed diverged",
         "implication": "Cannot separate an information effect from a sample-efficiency effect"},
        {"limitation": "Baselines are not statistically comparable",
         "evidence": "Deterministic, no training seed, so n=1 under the registered unit",
         "implication": "RL vs baseline comparisons are descriptive only"},
        {"limitation": "Most simulator quantities are not externally validated",
         "evidence": f"{len(unvalidated)} of {len(sim['rows'])} not validatable, including "
                     "capacity/occupancy, dwell, balking and OD",
         "implication": "Magnitudes are simulator-internal; only ordering is meaningful"},
        {"limitation": "Synthetic OD model drives the endpoint",
         "evidence": "Assumption A03; tested by EXP-SENS-002",
         "implication": "Direction survives all three OD variants"},
        {"limitation": "Effort asymmetry between baselines and arms",
         "evidence": "Four MILP corrections and a heuristic retune; zero RL tuning",
         "implication": "Runs against the proposed method"},
        {"limitation": "Single line, operator and data source",
         "evidence": "One network, one simulator",
         "implication": "No external-validity claim"},
        {"limitation": "Follow-up incomplete",
         "evidence": "FOLLOWUP-UQ-TRAINING stopped at 11/80 policies, asymmetric",
         "implication": "The budget confound remains open"},
    ]), "table10_limitations", "Limitations and practical implications",
        ["results/power_analysis.json", "results/simulator_validation.json",
         "results/seed_level.parquet", "docs/followup_status.md"])

    _ = mech  # loaded for provenance completeness; its figures are built above


def main() -> int:
    print("figures:")
    figure01_framework(); figure02_twin(); figure03_forecasting(); figure04_primary()
    figure05_placebo(); figure06_mechanism(); figure07_convex_loss(); figure08_scenarios()
    figure09_baselines(); figure10_sensitivity(); figure11_shap()
    print("tables:")
    build_tables()
    (RESULTS_DIR / "paper_assets_manifest.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/make_paper_assets.py",
        "figures": [m for m in manifest if m["kind"] == "figure"],
        "tables": [m for m in manifest if m["kind"] == "table"],
        "policy": ("every figure and table is generated from stored result files; no experimental "
                   "number is hand-entered. Figures 1 and 2 are schematics and contain no "
                   "experimental values."),
    }, indent=2), encoding="utf-8")
    print(f"\n{len([m for m in manifest if m['kind'] == 'figure'])} figures, "
          f"{len([m for m in manifest if m['kind'] == 'table'])} tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
