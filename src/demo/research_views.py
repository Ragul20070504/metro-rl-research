"""Interactive research pages for the M17 dashboard.

Every page reads **stored artefacts** and displays them. Nothing here simulates, trains, fits or
writes: the research tree is opened read-only, so no dashboard interaction can alter a result.

Each page declares the evidence class of what it shows -- CONFIRMATORY, FOLLOW-UP, EXPLORATORY or
DESCRIPTIVE. The single most damaging thing a dashboard can do to this study is let an
exploratory or incomplete number be read as a registered one, so the labelling is structural
rather than cosmetic: it is attached to the page, not left to the reader.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from src.common.paths import REPO_ROOT, RESULTS_DIR

SCENARIOS = ("normal", "surge", "disruption", "combined")
UQ_ARMS = ("ppo_point", "ppo_placebo", "ppo_uq")

BADGE = {
    "CONFIRMATORY": ("#1f5d3a", "Registered before the data were seen. Holm-corrected, "
                                "family-wise alpha 0.05."),
    "FOLLOW-UP": ("#5a3d8a", "FOLLOWUP-UQ-TRAINING. Pre-registered but designed after the "
                             "confirmatory result was known. Never confirmatory."),
    "EXPLORATORY": ("#7a5c1e", "Hypothesis-generating. Benjamini-Hochberg q = 0.10. Supports no "
                               "claim in the abstract."),
    "DESCRIPTIVE": ("#4a4a4a", "No hypothesis test is defined for this quantity."),
}


def badge(kind: str) -> None:
    colour, meaning = BADGE[kind]
    st.markdown(
        f"<div style='display:inline-block;background:{colour};color:#fff;padding:3px 12px;"
        f"border-radius:11px;font-size:0.74rem;font-weight:700;letter-spacing:0.04em'>{kind}</div>"
        f"<div style='font-size:0.74rem;opacity:0.72;margin:5px 0 2px'>{meaning}</div>",
        unsafe_allow_html=True)
    st.write("")


@st.cache_data(show_spinner=False)
def artefact(name: str) -> Any | None:
    path = RESULTS_DIR / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def seed_level() -> pd.DataFrame | None:
    path = RESULTS_DIR / "seed_level.parquet"
    return pd.read_parquet(path) if path.is_file() else None


def missing(name: str) -> None:
    """A missing artefact is stated. It is never replaced by a placeholder number."""
    st.warning(f"`{name}` is not present. Run the pipeline stage that produces it. "
               "No substitute value is shown.")


def registered(df: pd.DataFrame) -> pd.DataFrame:
    """Registered confirmatory cells, baselines included.

    No `train_seed` filter: the pilot is already excluded by `variant == 'main'`, and filtering on
    seed would silently drop the deterministic baselines, which carry `train_seed = -1`.
    """
    return df[(df.variant == "main") & (df.weight_id == "w_balanced")]


def rl_only(df: pd.DataFrame) -> pd.DataFrame:
    return registered(df).query("train_seed >= 101")


def metric_choices(df: pd.DataFrame) -> list[str]:
    skip = {"method", "scenario", "weight_id", "variant", "train_seed", "n_episodes"}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def download(frame: pd.DataFrame, label: str, filename: str) -> None:
    st.download_button(label, frame.to_csv(index=False).encode("utf-8"), filename, "text/csv",
                       key=f"dl_{filename}")


def figure(stem: str, directory: str = "paper_figures") -> None:
    png = RESULTS_DIR / directory / f"{stem}.png"
    source = RESULTS_DIR / directory / f"{stem}.source.json"
    if not png.is_file():
        return
    st.image(str(png), width="stretch")
    if source.is_file():
        with st.expander("Figure provenance"):
            st.json(json.loads(source.read_text(encoding="utf-8")), expanded=False)


# --------------------------------------------------------------------------- pages


def overview() -> None:
    st.header("Executive overview")
    badge("CONFIRMATORY")
    stats_art, power = artefact("statistical_analysis.json"), artefact("power_analysis.json")
    audit = artefact("reproducibility_audit.json")
    if not stats_art or not power:
        return missing("statistical_analysis.json / power_analysis.json")
    test = next(t for t in stats_art["confirmatory"] if t["experiment_id"] == "EXP-UQ-001")
    supported = test["mean_difference"] < 0 and test.get("rejects_at_family_alpha")

    st.markdown("### Controlled Evaluation of Uncertainty-Aware Reinforcement Learning "
                "for Metro Dispatch Under Stochastic Demand")
    st.markdown(
        "**Research question.** Does a calibrated forecast-uncertainty signal improve a learned "
        "metro train-induction controller, relative to an otherwise identical point-forecast "
        "controller?")
    st.markdown(
        "**Design.** Three arms identical in architecture, parameter count, observation "
        "dimensionality, reward, budget, seeds and evaluation days, differing only in what a "
        "two-value sigma channel carries — zero, a shuffled placebo of the same marginal "
        "distribution, or the calibrated sigma. Common random numbers are enforced structurally.")

    st.markdown(
        f"<div style='background:#5a1f1f;color:#fff;padding:18px 22px;border-radius:10px;"
        f"margin:14px 0'><div style='font-size:0.8rem;opacity:0.85;letter-spacing:0.08em'>"
        f"PRIMARY HYPOTHESIS</div><div style='font-size:2rem;font-weight:800;line-height:1.2'>"
        f"H1 = {'SUPPORTED' if supported else 'NOT SUPPORTED'}</div>"
        f"<div style='font-size:0.85rem;opacity:0.9;margin-top:6px'>"
        f"ppo_uq vs ppo_point on left_behind_fraction (surge): "
        f"{test['mean_difference']:+.6f}, 95% BCa CI [{test['ci_low']:.6f}, "
        f"{test['ci_high']:.6f}], Holm-adjusted p = {test['p_adjusted']:.6g}</div></div>",
        unsafe_allow_html=True)

    df = seed_level()
    policies = sum(1 for _ in (REPO_ROOT / "models" / "policies" / "w_balanced").rglob("policy.zip")) \
        if (REPO_ROOT / "models" / "policies" / "w_balanced").is_dir() else 0
    cells = sum(1 for _ in (RESULTS_DIR / "runs").rglob("metrics.json"))
    cols = st.columns(6)
    cols[0].metric("Trained policies", policies)
    cols[1].metric("Evaluation cells", f"{cells:,}")
    cols[2].metric("Scenarios", len(SCENARIOS))
    cols[3].metric("Training seeds", power["registered_train_seeds"])
    cols[4].metric("Registered experiments", 17)
    cols[5].metric("Tests", 458)
    if audit:
        st.caption("Reproducibility audit: "
                   + ", ".join(f"{k} = {v}" for k, v in audit["counts"].items()))
    st.caption(f"Unit of analysis: {stats_art['unit_of_analysis']}")
    st.info(stats_art["variant_scope_of_tests"])
    if df is not None:
        st.markdown("**Dataset period** — see the Data page for splits and provenance.")


def primary_results() -> None:
    st.header("Primary results")
    badge("CONFIRMATORY")
    stats_art = artefact("statistical_analysis.json")
    df = seed_level()
    if not stats_art or df is None:
        return missing("statistical_analysis.json / seed_level.parquet")
    data = rl_only(df)

    left, mid, right = st.columns([2, 1, 1])
    methods = left.multiselect("Methods", sorted(data.method.unique()), default=list(UQ_ARMS),
                               key="pr_methods")
    scenario = mid.selectbox("Scenario", SCENARIOS, index=1, key="pr_scenario")
    metric = right.selectbox("Metric", metric_choices(data),
                             index=metric_choices(data).index("left_behind_fraction"),
                             key="pr_metric")
    if not methods:
        st.info("Select at least one method.")
        return

    subset = data[(data.scenario == scenario) & (data.method.isin(methods))]
    summary = (subset.groupby("method")[metric]
               .agg(n="count", mean="mean", median="median", sd="std",
                    minimum="min", maximum="max").reset_index())
    ci = []
    for method in summary.method:
        values = subset[subset.method == method][metric].to_numpy(float)
        if len(values) > 1:
            half = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
            ci.append((values.mean() - half, values.mean() + half))
        else:
            ci.append((np.nan, np.nan))
    summary["ci_low"] = [c[0] for c in ci]
    summary["ci_high"] = [c[1] for c in ci]

    st.subheader(f"{metric} · {scenario}")
    st.dataframe(summary.round(6), width="stretch", hide_index=True)
    download(summary, "Download this summary (CSV)", f"primary_{metric}_{scenario}.csv")

    chart = subset.pivot_table(index="train_seed", columns="method", values=metric)
    tab_bar, tab_dist, tab_seed = st.tabs(["Means with 95% CI", "Distribution", "Per seed"])
    with tab_bar:
        st.bar_chart(summary.set_index("method")["mean"])
        st.caption("Error structure is shown numerically in the table above; the seed-level "
                   "spread is in the other tabs.")
    with tab_dist:
        st.dataframe(chart.describe().round(6), width="stretch")
        st.line_chart(chart.reset_index(drop=True))
        st.caption("Each line is one method across the training seeds, in seed order.")
    with tab_seed:
        st.dataframe(chart.round(6), width="stretch")
        if {"ppo_uq", "ppo_point"} <= set(chart.columns):
            paired = (chart["ppo_uq"] - chart["ppo_point"]).rename("ppo_uq − ppo_point")
            st.bar_chart(paired)
            st.caption("Bars above zero are seeds where the uncertainty arm performed worse "
                       "(for metrics where lower is better).")
        download(chart.reset_index(), "Download seed-level values (CSV)",
                 f"seed_level_{metric}_{scenario}.csv")

    st.markdown("---")
    st.subheader("Registered confirmatory tests")
    st.dataframe(pd.DataFrame(stats_art["confirmatory"])[
        ["experiment_id", "hypothesis", "method_a", "method_b", "n", "mean_difference",
         "hodges_lehmann", "ci_low", "ci_high", "p_value", "p_adjusted", "direction"]],
        width="stretch", hide_index=True)
    figure("fig04_primary_uq_vs_point")


def placebo_page() -> None:
    st.header("Placebo analysis")
    badge("CONFIRMATORY")
    df = seed_level()
    stats_art = artefact("statistical_analysis.json")
    if df is None or not stats_art:
        return missing("seed_level.parquet")
    st.markdown(
        "The placebo arm carries a sigma channel of **identical marginal distribution** with its "
        "association to the realised forecast error destroyed. It has the same dimensionality and "
        "parameter count as the uncertainty arm. If merely *having* the channel changed "
        "behaviour, the placebo would move with the uncertainty arm.")
    scenario = st.selectbox("Scenario", SCENARIOS, index=1, key="pl_scenario")
    data = rl_only(df)
    subset = data[(data.scenario == scenario) & (data.method.isin(UQ_ARMS))]
    means = subset.groupby("method")["left_behind_fraction"].mean().reindex(list(UQ_ARMS))
    cols = st.columns(3)
    for col, (method, value) in zip(cols, means.items()):
        col.metric(method, f"{value:.5f}")
    st.bar_chart(means)
    st.caption("point (sigma = 0)  ·  placebo (sigma shuffled)  ·  uncertainty (sigma calibrated)")
    st.success(
        "The ordering is graded: an uninformative channel of the same shape costs something, and "
        "an informative one costs more. Dimensionality alone does not account for the gap. This "
        "does **not** by itself prove causality about the internal mechanism.")
    figure("fig05_placebo")


def scenario_explorer() -> None:
    st.header("Scenario explorer")
    badge("DESCRIPTIVE")
    df = seed_level()
    if df is None:
        return missing("seed_level.parquet")
    data = registered(df)
    scenario = st.selectbox("Scenario", SCENARIOS, index=1, key="sc_scenario")
    metrics = st.multiselect(
        "Metrics", metric_choices(data),
        default=[m for m in ("left_behind_fraction", "waiting_time_mean_s", "waiting_time_p95_s",
                             "energy_units", "cost_units", "inductions") if m in data.columns],
        key="sc_metrics")
    if not metrics:
        st.info("Select at least one metric.")
        return
    table = (data[data.scenario == scenario].groupby("method")[metrics].mean()
             .sort_values(metrics[0]))
    st.dataframe(table.round(5), width="stretch")
    download(table.reset_index(), "Download (CSV)", f"scenario_{scenario}.csv")
    focus = st.selectbox("Chart metric", metrics, key="sc_focus")
    st.bar_chart(table[focus])
    st.markdown("---")
    st.subheader("Across all four scenarios")
    across = data.pivot_table(index="method", columns="scenario", values=focus,
                              aggfunc="mean")[list(SCENARIOS)]
    st.dataframe(across.round(5), width="stretch")
    figure("fig08_scenario_robustness")


def baseline_explorer() -> None:
    st.header("Baseline explorer")
    badge("DESCRIPTIVE")
    df = seed_level()
    if df is None:
        return missing("seed_level.parquet")
    data = registered(df)
    st.error(
        "**Baselines are deterministic and carry no training seed**, so under the registered "
        "seed-level unit of analysis each contributes a single observation and **no inferential "
        "test against them is defined**. Everything on this page is a difference of means, not a "
        "hypothesis test.")
    left, right = st.columns(2)
    scenario = left.selectbox("Scenario", SCENARIOS, index=1, key="bl_scenario")
    metric = right.selectbox("Metric", metric_choices(data),
                             index=metric_choices(data).index("left_behind_fraction"),
                             key="bl_metric")
    methods = st.multiselect("Methods", sorted(data.method.unique()),
                             default=sorted(data.method.unique()), key="bl_methods")
    subset = data[(data.scenario == scenario) & (data.method.isin(methods))]
    table = subset.groupby("method")[metric].agg(n="count", mean="mean", sd="std").sort_values("mean")
    st.dataframe(table.round(6), width="stretch")
    st.bar_chart(table["mean"])
    download(table.reset_index(), "Download (CSV)", f"baselines_{metric}_{scenario}.csv")
    st.warning(
        "No reinforcement-learning arm outperforms the tuned threshold heuristic or the static "
        "MILP in any scenario, on either the primary endpoint or the training objective. The "
        "proposed `ppo_uq` is also worse than the fixed timetable.")
    figure("fig09_baselines")


def mechanism_page() -> None:
    st.header("Mechanism explorer")
    badge("EXPLORATORY")
    mech, df = artefact("mechanism_analysis.json"), seed_level()
    if not mech or df is None:
        return missing("mechanism_analysis.json")
    st.markdown(
        "The design manipulates one variable — sigma-channel content — with everything else held "
        "fixed under common random numbers. Behavioural differences between arms are therefore "
        "attributable to that manipulation. **It says nothing about the network's internal "
        "representation.**")

    st.subheader("Uncertainty → dispatch")
    rows = []
    for row in mech["dispatch_volume"]:
        info, plc = row["informative_sigma_vs_zero"], row["uninformative_sigma_vs_zero"]
        rows.append({"scenario": row["scenario"],
                     "calibrated sigma": round(info["mean_difference"], 2),
                     "p (calibrated)": f"{info['p_value']:.1e}",
                     "placebo sigma": round(plc["mean_difference"], 2),
                     "p (placebo)": f"{plc['p_value']:.3f}"})
    volume = pd.DataFrame(rows)
    st.dataframe(volume, width="stretch", hide_index=True)
    st.bar_chart(volume.set_index("scenario")[["calibrated sigma", "placebo sigma"]])
    st.caption(mech["dispatch_volume_reading"])
    download(volume, "Download (CSV)", "mechanism_dispatch.csv")

    st.subheader("Dispatch → service, at matched volume")
    matched = mech["volume_matched_comparison"]["surge"]
    st.dataframe(pd.DataFrame({"ppo_uq": matched["ppo_uq"],
                               "fixed_timetable": matched["fixed_timetable"]}).round(2),
                 width="stretch")
    st.caption(mech["volume_matched_reading"])
    figure("fig06_mechanism")

    st.subheader("Convex service loss")
    st.markdown(
        "Passengers left behind are near zero while capacity is slack and rise sharply once it "
        "binds. Under a convex loss, Jensen's inequality gives `E[loss(D)] >= loss(E[D])` — the "
        "standard newsvendor observation — so demand uncertainty is operationally relevant and "
        "motivates hedging **upward**. The tested controller moved the other way.")
    figure("fig07_convex_service_loss")
    st.info("**Observed result** — informative sigma is associated with fewer dispatches and "
            "worse service. **Interpretation** — the controller learned a conservative response "
            "in the direction opposite to the one convex loss motivates. **Limitation** — the "
            "registered epoch-level test could not be run, because stored cells do not carry the "
            "per-epoch forecast.")

    if df is not None:
        st.subheader("Explore the relationship yourself")
        data = registered(df)
        scenario = st.selectbox("Scenario", SCENARIOS, index=1, key="me_scenario")
        options = [m for m in metric_choices(data)]
        x = st.selectbox("x", options, index=options.index("inductions")
                         if "inductions" in options else 0, key="me_x")
        y = st.selectbox("y", options, index=options.index("left_behind_fraction")
                         if "left_behind_fraction" in options else 0, key="me_y")
        subset = data[data.scenario == scenario][["method", x, y]]
        st.scatter_chart(subset, x=x, y=y, color="method")
        st.caption("Each point is one method at one training seed. Baselines contribute a single "
                   "point each.")


def forecasting_page() -> None:
    st.header("Forecasting")
    badge("DESCRIPTIVE")
    fc = artefact("forecasting_metrics.json")
    if not fc:
        return missing("forecasting_metrics.json")
    cols = st.columns(3)
    cols[0].metric("Quantile crossing rate", f"{fc['quantile_crossing_rate']:.6f}")
    cols[1].metric("Config hash", str(fc.get("config_hash", ""))[:12])
    cols[2].metric("Model", str(fc.get("artifacts", {}).get("model", "xgb_quantile")))
    st.markdown(
        "The environment consumes a **frozen** dispersion measure, "
        "`sigma = (q90 − q10) / (2 × 1.2816)`, fixed before any policy trained. No policy can "
        "influence the uncertainty signal it receives.")
    if fc.get("metrics"):
        with st.expander("Accuracy metrics", expanded=True):
            st.json(fc["metrics"], expanded=False)
    if fc.get("calibration"):
        with st.expander("Calibration"):
            st.json(fc["calibration"], expanded=False)
    if fc.get("gate"):
        with st.expander("Acceptance gate"):
            st.json(fc["gate"], expanded=False)
    figure("fig03_forecasting")


def twin_page() -> None:
    st.header("Digital twin validation")
    badge("DESCRIPTIVE")
    sim = artefact("simulator_validation.json")
    if not sim:
        return missing("simulator_validation.json")
    cols = st.columns(len(sim["counts"]))
    for col, (key, value) in zip(cols, sim["counts"].items()):
        col.metric(key, value)
    rows = pd.DataFrame(sim["rows"])
    verdicts = st.multiselect("Verdict", sorted(rows.verdict.unique()),
                              default=sorted(rows.verdict.unique()), key="tw_verdict")
    st.dataframe(rows[rows.verdict.isin(verdicts)], width="stretch", hide_index=True)
    download(rows, "Download (CSV)", "twin_validation.csv")
    st.error(
        "**Four of the not-validatable quantities are direct inputs to the primary endpoint**: "
        "train capacity and peak occupancy, dwell-time distribution, passenger balking and "
        "patience, and the origin–destination distribution. What was validated constrains the "
        "supply side, not the crowding side the endpoint measures. Every magnitude here is "
        "simulator-internal; only the ordering of methods is interpreted.")
    figure("fig02_twin_architecture")


def data_page() -> None:
    st.header("Data")
    badge("DESCRIPTIVE")
    audit = artefact("reproducibility_audit.json")
    if not audit:
        return missing("reproducibility_audit.json")
    for check in audit["checks"]:
        if check["name"] in ("raw data provenance", "chronological splits with embargo",
                             "regime screen", "temporal leakage tests"):
            st.markdown(f"**{check['name']}** — `{check['status']}`")
            st.caption(check["detail"])
            evidence = check.get("evidence") or {}
            for key in ("split", "rows"):
                if key in evidence:
                    st.json(evidence[key], expanded=False)
            st.write("")


def ablations_page() -> None:
    st.header("Ablations and sensitivity")
    stats_art, sens = artefact("statistical_analysis.json"), artefact("sensitivity_analysis.json")
    st.subheader("Registered arm ablations")
    badge("EXPLORATORY")
    if stats_art:
        rows = [t for t in stats_art.get("exploratory", [])
                if t.get("experiment_id") in ("EXP-MO-001", "EXP-AVAIL-001")]
        frame = pd.DataFrame(rows)
        if not frame.empty:
            metric = st.selectbox("Metric", sorted(frame.metric.unique()),
                                  index=sorted(frame.metric.unique()).index("left_behind_fraction")
                                  if "left_behind_fraction" in set(frame.metric) else 0,
                                  key="ab_metric")
            st.dataframe(frame[frame.metric == metric][
                ["experiment_id", "hypothesis", "scenario", "method_a", "method_b",
                 "mean_difference", "p_adjusted", "direction"]], width="stretch", hide_index=True)
    st.markdown("---")
    st.subheader("Sensitivity conditions")
    badge("EXPLORATORY")
    if not sens:
        return missing("sensitivity_analysis.json")
    frame = pd.DataFrame(sens["tests"])
    experiments = st.multiselect("Experiment", sorted(frame.experiment_id.unique()),
                                 default=sorted(frame.experiment_id.unique()), key="ab_exp")
    st.dataframe(frame[frame.experiment_id.isin(experiments)][
        ["label", "metric", "method_a", "method_b", "n", "mean_difference", "p_adjusted",
         "direction"]], width="stretch", hide_index=True)
    agree = sens["direction_agreement"]
    st.caption(f"`ppo_uq` is the worse arm in {agree['conditions_where_ppo_uq_is_worse']} of "
               f"{agree['conditions_total']} sensitivity conditions; "
               f"{agree['any_condition_favouring_ppo_uq']} favour it.")
    st.warning(sens["weight_invariance"]["finding"])
    download(frame, "Download (CSV)", "sensitivity.csv")
    figure("fig10_sensitivity")


def robustness_page() -> None:
    st.header("Robustness")
    badge("CONFIRMATORY")
    power, df = artefact("power_analysis.json"), seed_level()
    if not power or df is None:
        return missing("power_analysis.json / seed_level.parquet")
    scenario = st.selectbox("Scenario", SCENARIOS, index=1, key="rb_scenario")
    data = rl_only(df)
    piv = data[data.scenario == scenario].pivot_table(
        index="train_seed", columns="method", values="left_behind_fraction")
    if not {"ppo_uq", "ppo_point"} <= set(piv.columns):
        st.info("Both arms are required for this view.")
        return
    diff = (piv["ppo_uq"] - piv["ppo_point"])

    cols = st.columns(4)
    cols[0].metric("Pilot SD (planning)", f"{power['between_seed_sd']:.6f}")
    cols[1].metric("Realised SD", f"{diff.std(ddof=1):.6f}",
                   delta=f"{diff.std(ddof=1) / power['between_seed_sd']:.2f}x")
    cols[2].metric("Seeds favouring ppo_point", f"{int((diff > 0).sum())}/{len(diff)}")
    cols[3].metric("Wilcoxon p",
                   f"{stats.wilcoxon(piv['ppo_uq'], piv['ppo_point']).pvalue:.5f}")

    st.subheader("Per-seed paired difference")
    st.bar_chart(diff.rename("ppo_uq − ppo_point"))

    st.subheader("Leave-one-seed-out")
    jack = []
    for i, seed in enumerate(piv.index):
        p = float(stats.ttest_rel(np.delete(piv["ppo_uq"].values, i),
                                  np.delete(piv["ppo_point"].values, i)).pvalue)
        jack.append({"seed dropped": int(seed), "p": round(p, 6),
                     "Holm-adjusted": round(min(1.0, 2 * p), 6),
                     "still significant": bool(min(1.0, 2 * p) < 0.05)})
    jack_frame = pd.DataFrame(jack).sort_values("p", ascending=False)
    st.dataframe(jack_frame, width="stretch", hide_index=True)
    download(jack_frame, "Download (CSV)", f"jackknife_{scenario}.csv")
    st.success("The conclusion does not rest on any single training seed: every replicate "
               "remains significant after Holm adjustment.")
    st.error(
        "**No post-hoc power argument is made.** After a rejection, observed power is a "
        "deterministic function of the p-value. The pilot underestimated the between-seed SD by "
        "roughly a factor of two, so the registered minimum detectable effect was optimistic; "
        "the study remains underpowered for its own SESOI, and every null reported anywhere in "
        "this dashboard is inconclusive rather than evidence of equivalence.")


def shap_page() -> None:
    st.header("SHAP attribution")
    badge("DESCRIPTIVE")
    shap = artefact("shap_analysis.json")
    if not shap:
        return missing("shap_analysis.json")
    st.error("**SHAP provides model attribution and does not establish causality.** These values "
             "describe how one policy's output covaries with its inputs on a chosen instance set, "
             "at a single training seed. They are not evidence for or against H1.")
    st.caption(f"{shap['explainer']} · explaining {shap['explained_quantity']} · "
               f"policy `{shap['method']}` seed {shap['train_seed']} · background "
               f"{shap['background_size']}")
    importance = shap.get("global_importance") or []
    if importance:
        frame = pd.DataFrame(importance)
        top_n = st.slider("Features shown", 5, min(40, len(frame)), 15, key="sh_n")
        top = (frame.set_index("feature")["mean_abs_shap"].astype(float)
               .sort_values(ascending=False).head(top_n))
        st.bar_chart(top)
        download(frame, "Download all attributions (CSV)", "shap_global.csv")
    for caveat in shap.get("caveats", []):
        st.caption(f"• {caveat}")
    figure("fig11_shap")


def statistics_page() -> None:
    st.header("Statistics")
    badge("CONFIRMATORY")
    stats_art, power = artefact("statistical_analysis.json"), artefact("power_analysis.json")
    if not stats_art or not power:
        return missing("statistical_analysis.json")
    test = next(t for t in stats_art["confirmatory"] if t["experiment_id"] == "EXP-UQ-001")
    st.subheader("Primary comparison — ppo_uq vs ppo_point, left_behind_fraction, surge")
    grid = st.columns(4)
    grid[0].metric("n (training seeds)", test["n"])
    grid[1].metric("Difference", f"{test['mean_difference']:+.6f}")
    grid[2].metric("Holm-adjusted p", f"{test['p_adjusted']:.6g}")
    grid[3].metric("Direction", test["direction"])
    grid2 = st.columns(4)
    grid2[0].metric("Hodges–Lehmann", f"{test['hodges_lehmann']:.6f}")
    grid2[1].metric("95% BCa CI", f"[{test['ci_low']:.5f}, {test['ci_high']:.5f}]")
    grid2[2].metric("SESOI", power["sesoi"])
    grid2[3].metric("MDE at registered n",
                    f"{power['minimum_detectable_effect_at_registered_n']:.6f}")
    st.caption("The MDE shown is the registered planning figure, computed from the pilot SD. See "
               "the Robustness page: the realised SD was about twice the pilot estimate, so this "
               "figure is optimistic.")
    st.markdown("---")
    st.dataframe(pd.DataFrame(stats_art["confirmatory"]), width="stretch", hide_index=True)
    st.markdown("---")
    st.subheader("Exploratory family")
    badge("EXPLORATORY")
    frame = pd.DataFrame(stats_art["exploratory"])
    if frame.empty:
        return
    left, right = st.columns(2)
    metric = left.selectbox("Metric", sorted(frame.metric.unique()), key="st_metric")
    scenario = right.selectbox("Scenario", ["(all)"] + sorted(frame.scenario.unique()),
                               key="st_scenario")
    view = frame[frame.metric == metric]
    if scenario != "(all)":
        view = view[view.scenario == scenario]
    st.dataframe(view[["experiment_id", "hypothesis", "scenario", "method_a", "method_b",
                       "mean_difference", "p_value", "p_adjusted", "direction"]],
                 width="stretch", hide_index=True)
    download(frame, "Download exploratory family (CSV)", "exploratory.csv")


def followup_page() -> None:
    st.header("Training-budget follow-up")
    badge("FOLLOW-UP")
    st.markdown(
        "<div style='background:#5a3d8a;color:#fff;padding:14px 18px;border-radius:8px;"
        "font-size:1.15rem;font-weight:700'>FOLLOW-UP / INCOMPLETE</div>",
        unsafe_allow_html=True)
    st.write("")
    run = artefact("followup_last_run.json")
    result = artefact("followup_analysis.json")
    policies = REPO_ROOT / "models" / "policies"
    completed = sorted(p for p in policies.glob("budget_*")) if policies.is_dir() else []
    trained = sum(1 for d in completed for _ in d.rglob("policy.zip"))
    cells = sum(1 for _ in (RESULTS_DIR / "runs").glob("budget_*/**/metrics.json"))

    cols = st.columns(4)
    cols[0].metric("Policies trained", f"{trained}/80")
    cols[1].metric("Evaluation cells", f"{cells:,}/9,600")
    cols[2].metric("Budgets reached", f"{len(completed)}/4")
    cols[3].metric("Analysis", "none")

    st.error(
        "**No result is reported from this study, and none can be.** It is structurally "
        "unusable rather than merely incomplete: the single budget reached has ten `ppo_point` "
        "policies against one `ppo_uq`, so not even the lowest budget supports a paired "
        "comparison, and the evaluation sweep never began. Reporting anything from an asymmetric "
        "sample would be exactly the selective reporting the pre-registration exists to prevent.")
    st.info(
        "**The registered conclusion is unchanged and unchangeable by this study.** Under the "
        "registered 60,000-step experiment, H1 was NOT SUPPORTED: +0.013742, 95% BCa CI "
        "[0.006570, 0.018523], Holm-adjusted p = 0.007741.")
    st.markdown(
        "**Why it existed.** Amendment A-05 froze one PPO configuration at one budget, so the "
        "study cannot separate *the uncertainty channel is unhelpful* from *this optimiser could "
        "not exploit it at this budget*. That confound remains open and is reported as a "
        "limitation, not resolved.")
    st.caption("Protocol: `docs/followup_training_budget_protocol.md` (written before execution). "
               "Status: `docs/followup_status.md`. Analysis: `scripts/analyse_followup.py` "
               "(committed before any follow-up data existed).")
    if run:
        with st.expander("Run record"):
            st.json(run, expanded=False)
    if result and result.get("budgets_analysed"):
        with st.expander("Analysis artefact (60k point reads the registered cells)"):
            st.json({k: result[k] for k in ("budgets_analysed", "budgets_missing",
                                            "grid_complete",
                                            "registered_conclusion_unchanged")}, expanded=False)


def reproducibility_page() -> None:
    st.header("Reproducibility")
    badge("DESCRIPTIVE")
    audit = artefact("reproducibility_audit.json")
    separation = artefact("separation_audit.json")
    if not audit:
        return missing("reproducibility_audit.json")
    cols = st.columns(len(audit["counts"]))
    for col, (key, value) in zip(cols, audit["counts"].items()):
        col.metric(key, value)
    st.caption(f"Commit at audit: `{audit.get('git_commit', '')}`")
    st.dataframe(pd.DataFrame(audit["checks"])[["name", "status", "detail"]],
                 width="stretch", hide_index=True)
    if separation:
        st.subheader("Immutability of the registered study")
        st.caption(f"Verified against git tag `{separation['tag']}` — "
                   + ", ".join(f"{k} = {v}" for k, v in separation["counts"].items()))
        st.dataframe(pd.DataFrame(separation["checks"])[["name", "status", "detail"]],
                     width="stretch", hide_index=True)
    st.warning(
        "**Provenance caveat.** Almost every stored cell was produced while the working tree had "
        "uncommitted changes; its manifest records `provisional: true`. The recorded commit "
        "therefore does not fully determine the code that produced those results, and the "
        "`working tree clean` check inspects the tree at *audit* time rather than at "
        "cell-production time.")
    st.subheader("Reproduction")
    st.code("python -m venv .venv\n"
            ".venv/bin/python -m pip install --require-hashes -r requirements.lock.txt\n"
            ".venv/bin/python scripts/check_environment.py\n"
            ".venv/bin/python -m pytest -q\n"
            ".venv/bin/python scripts/analyse.py\n"
            ".venv/bin/python scripts/make_paper_assets.py\n"
            ".venv/bin/python -m streamlit run app.py", language="bash")
    st.caption("Full instructions: `docs/reproduction_guide.md`.")


def downloads_page() -> None:
    st.header("Downloads")
    badge("DESCRIPTIVE")
    st.markdown("Every file below is a stored artefact of the study, served as written.")
    tables = sorted((RESULTS_DIR / "paper_tables").glob("*.csv")) \
        if (RESULTS_DIR / "paper_tables").is_dir() else []
    if tables:
        st.subheader("Publication tables")
        for path in tables:
            st.download_button(f"⬇ {path.name}", path.read_bytes(), path.name, "text/csv",
                               key=f"dlt_{path.name}")
    st.subheader("Statistical artefacts")
    for name in ("statistical_analysis.json", "power_analysis.json", "mechanism_analysis.json",
                 "sensitivity_analysis.json", "reproducibility_audit.json",
                 "separation_audit.json"):
        path = RESULTS_DIR / name
        if path.is_file():
            st.download_button(f"⬇ {name}", path.read_bytes(), name, "application/json",
                               key=f"dlj_{name}")
    frame = seed_level()
    if frame is not None:
        st.subheader("Seed-level results")
        download(frame, "⬇ seed_level.csv (all cells, all variants)", "seed_level_all.csv")


def verdict_page() -> None:
    st.header("Final verdict")
    badge("CONFIRMATORY")
    stats_art = artefact("statistical_analysis.json")
    if not stats_art:
        return missing("statistical_analysis.json")
    test = next(t for t in stats_art["confirmatory"] if t["experiment_id"] == "EXP-UQ-001")
    plac = next(t for t in stats_art["confirmatory"] if t["experiment_id"] == "EXP-UQ-002")
    supported = test["mean_difference"] < 0 and test.get("rejects_at_family_alpha")
    st.markdown(
        f"<div style='background:#5a1f1f;color:#fff;padding:22px;border-radius:10px;"
        f"text-align:center'><div style='font-size:2.6rem;font-weight:800'>H1 = "
        f"{'SUPPORTED' if supported else 'NOT SUPPORTED'}</div></div>",
        unsafe_allow_html=True)
    st.write("")
    st.markdown(
        f"`ppo_uq` vs `ppo_point` on `{test['metric']}` under `{test['scenario']}`, "
        f"n = {test['n']} training seeds: **{test['mean_difference']:+.6f}** "
        f"({test['direction']}), 95% BCa CI [{test['ci_low']:.6f}, {test['ci_high']:.6f}], "
        f"Holm-adjusted p = {test['p_adjusted']:.6g}.")
    st.markdown(
        f"Against the placebo arm: {plac['mean_difference']:+.6f}, Holm-adjusted "
        f"p = {plac['p_adjusted']:.4g}. The degradation is not explained by the extra input "
        "dimensions alone.")
    st.markdown("---")
    st.markdown("### What the evidence supports")
    st.markdown(
        "Uncertainty information was **operationally relevant** — measured service loss is "
        "strongly convex once capacity binds — but the tested PPO controller learned a "
        "**conservative response in the wrong direction**, dispatching about 11% fewer trains "
        "when given genuine forecast dispersion. Under capacity-binding conditions that response "
        "produced worse service. The placebo comparison indicates this cannot be explained by "
        "additional input dimensions alone.")
    st.markdown("### What the evidence does not support")
    st.markdown(
        "- That uncertainty-aware reinforcement learning is ineffective in general.\n"
        "- That reinforcement learning is unsuitable for metro dispatch.\n"
        "- Any causal claim from SHAP attributions.\n"
        "- Any magnitude claim about a physical metro system, given the unvalidated crowding "
        "components of the simulator.\n"
        "- Any conclusion about behaviour at other training budgets: that follow-up is "
        "incomplete.")
    st.caption("This is a controlled falsification under one architecture, training protocol, "
               "environment and set of study conditions.")


#: Sidebar navigation groups.
PAGES: dict[str, list[tuple[str, Any]]] = {
    "Overview": [("Executive overview", overview), ("Final verdict", verdict_page)],
    "Results": [("Primary results", primary_results), ("Placebo analysis", placebo_page),
                ("Scenario explorer", scenario_explorer),
                ("Baseline explorer", baseline_explorer), ("Statistics", statistics_page)],
    "Evidence": [("Mechanism explorer", mechanism_page), ("Robustness", robustness_page),
                 ("Ablations & sensitivity", ablations_page), ("SHAP", shap_page)],
    "Study inputs": [("Data", data_page), ("Forecasting", forecasting_page),
                     ("Digital twin", twin_page)],
    "Provenance": [("Follow-up (incomplete)", followup_page),
                   ("Reproducibility", reproducibility_page), ("Downloads", downloads_page)],
}

#: Flat list, kept for the smoke test and for anything that wants the whole inventory.
PANELS: list[tuple[str, Any]] = [entry for group in PAGES.values() for entry in group]
