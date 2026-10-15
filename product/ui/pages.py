"""Pages of the Metro Intelligent Dispatch prototype.

Operational pages run the validated simulator live and are badged PRODUCT DEMONSTRATION. Research
pages read the frozen study artefacts through the read-only repository and are badged with the
evidence class the study assigned. Nothing here writes to a research path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from product.services.operations import (
    SCENARIO_LABEL, SCENARIOS, ProductServiceError, SessionService, available_methods,
    unavailable_methods,
)
from product.services.research_repository import (
    CONFIRMATORY, DESCRIPTIVE, EXPLORATORY, PRODUCT, ResearchArtefactMissing, repository,
)
from product.ui.components import (
    guard, metric_row, prototype_notice, research_vs_demo_notice, section, status_chip,
    unavailable,
)

RESULTS_FIGURES = "paper_figures"


# --------------------------------------------------------------------------- session plumbing


def _selection() -> tuple[str, str, int, int | None]:
    """The operating selection, kept in session state so pages share one episode."""
    return (st.session_state.get("op_method", "ppo_point"),
            st.session_state.get("op_scenario", "surge"),
            int(st.session_state.get("op_seed", 7)),
            st.session_state.get("op_train_seed"))


def live_session() -> SessionService | None:
    """One live episode per selection. Rebuilt only when the operator changes something."""
    method, scenario, seed, train_seed = _selection()
    key = f"{method}|{scenario}|{seed}|{train_seed}"
    if st.session_state.get("session_key") != key:
        try:
            with st.spinner(f"Loading {method} on {SCENARIO_LABEL.get(scenario, scenario)}…"):
                st.session_state.session = SessionService(method, scenario, seed, train_seed)
            st.session_state.session_key = key
        except ProductServiceError as error:
            st.error(f"**Could not start a session.** {error}")
            return None
    return st.session_state.get("session")


def operations_controls() -> None:
    """Sidebar controls shared by every operational page."""
    st.sidebar.markdown("### Operating selection")
    methods = available_methods()
    if not methods:
        st.sidebar.error("No policy artefacts are available.")
        return
    labels = {m: label for m, label, _ in methods}
    method = st.sidebar.selectbox("Strategy", [m for m, _, _ in methods],
                                  format_func=lambda m: labels[m], key="op_method")
    scenario = st.sidebar.selectbox("Scenario", SCENARIOS,
                                    format_func=lambda s: SCENARIO_LABEL[s],
                                    index=1, key="op_scenario")
    seeds = next((s for m, _, s in methods if m == method), ())
    if seeds:
        st.sidebar.selectbox("Policy instance (training seed)", seeds, key="op_train_seed",
                             help="Each training seed is a separate trained policy.")
    else:
        st.session_state["op_train_seed"] = None
    st.sidebar.number_input("Simulation day (seed)", 1, 99_999, key="op_seed", step=1,
                            help="The same seed reproduces the same simulated day exactly.")
    st.sidebar.markdown("---")
    controls = st.sidebar.columns(3)
    if controls[0].button("Step", use_container_width=True):
        st.session_state.pending_steps = 1
    if controls[1].button("Run", use_container_width=True):
        st.session_state.pending_steps = int(st.session_state.get("run_epochs", 24))
    if controls[2].button("Reset", use_container_width=True):
        st.session_state.pending_reset = True
    st.sidebar.slider("Epochs per Run", 6, 228, 24, 6, key="run_epochs")
    missing = unavailable_methods()
    if missing:
        st.sidebar.markdown("---")
        st.sidebar.caption("Unavailable strategies:")
        for method_id, reason in missing:
            st.sidebar.caption(f"• {method_id} — {reason}")
    _ = scenario


def apply_pending(session: SessionService) -> None:
    if st.session_state.pop("pending_reset", False):
        session.reset()
    steps = st.session_state.pop("pending_steps", 0)
    if steps:
        with st.spinner(f"Simulating {steps} decision epoch(s)…"):
            session.step(steps)


# --------------------------------------------------------------------------- operational pages


def home() -> None:
    st.title("METRO INTELLIGENT DISPATCH")
    st.caption("Decision Support System · prototype")
    prototype_notice()
    session = live_session()
    if session is None:
        return
    apply_pending(session)
    state = session.operating_state()

    section("Network status", PRODUCT,
            f"{SCENARIO_LABEL[state.scenario]} · epoch {state.epoch} of {state.total_epochs} · "
            f"{state.clock}")
    metric_row([
        ("PASSENGERS WAITING", f"{state.total_queue:,.0f}", "Total across all stations, now."),
        ("FORECAST (next hour)", f"{state.forecast_mean:,.0f}",
         "Line-level demand forecast from the frozen research forecaster."),
        ("UNCERTAINTY", f"±{state.forecast_sigma:,.0f}" if state.carries_uncertainty else "n/a",
         "Forecast dispersion. This strategy carries no uncertainty channel."
         if not state.carries_uncertainty else "Dispersion of the demand forecast."),
        ("TRAINS AVAILABLE", f"{state.trains_available} / {state.fleet_size}",
         "Trainsets not currently in service."),
    ])
    metric_row([
        ("TRAINS IN SERVICE", state.trains_in_service, None),
        ("BUSIEST STATION QUEUE", f"{state.max_station_queue:,.0f}", None),
        ("CAPACITY PRESSURE", state.capacity_pressure,
         "Operator-facing band derived from queue and fleet utilisation. A presentation device, "
         "not a research quantity."),
        ("UNCERTAINTY LEVEL", state.uncertainty_level, None),
    ])
    st.markdown(
        status_chip("CAPACITY", state.capacity_pressure) + "&nbsp;&nbsp;"
        + status_chip("UNCERTAINTY", state.uncertainty_level), unsafe_allow_html=True)
    st.write("")
    if session.terminated:
        st.success("The simulated operating day has ended. Press Reset to run it again.")

    st.markdown("---")
    section("Operator workflow", PRODUCT)
    steps = ["Observe network", "Inspect forecast", "Inspect uncertainty", "Check availability",
             "Check queue and capacity", "Receive recommendation", "Inspect explanation",
             "Run scenario", "Compare strategies", "Review consequences"]
    columns = st.columns(5)
    for index, label in enumerate(steps):
        with columns[index % 5]:
            st.markdown(f"**{index + 1}.** {label}")

    st.markdown("---")
    research_vs_demo_notice()
    with guard("Research headline"):
        repo = repository()
        verdict = repo.h1_verdict()
        test = repo.primary_result()
        section("Research headline", CONFIRMATORY,
                "The completed study behind this prototype.")
        st.markdown(
            f"**H1 = {verdict}.** Supplying the controller with calibrated forecast uncertainty "
            f"changed `{test['metric']}` by **{test['mean_difference']:+.6f}** under "
            f"`{test['scenario']}` (Holm-adjusted p = {test['p_adjusted']:.4g}); the "
            "uncertainty-aware arm performed worse. See the Research Results page.")


def network() -> None:
    st.title("Network")
    session = live_session()
    if session is None:
        return
    apply_pending(session)
    section("Line state", PRODUCT, "Schematic — station order is real, geometry is not.")
    state = session.network_state()
    stations = pd.DataFrame(state["stations"])
    trains = pd.DataFrame(state["trains"])
    live = trains[trains["in_service"]] if len(trains) else trains

    metric_row([
        ("STATIONS", len(stations), None),
        ("TRAINS IN SERVICE", int(state["trains_in_service"]), None),
        ("FLEET", int(state["fleet_size"]), None),
        ("TOTAL QUEUE", f"{stations['queue'].sum():,.0f}", None),
    ])
    st.bar_chart(stations.set_index("name")[["queue"]].rename(
        columns={"queue": "passengers waiting"}), height=280)
    left, right = st.columns([3, 2])
    with left:
        st.caption("Stations")
        st.dataframe(stations.assign(**{
            "waiting": stations["queue"].round(0),
            "longest wait (min)": (stations["oldest_wait_s"] / 60).round(1),
        })[["station_id", "name", "waiting", "longest wait (min)"]],
            width="stretch", hide_index=True, height=320)
    with right:
        st.caption("Trains in service")
        if len(live):
            st.dataframe(live.assign(**{
                "occupancy": (live["occupancy_fraction"] * 100).round(0).astype(int).astype(str) + "%",
            })[["train_id", "status", "station_name", "occupancy"]],
                width="stretch", hide_index=True, height=320)
        else:
            unavailable("no trains are in service yet",
                        "Step or Run the simulation to release trainsets.")


def forecast() -> None:
    st.title("Demand forecast")
    session = live_session()
    if session is None:
        return
    apply_pending(session)
    state = session.operating_state()
    section("Line forecast, next hour", PRODUCT,
            "Produced by the frozen research forecaster; the product does not refit it.")
    if not state.carries_uncertainty:
        st.info("This strategy carries **no uncertainty channel** — its sigma inputs are zeroed. "
                "That is the point-forecast condition, not a missing model.")
    metric_row([
        ("FORECAST MEAN", f"{state.forecast_mean:,.0f}", None),
        ("DISPERSION (sigma)", f"±{state.forecast_sigma:,.0f}" if state.carries_uncertainty
         else "NOT AVAILABLE", None),
        ("LOWER BOUND (80%)", f"{state.forecast_low:,.0f}" if state.carries_uncertainty
         else "NOT AVAILABLE", None),
        ("UPPER BOUND (80%)", f"{state.forecast_high:,.0f}" if state.carries_uncertainty
         else "NOT AVAILABLE", None),
    ])
    with guard("Per-station forecast"):
        payload = session.forecast_state()
        rows = payload.get("stations") or payload.get("per_station")
        if rows:
            frame = pd.DataFrame(rows)
            st.caption("Per-station forecast")
            st.dataframe(frame, width="stretch", hide_index=True, height=300)
        else:
            unavailable("per-station forecast detail is not exposed by the session",
                        "The line-level forecast above is what the policy observes at this epoch.")
    with guard("Realised demand"):
        realised = session.session.realised_demand()
        if realised:
            st.caption("Realised demand so far this simulated day")
            st.line_chart(pd.DataFrame(realised).set_index(
                pd.DataFrame(realised).columns[0]))

    st.markdown("---")
    with guard("Forecast quality"):
        repo = repository()
        fc = repo.forecasting_metrics()
        section("Forecast model quality", DESCRIPTIVE, "From the completed study.")
        metric_row([
            ("QUANTILE CROSSING", f"{fc['quantile_crossing_rate']:.6f}", None),
            ("CONFIG HASH", str(fc.get("config_hash", ""))[:12], None),
            ("MODEL", str(fc.get("artifacts", {}).get("model", "xgb_quantile")), None),
        ])
        st.caption("The environment consumes a frozen dispersion measure, "
                   "sigma = (q90 − q10) / (2 × 1.2816), fixed before any policy trained.")


def dispatch() -> None:
    st.title("Dispatch decision")
    prototype_notice()
    session = live_session()
    if session is None:
        return
    apply_pending(session)
    state = session.operating_state()

    section("Current operating state", PRODUCT)
    left, right = st.columns(2)
    with left:
        st.markdown(
            f"- **Passengers waiting:** {state.total_queue:,.0f}\n"
            f"- **Forecast (next hour):** {state.forecast_mean:,.0f}\n"
            f"- **Uncertainty:** "
            + (f"±{state.forecast_sigma:,.0f} ({state.uncertainty_level})"
               if state.carries_uncertainty else "not carried by this strategy") + "\n"
            f"- **Busiest station queue:** {state.max_station_queue:,.0f}")
    with right:
        st.markdown(
            f"- **Trains available:** {state.trains_available} of {state.fleet_size}\n"
            f"- **Trains in service:** {state.trains_in_service}\n"
            f"- **Capacity pressure:** {state.capacity_pressure}\n"
            f"- **Epoch:** {state.epoch} of {state.total_epochs} ({state.clock})")

    st.markdown("---")
    if session.terminated:
        st.success("The simulated day has ended; there is no further decision to take.")
        return
    try:
        recommendation = session.recommend()
    except ProductServiceError as error:
        st.error(f"**No recommendation could be produced.** {error}")
        return

    section("Recommended action", PRODUCT)
    st.markdown(
        f"<div style='background:#1a4971;color:#fff;padding:18px 22px;border-radius:10px'>"
        f"<div style='font-size:0.78rem;opacity:0.85;letter-spacing:0.08em'>RECOMMENDATION"
        f"</div><div style='font-size:1.9rem;font-weight:800'>{recommendation.label}</div>"
        f"<div style='font-size:0.82rem;opacity:0.9;margin-top:4px'>from "
        f"{recommendation.source} · feasible actions "
        f"{recommendation.feasible_actions}</div></div>", unsafe_allow_html=True)
    st.write("")
    st.error("**Operator approval is required.** This prototype does not act on the railway.")

    st.subheader("Why this recommendation?")
    for reason in recommendation.reasons:
        st.markdown(f"- {reason}")
    st.caption("These reasons describe the observable operating state. They are not a claim about "
               "the policy network's internal computation.")

    st.subheader("Expected effect")
    st.caption("Simulated, not guaranteed. Press **Step** in the sidebar to advance one epoch and "
               "compare the realised outcome.")
    before = session._snapshot()
    st.dataframe(pd.DataFrame([
        {"quantity": "Passengers waiting (now)", "value": round(state.total_queue, 1)},
        {"quantity": "Left-behind fraction so far",
         "value": round(float(before.get("left_behind_fraction") or 0.0), 6)},
        {"quantity": "Inductions so far", "value": before.get("inductions")},
        {"quantity": "Mean waiting time so far (s)",
         "value": round(float(before.get("waiting_time_mean_s") or 0.0), 1)},
    ]), width="stretch", hide_index=True)


def simulator() -> None:
    st.title("Scenario simulator")
    st.caption("PRODUCT DEMONSTRATION SCENARIO — not a confirmatory experiment.")
    session = live_session()
    if session is None:
        return
    section("Run a demonstration", PRODUCT,
            "Runs the validated simulator forward under the selected strategy. Nothing is "
            "written to the research results.")
    left, right = st.columns([2, 1])
    epochs = left.slider("Decision epochs to simulate", 6, 228, 60, 6, key="sim_epochs")
    if right.button("▶  RUN SIMULATION", use_container_width=True, type="primary"):
        with st.spinner(f"Simulating {epochs} epochs…"):
            st.session_state.sim_outcome = session.step(epochs)
    if right.button("Reset day", use_container_width=True):
        session.reset()
        st.session_state.pop("sim_outcome", None)

    outcome = st.session_state.get("sim_outcome")
    if outcome is None:
        st.info("Press **RUN SIMULATION** to advance the simulated day and compare before/after.")
        return

    st.subheader(f"Before / after — {outcome.epochs_run} epochs of "
                 f"{SCENARIO_LABEL.get(outcome.scenario, outcome.scenario)}")
    interesting = ["left_behind_fraction", "left_behind_passengers", "waiting_time_mean_s",
                   "waiting_time_p95_s", "waiting_accrued_passenger_hours", "inductions",
                   "passengers_boarded", "energy_units", "cost_units", "service_level"]
    rows = []
    for key in interesting:
        before, after = outcome.before.get(key), outcome.after.get(key)
        if before is None and after is None:
            rows.append({"metric": key, "before": "NOT AVAILABLE", "after": "NOT AVAILABLE",
                         "change": "NOT AVAILABLE"})
            continue
        delta = outcome.delta(key)
        rows.append({
            "metric": key,
            "before": round(float(before), 4) if before is not None else "NOT AVAILABLE",
            "after": round(float(after), 4) if after is not None else "NOT AVAILABLE",
            "change": round(delta, 4) if delta is not None else "NOT AVAILABLE",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("A cumulative episode summary: 'after' includes everything simulated so far, not "
               "only the epochs just run.")


def strategy_comparison() -> None:
    st.title("Strategy comparison")
    research_vs_demo_notice()
    with guard("Research comparison"):
        repo = repository()
        section("Research results — all strategies", CONFIRMATORY,
                "Measured by the completed study over its registered evaluation design.")
        metrics = repo.metrics()
        left, right = st.columns(2)
        metric = left.selectbox("Metric", metrics,
                                index=metrics.index("left_behind_fraction")
                                if "left_behind_fraction" in metrics else 0, key="cmp_metric")
        scenario = right.selectbox("Scenario", SCENARIOS, index=1,
                                   format_func=lambda s: SCENARIO_LABEL[s], key="cmp_scenario")
        table = repo.comparison_table(metric, scenario)
        chosen = st.multiselect("Strategies", list(table.method),
                                default=list(table.method), key="cmp_methods")
        view = table[table.method.isin(chosen)]
        st.dataframe(view.round(6), width="stretch", hide_index=True)
        st.bar_chart(view.set_index("method")["mean"])
        st.caption(
            "Baselines are deterministic and carry no training seed, so each contributes n = 1 "
            "under the study's unit of analysis. **No inferential test between a learned "
            "strategy and a baseline is defined**; these are differences of means.")
        st.warning(
            "In the completed study, no learned strategy outperformed the tuned threshold "
            "heuristic in any scenario.")
        st.download_button("Download this comparison (CSV)",
                           view.to_csv(index=False).encode("utf-8"),
                           f"research_{metric}_{scenario}.csv", "text/csv")

        st.markdown("---")
        section("Across scenarios", CONFIRMATORY)
        st.dataframe(repo.scenario_matrix(metric).round(6), width="stretch")


def mechanism() -> None:
    st.title("Mechanism")
    with guard("Mechanism"):
        repo = repository()
        mech = repo.mechanism()
        section("Uncertainty → dispatch → queue → passengers left behind", EXPLORATORY,
                "From the completed study's stored per-epoch data.")
        st.markdown(
            "```\nUNCERTAINTY  →  DISPATCH RESPONSE  →  CAPACITY PRESSURE  →  QUEUE  →  "
            "LEFT-BEHIND PASSENGERS\n```")
        rows = []
        for row in mech["dispatch_volume"]:
            info, plc = row["informative_sigma_vs_zero"], row["uninformative_sigma_vs_zero"]
            rows.append({"scenario": row["scenario"],
                         "calibrated sigma": round(info["mean_difference"], 2),
                         "p": f"{info['p_value']:.1e}",
                         "placebo sigma": round(plc["mean_difference"], 2),
                         "p (placebo)": f"{plc['p_value']:.3f}"})
        frame = pd.DataFrame(rows)
        st.dataframe(frame, width="stretch", hide_index=True)
        st.bar_chart(frame.set_index("scenario")[["calibrated sigma", "placebo sigma"]])
        st.caption(mech["dispatch_volume_reading"])

        st.subheader("At matched dispatch volume")
        matched = mech["volume_matched_comparison"]["surge"]
        st.dataframe(pd.DataFrame({"ppo_uq": matched["ppo_uq"],
                                   "fixed_timetable": matched["fixed_timetable"]}).round(2),
                     width="stretch")
        st.caption(mech["volume_matched_reading"])

        st.subheader("Convex service loss")
        st.markdown(
            "Passengers left behind stay near zero while capacity is slack, then rise sharply "
            "once it binds. For a convex loss, the expected loss under uncertain demand exceeds "
            "the loss at the expected demand — the standard newsvendor observation — so demand "
            "uncertainty is operationally relevant and motivates hedging **upward**.")
        _figure("fig07_convex_service_loss")
        st.info(
            "**What the study found.** The tested uncertainty-aware controller responded to "
            "genuine forecast dispersion by dispatching *fewer* trains — the opposite direction. "
            "This is a finding about the tested controller under the tested conditions. It does "
            "not establish universal causality, and it does not show that uncertainty-aware "
            "reinforcement learning is ineffective in general.")


def research_results() -> None:
    st.title("Research results")
    with guard("Research results"):
        repo = repository()
        test, plac = repo.primary_result(), repo.placebo_result()
        verdict = repo.h1_verdict()
        section("Primary hypothesis", CONFIRMATORY)
        st.markdown(
            f"<div style='background:#5a1f1f;color:#fff;padding:20px;border-radius:10px;"
            f"text-align:center'><div style='font-size:2.4rem;font-weight:800'>H1 = {verdict}"
            f"</div></div>", unsafe_allow_html=True)
        st.write("")
        st.markdown(
            f"Primary comparison **`{test['method_a']}` vs `{test['method_b']}`** on "
            f"**`{test['metric']}`** under **`{test['scenario']}`**, n = {test['n']} training "
            "seeds.")
        metric_row([
            (f"MEAN {test['method_a']}", f"{test['mean_a']:.6f}", None),
            (f"MEAN {test['method_b']}", f"{test['mean_b']:.6f}", None),
            ("DIFFERENCE", f"{test['mean_difference']:+.6f}", None),
            ("HOLM-ADJUSTED p", f"{test['p_adjusted']:.6g}", None),
        ])
        metric_row([
            ("HODGES–LEHMANN", f"{test['hodges_lehmann']:.6f}", None),
            ("95% BCa CI", f"[{test['ci_low']:.6f}, {test['ci_high']:.6f}]", None),
            ("DIRECTION", test["direction"], None),
            ("TEST", test["test"], None),
        ])
        st.markdown("---")
        section("Placebo control", CONFIRMATORY,
                "An uncertainty channel of identical marginal distribution, with its association "
                "to forecast error destroyed.")
        st.markdown(
            f"`{plac['method_a']}` vs `{plac['method_b']}`: **{plac['mean_difference']:+.6f}**, "
            f"95% BCa CI [{plac['ci_low']:.6f}, {plac['ci_high']:.6f}], Holm-adjusted "
            f"p = {plac['p_adjusted']:.4g}. The degradation is not explained by the extra input "
            "dimensions alone.")
        _figure("fig05_placebo")
        st.markdown("---")
        section("All confirmatory tests", CONFIRMATORY)
        st.dataframe(pd.DataFrame(repo.confirmatory_tests()), width="stretch", hide_index=True)


def statistics() -> None:
    st.title("Statistics")
    with guard("Statistics"):
        repo = repository()
        power = repo.power()
        section("Registered design and power", CONFIRMATORY)
        metric_row([
            ("TRAINING SEEDS", power["registered_train_seeds"], None),
            ("EVALUATION SEEDS", power["registered_eval_seeds"], None),
            ("SESOI", power["sesoi"], "Smallest effect size of interest, frozen before the pilot."),
            ("MDE (planning)", f"{power['minimum_detectable_effect_at_registered_n']:.6f}",
             "Computed from the pilot SD."),
        ])
        st.error(
            "The four-seed pilot underestimated the between-seed standard deviation by roughly a "
            "factor of two, so the planning MDE above is optimistic. The study is underpowered "
            "for its own SESOI, and **every null result is inconclusive rather than evidence of "
            "equivalence**. No post-hoc power argument is made.")
        st.markdown("---")
        section("Exploratory family", EXPLORATORY,
                "Benjamini–Hochberg at q = 0.10. Supports no headline claim.")
        frame = pd.DataFrame(repo.exploratory_tests())
        if frame.empty:
            unavailable("no exploratory tests are stored")
            return
        left, right = st.columns(2)
        metric = left.selectbox("Metric", sorted(frame.metric.unique()), key="stats_metric")
        scenario = right.selectbox("Scenario", ["(all)"] + sorted(frame.scenario.unique()),
                                   key="stats_scenario")
        view = frame[frame.metric == metric]
        if scenario != "(all)":
            view = view[view.scenario == scenario]
        st.dataframe(view[["experiment_id", "hypothesis", "scenario", "method_a", "method_b",
                           "mean_difference", "p_adjusted", "direction"]],
                     width="stretch", hide_index=True)


def policy_inspection() -> None:
    st.title("Policy inspection")
    with guard("Policy attribution"):
        repo = repository()
        shap = repo.shap()
        section("Feature attribution", DESCRIPTIVE,
                f"{shap['explainer']} · policy `{shap['method']}` at training seed "
                f"{shap['train_seed']}")
        st.error("**Feature attribution describes model association and does not by itself "
                 "establish causality.** These values come from a single policy at a single "
                 "training seed and support no hypothesis in the study.")
        importance = shap.get("global_importance") or []
        if not importance:
            unavailable("no stored attribution values")
            return
        frame = pd.DataFrame(importance)
        count = st.slider("Features shown", 5, min(40, len(frame)), 15, key="shap_n")
        top = (frame.set_index("feature")["mean_abs_shap"].astype(float)
               .sort_values(ascending=False).head(count))
        st.bar_chart(top)
        st.download_button("Download attributions (CSV)",
                           frame.to_csv(index=False).encode("utf-8"), "attributions.csv",
                           "text/csv")

    st.markdown("---")
    session = live_session()
    if session is None:
        return
    section("Live decision trace", PRODUCT)
    try:
        recommendation = session.recommend()
        st.markdown(f"**Action now:** {recommendation.label} "
                    f"(feasible {recommendation.feasible_actions})")
        for reason in recommendation.reasons:
            st.markdown(f"- {reason}")
    except ProductServiceError as error:
        st.info(f"No live action available: {error}")


def follow_up() -> None:
    st.title("Registered follow-up")
    with guard("Follow-up"):
        status = repository().follow_up_status()
        section("Training-budget follow-up", status["evidence"])
        st.markdown(
            "<div style='background:#5a3d8a;color:#fff;padding:14px 18px;border-radius:8px;"
            "font-size:1.1rem;font-weight:700'>FOLLOW-UP / INCOMPLETE</div>",
            unsafe_allow_html=True)
        st.write("")
        metric_row([
            ("POLICIES TRAINED", f"{status['policies_trained']}/{status['policies_planned']}", None),
            ("EVALUATION CELLS", f"{status['cells']:,}/{status['cells_planned']:,}", None),
            ("BUDGETS REACHED", f"{len(status['budgets_reached'])}/4", None),
            ("USABLE RESULT", "NO", None),
        ])
        st.error(f"**No result is reported from this study, and none can be.** {status['reason']}")
        st.info("The registered conclusion is unchanged and unchangeable by this study: under the "
                "registered experiment, H1 was NOT SUPPORTED.")


def reproducibility() -> None:
    st.title("Reproducibility")
    with guard("Reproducibility"):
        repo = repository()
        provenance = repo.provenance()
        section("Research provenance", DESCRIPTIVE)
        metric_row([
            ("EVALUATION CELLS", f"{provenance['cells']:,}", None),
            ("TRAINED POLICIES", provenance["policies"], None),
            ("PROTECTED ARTEFACTS", "UNCHANGED" if provenance["protected_unchanged"]
             else "MODIFIED", "Hashed at product start-up and re-checked now."),
            ("COMMIT", str(provenance["commit"])[:12], None),
        ])
        counts = provenance.get("audit_counts") or {}
        if counts:
            st.caption("Reproducibility audit: "
                       + ", ".join(f"{k} = {v}" for k, v in counts.items()))
        separation = repo.separation_audit()
        if separation:
            st.subheader("Immutability of the registered study")
            st.caption(f"Verified against git tag `{separation['tag']}` — "
                       + ", ".join(f"{k} = {v}" for k, v in separation["counts"].items()))
            st.dataframe(pd.DataFrame(separation["checks"])[["name", "status", "detail"]],
                         width="stretch", hide_index=True)
        st.subheader("Launch")
        st.code("# research dashboard\nstreamlit run app.py --server.port 8501\n\n"
                "# this product\nstreamlit run product/app.py --server.port 8502", language="bash")


def about() -> None:
    st.title("About")
    prototype_notice()
    st.markdown(
        "### Metro Intelligent Dispatch — Decision Support System\n\n"
        "A prototype decision-support interface built on the validated simulation, forecasting "
        "and policy artefacts of a completed reinforcement-learning study of metro train "
        "induction.\n\n"
        "**What it does**\n"
        "- Presents the simulated operating state of a metro line: demand, forecast, "
        "uncertainty, availability, queues and capacity pressure.\n"
        "- Produces a dispatch recommendation from a validated policy, with an explanation of the "
        "observable state behind it.\n"
        "- Simulates scenarios forward and reports before/after consequences.\n"
        "- Compares strategies using the completed study's measured results.\n\n"
        "**What it does not do**\n"
        "- It does not control railway infrastructure or train movements.\n"
        "- It is not safety-certified and is not deployment-ready.\n"
        "- It does not retrain or modify any research artefact; research files are opened "
        "read-only.\n"
        "- It does not guarantee outcomes. Simulated consequences depend on simulator "
        "assumptions, several of which are not externally validated.\n\n"
        "**Relationship to the research**\n"
        "The completed study found **H1 = NOT SUPPORTED**: supplying the controller with "
        "calibrated forecast uncertainty produced worse passenger service than a point forecast. "
        "This product presents that finding rather than contradicting it, and separates research "
        "evidence from live demonstration everywhere the two appear together.")
    st.caption("Research dashboard: http://localhost:8501 · This product: http://localhost:8502")


# --------------------------------------------------------------------------- helpers


def _figure(stem: str) -> None:
    from product.services.research_repository import RESULTS

    png = RESULTS / RESULTS_FIGURES / f"{stem}.png"
    if png.is_file():
        st.image(str(png), width="stretch")
    else:
        unavailable(f"figure `{stem}` has not been generated")


PAGES: dict[str, list[tuple[str, Any]]] = {
    "Operations": [("Home", home), ("Network", network), ("Forecast", forecast),
                   ("Dispatch", dispatch), ("Scenario simulator", simulator)],
    "Analysis": [("Strategy comparison", strategy_comparison), ("Mechanism", mechanism),
                 ("Policy inspection", policy_inspection)],
    "Research": [("Research results", research_results), ("Statistics", statistics),
                 ("Follow-up", follow_up), ("Reproducibility", reproducibility)],
    "Info": [("About", about)],
}

#: Pages whose sidebar shows the live operating controls.
OPERATIONAL = {"Home", "Network", "Forecast", "Dispatch", "Scenario simulator",
               "Policy inspection"}

FLAT: list[tuple[str, Any]] = [entry for group in PAGES.values() for entry in group]
