"""M17 — interactive research dashboard.

    streamlit run app.py

A demonstration layer over the validated M0–M16 artefacts. It loads the fitted forecaster, the
calibrated simulator, the frozen RL environment and the trained policies, and shows what they do.
It does not retrain, retune, or recompute anything: every action comes from the actual policy,
every metric from the actual simulator, every reward component from the actual reward function.

Nothing here is a confirmatory result. A single interactive episode is one draw from a
distribution the paper characterises over many seeds, and the interface says so wherever a number
could be mistaken for evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st

from src.demo import SCENARIOS, DemoSession, available_policies
from src.demo import research_views
from src.demo.session import MissingArtifactError, scenario_label

st.set_page_config(page_title="Metro RL — research demonstration",
                   page_icon="🚈", layout="wide")

DEMO_BANNER = (
    "**Interactive demonstration — not confirmatory statistical results.** "
    "One episode is a single draw. The paper's claims rest on many seeds, analysed with the "
    "registered tests; nothing on this page is evidence for or against a hypothesis."
)


# --------------------------------------------------------------------------- session state


def _session_key(method: str, scenario: str, seed: int, train_seed: int | None) -> str:
    return f"{method}|{scenario}|{seed}|{train_seed}"


def get_session(method: str, scenario: str, seed: int, train_seed: int | None) -> DemoSession:
    """One live session per (method, scenario, seed), rebuilt only when the selection changes."""
    key = _session_key(method, scenario, seed, train_seed)
    if st.session_state.get("session_key") != key:
        with st.spinner(f"Loading {method} on {scenario}…"):
            st.session_state.session = DemoSession(
                method, scenario, eval_seed=seed, train_seed=train_seed)
        st.session_state.session_key = key
    return st.session_state.session


# --------------------------------------------------------------------------- sidebar


def sidebar() -> tuple[str, str, int, int | None, int]:
    st.sidebar.title("Metro RL")
    st.sidebar.caption("Uncertainty-aware train induction — research demonstration")

    options = available_policies()
    usable = [o for o in options if o.available]
    missing = [o for o in options if not o.available]

    if not usable:
        st.sidebar.error("No policy artefacts found. Run the research pipeline first.")
        st.stop()

    scenario = st.sidebar.selectbox(
        "Scenario", SCENARIOS, format_func=scenario_label,
        help="Scenario families share a base world at a given seed (amendment A-03), so a "
             "scenario contrast measures the scenario rather than a different random draw.")

    labels = {o.method_id: o.label for o in usable}
    method = st.sidebar.selectbox(
        "Policy", [o.method_id for o in usable], format_func=lambda m: labels[m])

    chosen = next(o for o in usable if o.method_id == method)
    train_seed = None
    if chosen.needs_train_seed:
        train_seed = st.sidebar.selectbox(
            "Training seed", chosen.train_seeds,
            help="Which trained policy to demonstrate. Each training seed is a separate policy; "
                 "the paper treats the training seed as the unit of analysis.")

    seed = st.sidebar.number_input(
        "Evaluation seed", min_value=1, max_value=99_999, value=7, step=1,
        help="The same seed reproduces the same episode exactly.")

    st.sidebar.markdown("---")
    controls = st.sidebar.columns(3)
    if controls[0].button("Step", use_container_width=True):
        st.session_state.pending_steps = 1
    if controls[1].button("Run", use_container_width=True):
        st.session_state.pending_steps = int(st.session_state.get("run_steps", 60))
    if controls[2].button("Reset", use_container_width=True):
        st.session_state.pending_reset = True

    st.session_state.run_steps = st.sidebar.slider(
        "Epochs per Run", min_value=6, max_value=480, value=60, step=6)

    if missing:
        st.sidebar.markdown("---")
        st.sidebar.caption("Not available yet:")
        for option in missing:
            st.sidebar.caption(f"• {option.label} — {option.reason}")

    return method, scenario, int(seed), train_seed, int(st.session_state.run_steps)


# --------------------------------------------------------------------------- panels


def network_panel(session: DemoSession) -> None:
    state = session.network_state()
    st.subheader("Network")

    top = st.columns(4)
    top[0].metric("Clock", state["clock"])
    top[1].metric("Epoch", f"{state['epoch']} / {state['n_decision_epochs']}")
    top[2].metric("Trains in service", f"{state['trains_in_service']} / {state['fleet_size']}")
    top[3].metric("Total queue", f"{sum(s['queue'] for s in state['stations']):,.0f}")

    stations = pd.DataFrame(state["stations"])
    trains = pd.DataFrame(state["trains"])

    live = trains[trains["in_service"]]
    per_station = (live.groupby("station").size() if len(live)
                   else pd.Series(dtype=int)).reindex(stations["index"], fill_value=0)

    chart = pd.DataFrame({
        "station": stations["name"],
        "queue (passengers)": stations["queue"].to_numpy(),
        "trains present": per_station.to_numpy(),
    }).set_index("station")

    st.bar_chart(chart[["queue (passengers)"]], height=260)

    left, right = st.columns([3, 2])
    with left:
        st.caption("Stations — queue and longest current wait")
        st.dataframe(
            stations.assign(**{
                "oldest wait (min)": (stations["oldest_wait_s"] / 60.0).round(1),
                "queue": stations["queue"].round(0),
            })[["station_id", "name", "queue", "oldest wait (min)"]]
            .rename(columns={"station_id": "id", "name": "station"}),
            use_container_width=True, hide_index=True, height=300)
    with right:
        st.caption("Trains in service")
        if len(live):
            st.dataframe(
                live.assign(**{
                    "occupancy": (live["occupancy_fraction"] * 100).round(0).astype(int).astype(str) + "%",
                    "dir": np.where(live["direction"] == 0, "→", "←"),
                })[["train_id", "status", "station_name", "dir", "occupancy"]]
                .rename(columns={"train_id": "train", "station_name": "at"}),
                use_container_width=True, hide_index=True, height=300)
        else:
            st.info("No trains in service yet.")


def forecast_panel(session: DemoSession) -> None:
    forecast = session.forecast_state()
    st.subheader("Demand forecast")

    if not forecast["carries_uncertainty"]:
        st.info(f"This arm (`{forecast['arm']}`) carries no uncertainty channel: its sigma inputs "
                "are zeroed. That is the point-forecast condition, not a missing model.")

    columns = st.columns(3)
    columns[0].metric("Forecast mean, line, h+1", f"{forecast['mu_line_h1']:,.0f}")
    if forecast["carries_uncertainty"]:
        columns[1].metric("Forecast sigma, line, h+1", f"±{forecast['sigma_line_h1']:,.0f}")
        low = forecast["mu_line_h1"] - 1.2816 * forecast["sigma_line_h1"]
        high = forecast["mu_line_h1"] + 1.2816 * forecast["sigma_line_h1"]
        columns[2].metric("80% interval", f"{low:,.0f} – {high:,.0f}")

    mu = np.asarray(forecast["mu_per_station"], dtype=float)
    sigma = np.asarray(forecast["sigma_per_station"], dtype=float)
    if mu.size:
        names = [s["name"] for s in session.network_state()["stations"]]
        frame = pd.DataFrame({"station": names, "forecast mean": mu})
        if forecast["carries_uncertainty"] and sigma.size == mu.size:
            frame["lower (q10)"] = mu - 1.2816 * sigma
            frame["upper (q90)"] = mu + 1.2816 * sigma
        st.line_chart(frame.set_index("station"), height=240)

    st.caption(forecast["interval_note"] + ". This is the forecast the policy is looking at, in "
               "passengers, taken from the unscaled observation.")

    realised = session.realised_demand()
    if realised:
        st.caption("**Realised demand** — shown only for epochs already stepped. It is an "
                   "outcome, never an input: the policy had already acted before it existed.")
        st.line_chart(
            pd.DataFrame(realised).set_index("clock")[["arrived", "boarded", "left_behind"]],
            height=200)


def decision_panel(session: DemoSession) -> None:
    st.subheader("Decision")
    if not session.history:
        st.info("Step the simulation to see the policy's decision.")
        return

    record = session.history[-1]
    columns = st.columns(4)
    columns[0].metric("Action taken", f"{record.executed_action} trainset(s)")
    columns[1].metric("Requested", f"{record.requested_action}")
    columns[2].metric("Feasible", ", ".join(str(a) for a in record.feasible_actions))
    columns[3].metric("Reward this epoch", f"{record.reward:,.2f}")

    mask_text = " ".join(
        f"`{index}`{'✅' if allowed else '🚫'}" for index, allowed in enumerate(record.action_mask))
    st.caption(f"Action mask: {mask_text} — constraints are enforced structurally by masking, "
               "not by reward penalties (protocol §3.4/6.3). An infeasible induction is refused, "
               "never priced.")

    if record.requested_action != record.executed_action:
        st.warning(f"The policy asked for {record.requested_action} but only "
                   f"{record.executed_action} was feasible. Clipping is recorded for every "
                   "method, so a method limited by infrastructure is distinguishable from one "
                   "limited by its own logic.")


def metrics_panel(session: DemoSession) -> None:
    metrics = session.metrics()
    st.subheader("Outcomes")

    primary = metrics["primary_endpoint"]
    summary = metrics["summary"]

    top = st.columns(4)
    top[0].metric("Left behind (primary endpoint)", f"{primary['value'] * 100:.2f}%")
    top[1].metric("Cumulative reward", f"{metrics['cumulative_reward']:,.1f}")
    top[2].metric("Mean wait", f"{summary.get('waiting_time_mean_s', float('nan')) / 60:,.1f} min")
    top[3].metric("p95 wait", f"{summary.get('waiting_time_p95_s', float('nan')) / 60:,.1f} min")

    second = st.columns(4)
    second[0].metric("Distinct left behind", f"{summary.get('left_behind', 0):,.0f}")
    second[1].metric("Inductions", f"{summary.get('inductions', 0):,.0f}")
    second[2].metric("Energy (units)", f"{summary.get('energy_units', 0):,.1f}")
    second[3].metric("Cost (units)", f"{summary.get('cost_units', 0):,.1f}")

    st.caption(f"Primary endpoint: {primary['note']}. Energy and cost are in equivalent units, "
               "not physical ones — assumptions A15 and A16 are unresolved, so both are embargoed "
               "from headline claims.")

    components = metrics["current_components"]
    if components:
        st.caption("Reward decomposition for the last epoch, from the actual reward function")
        frame = pd.DataFrame({
            "component": ["Waiting (passenger-min)", "Left behind (passengers)",
                          "Energy (units)", "Cost (units)"],
            "value": [components.get("W_minutes", 0.0), components.get("U_passengers", 0.0),
                      components.get("E_units", 0.0), components.get("C_units", 0.0)],
        })
        st.dataframe(frame, use_container_width=True, hide_index=True)

    if session.history:
        history = pd.DataFrame([
            {"clock": r.clock, "reward": r.reward,
             "cumulative": total}
            for r, total in zip(session.history, np.cumsum([r.reward for r in session.history]))
        ]).set_index("clock")
        st.line_chart(history[["cumulative"]], height=200)


def traceability_panel(session: DemoSession) -> None:
    trace = session.traceability()
    with st.expander("Traceability — what produced this run"):
        left, right = st.columns(2)
        left.markdown(
            f"**Policy** `{trace['method']}`  \n"
            f"**Scenario** `{trace['scenario']}`  \n"
            f"**Evaluation seed** `{trace['eval_seed']}`  \n"
            f"**Training seed** `{trace['train_seed']}`  \n"
            f"**Reward weights** `{trace['weight_id']}`")
        right.markdown(
            f"**Git commit** `{(trace['git_commit'] or 'unknown')[:12]}`"
            + (" *(working tree dirty)*" if trace["git_dirty"] else "") + "  \n"
            f"**Stations** {trace['n_stations']}  \n"
            f"**Horizon** {trace['horizon_epochs']} epochs "
            f"× {trace['decision_interval_s']:.0f}s  \n"
            f"**Fleet** {trace['fleet_size']} trainsets")
        forecast = trace.get("forecast_model") or {}
        if forecast:
            st.markdown(
                f"**Forecast model** — {forecast.get('source')}; calibration factor "
                f"{forecast.get('calibration_factor'):.4f}; refit agreement "
                f"{forecast.get('refit_agreement')}")
        st.json(trace, expanded=False)


# --------------------------------------------------------------------------- main


def research_view() -> None:
    """The research dashboard: stored artefacts, read-only.

    Separated from the live demo because the two show different kinds of thing. Everything here
    is a result the study actually produced; everything in the demo tab is a fresh simulation run
    for illustration. Mixing them in one view is how a demo number ends up being read as a
    finding.
    """

    group = st.sidebar.radio("Section", list(research_views.PAGES), key="nav_group")
    pages = research_views.PAGES[group]
    name = st.sidebar.radio("Page", [n for n, _ in pages], key=f"nav_page_{group}")
    render = dict(pages)[name]
    st.sidebar.markdown("---")
    st.sidebar.caption("Every page reads stored artefacts read-only. No interaction here can "
                       "alter a research result.")
    try:
        render()
    except Exception as error:  # noqa: BLE001 - a broken page must not take the app down
        st.error(f"This page could not render: {error!r}")


def demo_view() -> None:
    method, scenario, seed, train_seed, _ = sidebar()

    try:
        session = get_session(method, scenario, seed, train_seed)
    except MissingArtifactError as error:
        st.error(str(error))
        st.stop()

    if st.session_state.pop("pending_reset", False):
        session.reset()
    pending = st.session_state.pop("pending_steps", 0)
    if pending:
        with st.spinner(f"Simulating {pending} epoch(s)…"):
            session.run(pending)

    st.title("Real-time metro train induction")
    st.caption(f"{session.option.label} · {scenario_label(scenario)} · evaluation seed {seed}"
               + (f" · training seed {train_seed}" if train_seed is not None else ""))
    st.warning(DEMO_BANNER)

    if session.terminated:
        st.success("Episode complete. Press Reset to run it again, or change the seed.")

    network_panel(session)
    st.markdown("---")
    left, right = st.columns([3, 2])
    with left:
        forecast_panel(session)
    with right:
        decision_panel(session)
    st.markdown("---")
    metrics_panel(session)
    traceability_panel(session)

    if st.button("Save this run to results/demo/"):
        path = session.save()
        st.success(f"Wrote {path}")


def main() -> None:
    # `set_page_config` is already called at import time and may only run once per session.
    mode = st.sidebar.radio(
        "View",
        ("Research results", "Live demonstration"),
        help=("Research results are the stored outputs of the registered study. The live "
              "demonstration runs a fresh episode for illustration and writes only to "
              "results/demo/."),
    )
    st.sidebar.markdown("---")
    if mode == "Research results":
        research_view()
    else:
        demo_view()


if __name__ == "__main__":
    main()
