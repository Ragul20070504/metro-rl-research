"""METRO INTELLIGENT DISPATCH — decision-support prototype.

    streamlit run product/app.py --server.port 8502

A separate application from the research dashboard on port 8501. This one is operator-facing:
it presents the simulated state of a metro line, produces dispatch recommendations from validated
policies, and lets an operator explore scenarios and compare strategies.

It reads the completed study's artefacts and never writes to them. The research dashboard, the
registered results and the frozen registry are untouched by anything in this package.

**This prototype does not control railway infrastructure.** Every recommendation requires
operator approval.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from product.services.research_repository import assert_read_only  # noqa: E402
from product.ui import pages as product_pages  # noqa: E402

st.set_page_config(page_title="Metro Intelligent Dispatch",
                   page_icon="🚇", layout="wide",
                   initial_sidebar_state="expanded")

#: The panel-demonstration route, in the order a presenter should walk it.
PANEL_DEMO = [
    ("Home", "The problem and the live operating state"),
    ("Network", "Where passengers are waiting"),
    ("Forecast", "What demand is expected next"),
    ("Dispatch", "The recommendation and why"),
    ("Scenario simulator", "What happens if we act"),
    ("Strategy comparison", "How strategies compare"),
    ("Research results", "What the study concluded"),
    ("Mechanism", "Why the uncertainty-aware controller behaved that way"),
    ("About", "Limitations and next steps"),
]


def _page_index() -> dict[str, tuple[str, object]]:
    return {name: (group, render)
            for group, entries in product_pages.PAGES.items()
            for name, render in entries}


def main() -> None:
    st.sidebar.markdown("## 🚇 METRO DISPATCH")
    st.sidebar.caption("Decision Support System · prototype")

    index = _page_index()
    demo_mode = st.sidebar.toggle(
        "Panel demo mode", value=False, key="panel_demo",
        help="A guided route through the demonstration, in presentation order.")

    if demo_mode:
        names = [name for name, _ in PANEL_DEMO if name in index]
        captions = dict(PANEL_DEMO)
        step = st.sidebar.radio("Demonstration step", names,
                                format_func=lambda n: f"{names.index(n) + 1}. {n}",
                                key="demo_step")
        st.sidebar.info(captions.get(step, ""))
        position = names.index(step) + 1
        st.progress(position / len(names),
                    text=f"Panel demonstration — step {position} of {len(names)}: {step}")
        selected = step
    else:
        group = st.sidebar.radio("Section", list(product_pages.PAGES), key="nav_group")
        selected = st.sidebar.radio(
            "Page", [name for name, _ in product_pages.PAGES[group]],
            key=f"nav_page_{group}")

    st.sidebar.markdown("---")
    if selected in product_pages.OPERATIONAL:
        product_pages.operations_controls()
        st.sidebar.markdown("---")

    # Cheap, and it is the guarantee the whole product rests on: if anything under this package
    # ever wrote to a research artefact, it would show here rather than in a later audit.
    violations = assert_read_only()
    if violations:
        st.sidebar.error("Research artefacts changed during this session: "
                         + ", ".join(violations))
    else:
        st.sidebar.success("Research artefacts: read-only, unchanged")
    st.sidebar.caption("Research dashboard runs separately on port 8501.")

    render = index[selected][1]
    try:
        render()
    except Exception as error:  # noqa: BLE001 - a broken page must not take the app down
        st.error(f"**This page could not be rendered.** {type(error).__name__}: {error}")
        st.caption("No substitute values are shown. Other pages remain usable.")


if __name__ == "__main__":
    main()
