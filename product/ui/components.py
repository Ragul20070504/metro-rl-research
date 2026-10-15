"""Shared UI pieces for the dispatch prototype.

The evidence badge is the most important component here. Every panel that shows a number declares
where it came from, because the one thing this product must never do is let a simulated
demonstration value be read as a research result, or an incomplete follow-up be read as evidence.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from product.services.research_repository import (
    CONFIRMATORY, DESCRIPTIVE, EXPLORATORY, FOLLOW_UP, PILOT, PRODUCT,
)

BADGE_STYLE: dict[str, tuple[str, str]] = {
    CONFIRMATORY: ("#1f5d3a", "Registered research result, fixed before the data were seen."),
    EXPLORATORY: ("#7a5c1e", "Hypothesis-generating research result. Not a registered claim."),
    DESCRIPTIVE: ("#4a4a4a", "Descriptive. No hypothesis test is defined for this quantity."),
    FOLLOW_UP: ("#5a3d8a", "Stopped before any comparison was possible. Not evidence."),
    PILOT: ("#7a1f1f", "Discarded by protocol. Never presented as a final result."),
    PRODUCT: ("#1a4971", "Live product simulation — one simulated draw, not a research result."),
}

PRESSURE_COLOUR = {"LOW": "#2f855a", "MODERATE": "#b7791f", "HIGH": "#c53030",
                   "NOT AVAILABLE": "#718096"}


def badge(kind: str) -> None:
    colour, meaning = BADGE_STYLE.get(kind, ("#4a4a4a", ""))
    st.markdown(
        f"<div style='display:inline-block;background:{colour};color:#fff;padding:3px 12px;"
        f"border-radius:11px;font-size:0.72rem;font-weight:700;letter-spacing:0.05em'>{kind}</div>"
        f"<div style='font-size:0.72rem;opacity:0.7;margin:5px 0 10px'>{meaning}</div>",
        unsafe_allow_html=True)


def status_chip(label: str, value: str) -> str:
    colour = PRESSURE_COLOUR.get(value, "#4a5568")
    return (f"<span style='background:{colour};color:#fff;padding:2px 10px;border-radius:9px;"
            f"font-size:0.75rem;font-weight:700'>{label}: {value}</span>")


def metric_row(items: list[tuple[str, Any, str | None]]) -> None:
    """A row of metric cards. `(label, value, help)` per card."""
    columns = st.columns(len(items))
    for column, (label, value, helptext) in zip(columns, items):
        column.metric(label, value, help=helptext)


def prototype_notice() -> None:
    st.warning(
        "**Decision-support prototype.** This system does not control railway infrastructure or "
        "train movements. Every recommendation requires operator approval. It is simulation-based, "
        "not safety-certified, and not deployment-ready.")


def unavailable(what: str, reason: str = "") -> None:
    """State what is missing. Never substitute a value."""
    st.info(f"**NOT AVAILABLE** — {what}." + (f" {reason}" if reason else ""))


def guard(section: str):
    """Context manager rendering an error instead of crashing the page.

    A demonstration that dies on one missing artefact is worse than one that says which panel
    could not be drawn and carries on.
    """
    from contextlib import contextmanager

    @contextmanager
    def _guard():
        try:
            yield
        except Exception as error:  # noqa: BLE001 - surfaced to the operator by design
            st.error(f"**{section} could not be shown.** {type(error).__name__}: {error}")

    return _guard()


def section(title: str, evidence: str, description: str = "") -> None:
    st.subheader(title)
    badge(evidence)
    if description:
        st.caption(description)


def research_vs_demo_notice() -> None:
    st.info(
        "**Research results and live simulation are never mixed.** Panels badged "
        f"`{CONFIRMATORY}` or `{EXPLORATORY}` are read from the frozen study artefacts. Panels "
        f"badged `{PRODUCT}` are produced live by this prototype and are a single simulated draw.")
