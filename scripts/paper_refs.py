"""Verified bibliography for the final manuscript.

**Zero-fabrication policy.** Every entry below was located by literature search during manuscript
preparation. Each carries an explicit `status`:

- ``full``    — authors, title, venue, year and pages were all returned by the search and are
                reproduced here.
- ``partial`` — title and venue were verified; one or more of authors, year, volume or pages was
                not returned by the search and is marked for confirmation against the publisher
                record before submission.

No DOI is stated unless it was actually observed. No page range, volume or author list is
reconstructed from memory. Where a field is unknown the manuscript prints
``[REFERENCE VERIFICATION REQUIRED]`` rather than a plausible guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Reference:
    key: str
    authors: str
    year: str
    title: str
    venue: str
    detail: str = ""
    url: str = ""
    status: str = "partial"
    note: str = ""
    topics: tuple[str, ...] = field(default_factory=tuple)

    def formatted(self) -> str:
        parts = [f"{self.authors} ({self.year}). {self.title}. *{self.venue}*"]
        if self.detail:
            parts.append(f", {self.detail}")
        parts.append(".")
        line = "".join(parts)
        if self.url:
            line += f" {self.url}"
        if self.status != "full":
            line += "  \n  `[REFERENCE VERIFICATION REQUIRED: " + (
                self.note or "confirm authors, volume and pages against the publisher record")
            line += "]`"
        return line


REFERENCES: tuple[Reference, ...] = (
    # ---------------------------------------------------------------- railway scheduling
    Reference(
        key="cacchiani2014",
        authors="Cacchiani, V., Huisman, D., Kidd, M., Kroon, L., Toth, P., Veelenturf, L., "
                "Wagenaar, J.",
        year="2014",
        title="An overview of recovery models and algorithms for real-time railway rescheduling",
        venue="Transportation Research Part B: Methodological",
        detail="63, 15–37",
        url="https://www.sciencedirect.com/science/article/abs/pii/S0191261514000198",
        status="full",
        topics=("rescheduling", "optimisation"),
    ),
    Reference(
        key="semrov2016",
        authors="Šemrov, D., Marsetič, R., Žura, M., Todorovski, L., Srdic, A.",
        year="2016",
        title="Reinforcement learning approach for train rescheduling on a single-track railway",
        venue="Transportation Research Part B: Methodological",
        detail="86, 250–267",
        url="https://www.sciencedirect.com/science/article/abs/pii/S0191261516000084",
        status="full",
        topics=("rl", "rescheduling"),
    ),
    Reference(
        key="yin2016",
        authors="Yin, J., Tang, T., Yang, L., Gao, Z., Ran, B.",
        year="2016",
        title="Energy-efficient metro train rescheduling with uncertain time-variant passenger "
              "demands: An approximate dynamic programming approach",
        venue="Transportation Research Part B: Methodological",
        detail="91, 178–210",
        url="https://www.sciencedirect.com/science/article/abs/pii/S0191261515302241",
        status="full",
        topics=("metro", "uncertainty", "scheduling"),
    ),
    Reference(
        key="yin2017",
        authors="Yin, J., Yang, L., Tang, T., Gao, Z., Ran, B.",
        year="2017",
        title="Dynamic passenger demand oriented metro train scheduling with energy-efficiency "
              "and waiting time minimization: Mixed-integer linear programming approaches",
        venue="Transportation Research Part B: Methodological",
        detail="97, 182–213",
        url="https://www.sciencedirect.com/science/article/abs/pii/S0191261516305082",
        status="full",
        topics=("metro", "milp", "scheduling"),
    ),
    Reference(
        key="trs_survey",
        authors="[REFERENCE VERIFICATION REQUIRED — author list not returned by search]",
        year="1998",
        title="A Survey of Optimization Models for Train Routing and Scheduling",
        venue="Transportation Science",
        detail="32(4), 380",
        url="https://pubsonline.informs.org/doi/10.1287/trsc.32.4.380",
        note="confirm authors and year",
        topics=("optimisation",),
    ),
    Reference(
        key="inflow_control",
        authors="[REFERENCE VERIFICATION REQUIRED — author list not returned by search]",
        year="2018",
        title="Reinforcement learning approach for coordinated passenger inflow control of urban "
              "rail transit in peak hours",
        venue="Transportation Research Part C: Emerging Technologies",
        url="https://www.sciencedirect.com/science/article/pii/S0968090X18300111",
        note="confirm authors, volume and pages",
        topics=("rl", "metro"),
    ),
    Reference(
        key="multitask_resched",
        authors="[REFERENCE VERIFICATION REQUIRED — author list not returned by search]",
        year="2024",
        title="A multi-task deep reinforcement learning approach to real-time railway train "
              "rescheduling",
        venue="Transportation Research Part E: Logistics and Transportation Review",
        url="https://www.sciencedirect.com/science/article/abs/pii/S1366554524004915",
        note="confirm authors, volume and pages",
        topics=("rl", "rescheduling"),
    ),
    Reference(
        key="madrl_shortturn",
        authors="[REFERENCE VERIFICATION REQUIRED — author list not returned by search]",
        year="2024",
        title="Adaptive rescheduling of rail transit services with short-turnings under "
              "disruptions via a multi-agent deep reinforcement learning approach",
        venue="Transportation Research Part B: Methodological",
        url="https://www.sciencedirect.com/science/article/abs/pii/S0191261524001917",
        note="confirm authors, volume and pages",
        topics=("rl", "disruption"),
    ),

    # ---------------------------------------------------------------- reinforcement learning
    Reference(
        key="schulman2017",
        authors="Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O.",
        year="2017",
        title="Proximal Policy Optimization Algorithms",
        venue="arXiv preprint",
        detail="arXiv:1707.06347",
        url="https://arxiv.org/abs/1707.06347",
        status="full",
        topics=("rl", "ppo"),
    ),
    Reference(
        key="henderson2018",
        authors="Henderson, P., et al.",
        year="2018",
        title="Deep Reinforcement Learning that Matters",
        venue="arXiv preprint / AAAI Conference on Artificial Intelligence",
        detail="arXiv:1709.06560",
        url="https://arxiv.org/abs/1709.06560",
        note="first author and title verified; confirm the full author list and AAAI page range",
        topics=("methodology", "reproducibility"),
    ),
    Reference(
        key="colas2018",
        authors="Colas, C., et al.",
        year="2018",
        title="How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning "
              "Experiments",
        venue="arXiv preprint",
        detail="arXiv:1806.08295",
        url="https://arxiv.org/pdf/1806.08295",
        note="title and identifier verified; confirm the full author list",
        topics=("methodology", "statistics"),
    ),

    # ---------------------------------------------------------------- uncertainty and forecasting
    Reference(
        key="uq_review",
        authors="[REFERENCE VERIFICATION REQUIRED — author list not returned by search]",
        year="2022",
        title="A Review of Uncertainty for Deep Reinforcement Learning",
        venue="Proceedings of the AAAI Conference on Artificial Intelligence and Interactive "
              "Digital Entertainment",
        url="https://www.researchgate.net/publication/364395789",
        note="confirm authors, venue and year",
        topics=("uncertainty", "rl"),
    ),
    Reference(
        key="clements2019",
        authors="[REFERENCE VERIFICATION REQUIRED — author list not returned by search]",
        year="2019",
        title="Estimating Risk and Uncertainty in Deep Reinforcement Learning",
        venue="arXiv preprint",
        detail="arXiv:1905.09638",
        url="https://arxiv.org/pdf/1905.09638",
        note="confirm authors and any peer-reviewed version",
        topics=("uncertainty", "rl"),
    ),
    Reference(
        key="uq_demand_gnn",
        authors="[REFERENCE VERIFICATION REQUIRED — author list not returned by search]",
        year="2023",
        title="Uncertainty Quantification of Spatiotemporal Travel Demand with Probabilistic "
              "Graph Neural Networks",
        venue="arXiv preprint",
        detail="arXiv:2303.04040",
        url="https://arxiv.org/pdf/2303.04040",
        note="confirm authors and any peer-reviewed version",
        topics=("forecasting", "uncertainty"),
    ),
    Reference(
        key="uq_ridesourcing",
        authors="[REFERENCE VERIFICATION REQUIRED — author list not returned by search]",
        year="2025",
        title="Uncertainty quantification for joint demand prediction of multi-modal "
              "ride-sourcing services using spatiotemporal Mixture-of-Expert neural network",
        venue="Transportation Research Part C: Emerging Technologies",
        url="https://www.sciencedirect.com/science/article/abs/pii/S0968090X2500511X",
        note="confirm authors, volume and pages",
        topics=("forecasting", "uncertainty"),
    ),

    # ---------------------------------------------------------------- convex loss under uncertainty
    Reference(
        key="arrow1951",
        authors="Arrow, K. J., Harris, T., Marschak, J.",
        year="1951",
        title="Optimal Inventory Policy",
        venue="Econometrica",
        detail="19(3), 250–272",
        url="https://www.econometricsociety.org/publications/econometrica/1951/07/01/optimal-inventory-policy",
        status="full",
        topics=("newsvendor", "convexity"),
    ),
    Reference(
        key="vanmieghem2002",
        authors="Van Mieghem, J. A., Rudi, N.",
        year="2002",
        title="Newsvendor Networks: Inventory Management and Capacity Investment with "
              "Discretionary Activities",
        venue="Manufacturing & Service Operations Management",
        detail="4(4)",
        url="https://www.kellogg.northwestern.edu/faculty/vanmieghem/htm/pubs/vanmieghem_rudi_2002_newsnet_msom.pdf",
        note="confirm page range",
        topics=("newsvendor", "capacity"),
    ),

    # ---------------------------------------------------------------- digital twins
    Reference(
        key="dt_railway_review",
        authors="[REFERENCE VERIFICATION REQUIRED — author list not returned by search]",
        year="2025",
        title="Digital twin in railway industry: a bibliometric analysis and systematic review",
        venue="Digital Transportation and Safety (Taylor & Francis)",
        url="https://www.tandfonline.com/doi/full/10.1080/27525783.2025.2533858",
        note="confirm authors, journal title, volume and pages",
        topics=("digital twin",),
    ),
)

BY_KEY = {reference.key: reference for reference in REFERENCES}


def cite(*keys: str) -> str:
    """Inline citation marker, e.g. ``[schulman2017]``. Fails loudly on an unknown key."""
    for key in keys:
        if key not in BY_KEY:
            raise KeyError(f"citation {key!r} is not in the verified bibliography")
    return "[" + "; ".join(keys) + "]"


def counts() -> dict[str, int]:
    full = sum(1 for r in REFERENCES if r.status == "full")
    return {"total": len(REFERENCES), "fully_verified": full,
            "needing_confirmation": len(REFERENCES) - full}
