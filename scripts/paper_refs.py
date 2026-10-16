"""Bibliography verified against publisher-deposited Crossref and arXiv metadata.

Missing fields are omitted rather than invented. Full source records, retrieval dates and
corrections are retained in results/reference_verification.json. Citation keys remain stable
when a corrected publication year differs from the year embedded in an old key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path


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
    citation_author: str = ""

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


# Each field is backed by Crossref publisher metadata or arXiv citation metadata.
# Retrieval evidence and corrections are retained in results/reference_verification.json.
REFERENCES = tuple(Reference(**row) for row in json.loads(
    (Path(__file__).resolve().parents[1] / "research_spec" / "bibliography.json").read_text()))

BY_KEY = {reference.key: reference for reference in REFERENCES}


def cite(*keys: str) -> str:
    """Author–year citation; fail loudly on an unknown key."""
    for key in keys:
        if key not in BY_KEY:
            raise KeyError(f"citation {key!r} is not in the verified bibliography")
    return "(" + "; ".join(f"{BY_KEY[key].citation_author}, {BY_KEY[key].year}" for key in keys) + ")"


def counts() -> dict[str, int]:
    full = sum(1 for r in REFERENCES if r.status == "full")
    return {"total": len(REFERENCES), "fully_verified": full,
            "needing_confirmation": len(REFERENCES) - full}
