"""Network model for the simulated line (M5).

Built from real data wherever the protocol allows: station ordering and inter-station running
times come from the GTFS static feed, station identity from ``mta_subway_stations``. Every
parameter that is *not* data-derived is carried with an explicit provenance label so a config
containing an ungrounded value cannot be loaded outside a tagged sensitivity sweep.

The line is modelled as a simple corridor: ``N`` stations in a fixed order, two directions,
no branching and no shared trackage. That abstraction is a property of the real line chosen
for the study (a single service on unshared track), not a convenience -- which is why it was
chosen (``docs/research_audit.md`` section F.3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.common.logging import get_logger
from src.common.paths import PROCESSED_DIR

logger = get_logger(__name__)

#: Direction 0 runs from index 0 to index N-1; direction 1 is the reverse.
DIRECTION_FORWARD = 0
DIRECTION_BACKWARD = 1
DIRECTIONS = (DIRECTION_FORWARD, DIRECTION_BACKWARD)

NETWORK_FILE = PROCESSED_DIR / "network" / "network.json"


@dataclass(frozen=True)
class Network:
    """Static description of the modelled line.

    Attributes:
        station_ids: MTA station-complex ids in travel order for direction 0.
        station_names: Human-readable names, same order.
        segment_seconds: ``segment_seconds[d][i]`` is the running time from station ``i`` to
            the next station in direction ``d``. Length ``N - 1`` in each direction.
        dwell_base_seconds: Minimum dwell at a stop, before any passenger-dependent term.
        segment_km: Distance of each segment, used by the energy model.
        min_headway_seconds: Minimum separation enforced between consecutive departures.
        turnaround_seconds: Terminal turnaround before a train can run the opposite direction.
        provenance: Parameter name -> provenance label, copied into every run manifest.
    """

    station_ids: tuple[str, ...]
    station_names: tuple[str, ...]
    segment_seconds: tuple[tuple[float, ...], tuple[float, ...]]
    dwell_base_seconds: float
    segment_km: tuple[float, ...]
    min_headway_seconds: float
    turnaround_seconds: float
    provenance: dict[str, str] = field(default_factory=dict)

    @property
    def n_stations(self) -> int:
        return len(self.station_ids)

    def index_of(self, station_id: str) -> int:
        return self.station_ids.index(station_id)

    def terminal_for(self, direction: int) -> int:
        """Index of the station where a train inducted in ``direction`` starts."""
        return 0 if direction == DIRECTION_FORWARD else self.n_stations - 1

    def next_station(self, station: int, direction: int) -> int | None:
        """Index of the next station, or ``None`` if ``station`` is the terminus."""
        step = 1 if direction == DIRECTION_FORWARD else -1
        nxt = station + step
        return nxt if 0 <= nxt < self.n_stations else None

    def segment_time(self, station: int, direction: int) -> float:
        """Running time from ``station`` to the next station in ``direction``."""
        if direction == DIRECTION_FORWARD:
            return self.segment_seconds[DIRECTION_FORWARD][station]
        # Backward segment i connects station i+1 -> i, stored at index i.
        return self.segment_seconds[DIRECTION_BACKWARD][station - 1]

    def segment_distance(self, station: int, direction: int) -> float:
        """Distance of the segment leaving ``station`` in ``direction``, in km."""
        index = station if direction == DIRECTION_FORWARD else station - 1
        return self.segment_km[index]

    def downstream_indices(self, station: int, direction: int) -> np.ndarray:
        """Station indices a passenger boarding at ``station`` could be travelling to."""
        if direction == DIRECTION_FORWARD:
            return np.arange(station + 1, self.n_stations)
        return np.arange(station - 1, -1, -1)

    def end_to_end_seconds(self, direction: int) -> float:
        """Scheduled end-to-end running time including base dwells."""
        return float(sum(self.segment_seconds[direction]) + self.dwell_base_seconds * (self.n_stations - 1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "station_ids": list(self.station_ids),
            "station_names": list(self.station_names),
            "n_stations": self.n_stations,
            "segment_seconds": [list(self.segment_seconds[0]), list(self.segment_seconds[1])],
            "segment_km": list(self.segment_km),
            "dwell_base_seconds": self.dwell_base_seconds,
            "min_headway_seconds": self.min_headway_seconds,
            "turnaround_seconds": self.turnaround_seconds,
            "end_to_end_seconds": {str(d): self.end_to_end_seconds(d) for d in DIRECTIONS},
            "provenance": self.provenance,
        }

    def save(self, path: Path = NETWORK_FILE) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path = NETWORK_FILE) -> "Network":
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: python -m src.simulator.build_network"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            station_ids=tuple(payload["station_ids"]),
            station_names=tuple(payload["station_names"]),
            segment_seconds=(tuple(payload["segment_seconds"][0]), tuple(payload["segment_seconds"][1])),
            dwell_base_seconds=float(payload["dwell_base_seconds"]),
            segment_km=tuple(payload["segment_km"]),
            min_headway_seconds=float(payload["min_headway_seconds"]),
            turnaround_seconds=float(payload["turnaround_seconds"]),
            provenance=payload.get("provenance", {}),
        )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres.

    Used only for segment distances in the energy model. It understates real track length
    (track is not a geodesic), which is recorded as a limitation of the energy term -- a term
    already flagged ``unresolved`` because no citable energy-per-train-km figure has been
    sourced.
    """
    radius = 6371.0088
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = np.radians(lon2 - lon1)
    a = np.sin(d_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2) ** 2
    return float(2 * radius * np.arcsin(np.sqrt(a)))


def build_toy_network(n_stations: int = 10, segment_seconds: float = 120.0) -> Network:
    """A synthetic corridor for unit tests and sanity environments.

    Never used for a reported result. Keeping the toy network structurally identical to the
    real one means the sanity tests exercise the same code paths as the experiments.
    """
    forward = tuple(float(segment_seconds) for _ in range(n_stations - 1))
    return Network(
        station_ids=tuple(f"T{i:02d}" for i in range(n_stations)),
        station_names=tuple(f"Toy {i}" for i in range(n_stations)),
        segment_seconds=(forward, forward),
        dwell_base_seconds=30.0,
        segment_km=tuple(1.0 for _ in range(n_stations - 1)),
        min_headway_seconds=150.0,
        turnaround_seconds=300.0,
        provenance={"all": "simulation_assumption:toy network for tests only"},
    )
