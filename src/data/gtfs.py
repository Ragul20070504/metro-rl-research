"""Parse the MTA subway GTFS static feed for the modelled line.

Serves two milestones. M3 needs the ordering of stations along the line so the forecaster can
express spatial structure; M5 needs scheduled inter-station running times, dwell times, trip
counts and headways as the simulator's real-data-backed parameters, plus the timetable itself
for the ``fixed_timetable`` baseline.

Two GTFS details that routinely cause silent errors, handled explicitly here:

* **Times past midnight.** GTFS encodes a trip continuing past midnight as ``25:14:00``.
  Parsing that with a normal time parser either fails or silently wraps to 01:14, corrupting
  overnight running times. Times are parsed to seconds-after-service-midnight instead.
* **Directional stop ids.** NYC subway stop ids carry a direction suffix (``L01N``, ``L01S``)
  whose ``parent_station`` is the physical stop (``L01``). Joining on the suffixed id against
  any other MTA dataset silently matches nothing.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.common.paths import RAW_DIR

logger = get_logger(__name__)

GTFS_ZIP = RAW_DIR / "mta_gtfs_static" / "gtfs_subway.zip"

#: GTFS tables this project reads. Anything else in the feed is ignored.
_TABLES = ("routes.txt", "trips.txt", "stop_times.txt", "stops.txt", "calendar.txt")


def gtfs_time_to_seconds(series: pd.Series) -> pd.Series:
    """Convert ``HH:MM:SS`` GTFS times to seconds after service midnight.

    Hours may exceed 23 for trips continuing past midnight; that is valid GTFS and must be
    preserved, not wrapped.
    """
    parts = series.astype(str).str.strip().str.split(":", expand=True)
    if parts.shape[1] < 3:
        raise ValueError("GTFS time column is not in HH:MM:SS form")
    hours = pd.to_numeric(parts[0], errors="coerce")
    minutes = pd.to_numeric(parts[1], errors="coerce")
    seconds = pd.to_numeric(parts[2], errors="coerce")
    return hours * 3600 + minutes * 60 + seconds


@lru_cache(maxsize=1)
def load_gtfs_tables(zip_path: str | None = None) -> dict[str, pd.DataFrame]:
    """Read the required GTFS tables out of the archive.

    Cached because ``stop_times.txt`` for the whole subway system is large and several
    callers need it.
    """
    path = Path(zip_path) if zip_path else GTFS_ZIP
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.data.download --only mta_gtfs_static"
        )
    tables: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(path) as archive:
        available = set(archive.namelist())
        for name in _TABLES:
            if name not in available:
                logger.warning("GTFS archive has no %s", name)
                continue
            with archive.open(name) as handle:
                tables[name] = pd.read_csv(BytesIO(handle.read()), dtype=str)
    logger.info("GTFS tables loaded: %s", {k: len(v) for k, v in tables.items()})
    return tables


@dataclass(frozen=True)
class LinePattern:
    """The dominant stopping pattern for one direction of a route.

    Attributes:
        direction_id: GTFS direction (``"0"`` / ``"1"``).
        parent_stations: Physical stop ids in travel order.
        segment_seconds: Median scheduled running time for each consecutive pair, in seconds.
            Length is ``len(parent_stations) - 1``.
        dwell_seconds: Median scheduled dwell at each stop, in seconds.
        n_trips: Number of trips matching this pattern in the feed.
    """

    direction_id: str
    parent_stations: tuple[str, ...]
    segment_seconds: tuple[float, ...]
    dwell_seconds: tuple[float, ...]
    n_trips: int


def _route_id_for(tables: dict[str, pd.DataFrame], route_short_name: str) -> str:
    routes = tables["routes.txt"]
    matches = routes[routes["route_short_name"].str.strip().str.upper() == route_short_name.upper()]
    if matches.empty:
        raise ValueError(f"route {route_short_name!r} not found in the GTFS feed")
    return str(matches.iloc[0]["route_id"])


def line_stop_times(route_short_name: str = "L", zip_path: str | None = None) -> pd.DataFrame:
    """Return stop_times for the route, joined to parent stations and parsed times."""
    tables = load_gtfs_tables(zip_path)
    route_id = _route_id_for(tables, route_short_name)

    trips = tables["trips.txt"]
    trips = trips[trips["route_id"] == route_id][["trip_id", "direction_id", "service_id"]]

    stop_times = tables["stop_times.txt"].merge(trips, on="trip_id", how="inner")
    stops = tables["stops.txt"][["stop_id", "stop_name", "parent_station"]]
    stop_times = stop_times.merge(stops, on="stop_id", how="left")

    # Directional stop ids (L01N/L01S) roll up to the physical stop (L01).
    stop_times["parent_station"] = stop_times["parent_station"].fillna(stop_times["stop_id"])
    stop_times["stop_sequence"] = pd.to_numeric(stop_times["stop_sequence"], errors="coerce")
    stop_times["arrival_s"] = gtfs_time_to_seconds(stop_times["arrival_time"])
    stop_times["departure_s"] = gtfs_time_to_seconds(stop_times["departure_time"])

    logger.info("route %s: %d trips, %d stop_times rows",
                route_short_name, trips["trip_id"].nunique(), len(stop_times))
    return stop_times.sort_values(["trip_id", "stop_sequence"]).reset_index(drop=True)


def dominant_patterns(route_short_name: str = "L", zip_path: str | None = None) -> dict[str, LinePattern]:
    """Extract the most common full-length stopping pattern for each direction.

    Short-turn and work-train trips produce many minority patterns. Taking the modal pattern
    per direction gives the line's normal end-to-end service, which is what the simulator
    models; the count of trips matching it is retained so the choice is auditable.
    """
    stop_times = line_stop_times(route_short_name, zip_path)
    patterns: dict[str, LinePattern] = {}

    for direction, block in stop_times.groupby("direction_id", observed=True):
        sequences = (
            block.groupby("trip_id", observed=True)["parent_station"]
            .apply(tuple)
        )
        counts = sequences.value_counts()
        modal_sequence = counts.index[0]
        n_trips = int(counts.iloc[0])

        matching_trips = sequences[sequences == modal_sequence].index
        matching = block[block["trip_id"].isin(matching_trips)].copy()

        # Median over trips of each consecutive-stop travel time and each dwell.
        matching = matching.sort_values(["trip_id", "stop_sequence"])
        matching["next_arrival_s"] = matching.groupby("trip_id", observed=True)["arrival_s"].shift(-1)
        matching["segment_s"] = matching["next_arrival_s"] - matching["departure_s"]
        matching["dwell_s"] = matching["departure_s"] - matching["arrival_s"]

        segment_median = (
            matching.groupby("stop_sequence", observed=True)["segment_s"].median().sort_index()
        )
        dwell_median = (
            matching.groupby("stop_sequence", observed=True)["dwell_s"].median().sort_index()
        )

        segments = tuple(float(v) for v in segment_median.to_numpy()[:-1])
        dwells = tuple(float(v) for v in dwell_median.to_numpy())

        patterns[str(direction)] = LinePattern(
            direction_id=str(direction),
            parent_stations=tuple(str(s) for s in modal_sequence),
            segment_seconds=segments,
            dwell_seconds=dwells,
            n_trips=n_trips,
        )
        logger.info("direction %s: %d stops, %d trips match the modal pattern, "
                    "median segment %.0fs, median dwell %.0fs",
                    direction, len(modal_sequence), n_trips,
                    float(np.median(segments)) if segments else float("nan"),
                    float(np.median(dwells)) if dwells else float("nan"))

    return patterns


def scheduled_headways(route_short_name: str = "L", zip_path: str | None = None) -> pd.DataFrame:
    """Scheduled headway statistics by direction and hour of the service day.

    Headways are measured at a **fixed reference station per direction** -- the origin
    terminus of that direction's modal pattern -- over every trip that calls there.

    Measuring instead at "each trip's own first stop" is wrong and produces spurious
    zero-second headways: short-turn trips begin mid-line, so their first departures are
    compared against full-length trips departing a different station entirely. The first
    implementation here did exactly that and reported a 0 s minimum weekday headway, which is
    physically impossible on a line with a ~90 s minimum. Anchoring to one station fixes it.
    """
    stop_times = line_stop_times(route_short_name, zip_path)
    patterns = dominant_patterns(route_short_name, zip_path)

    rows: list[dict[str, float | str | int]] = []
    for direction, pattern in patterns.items():
        reference_station = pattern.parent_stations[0]
        at_reference = stop_times[
            (stop_times["direction_id"] == direction)
            & (stop_times["parent_station"] == reference_station)
        ]
        for service, block in at_reference.groupby("service_id", observed=True):
            departures = np.sort(block["departure_s"].dropna().to_numpy())
            if departures.size < 2:
                continue
            gaps = np.diff(departures)
            hours = (departures[:-1] // 3600).astype(int)
            for hour in np.unique(hours):
                selected = gaps[hours == hour]
                rows.append({
                    "direction_id": str(direction),
                    "reference_station": reference_station,
                    "service_id": str(service),
                    "service_hour": int(hour),
                    "n_gaps": int(selected.size),
                    "headway_median_s": float(np.median(selected)),
                    "headway_min_s": float(np.min(selected)),
                    "headway_p90_s": float(np.percentile(selected, 90)),
                })
    return pd.DataFrame(rows)


def station_order(route_short_name: str = "L", zip_path: str | None = None) -> dict[str, int]:
    """Map GTFS parent station id -> position along the line, ``0`` at one terminus.

    Direction ``"0"``'s modal pattern defines the canonical ordering.
    """
    patterns = dominant_patterns(route_short_name, zip_path)
    if not patterns:
        raise ValueError(f"no stopping patterns found for route {route_short_name!r}")
    reference = patterns.get("0") or next(iter(patterns.values()))
    return {station: index for index, station in enumerate(reference.parent_stations)}


def complex_station_order(
    stations_table: pd.DataFrame, route_short_name: str = "L", zip_path: str | None = None
) -> dict[str, int]:
    """Map MTA ``station_complex_id`` -> position along the line.

    Bridges the two identifier systems: demand is published per *station complex*, while GTFS
    works in *stop ids*. ``mta_subway_stations`` carries both, so it is the join.
    """
    gtfs_order = station_order(route_short_name, zip_path)
    lookup = stations_table[["gtfs_stop_id", "complex_id"]].dropna()
    lookup = lookup.assign(
        gtfs_stop_id=lookup["gtfs_stop_id"].astype(str).str.strip(),
        complex_id=lookup["complex_id"].astype(str).str.strip(),
    )

    ordered: dict[str, int] = {}
    for _, row in lookup.iterrows():
        position = gtfs_order.get(row["gtfs_stop_id"])
        if position is None:
            continue
        # A complex can contain several stops; keep the smallest position so the mapping is
        # deterministic regardless of row order.
        existing = ordered.get(row["complex_id"])
        if existing is None or position < existing:
            ordered[row["complex_id"]] = position

    logger.info("mapped %d station complexes onto the line ordering", len(ordered))
    return ordered
