"""Derive the simulator network from real data and persist it (M5).

Every value this writes is either taken from the GTFS feed / stations table, or is a config
parameter carrying an explicit provenance label. Nothing is invented here.

Run::

    python -m src.simulator.build_network --config configs/simulator.yaml
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.common.logging import get_logger, setup_logging
from src.common.paths import RAW_DIR
from src.data.gtfs import dominant_patterns
from src.simulator.network import Network, haversine_km

logger = get_logger(__name__)


def build_network(config_path: str = "configs/simulator.yaml") -> Network:
    """Construct the :class:`Network` for the modelled line from GTFS + stations."""
    config = load_config(config_path, allow_unresolved=True, sweep_tag="sensitivity_sweep")
    route = str(config.get("network.route_short_name", "L"))

    patterns = dominant_patterns(route)
    if "0" not in patterns or "1" not in patterns:
        raise RuntimeError(f"route {route} does not have both directions in the GTFS feed")

    forward, backward = patterns["0"], patterns["1"]
    if len(forward.parent_stations) != len(backward.parent_stations):
        raise RuntimeError(
            "the two directions have different stop counts "
            f"({len(forward.parent_stations)} vs {len(backward.parent_stations)}); "
            "the simple-corridor abstraction does not hold"
        )

    # Direction 0's modal pattern defines the canonical station order.
    gtfs_order = list(forward.parent_stations)

    stations = pd.read_parquet(RAW_DIR / "mta_subway_stations" / "data.parquet")
    stations = stations.assign(
        gtfs_stop_id=stations["gtfs_stop_id"].astype(str).str.strip(),
        complex_id=stations["complex_id"].astype(str).str.strip(),
    ).drop_duplicates(subset="gtfs_stop_id")
    lookup = stations.set_index("gtfs_stop_id")

    missing = [s for s in gtfs_order if s not in lookup.index]
    if missing:
        raise RuntimeError(f"GTFS stops absent from mta_subway_stations: {missing}")

    complex_ids = tuple(str(lookup.loc[s, "complex_id"]) for s in gtfs_order)
    names = tuple(str(lookup.loc[s, "stop_name"]) for s in gtfs_order)
    latitudes = np.array([float(lookup.loc[s, "gtfs_latitude"]) for s in gtfs_order])
    longitudes = np.array([float(lookup.loc[s, "gtfs_longitude"]) for s in gtfs_order])

    segment_km = tuple(
        haversine_km(latitudes[i], longitudes[i], latitudes[i + 1], longitudes[i + 1])
        for i in range(len(gtfs_order) - 1)
    )

    # Backward running times are reported in the backward pattern's own stop order; reverse
    # them so index i refers to the same physical segment in both directions.
    backward_segments = tuple(reversed(backward.segment_seconds))

    # Conserve the real scheduled end-to-end time.
    #
    # This feed reports arrival == departure at intermediate stops, so its segment times
    # (next_arrival - departure) already absorb the station stop. Adding an explicit base dwell
    # on top double-counts it: doing so produced a 48.5 min end-to-end against a published
    # scheduled 33.5-37 min. The base dwell is therefore *carved out* of each segment rather
    # than added to it, which keeps sum(segment + dwell) equal to the GTFS scheduled runtime
    # while still letting dwell respond to passenger load during a run.
    base_dwell = float(config.get("dwell.base_seconds"))
    n_segments = len(gtfs_order) - 1

    def carve_out_dwell(segments: tuple[float, ...], label: str) -> tuple[float, ...]:
        adjusted = tuple(max(float(s) - base_dwell, 15.0) for s in segments)
        shortfall = sum(float(s) for s in segments) - sum(adjusted) - base_dwell * n_segments
        if abs(shortfall) > 1.0:
            logger.warning(
                "%s: %.0f s of base dwell could not be carved out of the shortest segments "
                "(floored at 15 s); simulated end-to-end will exceed the schedule by that much",
                label, abs(shortfall),
            )
        return adjusted

    forward_segments = carve_out_dwell(tuple(float(s) for s in forward.segment_seconds), "direction 0")
    backward_segments = carve_out_dwell(backward_segments, "direction 1")

    scheduled_end_to_end = sum(float(s) for s in forward.segment_seconds)
    logger.info("GTFS scheduled end-to-end %.1f min; base dwell %.0f s x %d stops carved out",
                scheduled_end_to_end / 60.0, base_dwell, n_segments)

    network = Network(
        station_ids=complex_ids,
        station_names=names,
        segment_seconds=(forward_segments, backward_segments),
        dwell_base_seconds=float(config.get("dwell.base_seconds")),
        segment_km=segment_km,
        min_headway_seconds=float(config.get("headway.min_seconds")),
        turnaround_seconds=float(config.get("turnaround.seconds")),
        provenance={
            "station_ids": "real_data:mta_subway_stations + GTFS modal pattern",
            "station_order": "real_data:mta_gtfs_static (direction 0 modal stopping pattern)",
            "segment_seconds": "real_data:mta_gtfs_static (median scheduled running time)",
            "segment_km": "real_data:mta_subway_stations coordinates via haversine; "
                          "understates track length, recorded as an energy-model limitation",
            "dwell_base_seconds": str(config.provenance.get("dwell.base_seconds", "unknown")),
            "min_headway_seconds": str(config.provenance.get("headway.min_seconds", "unknown")),
            "turnaround_seconds": str(config.provenance.get("turnaround.seconds", "unknown")),
        },
    )

    logger.info(
        "network: %d stations, end-to-end fwd %.1f min / bwd %.1f min, total %.2f km",
        network.n_stations,
        network.end_to_end_seconds(0) / 60.0,
        network.end_to_end_seconds(1) / 60.0,
        sum(segment_km),
    )
    return network


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/simulator.yaml")
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    network = build_network(args.config)
    path = network.save()

    print(f"stations        : {network.n_stations}")
    print(f"first / last    : {network.station_names[0]} -> {network.station_names[-1]}")
    print(f"end-to-end fwd  : {network.end_to_end_seconds(0)/60:.1f} min")
    print(f"end-to-end bwd  : {network.end_to_end_seconds(1)/60:.1f} min")
    print(f"total distance  : {sum(network.segment_km):.2f} km")
    print(f"min headway     : {network.min_headway_seconds:.0f} s")
    print(f"turnaround      : {network.turnaround_seconds:.0f} s")
    print(f"written         : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
