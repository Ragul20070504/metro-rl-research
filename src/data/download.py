"""Acquire raw data into ``data/raw/`` and record its provenance.

Design rules, from the research brief and ``docs/experimental_protocol.md`` section 2:

* ``data/raw/`` is written **once** and never modified. Re-running skips what is already
  present unless ``--force`` is given, so ingestion is resumable and idempotent.
* Every dataset directory gets a ``MANIFEST.json`` recording the source URL, retrieval time,
  per-file SHA-256, row counts, column list, upstream ``rowsUpdatedAt``, licence and
  attribution -- the ``dataset`` end of the traceability chain.
* Hourly ridership is fetched **month by month**. Each month is one request that stays far
  below the row cap (24 complexes x 24 h x 31 d = 17,856 rows), and each month is cached
  separately, so an interrupted download resumes instead of restarting.
* Aggregation over ``payment_method`` and ``fare_class_category`` happens **server-side**.
  This is the preprocessing the protocol specifies, and doing it in the query turns 6.9M
  fare-class rows into ~1.4M station-hour rows without changing the result.

Run::

    python -m src.data.download --config configs/data.yaml
    python -m src.data.download --config configs/data.yaml --only mta_hourly_ridership_2020
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import load_config
from src.common.logging import get_logger, setup_logging
from src.common.paths import RAW_DIR, ensure_dir
from src.data.sources import (
    ALL_SOURCES,
    MTA_ATTRIBUTION,
    SOURCES_BY_ID,
    Source,
    SourceFetchError,
    http_get,
    socrata_metadata,
    soql,
)

logger = get_logger(__name__)

MANIFEST_NAME = "MANIFEST.json"
LICENSE_NAME = "LICENSE.txt"

LICENSE_TEXT = """Data attribution
================

{attribution}

Dataset : {title}
Source  : {url}
Licence : {license}
{license_url}

Retrieved: {retrieved}

This project redistributes no raw data. Raw files are git-ignored; only manifests, hashes
and this attribution file are committed, so a fresh clone can verify it re-downloaded
identical bytes and fail loudly if upstream has changed.
"""


def sha256_of(path: Path) -> str:
    """Return the SHA-256 of a file, streamed so large feeds do not load into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def month_range(first: str, last: str) -> list[str]:
    """Inclusive list of ``YYYY-MM`` strings from ``first`` to ``last``."""
    start = pd.Period(first, freq="M")
    end = pd.Period(last, freq="M")
    if end < start:
        raise ValueError(f"last_month {last} precedes first_month {first}")
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def _complex_filter(complex_ids: list[str]) -> str:
    """SoQL ``in(...)`` clause restricting to the modelled line's station complexes."""
    quoted = ",".join(f"'{c}'" for c in complex_ids)
    return f"station_complex_id in({quoted})"


def fetch_hourly_month(
    source: Source,
    month: str,
    complex_ids: list[str],
    *,
    page_limit: int,
    timeout: int,
    retries: int,
) -> pd.DataFrame:
    """Fetch one calendar month of station-hour demand for the modelled line.

    Returns an empty frame when the month lies outside the dataset's coverage, which is the
    normal case at the seams between the three ridership datasets.
    """
    period = pd.Period(month, freq="M")
    start = period.start_time.strftime("%Y-%m-%dT%H:%M:%S")
    end = (period.end_time + pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S")
    where = (
        f"{_complex_filter(complex_ids)} "
        f"AND transit_timestamp >= '{start}' AND transit_timestamp < '{end}'"
    )
    params: dict[str, str] = {
        "$select": source.select or "*",
        "$where": where,
        "$limit": str(page_limit),
    }
    if source.group:
        params["$group"] = source.group
    if source.order:
        params["$order"] = source.order

    rows = soql(source.socrata_id or "", timeout=timeout, retries=retries, **params)
    if len(rows) >= page_limit:
        # Would mean silent truncation. The month partition is sized so this cannot happen;
        # if it ever does, fail loudly rather than lose data.
        raise SourceFetchError(
            f"{source.dataset_id} {month} returned {len(rows)} rows, at the page limit -- "
            "the month partition is no longer sufficient and ingestion would silently truncate"
        )
    if not rows:
        return pd.DataFrame(columns=["transit_timestamp", "station_complex_id", "ridership", "transfers"])

    frame = pd.DataFrame(rows)
    frame["transit_timestamp"] = pd.to_datetime(frame["transit_timestamp"])
    for column in ("ridership", "transfers"):
        # Socrata returns numerics as strings; cast explicitly rather than relying on inference.
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame["station_complex_id"] = frame["station_complex_id"].astype(str)
    return frame.sort_values(["transit_timestamp", "station_complex_id"]).reset_index(drop=True)


def download_hourly_source(source: Source, config: Any, *, force: bool) -> dict[str, Any]:
    """Download one hourly-ridership dataset, month by month, and write its manifest."""
    target = ensure_dir(RAW_DIR / source.dataset_id)
    months_dir = ensure_dir(target / "months")

    complex_ids = list(config.get("line.station_complex_ids"))
    ingestion = config.get("ingestion")
    # Intersect the configured span with this dataset's verified coverage. Requesting months
    # outside coverage merely returns empty results, but it triples the request count across
    # the three ridership datasets for no information.
    first = max(pd.Period(ingestion["first_month"], "M"), pd.Period(source.first_month or ingestion["first_month"], "M"))
    last = min(pd.Period(ingestion["last_month"], "M"), pd.Period(source.last_month or ingestion["last_month"], "M"))
    months = month_range(str(first), str(last)) if first <= last else []

    written: list[Path] = []
    total_rows = 0
    empty_months: list[str] = []

    for month in months:
        path = months_dir / f"{month}.parquet"
        if path.exists() and not force:
            frame = pd.read_parquet(path)
            total_rows += len(frame)
            written.append(path)
            continue
        try:
            frame = fetch_hourly_month(
                source, month, complex_ids,
                page_limit=int(ingestion["page_limit"]),
                timeout=int(ingestion["request_timeout_s"]),
                retries=int(ingestion["retries"]),
            )
        except SourceFetchError as exc:
            logger.error("%s %s: %s", source.dataset_id, month, exc)
            raise
        if frame.empty:
            empty_months.append(month)
            continue
        frame.to_parquet(path, index=False)
        written.append(path)
        total_rows += len(frame)
        logger.info("%s %s: %d station-hour rows", source.dataset_id, month, len(frame))

    metadata = socrata_metadata(source.socrata_id) if source.socrata_id else {}
    manifest = _build_manifest(
        source, written, total_rows, metadata,
        extra={
            "partition": "month",
            "months_requested": len(months),
            "months_with_data": len(written),
            "months_empty": empty_months,
            "server_side_aggregation": source.select,
            "station_complex_filter": complex_ids,
        },
    )
    _write_manifest(target, source, manifest, metadata)
    return manifest


def download_table_source(source: Source, config: Any, *, force: bool) -> dict[str, Any]:
    """Download a small reference/validation table in one request."""
    target = ensure_dir(RAW_DIR / source.dataset_id)
    path = target / "data.parquet"
    if path.exists() and not force:
        frame = pd.read_parquet(path)
        logger.info("%s: cached (%d rows)", source.dataset_id, len(frame))
    else:
        params: dict[str, str] = {"$limit": "50000"}
        if source.select:
            params["$select"] = source.select
        if source.where:
            params["$where"] = source.where
        if source.order:
            params["$order"] = source.order
        rows = soql(source.socrata_id or "", timeout=180, **params)
        frame = pd.DataFrame(rows)
        # Drop Socrata's computed-region columns: they are portal internals, not data.
        frame = frame[[c for c in frame.columns if not c.startswith(":@")]]
        frame.to_parquet(path, index=False)
        logger.info("%s: %d rows", source.dataset_id, len(frame))

    metadata = socrata_metadata(source.socrata_id) if source.socrata_id else {}
    manifest = _build_manifest(source, [path], len(frame), metadata,
                               extra={"columns": sorted(frame.columns)})
    _write_manifest(target, source, manifest, metadata)
    return manifest


def download_zip_source(source: Source, config: Any, *, force: bool) -> dict[str, Any]:
    """Download a binary feed (the GTFS static archive)."""
    target = ensure_dir(RAW_DIR / source.dataset_id)
    path = target / "gtfs_subway.zip"
    if path.exists() and not force:
        logger.info("%s: cached (%d bytes)", source.dataset_id, path.stat().st_size)
    else:
        blob = http_get(source.url, timeout=300, retries=3)
        if blob[:2] != b"PK":
            raise SourceFetchError(f"{source.url} did not return a zip archive")
        path.write_bytes(blob)
        logger.info("%s: %d bytes", source.dataset_id, len(blob))

    manifest = _build_manifest(source, [path], None, {}, extra={"content_type": "application/zip"})
    _write_manifest(target, source, manifest, {})
    return manifest


def _build_manifest(
    source: Source,
    files: list[Path],
    row_count: int | None,
    metadata: dict[str, Any],
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the manifest payload for one dataset."""
    return {
        "dataset_id": source.dataset_id,
        "title": source.title,
        "socrata_id": source.socrata_id,
        "source_url": source.url,
        "role": source.role,
        "kind": source.kind,
        "download_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": row_count,
        "files": [
            {
                "path": str(path.relative_to(RAW_DIR)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_of(path),
            }
            for path in sorted(files)
        ],
        "upstream_metadata": metadata,
        "license": metadata.get("license"),
        "attribution": metadata.get("attribution", MTA_ATTRIBUTION),
        "limitations": list(source.limitations),
        **(extra or {}),
    }


def _write_manifest(target: Path, source: Source, manifest: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Write ``MANIFEST.json`` and ``LICENSE.txt`` beside the data."""
    (target / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (target / LICENSE_NAME).write_text(
        LICENSE_TEXT.format(
            attribution=manifest.get("attribution", MTA_ATTRIBUTION),
            title=source.title,
            url=source.url,
            license=metadata.get("license") or "not stated by the publisher at retrieval time",
            license_url=metadata.get("license_url") or "",
            retrieved=manifest["download_utc"],
        ),
        encoding="utf-8",
    )


DOWNLOADERS = {
    "socrata_hourly": download_hourly_source,
    "socrata_table": download_table_source,
    "zip": download_zip_source,
}


def download_all(config_path: str | Path, *, only: list[str] | None = None, force: bool = False) -> dict[str, Any]:
    """Download every registered source (or the subset named by ``only``)."""
    config = load_config(config_path)
    selected = [SOURCES_BY_ID[name] for name in only] if only else list(ALL_SOURCES)

    summary: dict[str, Any] = {}
    for source in selected:
        logger.info("--- %s (%s) ---", source.dataset_id, source.title)
        summary[source.dataset_id] = DOWNLOADERS[source.kind](source, config, force=force)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--only", nargs="*", default=None, help="dataset_id(s) to fetch")
    parser.add_argument("--force", action="store_true", help="re-download cached partitions")
    args = parser.parse_args()

    setup_logging(level=logging.INFO, log_file="data_download.log")
    summary = download_all(args.config, only=args.only, force=args.force)

    print(f"\n{'dataset':34s} {'rows':>10s}  files")
    for dataset_id, manifest in summary.items():
        rows = manifest.get("row_count")
        print(f"{dataset_id:34s} {rows if rows is not None else '-':>10}  {len(manifest['files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
