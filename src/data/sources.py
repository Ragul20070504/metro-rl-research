"""Registry of the verified data sources, and the low-level fetch helpers.

Single source of truth for *what* is downloaded and *from where*. Every identifier here was
probed on 2026-08-26 by ``scripts/probe_sources.py``; evidence is in
``results/source_probe.json`` and the authoritative plan is ``docs/data_sources.md``.

Two identifiers recorded in the M0 audit do not work and are deliberately absent:
``4apg-4kt9`` (HTTP 403; superseded by ``r7qk-6tcy``) and ``fgm6-ccue`` (GTFS is not served
through the tabular API; the feed comes from the credential-free S3 endpoint below).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Literal

from src.common.logging import get_logger

logger = get_logger(__name__)

SOCRATA_HOST = "https://data.ny.gov"
USER_AGENT = {"User-Agent": "metro-rl-research/0.3 (research data pipeline)"}
GTFS_SUBWAY_URL = "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip"

SourceKind = Literal["socrata_hourly", "socrata_table", "zip"]


@dataclass(frozen=True)
class Source:
    """One verified data source.

    Attributes:
        dataset_id: Local identifier; also the directory name under ``data/raw/``.
        socrata_id: Socrata four-by-four, or ``None`` for non-Socrata sources.
        title: Published dataset title.
        kind: How it must be fetched.
        role: What the project uses it for.
        url: Canonical human-facing URL.
        select: SoQL ``$select`` clause, if server-side aggregation is used.
        group: SoQL ``$group`` clause.
        order: SoQL ``$order`` clause.
        where: Static SoQL ``$where`` clause (line filters etc.).
        first_month: Earliest ``YYYY-MM`` this dataset covers, from the verified probe.
        last_month: Latest ``YYYY-MM`` this dataset covers.
        limitations: Recorded limitations, copied into the manifest.
    """

    dataset_id: str
    title: str
    kind: SourceKind
    role: str
    url: str
    socrata_id: str | None = None
    select: str | None = None
    group: str | None = None
    order: str | None = None
    where: str | None = None
    first_month: str | None = None
    last_month: str | None = None
    limitations: tuple[str, ...] = field(default_factory=tuple)


# --- demand: three same-schema datasets forming one continuous hourly series --------------
# Aggregated server-side over payment_method and fare_class_category, which is exactly the
# preprocessing docs/data_sources.md section 1 specifies. Doing it server-side turns 6.9M
# fare-class rows into ~1.4M station-hour rows without changing the result.
_HOURLY_SELECT = (
    "transit_timestamp,station_complex_id,"
    "sum(ridership) as ridership,sum(transfers) as transfers"
)
_HOURLY_GROUP = "transit_timestamp,station_complex_id"
_HOURLY_ORDER = "transit_timestamp,station_complex_id"
_HOURLY_LIMITS = (
    "hourly resolution only; a within-hour disaggregation assumption (A02) is required",
    "entries and transfers only: no exits, no destinations, no OD matrix (assumption A03)",
    "station-complex granularity, not platform or direction",
    "MTA estimates derived from fare transactions, not raw turnstile counts",
)

HOURLY_RIDERSHIP: tuple[Source, ...] = (
    Source(
        dataset_id="mta_hourly_ridership_2017",
        socrata_id="t69i-h2me",
        title="MTA Subway Hourly Ridership: 2017-2019",
        kind="socrata_hourly",
        role="demand -- pre-pandemic regime, reserved for the out-of-regime check EXP-ROBUST-004",
        url="https://data.ny.gov/d/t69i-h2me",
        select=_HOURLY_SELECT, group=_HOURLY_GROUP, order=_HOURLY_ORDER,
        first_month="2017-01", last_month="2019-12",
        limitations=_HOURLY_LIMITS,
    ),
    Source(
        dataset_id="mta_hourly_ridership_2020",
        socrata_id="wujg-7c2s",
        title="MTA Subway Hourly Ridership: 2020-2024",
        kind="socrata_hourly",
        role="demand ground truth and forecasting target",
        url="https://data.ny.gov/d/wujg-7c2s",
        select=_HOURLY_SELECT, group=_HOURLY_GROUP, order=_HOURLY_ORDER,
        first_month="2020-01", last_month="2024-12",
        limitations=_HOURLY_LIMITS,
    ),
    Source(
        dataset_id="mta_hourly_ridership_2025",
        socrata_id="5wq4-mkjj",
        title="MTA Subway Hourly Ridership: Beginning 2025",
        kind="socrata_hourly",
        role="demand -- most recent regime",
        url="https://data.ny.gov/d/5wq4-mkjj",
        select=_HOURLY_SELECT, group=_HOURLY_GROUP, order=_HOURLY_ORDER,
        first_month="2025-01", last_month="2026-08",
        limitations=_HOURLY_LIMITS,
    ),
)

# --- network, timetable, calibration and validation ---------------------------------------
REFERENCE_SOURCES: tuple[Source, ...] = (
    Source(
        dataset_id="mta_subway_stations",
        socrata_id="39hk-dx4f",
        title="MTA Subway Stations",
        kind="socrata_table",
        role="complex/stop join, line membership, station ordering, coordinates",
        url="https://data.ny.gov/d/39hk-dx4f",
        order="complex_id",
    ),
    Source(
        dataset_id="mta_gtfs_static",
        socrata_id=None,
        title="MTA subway GTFS static feed",
        kind="zip",
        role="inter-station running times, scheduled headways, trip counts, fixed_timetable baseline",
        url=GTFS_SUBWAY_URL,
        limitations=(
            "scheduled, not realised",
            "snapshot only: older editions are not archived at this endpoint, so the hash "
            "and retrieval date must be pinned",
        ),
    ),
    Source(
        dataset_id="mta_mdbf",
        socrata_id="e2qc-xgxs",
        title="MTA Subway Mean Distance Between Failures: Beginning 2015",
        kind="socrata_table",
        role="calibrates the aggregate failure rate of the synthetic availability process (A13)",
        url="https://data.ny.gov/d/e2qc-xgxs",
        order="month",
        limitations=(
            "monthly and by car class: not per train, not per day, no sensor traces",
            "constrains the aggregate rate only; within-day timing structure stays synthetic",
        ),
    ),
    Source(
        dataset_id="mta_cjfm",
        socrata_id="r7qk-6tcy",
        title="MTA Subway Customer Journey-Focused Metrics: Beginning 2015",
        kind="socrata_table",
        role="validation: Additional Platform Time and Additional Train Time, peak vs offpeak",
        url="https://data.ny.gov/d/r7qk-6tcy",
        where="line='L'", order="month",
        limitations=("aggregated to month x line x period",),
    ),
    Source(
        dataset_id="mta_wait_assessment",
        socrata_id="s666-h6b7",
        title="MTA Subway Wait Assessment: Beginning 2015",
        kind="socrata_table",
        role="validation: headway regularity",
        url="https://data.ny.gov/d/s666-h6b7",
        where="line='L'", order="month",
        limitations=("selected peak timepoints only",),
    ),
    Source(
        dataset_id="mta_service_delivered",
        socrata_id="32ch-sei3",
        title="MTA Subway Service Delivered: Beginning 2015",
        kind="socrata_table",
        role="validation: scheduled vs actual trains; real reference for dispatch success rate",
        url="https://data.ny.gov/d/32ch-sei3",
        where="line='L'", order="month",
        limitations=(
            "the published service_delivered field is mis-scaled by 1/100; the pipeline "
            "recomputes the ratio from num_actual_trains / num_sched_trains and asserts the "
            "recomputed value lies in [0, 1.05]",
        ),
    ),
    Source(
        dataset_id="mta_running_times",
        socrata_id="sp9g-mzjh",
        title="MTA Subway End-to-End Running Times: Beginning 2019",
        kind="socrata_table",
        role="validation: realised end-to-end runtime distribution (25th/50th/75th percentiles)",
        url="https://data.ny.gov/d/sp9g-mzjh",
        where="line='L'", order="month",
        limitations=("aggregated to month; no per-train trajectories",),
    ),
    Source(
        dataset_id="mta_terminal_otp",
        socrata_id="f6rf-2a3t",
        title="MTA Subway Terminal On-Time Performance: Beginning 2015",
        kind="socrata_table",
        role="validation: punctuality (secondary)",
        url="https://data.ny.gov/d/f6rf-2a3t",
        where="line='L'", order="month",
    ),
)

ALL_SOURCES: tuple[Source, ...] = HOURLY_RIDERSHIP + REFERENCE_SOURCES

SOURCES_BY_ID: dict[str, Source] = {s.dataset_id: s for s in ALL_SOURCES}

MTA_ATTRIBUTION = (
    "Metropolitan Transportation Authority, published via NY State Open Data. "
    "See https://www.mta.info/open-data"
)


class SourceFetchError(RuntimeError):
    """Raised when a source cannot be retrieved after retries."""


def http_get(url: str, *, timeout: int = 180, retries: int = 4) -> bytes:
    """GET with exponential backoff.

    The portal drops connections intermittently under load; during source verification this
    manifested as sporadic DNS and read-timeout failures on otherwise healthy endpoints.
    HTTP errors are *not* retried -- a 403 is a fact about the resource, not a transient
    fault, and retrying it would only hide it.
    """
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=USER_AGENT)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            raise SourceFetchError(f"HTTP {exc.code} for {url}") from exc
        except Exception as exc:  # noqa: BLE001 - transient network faults are expected
            last = exc
            wait = 2 ** attempt
            logger.warning("fetch attempt %d/%d failed (%s); retrying in %ds",
                           attempt + 1, retries, type(exc).__name__, wait)
            time.sleep(wait)
    raise SourceFetchError(f"{url} failed after {retries} attempts: {last}")


def soql_url(socrata_id: str, **params: str) -> str:
    """Build a Socrata SoQL request URL."""
    query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    return f"{SOCRATA_HOST}/resource/{socrata_id}.json?{query}"


def soql(socrata_id: str, *, timeout: int = 180, retries: int = 4, **params: str) -> list[dict[str, Any]]:
    """Run a SoQL query and return the decoded rows."""
    return json.loads(http_get(soql_url(socrata_id, **params), timeout=timeout, retries=retries).decode("utf-8"))


def socrata_metadata(socrata_id: str) -> dict[str, Any]:
    """Fetch the dataset's published metadata (name, licence, last update).

    Recorded in the manifest so a later run can detect that upstream has changed rather than
    silently producing different numbers.
    """
    payload = json.loads(http_get(f"{SOCRATA_HOST}/api/views/{socrata_id}.json", timeout=90).decode("utf-8"))
    licence = payload.get("license") or {}
    return {
        "name": payload.get("name"),
        "socrata_id": socrata_id,
        "license": licence.get("name"),
        "license_url": licence.get("termsLink"),
        "attribution": payload.get("attribution") or MTA_ATTRIBUTION,
        "rows_updated_at": payload.get("rowsUpdatedAt"),
        "data_updated_at": payload.get("dataUpdatedAt"),
    }
