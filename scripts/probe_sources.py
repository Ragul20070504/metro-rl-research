"""Verify that every planned data source is reachable, readable, and covers the modelled line.

Rationale (M3 precursor). ``docs/data_sources.md`` records dataset identifiers taken from
search results and portal URLs. A portal URL is *not* evidence that a resource can be read:
during the first probe pass, two identifiers recorded in that document turned out to be
unusable -- one returned HTTP 403 and one is not served through the tabular API at all. This
script exists so that claim and evidence stay attached, and so the check can be repeated when
upstream datasets change.

It downloads nothing bulky: SoQL aggregates and 1-row samples only. Run it before M3
ingestion and whenever a source is added.

Run::

    .venv/Scripts/python.exe scripts/probe_sources.py
    .venv/Scripts/python.exe scripts/probe_sources.py --json results/source_probe.json
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SOCRATA_HOST = "https://data.ny.gov"
USER_AGENT = {"User-Agent": "metro-rl-research/0.2 (research data probe)"}

#: Station complexes served by the modelled line. Derived from `mta_subway_stations`
#: (``daytime_routes`` containing a standalone ``L`` token) and re-derived by this script.
L_COMPLEX_IDS: tuple[str, ...] = (
    "118", "119", "120", "122", "123", "124", "125", "126", "127", "129",
    "130", "131", "133", "134", "135", "136", "137", "138",
    "601", "602", "618", "621", "629", "630",
)


@dataclass
class ProbeResult:
    """Outcome of probing one source."""

    dataset_id: str
    label: str
    role: str
    status: str = "unknown"
    detail: str = ""
    fields: list[str] = field(default_factory=list)
    coverage: dict[str, str] = field(default_factory=dict)
    line_covered: bool | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _get(url: str, timeout: int = 90, retries: int = 3) -> bytes:
    """HTTP GET with backoff. The portal intermittently drops connections under load."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=USER_AGENT)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError:
            raise
        except Exception as exc:  # noqa: BLE001 - transient network faults are expected
            last = exc
            time.sleep(2 * (attempt + 1))
    assert last is not None
    raise last


def soql(dataset_id: str, timeout: int = 90, **params: str) -> list[dict]:
    """Run a SoQL query against a Socrata dataset and return the rows."""
    query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    url = f"{SOCRATA_HOST}/resource/{dataset_id}.json?{query}"
    return json.loads(_get(url, timeout).decode("utf-8"))


def probe_tabular(
    dataset_id: str,
    label: str,
    role: str,
    *,
    time_column: str | None = None,
    line_filter: str | None = None,
    timeout: int = 90,
) -> ProbeResult:
    """Check readability, capture the field list, coverage, and whether the line appears."""
    result = ProbeResult(dataset_id=dataset_id, label=label, role=role)
    try:
        sample = soql(dataset_id, timeout=timeout, **{"$limit": "1"})
    except urllib.error.HTTPError as exc:
        result.status = "unreadable"
        result.detail = f"HTTP {exc.code} from the tabular API"
        return result
    except Exception as exc:  # noqa: BLE001
        result.status = "error"
        result.detail = f"{type(exc).__name__}: {exc}"
        return result

    result.status = "ok"
    if sample:
        result.fields = sorted(k for k in sample[0] if not k.startswith(":"))

    if time_column:
        try:
            rows = soql(
                dataset_id,
                timeout=timeout * 2,
                **{"$select": f"min({time_column}) as first, max({time_column}) as last"},
            )
            if rows:
                result.coverage = {"first": rows[0].get("first", "?"), "last": rows[0].get("last", "?")}
        except Exception as exc:  # noqa: BLE001
            result.detail += f" coverage query failed ({type(exc).__name__});"

    if line_filter:
        try:
            rows = soql(dataset_id, timeout=timeout * 2, **{"$where": line_filter, "$limit": "1"})
            result.line_covered = bool(rows)
        except Exception as exc:  # noqa: BLE001
            result.detail += f" line query failed ({type(exc).__name__});"

    return result


def probe_gtfs() -> ProbeResult:
    """Resolve how the subway GTFS static feed is actually obtainable.

    The NY Open Data entry for GTFS Static is not served through the tabular API (its only
    attachment is an overview PDF), so the feed itself must come from the MTA's public bucket.
    That endpoint needs no API key, which keeps a fresh clone reproducible without credentials.
    """
    result = ProbeResult(
        dataset_id="gtfs_subway.zip",
        label="MTA subway GTFS static feed",
        role="running times, scheduled headways, trip counts, fixed-timetable baseline",
    )
    url = "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip"
    try:
        blob = _get(url, timeout=180, retries=2)
    except urllib.error.HTTPError as exc:
        result.status = "unreadable"
        result.detail = f"HTTP {exc.code} from {url}"
        return result
    except Exception as exc:  # noqa: BLE001
        result.status = "error"
        result.detail = f"{type(exc).__name__}: {exc}"
        return result

    if blob[:2] != b"PK":
        result.status = "error"
        result.detail = f"{url} did not return a zip archive (first bytes {blob[:4]!r})"
        return result

    result.status = "ok"
    result.detail = f"{len(blob):,} bytes from {url} (no API key required)"
    return result


def derive_line_complexes(timeout: int = 90) -> ProbeResult:
    """Re-derive the modelled line's station complexes from the stations dataset."""
    result = ProbeResult(
        dataset_id="39hk-dx4f",
        label="MTA Subway Stations -- modelled line membership",
        role="complex/stop join, line membership, station ordering",
    )
    try:
        rows = soql(
            "39hk-dx4f",
            timeout=timeout,
            **{
                "$select": "complex_id,stop_name,daytime_routes,borough",
                "$where": "daytime_routes like '%L%'",
                "$limit": "200",
            },
        )
    except Exception as exc:  # noqa: BLE001
        result.status = "error"
        result.detail = f"{type(exc).__name__}: {exc}"
        return result

    # `like '%L%'` also matches routes whose label merely contains the letter, so require a
    # standalone token.
    exact = [r for r in rows if "L" in str(r.get("daytime_routes", "")).split()]
    complexes = sorted({r["complex_id"] for r in exact}, key=int)
    result.status = "ok"
    result.detail = f"{len(complexes)} station complexes serve the line"
    result.fields = complexes
    result.line_covered = set(complexes) == set(L_COMPLEX_IDS)
    if not result.line_covered:
        result.detail += (
            f" -- CHANGED from the recorded set: added {sorted(set(complexes) - set(L_COMPLEX_IDS))}, "
            f"removed {sorted(set(L_COMPLEX_IDS) - set(complexes))}"
        )
    return result


def build_probes() -> list[ProbeResult]:
    """Probe every planned source. Order matches docs/data_sources.md."""
    where_l_complexes = "station_complex_id in(" + ",".join(f"'{c}'" for c in L_COMPLEX_IDS) + ")"
    return [
        derive_line_complexes(),
        probe_tabular(
            "t69i-h2me", "Subway Hourly Ridership 2017-2019", "demand -- pre-pandemic regime",
            time_column="transit_timestamp", line_filter=where_l_complexes, timeout=120,
        ),
        probe_tabular(
            "wujg-7c2s", "Subway Hourly Ridership 2020-2024", "demand -- primary forecasting target",
            time_column="transit_timestamp", line_filter=where_l_complexes, timeout=120,
        ),
        probe_tabular(
            "5wq4-mkjj", "Subway Hourly Ridership Beginning 2025", "demand -- most recent regime",
            time_column="transit_timestamp", line_filter=where_l_complexes, timeout=120,
        ),
        probe_gtfs(),
        probe_tabular(
            "e2qc-xgxs", "Subway Mean Distance Between Failures", "failure-rate calibration",
            time_column="month", line_filter="car_class='R143'",
        ),
        probe_tabular(
            "r7qk-6tcy", "Subway Customer Journey-Focused Metrics", "validation -- excess platform/train time",
            time_column="month", line_filter="line='L'",
        ),
        probe_tabular(
            "s666-h6b7", "Subway Wait Assessment", "validation -- headway regularity",
            time_column="month", line_filter="line='L'",
        ),
        probe_tabular(
            "32ch-sei3", "Subway Service Delivered", "validation -- scheduled vs actual trains",
            time_column="month", line_filter="line='L'",
        ),
        probe_tabular(
            "sp9g-mzjh", "Subway End-to-End Running Times", "validation -- realised running times",
            time_column="month", line_filter="line='L'",
        ),
        probe_tabular(
            "f6rf-2a3t", "Subway Terminal On-Time Performance", "validation -- punctuality",
            time_column="month", line_filter="line='L'",
        ),
        # Recorded in the original audit but found unusable. Kept so a regression is visible
        # if it ever becomes readable, and so the audit's error is not quietly forgotten.
        probe_tabular(
            "4apg-4kt9", "Customer Journey-Focused Metrics (id from the M0 audit)",
            "SUPERSEDED by r7qk-6tcy -- recorded in error",
        ),
        probe_tabular(
            "fgm6-ccue", "GTFS Static portal entry (id from the M0 audit)",
            "NOT served through the tabular API -- use gtfs_subway.zip",
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None, help="write a machine-readable report here")
    args = parser.parse_args()

    results = build_probes()

    width = 78
    print("=" * width)
    print("DATA SOURCE PROBE")
    print(f"utc: {datetime.now(timezone.utc).isoformat()}")
    print("=" * width)

    for result in results:
        marker = {"ok": "OK  ", "unreadable": "DEAD", "error": "ERR "}.get(result.status, "??  ")
        print(f"\n{marker} {result.dataset_id:12s} {result.label}")
        print(f"     role: {result.role}")
        if result.coverage:
            print(f"     coverage: {result.coverage.get('first', '?')} .. {result.coverage.get('last', '?')}")
        if result.line_covered is not None:
            print(f"     modelled line present: {result.line_covered}")
        if result.detail:
            print(f"     note: {result.detail}")
        if result.fields and result.dataset_id != "39hk-dx4f":
            print(f"     fields: {', '.join(result.fields)}")

    usable = [r for r in results if r.ok and "SUPERSEDED" not in r.role and "NOT served" not in r.role]
    dead = [r for r in results if not r.ok]
    print("\n" + "=" * width)
    print(f"usable sources: {len(usable)}")
    if dead:
        print("unusable (expected, if they are the two identifiers the M0 audit got wrong):")
        for r in dead:
            print(f"  {r.dataset_id}: {r.detail}")
    print("=" * width)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "probed_utc": datetime.now(timezone.utc).isoformat(),
            "results": [asdict(r) for r in results],
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    # Only the sources the project actually depends on gate the exit code.
    return 0 if all(r.ok for r in usable) else 1


if __name__ == "__main__":
    sys.exit(main())
