"""Recover raw inputs only when their bytes match the original recorded SHA-256.

Never rewrite reference manifests, licenses or nonmatching existing files. Changed upstream
downloads are quarantined and reported, rather than accepted as the historical snapshot.
"""
from __future__ import annotations

try:
    import _path  # noqa: F401
except ModuleNotFoundError:
    pass

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.common.config import load_config
from src.common.paths import RAW_DIR, RESULTS_DIR
from src.data.download import fetch_hourly_month
from src.data.sources import SOURCES_BY_ID, http_get, soql


def promote(candidate: Path, target: Path, expected: str) -> str:
    """Verify before promoting; an upstream change must never replace the reference hash."""
    if target.exists():
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Existing file has unexpected bytes: {target}")
        return "ALREADY_VERIFIED"
    if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
        return "UPSTREAM_CHANGED"
    target.parent.mkdir(parents=True, exist_ok=True)
    candidate.replace(target)
    return "RECOVERED"


def recover(source_id: str, item: dict, complex_ids: list[str]) -> dict:
    source = SOURCES_BY_ID[source_id]
    relative = Path(item["path"].replace("\\", "/"))
    target = (RAW_DIR / relative).resolve()
    if not target.is_relative_to(RAW_DIR.resolve()):
        raise ValueError("Manifest path escapes raw-data directory")
    candidate = RAW_DIR / ".recovery-quarantine" / relative
    row = {"dataset_id": source_id, "path": relative.as_posix(),
           "expected_sha256": item["sha256"], "source_url": source.url}
    try:
        if target.exists():
            row["status"] = promote(candidate, target, item["sha256"])
            return row
        candidate.parent.mkdir(parents=True, exist_ok=True)
        if source.kind == "socrata_hourly":
            frame = fetch_hourly_month(source, relative.stem, complex_ids,
                                       page_limit=50000, timeout=180, retries=2)
            frame.to_parquet(candidate, index=False)
        elif source.kind == "socrata_table":
            params = {"$limit": "50000"}
            for name in ("select", "where", "order"):
                if value := getattr(source, name):
                    params["$" + name] = value
            frame = pd.DataFrame(soql(source.socrata_id, timeout=180, **params))
            frame = frame[[column for column in frame if not column.startswith(":@")]]
            frame.to_parquet(candidate, index=False)
        elif source.kind == "zip":
            candidate.write_bytes(http_get(source.url, timeout=180, retries=2))
        else:
            raise ValueError(f"Unsupported source kind {source.kind}")
        row["downloaded_sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
        row["status"] = promote(candidate, target, item["sha256"])
        if candidate.exists():
            row["quarantined_file"] = str(candidate.relative_to(RAW_DIR))
    except Exception as error:
        row.update(status="ERROR", error=str(error))
    print(row["status"], row["path"], flush=True)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    config = load_config("configs/data.yaml")
    ids = list(config.get("line.station_complex_ids"))
    jobs = [(payload["dataset_id"], item) for path in sorted(RAW_DIR.glob("*/MANIFEST.json"))
            for payload in [json.loads(path.read_text())] for item in payload["files"]]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        rows = list(pool.map(lambda job: recover(*job, ids), jobs))
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(), "files": rows,
              "complete": all(row["status"] in ("RECOVERED", "ALREADY_VERIFIED") for row in rows)}
    (RESULTS_DIR / "raw_recovery_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print({status: sum(row["status"] == status for row in rows)
           for status in sorted({row["status"] for row in rows})})
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
