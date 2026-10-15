"""Add the protocol section 4.6 provenance columns to forecasts already on disk.

`write_forecast_parquet` did not emit `model_id`, `model_version` or `calibrated`, so the M4 run
that completed on 2026-08-27 wrote forecasts without them (validation check 3d). The code is fixed
for future runs; this script repairs the stored artefact rather than spending five hours refitting
a model to obtain three metadata columns.

**No forecast value is touched.** Every existing column is compared against the immutable
reference snapshot in `results/reference/m4_full_fit/`, and the write is abandoned if any of them
differs. The added values come from the run's own manifest, not from anything chosen here.

The reference snapshot records a SHA-256 per file precisely so that a later edit cannot pass
unnoticed. Adding columns changes those hashes, so this script appends a `provenance_backfill`
record to `REFERENCE_MANIFEST.json` stating what changed, what was verified, and the new hash --
rather than leaving a reference that silently no longer matches.

    python scripts/backfill_forecast_provenance.py            # repair
    python scripts/backfill_forecast_provenance.py --check    # report, change nothing
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.logging import get_logger
from src.common.paths import PROCESSED_DIR, RESULTS_DIR

logger = get_logger(__name__)

REQUIRED = ("model_id", "model_version", "calibrated")
FORECAST_DIR = PROCESSED_DIR / "forecasts"
REFERENCE_DIR = RESULTS_DIR / "reference" / "m4_full_fit"
MANIFEST = RESULTS_DIR / "forecasting_metrics.json"


def _manifest() -> dict[str, Any]:
    if not MANIFEST.exists():
        raise SystemExit(f"no run manifest at {MANIFEST}; nothing to take provenance from")
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if payload.get("quick_mode") is not False:
        raise SystemExit(
            "the stored manifest is from a quick-mode run; refusing to stamp provenance onto "
            "forecasts that are not a reportable result"
        )
    return payload


def _targets(manifest: dict[str, Any]) -> list[tuple[Path, str, bool]]:
    """(path, model_id, calibrated) for each stored forecast file."""
    factor = float(manifest.get("calibration", {}).get("factor", 1.0))
    return [
        (FORECAST_DIR / "forecast.parquet", "xgb_quantile", factor != 1.0),
        # The point model emits no interval, so there is nothing to calibrate.
        (FORECAST_DIR / "forecast_point.parquet", "xgb_point", False),
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_for(path: Path) -> Path | None:
    candidate = REFERENCE_DIR / path.name
    return candidate if candidate.is_file() else None


def backfill(path: Path, *, model_id: str, model_version: str, calibrated: bool,
             dry_run: bool) -> str:
    if not path.exists():
        return f"{path.name}: ABSENT"

    frame = pd.read_parquet(path)
    present = [c for c in REQUIRED if c in frame.columns]
    if len(present) == len(REQUIRED):
        return f"{path.name}: already carries all three provenance columns"
    if present:
        raise SystemExit(
            f"{path.name} carries {present} but not the rest; a partially-stamped file is not "
            "something to guess at"
        )

    # Compare against the immutable reference rather than against the file's own in-memory copy.
    # A self-comparison only proves this script did not corrupt what it just read; the reference
    # proves the file still holds the forecasts that were validated.
    reference_path = _reference_for(path)
    if reference_path is None:
        raise SystemExit(
            f"no reference copy of {path.name} in {REFERENCE_DIR}; refusing to modify a stored "
            "forecast artefact with nothing to verify it against"
        )
    reference = pd.read_parquet(reference_path)
    if list(reference.columns) != list(frame.columns) or len(reference) != len(frame):
        raise SystemExit(
            f"{path.name} no longer matches its reference in shape "
            f"({frame.shape} vs {reference.shape}); this is a drift to investigate, not to patch"
        )
    for column in reference.columns:
        if not reference[column].equals(frame[column]):
            raise SystemExit(
                f"{path.name}: column {column!r} already differs from the reference copy; "
                "the artefact drifted before this script ran"
            )

    frame["model_id"] = model_id
    frame["model_version"] = model_version
    frame["calibrated"] = bool(calibrated)

    if dry_run:
        return (f"{path.name}: WOULD ADD {list(REQUIRED)} ({len(frame):,} rows); "
                "verified identical to the reference copy")

    frame.to_parquet(path, index=False)

    # Read back from disk: an in-memory check does not prove what the parquet round-trip stored.
    reloaded = pd.read_parquet(path)
    for column in reference.columns:
        if not reference[column].equals(reloaded[column]):
            raise SystemExit(
                f"{path.name}: column {column!r} did not survive the round-trip unchanged"
            )
    if list(reloaded["model_id"].unique()) != [model_id]:
        raise SystemExit(f"{path.name}: model_id did not store as expected")

    _record_in_reference(path, model_id=model_id, model_version=model_version,
                         calibrated=calibrated, rows=len(reloaded))

    return (f"{path.name}: added {list(REQUIRED)} to {len(reloaded):,} rows "
            f"(model_id={model_id}, calibrated={calibrated}); every original column verified "
            "identical to the reference copy")


def _record_in_reference(path: Path, *, model_id: str, model_version: str,
                         calibrated: bool, rows: int) -> None:
    """Append a backfill record so the reference manifest never silently stops matching."""
    manifest_path = REFERENCE_DIR / "REFERENCE_MANIFEST.json"
    if not manifest_path.exists():
        logger.warning("no reference manifest at %s; the backfill is unrecorded", manifest_path)
        return

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.setdefault("provenance_backfill", [])
    records.append({
        "file": path.name,
        "applied_utc": datetime.now(timezone.utc).isoformat(),
        "by": "scripts/backfill_forecast_provenance.py",
        "reason": (
            "write_forecast_parquet omitted the three columns protocol section 4.6 requires "
            "(validation check 3d). The code is fixed for future runs; refitting a five-hour "
            "model to obtain metadata columns would not be a defensible use of compute."
        ),
        "columns_added": list(REQUIRED),
        "values": {"model_id": model_id, "model_version": model_version,
                   "calibrated": bool(calibrated)},
        "rows": rows,
        "verification": (
            "every column present in the reference copy was compared to the reference before the "
            "write and re-compared after reading the file back from disk; no forecast value "
            "changed"
        ),
        "sha256_before": payload.get("files", {}).get(path.name, {}).get("sha256"),
        "sha256_after": _sha256(path),
        "reference_copy_unchanged": (
            "results/reference/m4_full_fit/ still holds the pre-backfill artefact; the hashes "
            "under `files` describe that copy and remain correct for it"
        ),
    })
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("recorded the backfill of %s in %s", path.name, manifest_path.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report without writing")
    args = parser.parse_args()

    manifest = _manifest()
    version = str(manifest.get("config_hash", ""))
    if not version:
        raise SystemExit("the manifest records no config_hash to use as the model version")

    for path, model_id, calibrated in _targets(manifest):
        message = backfill(path, model_id=model_id, model_version=version,
                           calibrated=calibrated, dry_run=args.check)
        logger.info("%s", message)
        print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
