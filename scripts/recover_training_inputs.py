"""Reconstruct train/validation features using verified raw bytes and frozen split boundaries.

Outputs stay under data/processed/recovery. Missing or revised test-period inputs cannot move
the split boundaries or be mistaken for a complete reconstruction of the original dataset.
"""
from __future__ import annotations

try:
    import _path  # noqa: F401
except ModuleNotFoundError:
    pass

from datetime import datetime, timezone
import hashlib
import json

import pandas as pd

from src.common.paths import PROCESSED_DIR, RAW_DIR, RESULTS_DIR
from src.data.features import FeatureSpec, build_supervised_frame
from src.data.prepare import apply_exclusions
from src.data.splits import SplitBoundaries
from src.data.validate import load_hourly_demand


def main() -> int:
    original = json.loads((PROCESSED_DIR / "split_manifest.json").read_text())
    boundaries = SplitBoundaries(**{key: value if key == "embargo_days" else pd.Timestamp(value)
                                   for key, value in original["split"].items()})
    boundaries.validate()
    verified = []
    for manifest in RAW_DIR.glob("mta_hourly_ridership_*/MANIFEST.json"):
        for item in json.loads(manifest.read_text())["files"]:
            path = RAW_DIR / item["path"]
            if path.is_file():
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != item["sha256"]:
                    raise ValueError(f"Unverified raw file: {path}")
                verified.append(item)
    # The raw loader reads only promoted files, never quarantine candidates.
    demand, _ = apply_exclusions(load_hourly_demand())
    demand = demand[demand.transit_timestamp >= boundaries.train_start].reset_index(drop=True)
    if demand.transit_timestamp.max().normalize() <= boundaries.val_end:
        raise ValueError("Verified data do not cover the complete validation period")
    order = json.loads((PROCESSED_DIR / "network/station_order.json").read_text())["station_complex_to_index"]
    features = build_supervised_frame(demand, FeatureSpec(), order)
    features["split"] = boundaries.label_for(features["target_time"])
    demand["split"] = boundaries.label_for(demand["transit_timestamp"])
    output = PROCESSED_DIR / "recovery"
    output.mkdir(exist_ok=True)
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "split_boundaries": original["split"], "test_recovered": False,
              "verified_raw_files": len(verified), "outputs": {}}
    for name in ("train", "val"):
        block = features[features.split == name]
        expected = original["rows"]["supervised_" + name]
        if len(block) != expected:
            raise ValueError(f"{name}: {len(block)} rows, expected {expected}")
        path = output / f"supervised_{name}.parquet"
        block.to_parquet(path, index=False)
        report["outputs"][name] = {"rows": len(block), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    demand[demand.split == "train"].to_parquet(output / "demand_train.parquet", index=False)
    (RESULTS_DIR / "training_input_recovery.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
