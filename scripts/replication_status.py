"""Show counts and training progress without inspecting interim scientific outcomes."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "replication" / "20261015"


def progress() -> dict:
    path = ROOT / "results" / "replication_20261015_status.json"
    state = json.loads(path.read_text()) if path.exists() else {"stage": "NOT_STARTED"}
    policies = BASE / "models" / "policies"
    active = []
    completed = 0
    for curve in sorted(policies.glob("*/*/*/learning_curve.monitor.csv")):
        if "software_smoke" in curve.parts:
            continue
        if curve.with_name("policy.zip").exists() and curve.with_name("metadata.json").exists():
            completed += 1
            continue
        with curve.open() as handle:
            rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
        active.append({"policy": str(curve.parent.relative_to(policies)),
                       "completed_episode_steps": sum(int(float(row["l"])) for row in rows if row.get("l"))})
    state.update(completed_policies=completed, planned_policies=110, active_training=active,
                 stored_evaluation_cells=sum(1 for _ in (BASE / "results" / "runs").rglob("metrics.json")),
                 planned_evaluation_cells=13_680,
                 note="Progress only; no interim hypothesis comparisons are computed.")
    return state


if __name__ == "__main__":
    print(json.dumps(progress(), indent=2))
