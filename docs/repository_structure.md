# Repository structure

| Location | Purpose |
|---|---|
| `src/data/`, `src/forecasting/` | Transit data preparation and demand forecasts |
| `src/simulator/`, `src/environment/` | Passenger/train simulation, observations, rewards and feasible actions |
| `src/rl/`, `src/baselines/` | Learned controllers and comparison methods |
| `src/evaluation/` | Repeated evaluations, aggregation and statistical procedures |
| `src/common/` | Shared configuration, paths, random streams and manifests |
| `src/explainability/`, `src/visualization/` | Model attribution and historical result plots |
| `src/demo/`, `app.py` | Research dashboard and demonstration |
| `product/` | Operator-facing prototype, services and its tests |
| `scripts/` | Data/experiment tools, audits and the current paper builder |
| `tests/`, `product/tests/` | Software and research-integrity checks |
| `configs/`, `experiments/` | Scientific settings and the original experiment registry |
| `research_spec/` | Specification provenance and verified bibliography data |
| `data/`, `models/` | Historical data/model locations; large binaries are ignored by Git |
| `results/` | Historical stored results, source companions and current audit/status reports |
| `replication/20261015/` | Isolated new inputs, source snapshot, policies, curves and evaluation cells; ignored by Git |
| `docs/` | Current paper, protocols, method descriptions and submission materials |
| `docs/archive/` | Superseded status reports, handovers and earlier manuscript/review documents |
| `dist/`, `logs/` | Local review archives and runtime logs; ignored by Git |

Use `requirements-linux.lock.txt` for the current Linux CPU environment. Original research
requirements and their historical Windows lock remain available for provenance. The separate
demo requirements describe the interface layer; do not replace the experiment's lock.

[Current commands](../scripts/README.md) and the [documentation index](README.md) replace the
old milestone-by-milestone navigation. This repository contains two interfaces because the
research dashboard and operator prototype have different purposes.

The active experiment verifies 69 computational files and 136 frozen inputs against their
recorded hashes. Its source archive and original results must remain intact. Editorial
cleanup does not change those scientific settings, models or analysis procedures.
