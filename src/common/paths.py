"""Canonical filesystem locations for the project.

Every module resolves paths through this file. Nothing in ``src/`` may contain a hard-coded
absolute path (``docs/repository_structure.md``, structural rules), because a hard-coded path
is the most common reason a "reproducible" repository fails to reproduce on another machine.

Overrides come from the environment (see ``.env.example``) so that large artefacts can live
off the repository volume without editing code.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "REPO_ROOT",
    "CONFIGS_DIR",
    "DATA_DIR",
    "RAW_DIR",
    "INTERIM_DIR",
    "PROCESSED_DIR",
    "EXTERNAL_DIR",
    "MODELS_DIR",
    "RESULTS_DIR",
    "LOGS_DIR",
    "DOCS_DIR",
    "EXPERIMENTS_DIR",
    "run_dir",
    "ensure_dir",
]


def _find_repo_root() -> Path:
    """Walk upwards from this file until a directory containing ``configs/`` is found.

    Anchoring on a committed directory rather than on ``__file__.parents[2]`` keeps the
    resolution correct if the package is ever moved or installed in editable mode.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
    # Fall back to the conventional layout: src/common/paths.py -> repo root.
    return here.parents[2]


def _from_env(variable: str, default: Path) -> Path:
    """Return an environment override if it is set and non-empty, else ``default``."""
    raw = os.environ.get(variable, "").strip()
    return Path(raw).expanduser().resolve() if raw else default


REPO_ROOT: Path = _find_repo_root()

CONFIGS_DIR: Path = REPO_ROOT / "configs"
DOCS_DIR: Path = REPO_ROOT / "docs"
EXPERIMENTS_DIR: Path = REPO_ROOT / "experiments"

DATA_DIR: Path = _from_env("METRORL_DATA_DIR", REPO_ROOT / "data")
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DIR: Path = DATA_DIR / "external"

MODELS_DIR: Path = _from_env("METRORL_MODELS_DIR", REPO_ROOT / "models")
RESULTS_DIR: Path = _from_env("METRORL_RESULTS_DIR", REPO_ROOT / "results")
LOGS_DIR: Path = _from_env("METRORL_LOGS_DIR", REPO_ROOT / "logs")


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if absent and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(scenario: str, method: str, train_seed: int | None, eval_seed: int) -> Path:
    """Return the canonical, append-only output directory for a single evaluation run.

    Layout is fixed by ``docs/experimental_protocol.md`` section 8.5::

        results/<scenario>/<method>/<train_seed>/<eval_seed>/

    Non-learning baselines have no training seed; they use the literal segment ``na`` so the
    depth of the tree stays uniform and globbing stays simple.

    This function only *computes* the path. Refusing to overwrite an existing run is enforced
    where results are written (``src.common.manifest``), so that the guard sits next to the
    write rather than next to the path arithmetic.
    """
    seed_segment = "na" if train_seed is None else str(train_seed)
    return RESULTS_DIR / scenario / method / seed_segment / str(eval_seed)
