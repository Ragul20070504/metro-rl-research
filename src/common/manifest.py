"""Run manifests: the traceability chain, and the append-only guard on ``results/``.

Research brief section 25 requires that every result be traceable
``result -> experiment -> configuration -> code commit -> dataset``, and section 21 requires
that experiments are never overwritten. Both are implemented here, next to the write, rather
than being left to convention.

Two deliberate behaviours are worth knowing about:

* :func:`open_run_dir` **refuses** to write into a directory that already holds a manifest.
  Re-running an experiment does not silently replace its result; the caller must delete it
  explicitly, which leaves a trace in the shell history and in git.
* A run made from a dirty working tree is recorded with ``git_dirty: true`` and
  ``provisional: true``. Such a run is reproducible only by accident and must not appear in
  the paper. This is recorded rather than blocked, because provisional runs are useful during
  development; the final reproducibility audit (M16) checks the flag.
"""

from __future__ import annotations

import json
import os
import platform as _platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from src.common.paths import REPO_ROOT, ensure_dir

__all__ = ["GitInfo", "RunManifest", "git_info", "package_versions", "open_run_dir"]

MANIFEST_NAME = "run_manifest.json"

#: Packages whose versions are recorded. Every one of these can change a numerical result.
_TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "statsmodels", "xgboost",
    "torch", "gymnasium", "stable-baselines3", "sb3-contrib", "ortools", "shap", "matplotlib",
)


def _git_executable() -> str | None:
    """Locate git, tolerating the common Windows case where it is installed but off PATH."""
    found = shutil.which("git")
    if found:
        return found
    for candidate in (
        Path(r"C:\Program Files\Git\cmd\git.exe"),
        Path(r"C:\Program Files (x86)\Git\cmd\git.exe"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Git" / "cmd" / "git.exe",
    ):
        if candidate.is_file():
            return str(candidate)
    return None


@dataclass(frozen=True)
class GitInfo:
    """Repository state at the moment a run started."""

    commit: str | None
    branch: str | None
    dirty: bool
    available: bool


def git_info() -> GitInfo:
    """Return the current commit, branch and dirty flag, or a null record if git is absent."""
    executable = _git_executable()
    if executable is None:
        return GitInfo(commit=None, branch=None, dirty=True, available=False)

    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                [executable, *args],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    status = run("status", "--porcelain")
    return GitInfo(
        commit=commit,
        branch=branch,
        dirty=bool(status) if status is not None else True,
        available=True,
    )


def package_versions() -> dict[str, str]:
    """Return installed versions of the packages that can change a numerical result."""
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


@dataclass
class RunManifest:
    """Everything needed to trace, audit and reproduce one run.

    Fields map directly onto research brief section 25. ``provisional`` is set whenever the
    run cannot be tied to a clean commit; the M16 audit refuses to accept a provisional run as
    evidence for a paper claim.
    """

    experiment_id: str
    method: str
    scenario: str
    train_seed: int | None
    eval_seed: int
    config_path: str
    config_hash: str
    config_values: dict[str, Any]
    dataset_manifests: dict[str, str] = field(default_factory=dict)
    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool = True
    git_available: bool = False
    provisional: bool = True
    sweep_tag: str | None = None
    unresolved_parameters: list[str] = field(default_factory=list)
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    python_executable: str = field(default_factory=lambda: sys.executable)
    platform: str = field(default_factory=_platform.platform)
    machine: str = field(default_factory=_platform.machine)
    processor: str = field(default_factory=_platform.processor)
    cpu_count: int | None = field(default_factory=os.cpu_count)
    thread_env: dict[str, str] = field(default_factory=dict)
    packages: dict[str, str] = field(default_factory=dict)
    started_utc: str | None = None
    finished_utc: str | None = None
    wall_clock_s: float | None = None
    notes: str = ""

    @classmethod
    def create(
        cls,
        *,
        experiment_id: str,
        method: str,
        scenario: str,
        train_seed: int | None,
        eval_seed: int,
        config_path: str | Path,
        config_hash: str,
        config_values: dict[str, Any],
        dataset_manifests: dict[str, str] | None = None,
        sweep_tag: str | None = None,
        unresolved_parameters: list[str] | None = None,
        notes: str = "",
    ) -> RunManifest:
        """Build a manifest, capturing git, environment and hardware state now."""
        info = git_info()
        return cls(
            experiment_id=experiment_id,
            method=method,
            scenario=scenario,
            train_seed=train_seed,
            eval_seed=eval_seed,
            config_path=str(config_path),
            config_hash=config_hash,
            config_values=config_values,
            dataset_manifests=dataset_manifests or {},
            git_commit=info.commit,
            git_branch=info.branch,
            git_dirty=info.dirty,
            git_available=info.available,
            provisional=info.dirty or info.commit is None,
            sweep_tag=sweep_tag,
            unresolved_parameters=unresolved_parameters or [],
            thread_env={
                key: os.environ.get(key, "unset")
                for key in ("PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
            packages=package_versions(),
            started_utc=datetime.now(timezone.utc).isoformat(),
            notes=notes,
        )

    def finish(self) -> None:
        """Stamp completion time and wall clock."""
        finished = datetime.now(timezone.utc)
        self.finished_utc = finished.isoformat()
        if self.started_utc is not None:
            self.wall_clock_s = (finished - datetime.fromisoformat(self.started_utc)).total_seconds()

    def write(self, directory: Path) -> Path:
        """Write the manifest into ``directory`` and return the file path."""
        ensure_dir(directory)
        target = directory / MANIFEST_NAME
        with target.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        return target


def open_run_dir(directory: Path, *, overwrite: bool = False) -> Path:
    """Create and return a run directory, refusing to clobber a completed run.

    Args:
        directory: Target run directory, normally from :func:`src.common.paths.run_dir`.
        overwrite: Explicit opt-in to reuse a directory that already holds a manifest.

    Raises:
        FileExistsError: If a manifest already exists and ``overwrite`` is False. Protocol
            section 8.5: results are append-only and experiments are not overwritten.
    """
    if (directory / MANIFEST_NAME).exists() and not overwrite:
        raise FileExistsError(
            f"{directory} already contains {MANIFEST_NAME}. Results are append-only "
            "(docs/experimental_protocol.md section 8.5). Delete the directory explicitly, or "
            "pass overwrite=True, if you really intend to discard that run."
        )
    return ensure_dir(directory)
