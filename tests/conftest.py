"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def write_config(tmp_path: Path):
    """Return a helper that writes YAML text to a temporary file and returns its path."""

    def _write(text: str, name: str = "cfg.yaml") -> Path:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return path

    return _write
