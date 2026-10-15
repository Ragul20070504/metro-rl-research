"""Configuration loading with mandatory parameter provenance.

Protocol section 5.2 requires that *every simulation parameter has provenance*. That is easy
to promise in a document and easy to forget in a YAML file, so it is enforced here: a config
declaring ``provenance_required: true`` will not load unless every scalar parameter states
where its value came from, and a config containing an ``unresolved`` parameter will not load
at all unless the run is explicitly tagged as a sensitivity sweep.

The intent is narrow and deliberate. Six of the assumptions this project depends on are
currently unresolved or synthetic (``docs/research_audit.md`` section E), and four of them feed
the reward function. Making an unresolved value *impossible to use by accident* is cheaper
than catching it in review.

A parameter is written as a mapping with a ``value`` key::

    min_headway_s:
      value: 120
      provenance: simulation_assumption
      note: "swept; cross-checked against the minimum scheduled headway in GTFS"

and is read back as a plain scalar by :func:`resolve`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.common.paths import CONFIGS_DIR

__all__ = [
    "PROVENANCE_CLASSES",
    "UNRESOLVED",
    "ProvenanceError",
    "Config",
    "load_config",
    "config_hash",
]

#: Allowed provenance classes. ``real_data:`` and ``literature:`` take a suffix identifying the
#: dataset id or citation, e.g. ``real_data:mta_gtfs_static``.
PROVENANCE_PREFIXES: tuple[str, ...] = ("real_data:", "literature:")
PROVENANCE_LITERALS: tuple[str, ...] = ("simulation_assumption", "synthetic", "unresolved")
PROVENANCE_CLASSES: tuple[str, ...] = PROVENANCE_PREFIXES + PROVENANCE_LITERALS

UNRESOLVED = "unresolved"

#: Top-level keys that describe the file rather than the experiment, and so are exempt from
#: the provenance requirement.
_METADATA_KEYS: frozenset[str] = frozenset(
    {"provenance_required", "config_id", "description", "version", "extends", "notes"}
)


class ProvenanceError(ValueError):
    """Raised when a configuration violates the provenance rules."""


def _is_parameter_node(node: Any) -> bool:
    """A parameter node is a mapping carrying an explicit ``value`` key."""
    return isinstance(node, dict) and "value" in node


def _check_provenance_string(provenance: Any, where: str) -> str:
    """Validate one provenance string and return it."""
    if not isinstance(provenance, str):
        raise ProvenanceError(f"{where}: provenance must be a string, got {type(provenance).__name__}")
    if provenance in PROVENANCE_LITERALS:
        return provenance
    for prefix in PROVENANCE_PREFIXES:
        if provenance.startswith(prefix) and len(provenance) > len(prefix):
            return provenance
    raise ProvenanceError(
        f"{where}: provenance {provenance!r} is not one of {PROVENANCE_CLASSES} "
        "(prefixed classes require a suffix, e.g. 'real_data:mta_gtfs_static')"
    )


@dataclass
class Config:
    """A loaded configuration, with its provenance audit attached.

    Attributes:
        path: File the configuration was loaded from.
        raw: The document exactly as parsed, parameter nodes intact.
        values: The same document with parameter nodes collapsed to their scalars.
        provenance: Dotted parameter path -> provenance string.
        unresolved: Dotted paths whose provenance is ``unresolved``.
        undocumented: Dotted paths of scalars carrying no provenance (strict mode reports
            these as errors; permissive mode records them so they are still visible).
    """

    path: Path
    raw: dict[str, Any]
    values: dict[str, Any]
    provenance: dict[str, str] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)
    undocumented: list[str] = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        """Look up a value by dotted path, e.g. ``config.get('simulator.min_headway_s')``."""
        node: Any = self.values
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def hash(self) -> str:
        """Stable SHA-256 of the resolved values; recorded in every run manifest."""
        return config_hash(self.values)


def _walk(node: Any, prefix: str, config: Config, *, strict: bool) -> Any:
    """Recursively validate and collapse ``node``; return the resolved structure."""
    if _is_parameter_node(node):
        provenance = node.get("provenance")
        if provenance is None:
            raise ProvenanceError(f"{prefix}: parameter node has 'value' but no 'provenance'")
        provenance = _check_provenance_string(provenance, prefix)
        config.provenance[prefix] = provenance
        if provenance == UNRESOLVED:
            config.unresolved.append(prefix)
        return node["value"]

    if isinstance(node, dict):
        return {
            key: _walk(child, f"{prefix}.{key}" if prefix else key, config, strict=strict)
            for key, child in node.items()
        }

    if isinstance(node, list):
        return [_walk(item, f"{prefix}[{i}]", config, strict=strict) for i, item in enumerate(node)]

    # A bare scalar. In strict mode every scalar must have been reached through a parameter
    # node; reaching one here means its provenance was never stated.
    if prefix and prefix.split(".")[0] not in _METADATA_KEYS:
        config.undocumented.append(prefix)
    return node


def load_config(
    path: str | Path,
    *,
    allow_unresolved: bool = False,
    sweep_tag: str | None = None,
) -> Config:
    """Load a YAML configuration and enforce its provenance rules.

    Args:
        path: Path to the YAML file, absolute or relative to ``configs/``.
        allow_unresolved: Permit parameters whose provenance is ``unresolved``. This is the
            escape hatch required by protocol section 5.2 and it is intentionally awkward:
            it must be paired with ``sweep_tag='sensitivity_sweep'`` so that a run using an
            ungrounded parameter is self-identifying in its own manifest.
        sweep_tag: Must be ``'sensitivity_sweep'`` when ``allow_unresolved`` is set.

    Raises:
        ProvenanceError: If a parameter lacks provenance in strict mode, if a provenance
            string is not from the allowed vocabulary, or if an unresolved parameter is used
            without the sweep escape hatch.
    """
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIGS_DIR / path
    path = path.resolve()

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ProvenanceError(f"{path}: top level of a config must be a mapping")

    strict = bool(raw.get("provenance_required", False))
    config = Config(path=path, raw=raw, values={})
    config.values = _walk(raw, "", config, strict=strict)

    if strict and config.undocumented:
        listed = "\n  ".join(config.undocumented[:20])
        more = "" if len(config.undocumented) <= 20 else f"\n  ... and {len(config.undocumented) - 20} more"
        raise ProvenanceError(
            f"{path} declares provenance_required: true but these values state no provenance:"
            f"\n  {listed}{more}\n"
            "Wrap each as {value: ..., provenance: ...}. See docs/experimental_protocol.md section 5.2."
        )

    if config.unresolved and not allow_unresolved:
        listed = ", ".join(config.unresolved)
        raise ProvenanceError(
            f"{path} contains parameters with provenance 'unresolved': {listed}.\n"
            "These have no defensible value yet (see docs/research_audit.md section E), so they may "
            "only be used in a tagged sensitivity sweep: "
            "load_config(..., allow_unresolved=True, sweep_tag='sensitivity_sweep')."
        )
    if config.unresolved and sweep_tag != "sensitivity_sweep":
        raise ProvenanceError(
            "allow_unresolved=True requires sweep_tag='sensitivity_sweep' so that the resulting "
            "run is self-identifying in its manifest."
        )

    return config


def config_hash(values: dict[str, Any]) -> str:
    """Return a stable SHA-256 over resolved configuration values.

    Keys are sorted and the encoding is fixed so the hash depends on content alone, not on
    dictionary insertion order or platform. This hash is part of the traceability chain
    ``result -> run_manifest -> config -> commit -> dataset``.
    """
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
