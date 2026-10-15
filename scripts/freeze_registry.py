"""Freeze the experiment registry (pre-registration) before the first confirmatory run.

`experiments/registry.yaml` carries the pre-registered design. Protocol section 12 and the
registry's own header require it to be **frozen and committed before the first M11 run**; after
freezing, entries are never edited and deviations are appended as numbered amendments instead.

This script performs that freeze, and `require_frozen()` is the guard the sweep and the ablations
call so a confirmatory run cannot start against an unfrozen -- that is, still editable -- design.

The freeze happens **after** the R-1 power gate and **before** M11, because R-1's registered job is
to fix the one quantity the design deliberately leaves open (protocol section 8.4 marks the power
analysis `TBD@M11, frozen before the sweep`). Everything else in the registry is already fixed;
the only value this script writes into it is the sample size that gate decided, together with the
decision text that justifies it.

    python scripts/freeze_registry.py            # freeze, using the recorded R-1 decision
    python scripts/freeze_registry.py --check    # report status, change nothing
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import argparse
import json
import re
from datetime import datetime, timezone
from typing import Any

import yaml

from src.common.logging import get_logger
from src.common.manifest import git_info
from src.common.paths import EXPERIMENTS_DIR, RESULTS_DIR

logger = get_logger(__name__)

REGISTRY = EXPERIMENTS_DIR / "registry.yaml"
POWER = RESULTS_DIR / "power_analysis.json"


def load_registry() -> dict[str, Any]:
    if not REGISTRY.exists():
        raise SystemExit(f"the experiment registry is missing: {REGISTRY}")
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))


def is_frozen() -> bool:
    return bool(load_registry().get("meta", {}).get("frozen") is True)


def require_frozen(purpose: str) -> dict[str, Any]:
    """Refuse to start a confirmatory run against an editable pre-registration."""
    registry = load_registry()
    meta = registry.get("meta", {})
    if meta.get("frozen") is not True:
        raise SystemExit(
            f"{purpose} must not run against an unfrozen registry. "
            "`experiments/registry.yaml` still has `frozen: false`, which means the "
            "pre-registered design is still editable and a result obtained now could not be "
            "distinguished from one obtained after the design was adjusted to suit it. "
            "Run `python scripts/freeze_registry.py` first."
        )
    logger.info("registry frozen at %s (%s); proceeding with %s",
                meta.get("frozen_at_commit"), meta.get("frozen_utc"), purpose)
    return registry


def registered_train_seed_count() -> int:
    """The sample size recorded in the frozen registry -- the authority once frozen."""
    meta = load_registry().get("meta", {})
    value = meta.get("train_seed_count")
    if value is None:
        raise SystemExit("the frozen registry records no training-seed count")
    return int(value)


def _power_decision() -> dict[str, Any]:
    if not POWER.exists():
        raise SystemExit(
            "the registry cannot be frozen before the R-1 power gate has produced a decision: "
            f"{POWER} is missing. The sample size is the one quantity the design leaves open, "
            "and freezing without it would either invent a number or freeze a placeholder."
        )
    payload = json.loads(POWER.read_text(encoding="utf-8"))
    n = int(payload.get("amended_train_seeds", 0))
    if n < 3:
        raise SystemExit(f"the power gate returned an unusable sample size ({n})")
    return payload


def _substitute(text: str, pattern: str, replacement, *, what: str) -> str:
    """Replace exactly one line, or refuse. A plain string replacement is treated as literal."""
    if isinstance(replacement, str):
        literal = replacement
        replacement = lambda _m: literal  # noqa: E731 - escaping is the whole point
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"could not locate {what} in {REGISTRY.name}; refusing to guess")
    return updated


def freeze() -> dict[str, Any]:
    power = _power_decision()
    seeds = int(power["amended_train_seeds"])
    decision = str(power.get("decision", ""))

    registry = load_registry()
    meta = registry.get("meta", {})
    if meta.get("frozen") is True:
        recorded = int(meta.get("train_seed_count", -1))
        if recorded != seeds:
            raise SystemExit(
                f"the registry is already frozen at S = {recorded} but the power gate decided "
                f"S = {seeds}. A frozen pre-registration is never silently rewritten: record an "
                "amendment in docs/experimental_protocol.md section 12 and supersede the entry."
            )
        logger.info("registry already frozen at S = %d; nothing to do", recorded)
        return meta

    git = git_info()
    now = datetime.now(timezone.utc)
    seed_list = list(range(101, 101 + seeds))

    text = REGISTRY.read_text(encoding="utf-8")
    text = _substitute(
        text,
        r"^# STATUS: SKELETON\. `frozen: false`.*$",
        "# STATUS: FROZEN. Entries below are the pre-registered design and are never edited.",
        what="the status header",
    )
    text = _substitute(text, r"^  frozen: false$", "  frozen: true", what="the frozen flag")
    text = _substitute(
        text,
        r"^  frozen_at_commit: null.*$",
        f"  frozen_at_commit: {git.commit or 'null'}"
        + ("        # working tree dirty at freeze time" if git.dirty else ""),
        what="the frozen_at_commit field",
    )
    # `re.subn` interprets backslashes and \g<..> in the replacement, so every substitution below
    # passes a *function* returning the literal text rather than a template string.
    frozen_block = "\n".join((
        f"  frozen_utc: {now.isoformat()}",
        f"  train_seed_count: {seeds}",
        "  train_seed_source: R-1 power gate, results/power_analysis.json",
        f"  train_seed_decision: {json.dumps(decision)}",
    ))
    text = _substitute(
        text, r"^  frozen_utc: null$", lambda _m: frozen_block, what="the frozen_utc field")
    text = _substitute(
        text,
        r"^  train_seeds: \[.*$",
        lambda _m: (f"  train_seeds: {seed_list}   "
                    "# fixed by the R-1 power gate; see results/power_analysis.json"),
        what="the train_seeds default",
    )

    REGISTRY.write_text(text, encoding="utf-8")

    # Re-parse, so a freeze that produced invalid YAML fails here rather than inside the sweep.
    meta = load_registry().get("meta", {})
    if meta.get("frozen") is not True or int(meta.get("train_seed_count", -1)) != seeds:
        raise SystemExit("the freeze wrote a registry that does not read back as frozen")

    logger.info("registry frozen at commit %s, S = %d (%s)", git.commit, seeds, decision)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report status without changing it")
    args = parser.parse_args()

    if args.check:
        meta = load_registry().get("meta", {})
        state = "FROZEN" if meta.get("frozen") is True else "NOT FROZEN"
        print(f"{state}: commit={meta.get('frozen_at_commit')} utc={meta.get('frozen_utc')} "
              f"S={meta.get('train_seed_count')}")
        return 0 if meta.get("frozen") is True else 1

    meta = freeze()
    print(f"registry frozen: S = {meta.get('train_seed_count')} "
          f"at commit {meta.get('frozen_at_commit')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
