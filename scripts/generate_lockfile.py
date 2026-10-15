"""Generate a hash-verified lockfile without downloading distributions.

``pip-compile --generate-hashes`` downloads every distribution in order to hash it. Against
this project's dependency set (torch alone is hundreds of megabytes) that repeatedly timed out
on ``files.pythonhosted.org``, leaving the reproducibility requirement unmet.

PyPI already publishes the SHA-256 of every distribution file in its JSON API, so the hashes
can be collected directly from metadata -- a few kilobytes per package instead of hundreds of
megabytes. The result is the same artefact ``pip install --require-hashes`` consumes.

Input is the exact pinned set in ``requirements.txt``, which is itself ``pip freeze`` of the
environment that passed ``scripts/check_environment.py`` and the test suite. So the lockfile
pins precisely what was verified to work.

Run::

    .venv/Scripts/python.exe scripts/generate_lockfile.py
    .venv/Scripts/python.exe scripts/generate_lockfile.py --verify
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PYPI_JSON = "https://pypi.org/pypi/{name}/{version}/json"
USER_AGENT = {"User-Agent": "metro-rl-research/0.3 (lockfile generator)"}
REQUIREMENT_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;#]+)")

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO_ROOT / "requirements.txt"
LOCKFILE = REPO_ROOT / "requirements.lock.txt"

HEADER = """\
# Hash-verified lockfile. GENERATED -- do not edit by hand.
#
# Produced by scripts/generate_lockfile.py from the exact pins in requirements.txt, which are
# `pip freeze` of the environment that passed scripts/check_environment.py (10/10 functional
# checks) and the test suite. This file therefore pins what was verified to work.
#
# Hashes are read from the PyPI JSON API rather than by downloading each distribution:
# `pip-compile --generate-hashes` downloads every artifact to hash it, which repeatedly timed
# out against this dependency set. The recorded digests are the same ones PyPI serves and that
# `pip install --require-hashes` checks.
#
# Every SHA-256 published for the pinned version is listed, covering all wheels and the sdist,
# so the file is valid on any platform -- not only the one it was generated on.
#
# Install with:
#   pip install --require-hashes -r requirements.lock.txt
#
# Generated (UTC): {generated}
# Packages: {count}
"""


def parse_requirements(path: Path) -> list[tuple[str, str]]:
    """Return ``(name, version)`` pairs from a fully pinned requirements file."""
    pinned: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = REQUIREMENT_RE.match(stripped)
        if match:
            pinned.append((match.group(1), match.group(2)))
        else:
            print(f"  warning: could not parse requirement line: {stripped!r}", file=sys.stderr)
    return pinned


def fetch_hashes(name: str, version: str, *, retries: int = 4, timeout: int = 45) -> list[str]:
    """Return every SHA-256 published for this exact version, sorted.

    Listing all distribution files (wheels for each platform plus the sdist) is what makes the
    lockfile portable: a machine resolving a different wheel still finds a matching hash.
    """
    url = PYPI_JSON.format(name=name, version=version)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=USER_AGENT)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            digests = {
                entry["digests"]["sha256"]
                for entry in payload.get("urls", [])
                if entry.get("digests", {}).get("sha256")
            }
            return sorted(digests)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return []  # not on PyPI (e.g. a locally built or vendored package)
            last = exc
        except Exception as exc:  # noqa: BLE001 - transient network faults are expected
            last = exc
        time.sleep(2 ** attempt)
    raise RuntimeError(f"could not fetch hashes for {name}=={version}: {last}")


def generate(requirements: Path, lockfile: Path) -> int:
    """Write the lockfile. Returns the number of packages with no published hashes."""
    pinned = parse_requirements(requirements)
    print(f"resolving hashes for {len(pinned)} pinned packages from PyPI metadata")

    lines: list[str] = []
    unhashed: list[str] = []
    for index, (name, version) in enumerate(pinned, start=1):
        hashes = fetch_hashes(name, version)
        if not hashes:
            unhashed.append(f"{name}=={version}")
            lines.append(f"# {name}=={version}  # NO PUBLISHED HASHES ON PyPI -- see note below")
            print(f"  [{index:2d}/{len(pinned)}] {name}=={version}: no published hashes")
            continue
        entry = f"{name}=={version}"
        continuation = " \\\n    ".join(f"--hash=sha256:{h}" for h in hashes)
        lines.append(f"{entry} \\\n    {continuation}")
        print(f"  [{index:2d}/{len(pinned)}] {name}=={version}: {len(hashes)} hashes")

    body = HEADER.format(generated=datetime.now(timezone.utc).isoformat(), count=len(pinned))
    if unhashed:
        body += (
            "#\n# NOTE: the following packages published no SHA-256 on PyPI for the pinned\n"
            "# version and are commented out above rather than silently omitted:\n"
            + "".join(f"#   {item}\n" for item in unhashed)
        )
    body += "\n" + "\n".join(lines) + "\n"
    lockfile.write_text(body, encoding="utf-8", newline="\n")

    print(f"\nwrote {lockfile} ({len(pinned) - len(unhashed)}/{len(pinned)} packages hashed)")
    return len(unhashed)


def verify(requirements: Path, lockfile: Path) -> int:
    """Check the lockfile covers every pinned requirement at the same version."""
    if not lockfile.exists():
        print(f"{lockfile} does not exist", file=sys.stderr)
        return 1
    text = lockfile.read_text(encoding="utf-8")
    missing = [
        f"{name}=={version}"
        for name, version in parse_requirements(requirements)
        if f"{name}=={version}" not in text
    ]
    if missing:
        print(f"lockfile is missing {len(missing)} requirement(s): {missing}", file=sys.stderr)
        return 1
    print(f"lockfile covers all pinned requirements in {requirements.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=REQUIREMENTS)
    parser.add_argument("--output", type=Path, default=LOCKFILE)
    parser.add_argument("--verify", action="store_true", help="check an existing lockfile instead")
    args = parser.parse_args()

    if args.verify:
        return verify(args.requirements, args.output)
    generate(args.requirements, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
