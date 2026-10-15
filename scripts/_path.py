"""Put the repository root on ``sys.path``, as an import side effect.

Running `python scripts/analyse.py` puts `scripts/` on `sys.path`, not the repository root, so
`import src.…` fails -- even though every script's docstring documents exactly that invocation.
The project has no installed package to fall back on, so each entry point imports this module
first::

    try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
        import _path  # noqa: F401
    except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
        pass

The guard covers the second way these modules are reached: `scripts/run_experiments.py` imports
`scripts.power_analysis`, and in *that* direction `scripts/` is not on `sys.path` -- but the
repository root necessarily is, or the import could not have resolved in the first place.

Nothing else in the repository depends on this: `src` imports `src`, and the tests get the root
from pytest's rootdir.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
