"""Verify the research dashboard and the product prototype run side by side.

The research dashboard on 8501 is a completed deliverable; the product on 8502 must not disturb
it. This launches both, checks both serve, confirms the research artefacts are unchanged
afterwards, and then stops only the processes it started.

Writes ``results/dual_app_check.json``.

    python scripts/check_both_apps.py
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from src.common.paths import REPO_ROOT, RESULTS_DIR

APPS = (
    ("research dashboard", "app.py", 8501),
    ("product prototype", "product/app.py", 8502),
)
BOOT_TIMEOUT_S = 150

PROTECTED = (
    "experiments/registry.yaml",
    "results/statistical_analysis.json",
    "results/power_analysis.json",
    "app.py",
)


def _fingerprint() -> dict[str, str]:
    return {name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
            for name in PROTECTED if (REPO_ROOT / name).is_file()}


def _get(url: str, timeout: float = 5.0) -> tuple[int, int]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, len(response.read(4096))
    except urllib.error.HTTPError as error:
        return error.code, 0
    except (urllib.error.URLError, OSError, TimeoutError):
        return 0, 0


def _port_busy(port: int) -> bool:
    """Is something already serving here before we start?

    Streamlit exits with "Port N is not available" if the port is taken, so a stale instance from
    an earlier run makes a fresh launch look like a crash while health checks still pass against
    the occupant. Detecting this up front turns a confusing failure into an accurate message.
    """
    return _get(f"http://localhost:{port}/_stcore/health", timeout=2.0)[0] == 200


def _wait(port: int) -> tuple[bool, float]:
    started = time.monotonic()
    while time.monotonic() - started < BOOT_TIMEOUT_S:
        status, _ = _get(f"http://localhost:{port}/_stcore/health")
        if status == 200:
            return True, time.monotonic() - started
        time.sleep(2)
    return False, time.monotonic() - started


def _write(checks: list[dict], summary: str = "") -> None:
    failed = [c for c in checks if c["status"] != "PASS"]
    (RESULTS_DIR / "dual_app_check.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "research_dashboard": "http://localhost:8501",
        "product": "http://localhost:8502",
        "checks": checks,
        "summary": summary or (f"{len(checks) - len(failed)}/{len(checks)} checks pass"),
    }, indent=2), encoding="utf-8")


def main() -> int:
    before = _fingerprint()
    checks: list[dict] = []
    processes: list[tuple[str, subprocess.Popen]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    occupied = [f"{label} (:{port})" for label, _script, port in APPS if _port_busy(port)]
    record("both ports are free before launching", not occupied,
           "8501 and 8502 are unoccupied" if not occupied
           else f"already served by an earlier instance: {', '.join(occupied)}; stop those "
                "first, otherwise a fresh launch exits with 'Port is not available'")
    if occupied:
        _write(checks)
        for check in checks:
            print(f"[{check['status']}] {check['name']}: {check['detail']}")
        return 1

    try:
        for label, script, port in APPS:
            processes.append((label, subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", script,
                 "--server.port", str(port), "--server.address", "localhost",
                 "--server.headless", "true"],
                cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)))
        # Both are started before either is polled, so the check exercises them running together
        # rather than one after the other.
        for label, script, port in APPS:
            booted, seconds = _wait(port)
            record(f"{label} serves on :{port}", booted,
                   f"health endpoint responded in {seconds:.1f}s from {script}" if booted
                   else f"no healthy response within {BOOT_TIMEOUT_S}s")
        for label, _script, port in APPS:
            status, size = _get(f"http://localhost:{port}")
            record(f"{label} returns a page on :{port}", status == 200,
                   f"HTTP {status}, {size} bytes")
        alive = [label for label, process in processes if process.poll() is None]
        both_serving = all(_port_busy(port) for _label, _script, port in APPS)
        record("both applications run simultaneously", len(alive) == len(APPS) and both_serving,
               f"{len(alive)} of {len(APPS)} processes alive and both ports serving"
               if len(alive) == len(APPS) and both_serving
               else f"alive: {alive or 'none'}; both ports serving: {both_serving}")
    finally:
        for _label, process in processes:
            process.terminate()
        for _label, process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()

    after = _fingerprint()
    unchanged = before == after
    record("research artefacts unchanged by running the product", unchanged,
           "protected artefacts hash identically before and after" if unchanged
           else f"CHANGED: {sorted(k for k in before if before[k] != after.get(k))}")

    stopped_cleanly = all(process.poll() is not None for _, process in processes)
    record("stopping the product leaves no stray process", stopped_cleanly,
           "every process started by this check has exited")

    failed = [c for c in checks if c["status"] != "PASS"]
    summary = (f"both applications verified; {len(checks)}/{len(checks)} checks pass"
               if not failed else
               f"FAILING: {len(failed)}/{len(checks)} — "
               + ", ".join(c["name"] for c in failed))
    _write(checks, summary)

    for check in checks:
        print(f"[{check['status']}] {check['name']}: {check['detail']}")
    print(f"\n{summary}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
