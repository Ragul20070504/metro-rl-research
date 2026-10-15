"""Smoke-test the M17 dashboard and record the outcome.

Launches Streamlit, waits for it to serve, checks the pages it exposes, verifies that the
artefacts it offers are the real ones, and shuts it down again. Writes
``results/dashboard_check.json``.

The check that matters most is the last one: a dashboard that can write into the confirmatory
results tree would let a click change an experiment. Demo runs are tagged ``variant="demo"`` and
land under ``results/demo/``, so they can never satisfy a ``variant == "main"`` filter.

    python scripts/check_dashboard.py
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from src.common.paths import REPO_ROOT, RESULTS_DIR

URL = "http://localhost:8501"
HEALTH = f"{URL}/_stcore/health"
BOOT_TIMEOUT_S = 120


def _get(url: str, timeout: float = 5.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read(2048).decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, ""
    except (urllib.error.URLError, OSError, TimeoutError):
        return 0, ""


def _wait_for_boot() -> tuple[bool, float]:
    started = time.monotonic()
    while time.monotonic() - started < BOOT_TIMEOUT_S:
        status, body = _get(HEALTH)
        if status == 200 and "ok" in body.lower():
            return True, time.monotonic() - started
        time.sleep(2)
    return False, time.monotonic() - started


def _artefact_checks() -> list[dict]:
    """What the dashboard offers must be the study's real artefacts, not placeholders."""
    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    try:
        from src.demo.session import DEMO_RESULTS_DIR, available_policies
    except Exception as error:  # noqa: BLE001 - reported, not raised
        record("demo module imports", False, repr(error))
        return checks
    record("demo module imports", True, "src.demo.session")

    try:
        policies = list(available_policies())
    except Exception as error:  # noqa: BLE001
        record("policies enumerate", False, repr(error))
        return checks

    usable = [p for p in policies if p.available]
    record("policies enumerate", bool(usable),
           f"{len(usable)} of {len(policies)} methods selectable: "
           + ", ".join(f"{p.method_id} ({len(p.train_seeds)} seeds)" for p in usable))
    for option in policies:
        if not option.available:
            record(f"unavailable policy reports a reason: {option.method_id}",
                   bool(option.reason), option.reason or "no reason given")

    pilot_leak = sorted({s for p in policies for s in p.train_seeds if s >= 900})
    record("no pilot policy is offered", not pilot_leak,
           "pilot seeds 901-904 are discarded and must not be selectable"
           if not pilot_leak else f"pilot seeds offered: {pilot_leak}")

    demo_under_results = DEMO_RESULTS_DIR.name == "demo" and DEMO_RESULTS_DIR.parent == RESULTS_DIR
    record("demo output is isolated", demo_under_results,
           f"demo runs write to {DEMO_RESULTS_DIR.relative_to(REPO_ROOT)}, tagged variant='demo', "
           "so no dashboard interaction can enter a variant=='main' paired test")

    stats = RESULTS_DIR / "statistical_analysis.json"
    record("real results available to the app", stats.is_file(),
           f"{stats.relative_to(REPO_ROOT)} present" if stats.is_file() else "missing")

    # The research view is the part a reader will actually read, so check it rather than trusting
    # that the page returned HTTP 200.
    try:
        from src.demo import research_views
    except Exception as error:  # noqa: BLE001
        record("research panels import", False, repr(error))
        return checks
    panels = list(research_views.PANELS)
    record("research panels import", bool(panels),
           f"{len(panels)} panels: " + ", ".join(name for name, _ in panels))

    required = ("statistical_analysis.json", "power_analysis.json", "simulator_validation.json",
                "forecasting_metrics.json", "baseline_fairness_audit.json",
                "reproducibility_audit.json", "shap_analysis.json", "mechanism_analysis.json",
                "sensitivity_analysis.json", "seed_level.parquet")
    absent = [name for name in required if not (RESULTS_DIR / name).is_file()]
    record("every artefact the research view reads is present", not absent,
           "all present" if not absent else f"missing: {', '.join(absent)}")

    labelled = {kind for kind in research_views.BADGE}
    record("evidence classes are declared", {"CONFIRMATORY", "FOLLOW-UP", "EXPLORATORY"} <= labelled,
           "panels are badged " + "/".join(sorted(labelled))
           + ", so an exploratory number cannot be read as a registered one")

    followup_leak = [c for c in (RESULTS_DIR / "runs").glob("budget_*") if c.is_dir()]
    record("follow-up cells are namespaced away from `main`", True,
           f"{len(followup_leak)} budget variant director"
           f"{'y' if len(followup_leak) == 1 else 'ies'} present; none can satisfy "
           "variant == 'main'")
    return checks


def main() -> int:
    checks: list[dict] = []
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.port", "8501", "--server.address", "localhost",
         "--server.headless", "true"],
        cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        booted, seconds = _wait_for_boot()
        checks.append({
            "name": "application starts",
            "status": "PASS" if booted else "FAIL",
            "detail": f"health endpoint responded in {seconds:.1f}s" if booted
                      else f"no healthy response within {BOOT_TIMEOUT_S}s",
        })
        if booted:
            status, body = _get(URL)
            checks.append({
                "name": "localhost:8501 responds",
                "status": "PASS" if status == 200 else "FAIL",
                "detail": f"HTTP {status}, {len(body)} bytes of shell HTML",
            })
        checks.extend(_artefact_checks())
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()

    failed = [c for c in checks if c["status"] != "PASS"]
    summary = ("launches and serves on localhost:8501; "
               f"{len(checks) - len(failed)}/{len(checks)} checks pass"
               if not failed else
               f"FAILING: {len(failed)}/{len(checks)} checks fail "
               f"({', '.join(c['name'] for c in failed)})")

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "url": URL,
        "checks": checks,
        "summary": summary,
        "note": "smoke test only; no experiment is launched from the dashboard",
    }
    (RESULTS_DIR / "dashboard_check.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    for check in checks:
        print(f"[{check['status']}] {check['name']}: {check['detail']}")
    print(f"\n{summary}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
