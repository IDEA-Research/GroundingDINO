"""Tier 2 — API observability scenario end-to-end over HTTP.

Drives the FastAPI backend via TestClient through the complete user
journey for the API observability scenario:

  1. POST /api/chat/message — create the API observability dashboard.
  2. Confirm schema validation + save to disk.
  3. POST /api/chat/message — answer the save prompt ("api obs").
  4. GET  /api/saved_dashboards/ — library now contains the entry.
  5. POST /api/chat/message — "move critical widgets to the top".
  6. POST /api/chat/message — "add error-rate threshold at 2%".
  7. POST /api/chat/message — "change CPU chart to memory chart".
  8. GET  /api/dashboard/{id} — reload and verify every patch
     landed correctly.

No Playwright. No browser. Deterministic.

Usage:
    cd helper-dashboard
    python3 scripts/tier2_api_observability.py           # mock (default)
    TIER2_LLM=opencode python3 scripts/tier2_api_observability.py

Exit code: 0 on success, non-zero otherwise.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"


def _reset_storage() -> None:
    for sub in ("dashboards", "tickets", "saved_dashboards"):
        d = BACKEND / "app" / "storage" / sub
        if d.exists():
            for p in d.glob("*.json"):
                p.unlink()


def fmt(ok: bool) -> str:
    return "\033[92mPASS\033[0m" if ok else "\033[91mFAIL\033[0m"


def main() -> int:
    mode = os.environ.get("TIER2_LLM", "mock").strip().lower()
    os.environ["HELPER_DASHBOARD_OPENCODE"] = mode
    os.environ["HELPER_DASHBOARD_PRE_OUTPUT_REVIEW"] = "0"
    if mode == "opencode":
        os.environ["HELPER_DASHBOARD_OPENCODE_BIN"] = str(
            ROOT / "bin" / "opencode"
        )
        os.environ["HELPER_DASHBOARD_OPENCODE_CMD"] = json.dumps(["{bin}"])

    _reset_storage()

    sys.path.insert(0, str(BACKEND))
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    print(f"Tier 2 — API observability E2E  (mode={mode})")
    print()

    passes = 0
    total = 0

    def record(label: str, ok: bool, detail: str = "") -> None:
        nonlocal passes, total
        total += 1
        if ok:
            passes += 1
        print(f"  {fmt(ok)}  {label}")
        if detail:
            print(f"        {detail}")

    # 1. create via chat -------------------------------------------
    r = client.post("/api/chat/message", json={
        "session_id": "t2-api",
        "message": (
            "Build me an API observability dashboard with HTTP request "
            "rate, 5xx error rate, p95 latency, p99 latency, service "
            "availability, CPU, memory, and firing alerts."
        ),
    })
    ok = r.status_code == 200 and r.json().get("intent_type") == "DashboardSpec"
    body = r.json() if ok else {}
    did = (body.get("dashboard") or {}).get("dashboard_id")
    widgets = body.get("dashboard", {}).get("widgets", [])
    record(
        "POST /api/chat/message (create) -> DashboardSpec",
        ok,
        f"id={did}  widgets={len(widgets)}",
    )
    if not ok:
        print(f"        reply={body.get('user_reply', '')[:200]}")
        return 1

    # 2. widget quality --------------------------------------------
    types = [w["type"] for w in widgets]
    record("widgets >= 6", len(widgets) >= 6, f"got {len(widgets)}")
    record(
        "every widget uses an allowed type",
        all(t in {"line_chart", "stat_card", "gauge", "table", "alert_list"}
            for t in types),
        f"types={types}",
    )
    record(
        "includes latency percentile widget",
        any("histogram_quantile" in w["query"]["promql"]
            for w in widgets),
    )
    record(
        "includes request-rate widget",
        any("rate(http_requests_total" in w["query"]["promql"]
            for w in widgets),
    )
    record(
        "has at least one threshold",
        any((w.get("thresholds") or []) for w in widgets),
    )

    # 3. answer save prompt ----------------------------------------
    record("save prompt issued", bool(body.get("save_prompt_for")),
           f"save_prompt_for={body.get('save_prompt_for')}")

    r = client.post("/api/chat/message", json={
        "session_id": "t2-api",
        "message": "api obs",
    })
    ok = r.status_code == 200 and "Saved as" in (r.json().get("user_reply") or "")
    record("save prompt answered -> library updated", ok,
           f"reply={(r.json() or {}).get('user_reply', '')[:100]}")

    r = client.get("/api/saved_dashboards/")
    names = [s["name"] for s in r.json().get("saved", [])]
    record("library contains 'api obs'", "api obs" in names, f"library={names}")

    # 4. patch: move critical widgets to top -----------------------
    r = client.post("/api/chat/message", json={
        "session_id": "t2-api",
        "message": "move critical widgets to the top",
        "current_dashboard_id": did,
    })
    body = r.json() if r.status_code == 200 else {}
    ok = (r.status_code == 200
          and body.get("intent_type") == "PatchSpec"
          and any(op["op"] == "reorder_widgets"
                  for op in body.get("patch", {}).get("operations", [])))
    record("patch: 'move critical widgets to the top' -> reorder", ok)

    # Skip another save prompt by answering "no".
    client.post("/api/chat/message",
                 json={"session_id": "t2-api", "message": "no"})

    # 5. patch: add error-rate threshold --------------------------
    r = client.post("/api/chat/message", json={
        "session_id": "t2-api",
        "message": "add error-rate threshold at 2%",
        "current_dashboard_id": did,
    })
    body = r.json() if r.status_code == 200 else {}
    ops = body.get("patch", {}).get("operations", [])
    error_upd = next(
        (op for op in ops if op["op"] == "update_widget"
         and "error" in op.get("widget_id", "").lower()),
        None,
    )
    threshold_added = False
    if error_upd:
        thrs = error_upd.get("fields", {}).get("thresholds") or []
        threshold_added = any(abs(t.get("value", 0) - 0.02) < 1e-6 for t in thrs)
    record(
        "patch: 'add error-rate threshold at 2%' landed on error widget",
        threshold_added,
        f"op_target={error_upd and error_upd.get('widget_id')}",
    )
    client.post("/api/chat/message",
                 json={"session_id": "t2-api", "message": "no"})

    # 6. patch: change CPU chart to memory chart ------------------
    r = client.post("/api/chat/message", json={
        "session_id": "t2-api",
        "message": "change CPU chart to memory chart",
        "current_dashboard_id": did,
    })
    body = r.json() if r.status_code == 200 else {}
    ops = body.get("patch", {}).get("operations", [])
    cpu_upd = next(
        (op for op in ops if op["op"] == "update_widget"
         and "cpu" in op.get("widget_id", "").lower()),
        None,
    )
    swapped = False
    if cpu_upd:
        promql = cpu_upd.get("fields", {}).get("query", {}).get("promql", "")
        swapped = "memory" in promql.lower()
    record(
        "patch: 'change CPU chart to memory chart' swaps to memory PromQL",
        swapped,
        f"op_target={cpu_upd and cpu_upd.get('widget_id')}",
    )
    client.post("/api/chat/message",
                 json={"session_id": "t2-api", "message": "no"})

    # 7. reload the final dashboard and verify the patches stuck --
    r = client.get(f"/api/dashboard/{did}")
    if r.status_code != 200:
        record("reload dashboard after all patches", False,
               f"HTTP {r.status_code}")
        return 1
    final = r.json()["spec"]
    by_id = {w["id"]: w for w in final["widgets"]}

    # After CPU->memory swap, the widget still has id "w-cpu" but
    # new title/query.
    cpu = by_id.get("w-cpu")
    record(
        "reload: w-cpu now has memory PromQL",
        cpu is not None and "memory" in cpu["query"]["promql"].lower(),
        f"cpu.promql={cpu and cpu['query']['promql'][:80]}",
    )

    err_w = by_id.get("w-error-rate")
    has_2pct = False
    if err_w:
        has_2pct = any(
            abs(t["value"] - 0.02) < 1e-6 for t in err_w.get("thresholds", [])
        )
    record(
        "reload: error widget has 0.02 threshold",
        has_2pct,
        f"thresholds={err_w and [t['value'] for t in err_w.get('thresholds', [])]}",
    )

    # 8. library still has the original saved spec ---------------
    r = client.get("/api/saved_dashboards/")
    names = [s["name"] for s in r.json().get("saved", [])]
    record("library still has 'api obs' after patches",
           "api obs" in names, f"library={names}")

    print()
    print(f"Tier 2 API observability: {passes}/{total} checks")
    if passes == total:
        print(fmt(True) + "  end-to-end scenario verified")
        return 0
    print(fmt(False) + "  scenario degraded")
    return 1


if __name__ == "__main__":
    sys.exit(main())
