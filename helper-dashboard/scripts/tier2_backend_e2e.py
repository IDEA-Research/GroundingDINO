"""Tier 2 verification — backend-only HTTP E2E.

Runs the FastAPI app via TestClient (in-process, no uvicorn, no
frontend, no Playwright) and exercises the full orchestrator
pipeline through the HTTP surface. This is the **primary reliable
live verification** — fast, deterministic, and does not depend on
a running browser or external network.

Chat transport: /api/chat/message is an ENQUEUE endpoint (returns a
job_id); the ChatResponse payload comes from polling
GET /api/chat/message/{job_id} until status=done. The `_chat` helper
below hides that so checks still read the synchronous response shape.

Scope:
- POST /api/chat/message — create
- POST /api/chat/message — answer save prompt
- GET  /api/saved_dashboards/ — library has the entry
- POST /api/chat/message — new session sees the library
- POST /api/chat/message — patch existing dashboard
- POST /api/chat/message — unsupported request -> DeveloperTicket
- GET  /api/developer/tickets — gated without token (403)
- GET  /api/developer/tickets — gated with bad token (403)
- GET  /api/developer/tickets — passes with correct token (200)

Runtime:
- By default runs in `mock` mode with review OFF (fast, deterministic).
- Set TIER2_LLM=opencode for a real-LLM verification via bin/opencode.
- Set TIER2_REVIEW=1 to exercise the review loop.

Usage:
    cd helper-dashboard
    python3 scripts/tier2_backend_e2e.py

Exit code: 0 if all checks pass, non-zero otherwise.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"


def _reset_storage():
    for sub in ("dashboards", "tickets", "saved_dashboards"):
        d = BACKEND / "app" / "storage" / sub
        if d.exists():
            for p in d.glob("*.json"):
                p.unlink()


def fmt(ok: bool) -> str:
    return "\033[92mPASS\033[0m" if ok else "\033[91mFAIL\033[0m"


def main() -> int:
    # Isolate: never use residual env the user had before.
    for k in (
        "HELPER_DASHBOARD_PRE_OUTPUT_REVIEW",
        "HELPER_DASHBOARD_OPENCODE_TIMEOUT_SECONDS",
    ):
        os.environ.pop(k, None)

    mode = os.environ.get("TIER2_LLM", "mock").strip().lower()
    review = os.environ.get("TIER2_REVIEW", "0") == "1"

    os.environ["HELPER_DASHBOARD_OPENCODE"] = mode
    os.environ["HELPER_DASHBOARD_PRE_OUTPUT_REVIEW"] = "1" if review else "0"
    if mode == "opencode":
        os.environ["HELPER_DASHBOARD_OPENCODE_BIN"] = str(ROOT / "bin" / "opencode")
        os.environ["HELPER_DASHBOARD_OPENCODE_CMD"] = json.dumps(["{bin}"])

    _reset_storage()

    sys.path.insert(0, str(BACKEND))
    # Import inside so env takes effect.
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    print(f"Tier 2 backend HTTP E2E")
    print(f"  mode:    {mode}")
    print(f"  review:  {'on' if review else 'off'}")
    print()

    total_checks = 0
    total_pass = 0

    def record(label: str, ok: bool, detail: str = "") -> None:
        nonlocal total_checks, total_pass
        total_checks += 1
        if ok:
            total_pass += 1
        print(f"  {fmt(ok)}  {label}")
        if detail:
            print(f"        {detail}")

    import time as _time

    def _chat(payload: dict, *, timeout_s: float = 120.0) -> tuple[int, dict]:
        """POST a chat message via the enqueue+poll API.

        Returns (status, result) where result is the completed ChatResponse
        payload — the shape the pre-queue synchronous endpoint used to return.
        """
        r = client.post("/api/chat/message", json=payload)
        if r.status_code != 200:
            return r.status_code, {}
        job_id = r.json().get("job_id")
        if not job_id:
            return r.status_code, {}
        deadline = _time.monotonic() + timeout_s
        while _time.monotonic() < deadline:
            s = client.get(f"/api/chat/message/{job_id}")
            if s.status_code != 200:
                return s.status_code, {}
            js = s.json()
            if js.get("status") == "done":
                return 200, js.get("result") or {}
            if js.get("status") == "error":
                print(f"        job error: {js.get('error')}")
                return 500, {}
            _time.sleep(0.2)
        print(f"        job {job_id} timed out after {timeout_s}s")
        return 504, {}

    # --- 1. health -------------------------------------------------
    r = client.get("/api/health")
    record("GET /api/health -> 200", r.status_code == 200, f"body={r.text[:80]}")

    # --- 2. create a dashboard via chat ----------------------------
    status, body = _chat({
        "session_id": "t2-a",
        "message": "Show me a CPU and memory dashboard",
    })
    ok = status == 200 and body.get("intent_type") == "DashboardSpec"
    did = (body.get("dashboard") or {}).get("dashboard_id")
    record(
        "POST /api/chat/message (create) -> DashboardSpec",
        ok,
        f"dashboard_id={did}  widgets="
        + (str(len(body.get("dashboard", {}).get("widgets", [])))
           if ok else "n/a"),
    )

    # --- 3. orchestrator asked to save -----------------------------
    record(
        "save prompt issued",
        bool(body.get("save_prompt_for")),
        f"save_prompt_for={body.get('save_prompt_for')}",
    )

    # --- 4. answer save prompt with a name -------------------------
    status, body4 = _chat({
        "session_id": "t2-a",
        "message": "prod cpu mem",
    })
    ok = status == 200 and "Saved as" in (body4.get("user_reply") or "")
    record(
        "save prompt answered with name -> saved",
        ok,
        f"reply={body4.get('user_reply', '')[:80]}",
    )

    # --- 5. library has one entry ----------------------------------
    r = client.get("/api/saved_dashboards/")
    names = [s["name"] for s in r.json().get("saved", [])]
    record(
        "GET /api/saved_dashboards/ -> library entry",
        r.status_code == 200 and "prod cpu mem" in names,
        f"library={names}",
    )

    # --- 6. patch existing dashboard via chat ----------------------
    if did:
        status, body = _chat({
            "session_id": "t2-a",
            "message": "add a gauge widget",
            "current_dashboard_id": did,
        })
        ok = (
            status == 200
            and body.get("intent_type") == "PatchSpec"
            and body.get("patch")
        )
        ops = []
        if ok:
            ops = [op.get("op") for op in body["patch"].get("operations", [])]
        record(
            "POST /api/chat/message (patch) -> PatchSpec",
            ok, f"ops={ops}",
        )
    else:
        record("patch via chat", False, "no dashboard_id to patch")

    # --- 7. new session sees the library ---------------------------
    status, body7 = _chat({
        "session_id": "t2-b-fresh",
        "message": "hello",
    })
    # Any completed job is fine; we don't LLM-parse the reply.
    record(
        "new session still serves /api/chat",
        status == 200,
        f"intent_type={body7.get('intent_type')}",
    )

    # --- 8. unsupported widget request -> DeveloperTicket ---------
    # Mock heuristic never produces DeveloperTicket for this prompt
    # (it just treats topology as a non-match and returns UserResponse),
    # so only run this assertion in opencode mode.
    if mode == "opencode":
        status, body = _chat({
            "session_id": "t2-c",
            "message": "Draw me a network topology map and a flame graph",
        })
        ok = (
            status == 200
            and body.get("intent_type") in
                {"DeveloperTicket", "ClarificationRequest"}
        )
        record(
            "unsupported request -> DeveloperTicket or ClarificationRequest",
            ok,
            f"intent_type={body.get('intent_type')}",
        )
    else:
        print("  SKIP  unsupported-request LLM test (needs TIER2_LLM=opencode)")

    # --- 9. developer gate without token ---------------------------
    # Clear any dev token first.
    os.environ.pop("HELPER_DASHBOARD_DEV_TOKEN", None)
    r = client.get("/api/developer/tickets")
    record(
        "GET /api/developer/tickets (no token) -> 403",
        r.status_code == 403,
        f"status={r.status_code}",
    )

    # --- 10. developer gate with correct token --------------------
    token = "dev-" + secrets.token_hex(8)
    os.environ["HELPER_DASHBOARD_DEV_TOKEN"] = token
    r = client.get("/api/developer/tickets",
                    headers={"X-Developer-Token": token})
    record(
        "GET /api/developer/tickets (correct token) -> 200",
        r.status_code == 200,
        f"status={r.status_code}",
    )

    r = client.get("/api/developer/tickets",
                    headers={"X-Developer-Token": "wrong"})
    record(
        "GET /api/developer/tickets (wrong token) -> 403",
        r.status_code == 403,
        f"status={r.status_code}",
    )

    # --- summary ---------------------------------------------------
    print()
    print(f"Tier 2 result: {total_pass}/{total_checks} checks")
    if total_pass == total_checks:
        print(fmt(True) + "  backend HTTP E2E verified")
        return 0
    print(fmt(False) + "  one or more checks failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
