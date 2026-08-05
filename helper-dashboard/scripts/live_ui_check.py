"""Live UI <-> backend round-trip test.

Starts uvicorn (FastAPI) and `next start` (production Next.js),
hits them with HTTP exactly like a real browser would, prints a
human-readable trace, and cleans up.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"


def hr(c="="): print(c * 72)
def banner(t): hr(); print("  " + t); hr()
def sub(t): print(f"\n--- {t} ---")


def post(url, body, timeout=8):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())

def get(url, timeout=5, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def wait_healthy(url, tries=30):
    for _ in range(tries):
        try:
            s, b = get(url, timeout=1)
            if s == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main():
    env = {**os.environ, "HELPER_DASHBOARD_OPENCODE": "mock",
           "PYTHONPATH": str(BACKEND_DIR)}
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--host", "127.0.0.1", "--port", "8000", "--log-level", "warning"],
        env=env, cwd=str(BACKEND_DIR),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    frontend = subprocess.Popen(
        ["npx", "next", "start", "-p", "3010"],
        cwd=str(FRONTEND_DIR),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    try:
        print(f"backend pid={backend.pid}, frontend pid={frontend.pid}; waiting...")
        assert wait_healthy("http://127.0.0.1:8000/api/health"), "backend"
        assert wait_healthy("http://127.0.0.1:3010/", tries=40), "frontend"

        banner("LIVE UI <-> BACKEND in MOCK mode (default)")

        sub("User lands on / (Next.js three-panel UI, SSR)")
        s, html = get("http://127.0.0.1:3010/")
        html = html.decode()
        print(f"GET /  -> HTTP {s}  bytes={len(html)}")
        for lbl in ["Helper", "No dashboard yet", "Ask Helper", "Send",
                    "Run browser evaluation", "Spec", "Last patch", "Evaluation"]:
            print(f"  UI element {lbl!r}:"
                  f" {'present' if lbl in html else 'MISSING'}")

        sub("User types a message (browser -> Next.js proxy -> FastAPI)")
        s, body = post(
            "http://127.0.0.1:3010/api/chat/message",
            {"session_id": "live-ui",
             "message": "Show me a CPU memory and alert dashboard"},
        )
        print(f"POST /api/chat/message -> HTTP {s}")
        print(f"  intent_type:    {body['intent_type']}")
        print(f"  runtime_used:   {body['runtime_used']}")
        print(f"  user_reply:     {body['user_reply']}")
        d = body["dashboard"]
        did = d["dashboard_id"]
        print(f"  dashboard_id:   {did}")
        print(f"  widgets:")
        for w in d["widgets"]:
            print(f"    - {w['id']:25s}  type={w['type']:12s}  "
                  f"promql={w['query']['promql']}")

        sub(f"Browser renders /dashboard/{did} (loading then client fetch)")
        s, html = get(f"http://127.0.0.1:3010/dashboard/{did}")
        m = re.search(rb'data-dashboard-state="([a-z]+)"', html)
        print(f"GET /dashboard/{did}  -> HTTP {s}")
        print(f"  data-dashboard-state: {m.group(1).decode() if m else 'MISSING'}"
              f"  (loading until JS hydrates, then 'ready')")

        sub("User patches via chat")
        s, body = post(
            "http://127.0.0.1:3010/api/chat/message",
            {"session_id": "live-ui", "message": "add a gauge",
             "current_dashboard_id": did},
        )
        print(f"POST /api/chat/message -> HTTP {s}")
        print(f"  intent_type:   {body['intent_type']}")
        print(f"  runtime_used:  {body['runtime_used']}")
        print(f"  operations:    {[op['op'] for op in body['patch']['operations']]}")
        print(f"  widget types now: "
              f"{[w['type'] for w in body['dashboard']['widgets']]}")

        sub("User clicks 'Run browser evaluation' in the Inspector panel")
        s, body = post(
            "http://127.0.0.1:3010/api/evaluate/run",
            {"dashboard_id": did},
        )
        r = body["report"]
        print(f"POST /api/evaluate/run -> HTTP {s}")
        print(f"  page_loaded:     {r['page_loaded']}")
        print(f"  widgets found:   {r['widgets_rendered']}")
        print(f"  missing widgets: {r['missing_widgets']}")
        print(f"  console_errors:  {r['console_errors']}")
        print(f"  recommendation:  {r['recommendation']}")

        sub("Developer endpoint still gated (no user path)")
        s, _ = get("http://127.0.0.1:3010/api/developer/tickets")
        print(f"  no token  -> HTTP {s}")
        s, _ = get("http://127.0.0.1:3010/api/developer/tickets",
                   headers={"X-Developer-Token": "bad"})
        print(f"  bad token -> HTTP {s}")

        banner("DONE")
    finally:
        for p in (backend, frontend):
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
        for p in (backend, frontend):
            try:
                p.wait(timeout=5)
            except Exception:
                try: p.kill()
                except Exception: pass


if __name__ == "__main__":
    main()
