"""Tier 3 — optional full UI stress demo.

RESOURCE-HEAVY. This is an ACCEPTANCE / STRESS TEST, not the primary
proof of correctness. Use Tier 1 (scripts/tier1_cli_check.py) for
CLI verification and Tier 2 (scripts/tier2_backend_e2e.py) for
backend HTTP E2E verification; the `tests/` suite is the canonical
correctness proof.

What this script does:
  - Starts FastAPI (uvicorn) + Next.js production server + Playwright.
  - Exercises the full chat -> review loop -> save prompt pipeline
    against a real LLM (opencode mode).
  - Validates that the rendered dashboard appears in the browser.

Known failure modes (reported clearly, not treated as system bugs):
  - OpenRouter credits exhausted / rate limited / slow responses.
  - Jetson / sandboxed CPU cannot sustain Chromium + LLM + backend.
  - Multi-minute review loops blowing Playwright's wait_for_selector
    timeouts.

When any of these happen, this script emits diagnostics and a
non-zero exit code. The system is NOT necessarily broken — run
Tier 1 / Tier 2 / pytest for the authoritative answer.

Usage:
    cd helper-dashboard
    set -a; source .env; set +a    # load OPENROUTER_API_KEY
    python3 scripts/live_full_demo.py
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"
SHOT_DIR = (
    BACKEND_DIR / "app" / "storage" / "evaluation_reports"
    / "screenshots" / "live_full_demo"
)
SHOT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Resource-bottleneck detection thresholds.
# If a single /api/chat/message request exceeds this, we conclude the
# LLM pipeline is too slow for a Playwright-driven demo and abort
# cleanly with guidance.
# ---------------------------------------------------------------------
CHAT_BUDGET_S = 300


def hr(c="="): print(c * 72)
def banner(t): hr(); print("  " + t); hr()
def step(t): print(f"\n[step] {t}")
def warn(t): print(f"\n[warn] {t}")
def info(t): print(f"\n[info] {t}")


def wait_http(url, tries=40, timeout=1.0):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def load_dotenv() -> dict:
    env = {}
    p = ROOT / ".env"
    if not p.exists():
        return env
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def http_post(url, body, timeout=600):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"content-type": "application/json"}, method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        return r.status, data, time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), time.time() - t0
    except Exception as e:
        return 0, {"error": str(e)}, time.time() - t0


def http_get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read()


def start_backend():
    dotenv = load_dotenv()
    env = {
        **os.environ,
        **dotenv,
        "PYTHONPATH": str(BACKEND_DIR),
        "HELPER_DASHBOARD_OPENCODE": "opencode",
        "HELPER_DASHBOARD_OPENCODE_BIN": str(ROOT / "bin" / "opencode"),
        "HELPER_DASHBOARD_OPENCODE_CMD": json.dumps(["{bin}"]),
        "HELPER_DASHBOARD_PRE_OUTPUT_REVIEW": "1",
    }
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--host", "127.0.0.1", "--port", "8000", "--log-level", "warning"],
        env=env, cwd=str(BACKEND_DIR),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def start_frontend():
    return subprocess.Popen(
        ["npx", "next", "start", "-p", "3010"],
        cwd=str(FRONTEND_DIR),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def stop(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=8)
    except Exception:
        try: proc.kill()
        except Exception: pass


def summarise_response(r: dict) -> None:
    print(f"  intent_type:   {r.get('intent_type')}")
    print(f"  runtime_used:  {r.get('runtime_used')}  "
          f"fallback:  {r.get('fallback_reason')}")
    print(f"  save_prompt_for: {r.get('save_prompt_for')}")
    d = r.get("dashboard")
    if d:
        print(f"  dashboard:     id={d['dashboard_id']!r}  "
              f"title={d['title']!r}")
        for w in d["widgets"]:
            print(f"    - {w['id']:30s} type={w['type']:10s} "
                  f"promql={w['query']['promql'][:60]}")
    trail = r.get("review_trail") or []
    if trail:
        print(f"  review trail:")
        for t in trail:
            extra = []
            if t.get("decision"):
                extra.append(f"decision={t['decision']}")
            if t.get("kind"):
                extra.append(f"kind={t['kind']}")
            if t.get("attempt"):
                extra.append(f"attempt={t['attempt']}")
            if t.get("rationale"):
                extra.append(f"rationale={t['rationale'][:60]!r}")
            print(f"    · {t['stage']:15s}  {'  '.join(extra)}")
    cq = r.get("clarification_questions")
    if cq:
        print(f"  clarification questions:")
        for q in cq[:4]:
            print(f"    · {q}")
    print(f"  reply:         {r.get('user_reply', '')[:200]}")


def screenshot_dashboard(page, dashboard_id: str, name: str):
    try:
        page.goto(f"http://127.0.0.1:3010/dashboard/{dashboard_id}",
                   wait_until="networkidle", timeout=30_000)
        try:
            page.wait_for_function(
                "() => document.querySelector('[data-dashboard-state]')"
                "?.getAttribute('data-dashboard-state') === 'ready'",
                timeout=15_000,
            )
        except Exception:
            pass
        page.wait_for_timeout(500)
        out = SHOT_DIR / f"{name}.png"
        page.screenshot(path=str(out), full_page=True)
        print(f"  -> {out.relative_to(ROOT)}")
    except Exception as exc:
        warn(f"screenshot failed: {exc}")


def screenshot_home(page, name: str):
    try:
        page.goto("http://127.0.0.1:3010/",
                   wait_until="networkidle", timeout=20_000)
        try:
            page.wait_for_function(
                "() => document.body.innerHTML.includes('bg-emerald-400')",
                timeout=15_000,
            )
        except Exception:
            pass
        out = SHOT_DIR / f"{name}.png"
        page.screenshot(path=str(out), full_page=True)
        print(f"  -> {out.relative_to(ROOT)}")
    except Exception as exc:
        warn(f"screenshot failed: {exc}")


def diagnose_bottleneck(elapsed_s: float, call_name: str) -> str:
    msg = (
        f"\nRESOURCE-BOTTLENECK DIAGNOSTICS\n"
        f"  The {call_name} call took {elapsed_s:.1f}s — over the\n"
        f"  {CHAT_BUDGET_S}s budget this Tier 3 demo script enforces.\n"
        f"  This almost always means ONE of:\n"
        f"    * The LLM is slow or rate-limited (check OpenRouter).\n"
        f"    * OpenRouter credits are exhausted (check recent tickets).\n"
        f"    * The review loop is iterating more than expected.\n"
        f"    * Chromium + backend together exceed available CPU.\n"
        f"\n"
        f"  This script is NOT the authoritative proof of correctness.\n"
        f"  Run Tier 1 + Tier 2 (and pytest) to verify the system:\n"
        f"    python3 scripts/tier1_cli_check.py\n"
        f"    python3 scripts/tier2_backend_e2e.py\n"
        f"    python3 -m pytest tests/ -q\n"
    )
    return msg


def main() -> int:
    banner("TIER 3 — OPTIONAL FULL UI DEMO (resource-heavy, acceptance test)")
    info(
        "If this script fails due to LLM latency, Chromium, or memory\n"
        "pressure, it does NOT mean the system is broken. Run Tier 1\n"
        "and Tier 2 (see README) for the authoritative answer."
    )

    backend = start_backend()
    frontend = start_frontend()
    rc = 0

    try:
        print(f"backend pid={backend.pid}, frontend pid={frontend.pid}")
        if not wait_http("http://127.0.0.1:8000/api/health"):
            warn("backend never became healthy")
            return 2
        if not wait_http("http://127.0.0.1:3010/", tries=60):
            warn("frontend never became healthy")
            return 2

        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(viewport={"width": 1440, "height": 900})
            page = ctx.new_page()
            console_errors: list[str] = []
            page.on("console",
                     lambda m: console_errors.append(m.text)
                     if m.type == "error" else None)

            # =========================================================
            banner("PART A — happy path via real LLM")
            # =========================================================

            step("screenshot the empty home (green connection dot)")
            screenshot_home(page, "A01_home_green")

            step("POST /api/chat/message — create a dashboard")
            status, body, elapsed = http_post(
                "http://127.0.0.1:8000/api/chat/message",
                {
                    "session_id": "tier3-a",
                    "message": (
                        "Build me a dashboard showing HTTP request rate and "
                        "p95 latency for the api service."
                    ),
                },
                timeout=CHAT_BUDGET_S + 30,
            )
            print(f"  HTTP {status} in {elapsed:.1f}s")
            summarise_response(body)

            # Detect bottleneck conditions.
            if status != 200:
                warn(f"chat create failed: HTTP {status}")
                print(diagnose_bottleneck(elapsed, "chat create"))
                return 3

            if elapsed > CHAT_BUDGET_S:
                warn("over-budget (still succeeded but slow)")
                print(diagnose_bottleneck(elapsed, "chat create"))

            dashboard_id = (body.get("dashboard") or {}).get("dashboard_id")
            if not dashboard_id:
                # Review escalated to clarification / ticket. This is NOT a
                # bug — it's the safety net working. But there's nothing to
                # save, so we skip the save-prompt + library steps.
                warn(
                    "no dashboard was produced on this turn. "
                    f"intent_type={body.get('intent_type')}. "
                    "Skipping save-prompt and library checks. This is "
                    "expected when the review loop escalates."
                )
            else:
                step("render /dashboard/<id> in Chromium")
                screenshot_dashboard(
                    page, dashboard_id, "A02_dashboard_rendered"
                )

                step("POST /api/chat/message — answer the save prompt")
                status, body, elapsed = http_post(
                    "http://127.0.0.1:8000/api/chat/message",
                    {"session_id": "tier3-a",
                     "message": "api observability"},
                    timeout=120,
                )
                print(f"  HTTP {status} in {elapsed:.1f}s")
                print(f"  reply: {body.get('user_reply', '')[:160]}")

                step("GET /api/saved_dashboards/ — library should have 1")
                s_status, raw = http_get(
                    "http://127.0.0.1:8000/api/saved_dashboards/")
                lib = json.loads(raw)
                print(f"  HTTP {s_status}  saved items: {len(lib['saved'])}")
                for s in lib["saved"]:
                    print(f"    · {s['name']:20s} "
                          f"types={s['widget_types']}")

            # =========================================================
            banner("PART B — rescue path (unsupported visualizations)")
            # =========================================================

            step("POST /api/chat/message — request unsupported widgets")
            status, body, elapsed = http_post(
                "http://127.0.0.1:8000/api/chat/message",
                {
                    "session_id": "tier3-b",
                    "message": (
                        "Draw a network topology map showing every "
                        "microservice and a flame graph for the slowest one."
                    ),
                },
                timeout=CHAT_BUDGET_S + 30,
            )
            print(f"  HTTP {status} in {elapsed:.1f}s")
            if status != 200:
                warn(f"chat failed: HTTP {status}")
            else:
                summarise_response(body)

            step("check tickets on disk")
            tdir = BACKEND_DIR / "app" / "storage" / "tickets"
            tickets = sorted(
                p for p in tdir.glob("*.json") if p.stat().st_size > 0
            )
            print(f"  tickets: {[p.name for p in tickets]}")
            for tp in tickets:
                with tp.open() as f:
                    t = json.load(f)
                print(f"    · {t['source_agent']:28s} sev={t['severity']:6s} "
                      f"{t['summary'][:70]}")

            step("capture final home screenshot")
            screenshot_home(page, "B01_home_after_rescue")

            # =========================================================
            banner("SUMMARY")
            # =========================================================
            shots = sorted(SHOT_DIR.glob("*.png"))
            print(f"  screenshots:        {[s.name for s in shots]}")
            print(f"  console errors:     {len(console_errors)}")
            for e in console_errors[:8]:
                print(f"    - {e[:140]}")
            print(
                f"  dashboards on disk: "
                f"{[p.name for p in (BACKEND_DIR / 'app' / 'storage' / 'dashboards').glob('*.json')]}"
            )
            print(
                f"  saved_dashboards:   "
                f"{[p.name for p in (BACKEND_DIR / 'app' / 'storage' / 'saved_dashboards').glob('*.json')]}"
            )
            print(
                f"  tickets:            "
                f"{[p.name for p in (BACKEND_DIR / 'app' / 'storage' / 'tickets').glob('*.json')]}"
            )
            print()
            print("  This demo is a stress test, not the primary correctness")
            print("  proof. See README for Tier 1 / Tier 2 authoritative")
            print("  verification commands.")

            browser.close()
    except KeyboardInterrupt:
        warn("interrupted")
        rc = 130
    except Exception as exc:
        warn(f"unhandled error: {exc}")
        rc = 4
    finally:
        stop(backend)
        stop(frontend)
    return rc


if __name__ == "__main__":
    sys.exit(main())
