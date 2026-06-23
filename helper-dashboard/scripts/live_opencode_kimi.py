"""End-to-end live demo using real OpenCode CLI + Kimi K2.

Starts backend in `opencode` mode with the real bin/opencode binary
and OPENROUTER_API_KEY loaded from .env. Drives a real Chromium via
Playwright through the whole journey:

  1. User lands on /, sees connection dot green.
  2. User asks for a dashboard — Helper routes via real LLM.
  3. Review loop runs in the backend (synchronous).
  4. Review trail appears in the Inspector's Review tab.
  5. Helper asks "Would you like to save this dashboard?"
  6. User names it; it goes to the library.
  7. User starts a new session — Helper sees the saved one.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"
SHOT_DIR = (
    BACKEND_DIR / "app" / "storage" / "evaluation_reports"
    / "screenshots" / "opencode_live"
)
SHOT_DIR.mkdir(parents=True, exist_ok=True)


def hr(c="="): print(c * 72)
def banner(t): hr(); print("  " + t); hr()
def step(t): print(f"\n[step] {t}")


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
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def start_backend():
    dotenv = load_dotenv()
    env = {
        **os.environ,
        **dotenv,
        "PYTHONPATH": str(BACKEND_DIR),
        # Force opencode mode explicitly for this demo.
        "HELPER_DASHBOARD_OPENCODE": "opencode",
        "HELPER_DASHBOARD_OPENCODE_BIN": str(ROOT / "bin" / "opencode"),
        "HELPER_DASHBOARD_OPENCODE_CMD": json.dumps(["{bin}"]),
        "HELPER_DASHBOARD_PRE_OUTPUT_REVIEW": "0",
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


def shot(page, name: str):
    p = SHOT_DIR / f"{name}.png"
    page.screenshot(path=str(p), full_page=True)
    print(f"  -> {p.relative_to(ROOT)}")
    return p


def main():
    backend = start_backend()
    frontend = start_frontend()
    try:
        print(f"backend pid={backend.pid}, frontend pid={frontend.pid}")
        assert wait_http("http://127.0.0.1:8000/api/health"), "backend"
        assert wait_http("http://127.0.0.1:3010/", tries=60), "frontend"

        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(viewport={"width": 1440, "height": 900})
            page = ctx.new_page()

            console_errors = []
            page.on(
                "console",
                lambda m: console_errors.append(m.text)
                if m.type == "error" else None,
            )

            banner("PART 1 — LOAD /, CONNECTION DOT SHOULD BE GREEN")
            page.goto("http://127.0.0.1:3010/", wait_until="networkidle")
            # Wait for the health probe to succeed.
            page.wait_for_function(
                "() => document.body.innerHTML.includes('bg-emerald-400')",
                timeout=20_000,
            )
            print("  connection dot: green ✓")
            shot(page, "01_home_green_dot")

            banner("PART 2 — USER ASKS FOR A DASHBOARD (opencode / Kimi K2)")
            box = page.get_by_placeholder(
                "Ask Helper for a dashboard or a change…"
            )
            box.click()
            box.fill(
                "Build me a dashboard showing the HTTP request rate and "
                "the p95 request latency for the api service."
            )
            shot(page, "02_user_typed")
            page.get_by_role("button", name="Send").click()

            # This is a real LLM call plus a review loop. Be patient.
            step("waiting for review loop to finish (can take 1-3 min)...")
            page.wait_for_selector("[data-widget-id]", timeout=240_000)
            widgets = page.eval_on_selector_all(
                "[data-widget-id]",
                "els => els.map(e => e.getAttribute('data-widget-id'))",
            )
            print(f"  widgets on screen: {widgets}")
            shot(page, "03_dashboard_rendered_via_opencode")

            banner("PART 3 — INSPECTOR REVIEW TAB SHOWS THE TRAIL")
            page.get_by_role("button", name="Review", exact=True).click()
            page.wait_for_function(
                "() => document.body.innerText.includes('render')",
                timeout=5_000,
            )
            print("  review tab: populated")
            shot(page, "04_inspector_review_tab")

            banner("PART 4 — SAVE PROMPT APPEARS IN CHAT")
            # The helper reply contains the save-prompt text.
            ok = page.wait_for_function(
                """() => Array.from(document.querySelectorAll('*'))
                    .some(e => e.textContent &&
                               e.textContent.includes('save this dashboard'))""",
                timeout=5_000,
            )
            print("  save prompt visible: yes")
            shot(page, "05_save_prompt")

            step("User types a name into the save quick-input")
            save_input = page.get_by_placeholder("name…")
            save_input.click()
            save_input.fill("api observability")
            page.get_by_role("button", name="save").click()
            page.wait_for_function(
                """() => Array.from(document.querySelectorAll('*'))
                    .some(e => e.textContent &&
                               e.textContent.includes('Saved as'))""",
                timeout=10_000,
            )
            print("  library: 'api observability' saved")
            shot(page, "06_saved_to_library")

            banner("PART 5 — RESTART SESSION, HELPER SEES THE LIBRARY")
            ctx.clear_cookies()
            page.goto("http://127.0.0.1:3010/?fresh=1",
                       wait_until="networkidle")
            box = page.get_by_placeholder(
                "Ask Helper for a dashboard or a change…"
            )
            box.click()
            box.fill("Do we have an api observability dashboard saved?")
            page.get_by_role("button", name="Send").click()

            step("waiting for Helper to respond (LLM call)...")
            page.wait_for_function(
                """() => {
                    const msgs = document.querySelectorAll('.bg-slate-800');
                    return msgs.length >= 1;
                }""",
                timeout=60_000,
            )
            # Give the LLM reply a moment to render.
            time.sleep(2)
            shot(page, "07_new_session_recognises_library")

            banner("SUMMARY")
            print(f"  console errors: {len(console_errors)}")
            for e in console_errors[:6]:
                print(f"    - {e[:160]}")
            browser.close()
    finally:
        stop(backend)
        stop(frontend)


if __name__ == "__main__":
    main()
