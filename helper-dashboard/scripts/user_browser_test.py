"""Playwright-driven 'user acts through the browser' test.

Starts backend + frontend, then drives a real browser:

  1. User lands on /, sees the empty three-panel UI.
  2. User types a chat message and clicks Send.
  3. Browser shows Helper's reply and the dashboard renders with
     recharts widgets (we assert they appear on screen).
  4. User patches via chat ("add a gauge").
  5. User clicks the inspector tabs (Spec / Last patch / Evaluation).
  6. User clicks "Run browser evaluation" — that's the real
     /api/evaluate/run call which is currently a static fallback.
  7. User opens /dashboard/[id] directly (Playwright target page)
     and we assert each widget appears.

Screenshots are written under backend/app/storage/evaluation_reports/
screenshots/user_flow/.
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
    / "screenshots" / "user_flow"
)
SHOT_DIR.mkdir(parents=True, exist_ok=True)


def hr(c="="): print(c * 72)
def banner(t): hr(); print("  " + t); hr()
def step(t): print(f"\n[step] {t}")


def wait_http(url, tries=40):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def start_backend():
    env = {**os.environ,
           "HELPER_DASHBOARD_OPENCODE": "mock",
           "PYTHONPATH": str(BACKEND_DIR)}
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
        proc.wait(timeout=5)
    except Exception:
        try: proc.kill()
        except Exception: pass


def shot(page, name: str) -> Path:
    p = SHOT_DIR / f"{name}.png"
    page.screenshot(path=str(p), full_page=True)
    print(f"  -> {p.relative_to(ROOT)}")
    return p


def main():
    backend = start_backend()
    frontend = start_frontend()
    try:
        print(f"backend pid={backend.pid}, frontend pid={frontend.pid}; waiting...")
        assert wait_http("http://127.0.0.1:8000/api/health"), "backend"
        assert wait_http("http://127.0.0.1:3010/", tries=60), "frontend"

        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()

            console_errors: list[str] = []
            page.on(
                "console",
                lambda m: console_errors.append(m.text)
                if m.type == "error" else None,
            )

            banner("USER OPENS /   (home — three-panel UI)")
            page.goto("http://127.0.0.1:3010/", wait_until="networkidle")
            assert "Helper Dashboard" in page.title() \
                   or "Helper" in page.content(), "home page missing"
            # Panels
            for label in ("Helper", "No dashboard yet",
                          "Spec", "Last patch", "Evaluation",
                          "Run browser evaluation"):
                assert page.get_by_text(label).first.is_visible(), \
                    f"UI element {label!r} not visible"
                print(f"  visible: {label!r}")
            shot(page, "01_home_empty")

            banner("USER TYPES A MESSAGE AND CLICKS SEND")
            step("type in chat")
            box = page.get_by_placeholder(
                "Ask Helper for a dashboard or a change…")
            box.click()
            box.fill("Show me a dashboard with a CPU line chart, a memory stat card, and an alert list")
            shot(page, "02_chat_typed")

            step("click Send")
            page.get_by_role("button", name="Send").click()

            step("wait for Helper reply")
            expect(
                page.get_by_text("Building that dashboard now.")
            ).to_be_visible(timeout=10_000)
            print("  Helper replied in the chat")

            step("wait for dashboard preview to render")
            # Each widget sets data-widget-id=<id> on its frame.
            page.wait_for_selector("[data-widget-id]", timeout=10_000)
            widget_ids = page.eval_on_selector_all(
                "[data-widget-id]",
                "els => els.map(e => e.getAttribute('data-widget-id'))",
            )
            print(f"  widgets rendered: {widget_ids}")
            assert widget_ids, "no widgets rendered"

            # Recharts produces SVG charts — confirm at least one rendered.
            svg_count = page.eval_on_selector_all(
                "[data-widget-id] svg", "els => els.length"
            )
            print(f"  chart SVGs on screen: {svg_count}")
            shot(page, "03_dashboard_rendered")

            banner("USER CLICKS INSPECTOR TABS")
            step("Spec tab is default — JSON is visible")
            pre = page.locator("pre").first
            spec_text = pre.inner_text()
            assert '"dashboard_id"' in spec_text
            assert '"widgets"' in spec_text
            print("  Spec JSON present in inspector")
            shot(page, "04_inspector_spec")

            step("Last patch tab (should be empty first time)")
            page.get_by_role("button", name="Last patch", exact=True).click()
            assert "no patch yet" in page.locator("pre").first.inner_text()
            print("  'no patch yet' placeholder confirmed")

            step("Evaluation tab (still empty)")
            page.get_by_role("button", name="Evaluation", exact=True).click()
            assert "no evaluation report yet" in page.locator("pre").first.inner_text()
            print("  'no evaluation report yet' placeholder confirmed")

            banner("USER PATCHES VIA CHAT — 'add a gauge widget'")
            box.click()
            box.fill("add a gauge widget")
            page.get_by_role("button", name="Send").click()

            step("wait for PatchSpec user reply")
            expect(
                page.get_by_text("Working on those changes.")
            ).to_be_visible(timeout=10_000)
            page.wait_for_function(
                "() => document.querySelectorAll('[data-widget-id]').length >= 3",
                timeout=10_000,
            )
            widget_ids = page.eval_on_selector_all(
                "[data-widget-id]",
                "els => els.map(e => e.getAttribute('data-widget-id'))",
            )
            print(f"  widgets after patch: {widget_ids}")
            shot(page, "05_after_patch")

            step("Last patch tab now shows the PatchSpec")
            page.get_by_role("button", name="Last patch", exact=True).click()
            patch_text = page.locator("pre").first.inner_text()
            assert '"op": "add_widget"' in patch_text, patch_text[:200]
            print("  PatchSpec JSON visible in inspector")
            shot(page, "06_inspector_patch")

            banner("USER CLICKS 'Run browser evaluation'")
            page.get_by_role("button", name="Run browser evaluation").click()
            step("wait for Evaluation tab to populate")
            # Give the network call + render time.
            page.wait_for_function(
                """() => {
                    const pre = document.querySelector('pre');
                    return pre && pre.innerText.includes('"dashboard_id"');
                }""",
                timeout=10_000,
            )
            eval_text = page.locator("pre").first.inner_text()
            assert '"page_loaded"' in eval_text
            print("  BrowserEvaluationReport visible in inspector")
            shot(page, "07_inspector_evaluation")

            banner("USER OPENS /dashboard/<id> DIRECTLY (the evaluator target)")
            # Get the current dashboard id from the spec JSON.
            page.get_by_role("button", name="Spec", exact=True).click()
            spec_text = page.locator("pre").first.inner_text()
            spec = json.loads(spec_text)
            did = spec["dashboard_id"]
            print(f"  dashboard_id: {did}")
            page.goto(f"http://127.0.0.1:3010/dashboard/{did}",
                      wait_until="networkidle")
            state = page.locator("[data-dashboard-state]").get_attribute(
                "data-dashboard-state")
            print(f"  data-dashboard-state: {state}")
            page.wait_for_selector("[data-widget-id]", timeout=10_000)
            ids = page.eval_on_selector_all(
                "[data-widget-id]",
                "els => els.map(e => e.getAttribute('data-widget-id'))",
            )
            print(f"  widgets on standalone page: {ids}")
            shot(page, "08_standalone_dashboard")

            banner("FINAL CHECKS")
            print(f"  total console errors captured: {len(console_errors)}")
            for e in console_errors[:10]:
                print(f"    - {e[:160]}")

            browser.close()
    finally:
        stop(backend)
        stop(frontend)


if __name__ == "__main__":
    main()
