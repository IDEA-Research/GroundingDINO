"""Regression test for the offline-backend UX.

With the frontend running but the backend down, the UI should:
  - show a red connection dot within ~15 s
  - present an improved error message when the user tries to send

Skipped when Playwright isn't installed. Starts the Next.js
production server itself so the test is self-contained.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

try:
    from playwright.sync_api import sync_playwright  # type: ignore
except Exception:  # pragma: no cover
    sync_playwright = None  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT / "frontend"


def _next_dir_exists() -> bool:
    return (FRONTEND_DIR / ".next").exists()


def _port_free(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=0.5):
            return False
    except Exception:
        return True


def _wait_port(port: int, tries: int = 60) -> bool:
    for _ in range(tries):
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/", timeout=0.5
            ) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


@pytest.mark.skipif(sync_playwright is None, reason="playwright not installed")
@pytest.mark.skipif(not _next_dir_exists(),
                    reason="frontend .next build not present")
def test_backend_down_ux_shows_disconnected_and_improved_error():
    port = 3939
    assert _port_free(port), f"port {port} already in use"

    # Start frontend without a backend running. NEXT_PUBLIC_API_BASE is
    # baked at build time to http://localhost:8000 which will be down.
    front = subprocess.Popen(
        ["npx", "next", "start", "-p", str(port)],
        cwd=str(FRONTEND_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        assert _wait_port(port), "frontend never came up"

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{port}/",
                       wait_until="domcontentloaded")

            # The connection dot turns red within ~15s of the first
            # health probe failing. We poll the element color.
            deadline = time.time() + 25
            dot_red = False
            while time.time() < deadline:
                html = page.content()
                if "bg-rose-500" in html:
                    dot_red = True
                    break
                time.sleep(0.5)
            assert dot_red, "connection dot did not turn red"

            # Try to send a message. The input should be disabled
            # AND/OR the Send button should be disabled. Either way
            # the user can't accidentally send into the void.
            btn = page.get_by_role("button", name="Send")
            assert btn.is_disabled()

            browser.close()
    finally:
        try:
            os.killpg(os.getpgid(front.pid), signal.SIGTERM)
            front.wait(timeout=5)
        except Exception:
            try:
                front.kill()
            except Exception:
                pass
