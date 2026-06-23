"""Playwright-based browser evaluator.

Runs the frontend in a headless browser and inspects the rendered
dashboard for a given `DashboardSpec`. Produces a
`BrowserEvaluationReport`.

Playwright is optional at install time. If it is not importable, the
evaluator falls back to a conservative static check that still produces
a report, so the rest of the pipeline keeps working.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..specs import DashboardSpec
from ..specs.evaluation_report import BrowserEvaluationReport


FRONTEND_BASE_URL = os.getenv("HELPER_DASHBOARD_FRONTEND_URL", "http://localhost:3000")
SCREENSHOT_DIR = Path(__file__).resolve().parent.parent / "storage" / "evaluation_reports" / "screenshots"


def _expected_widget_ids(spec: DashboardSpec) -> list[str]:
    return [w.id for w in spec.widgets]


class BrowserEvaluator:
    """Evaluate a rendered dashboard with Playwright, with a safe fallback."""

    # Class-level "have we logged the resolved base URL yet?" guard, so
    # operators see the value once at first use without spamming every
    # evaluation. The misalignment (backend evaluates port 3000 while
    # the real frontend is on 3050) was previously silent and caused
    # every dashboard to look "all widgets missing".
    _base_url_logged: bool = False

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or FRONTEND_BASE_URL
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        if not BrowserEvaluator._base_url_logged:
            BrowserEvaluator._base_url_logged = True
            print(
                f"[browser_evaluator] using frontend base_url={self.base_url!r} "
                f"(override via HELPER_DASHBOARD_FRONTEND_URL)"
            )

    # ------------------------------------------------------------
    def evaluate(self, spec: DashboardSpec) -> BrowserEvaluationReport:
        try:
            return self._evaluate_with_playwright(spec)
        except _PlaywrightUnavailable:
            return self._evaluate_static_fallback(spec)

    # ------------------------------------------------------------
    def _evaluate_with_playwright(self, spec: DashboardSpec) -> BrowserEvaluationReport:
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            raise _PlaywrightUnavailable(str(exc))

        console_errors: list[str] = []
        layout_errors: list[str] = []
        prometheus_errors: list[str] = []
        missing: list[str] = []
        rendered: list[str] = []
        page_loaded = False
        screenshot_path: str | None = None
        dashboard_state: str | None = None

        url = f"{self.base_url}/dashboard/{spec.dashboard_id}"
        expected = _expected_widget_ids(spec)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            def on_console(msg) -> None:
                if msg.type == "error":
                    console_errors.append(msg.text)

            page.on("console", on_console)

            # `networkidle` is unreliable under Next.js dev mode (HMR
            # websocket keeps the network busy forever, or, conversely,
            # idles before React has even committed). `domcontentloaded`
            # is the deterministic gate; we then wait_for_function below
            # for the React-mounted markers.
            try:
                resp = page.goto(url, wait_until="domcontentloaded", timeout=15000)
                page_loaded = bool(resp and resp.ok)
            except Exception as exc:
                console_errors.append(f"goto failed: {exc}")

            if page_loaded:
                # Wait until the dashboard page either declares ready
                # (with the expected number of widget frames mounted)
                # or declares error. Falling through on timeout still
                # produces a useful missing-widgets count.
                try:
                    page.wait_for_function(
                        """(expectedCount) => {
                            const err   = document.querySelector("[data-dashboard-state='error']");
                            if (err) return true;
                            const ready = document.querySelector("[data-dashboard-state='ready']");
                            const n     = document.querySelectorAll("[data-widget-id]").length;
                            return Boolean(ready) && n >= expectedCount;
                        }""",
                        arg=len(expected),
                        timeout=8000,
                    )
                except Exception:
                    # Real mount failure — fall through; counts will
                    # reflect what's actually in the DOM.
                    pass

                try:
                    dashboard_state = page.evaluate(
                        """() => {
                            const el = document.querySelector("[data-dashboard-state]");
                            return el ? el.getAttribute("data-dashboard-state") : null;
                        }"""
                    )
                except Exception:
                    dashboard_state = None

                for widget_id in expected:
                    try:
                        loc = page.locator(f"[data-widget-id='{widget_id}']")
                        if loc.count() > 0:
                            rendered.append(widget_id)
                        else:
                            missing.append(widget_id)
                    except Exception as exc:
                        missing.append(widget_id)
                        console_errors.append(f"locator error for {widget_id}: {exc}")

                # Layout overflow — quick sanity check via JS.
                try:
                    overflow = page.evaluate(
                        "() => document.body.scrollWidth - document.body.clientWidth"
                    )
                    if isinstance(overflow, (int, float)) and overflow > 20:
                        layout_errors.append(
                            f"horizontal overflow: {overflow}px"
                        )
                except Exception:
                    pass

                shot = SCREENSHOT_DIR / f"{spec.dashboard_id}.png"
                try:
                    page.screenshot(path=str(shot), full_page=True)
                    screenshot_path = str(shot)
                except Exception:
                    screenshot_path = None

            browser.close()

        return BrowserEvaluationReport(
            dashboard_id=spec.dashboard_id,
            page_loaded=page_loaded,
            widgets_rendered=rendered,
            missing_widgets=missing,
            console_errors=console_errors,
            layout_errors=layout_errors,
            prometheus_errors=prometheus_errors,
            screenshot_path=screenshot_path,
            recommendation=_recommendation(
                page_loaded,
                missing,
                console_errors,
                expected_total=len(expected),
                dashboard_state=dashboard_state,
            ),
        )

    # ------------------------------------------------------------
    def _evaluate_static_fallback(self, spec: DashboardSpec) -> BrowserEvaluationReport:
        """Produce a minimal report when Playwright is unavailable.

        We cannot verify actual rendering, but we can report what the
        expected layout is so the pipeline still generates a report.
        """
        return BrowserEvaluationReport(
            dashboard_id=spec.dashboard_id,
            page_loaded=False,
            widgets_rendered=[],
            missing_widgets=[w.id for w in spec.widgets],
            console_errors=[],
            layout_errors=[],
            prometheus_errors=[],
            screenshot_path=None,
            recommendation="playwright not installed; browser evaluation skipped",
        )


class _PlaywrightUnavailable(Exception):
    pass


def _recommendation(
    page_loaded: bool,
    missing: list[str],
    console_errors: list[str],
    *,
    expected_total: int = 0,
    dashboard_state: str | None = None,
) -> str:
    if not page_loaded:
        return "page failed to load — investigate frontend or network"

    # All widgets missing in DOM is almost never a spec problem.
    # Two common causes, both unfixable by a PatchSpec:
    #   (a) the page is in 'error' state because /api/dashboard/<id>
    #       returned a non-2xx — the LLM-authored spec never reached
    #       the renderer at all;
    #   (b) the page is 'ready' but React hadn't committed widgets
    #       before the evaluator counted (the new wait_for_function
    #       guards against this, but the wait may have timed out).
    # In both cases the console errors are network/fetch errors, not
    # widget-implementation errors. Tell the LLM not to "fix the spec".
    if expected_total > 0 and len(missing) == expected_total:
        fetch_like = lambda s: (
            "Failed to load resource" in s
            or "404" in s
            or "ERR_CONNECTION" in s
            or "NetworkError" in s
            or "Failed to fetch" in s
        )
        non_fetch_errors = [e for e in console_errors if not fetch_like(e)]
        if not non_fetch_errors and dashboard_state in (None, "loading", "error"):
            return (
                "all widgets missing but only fetch/network errors and "
                f"dashboard_state={dashboard_state!r} — likely the spec "
                "never reached the renderer (fetch failure) or React did "
                "not commit in time; NOT a spec schema issue; do not "
                "patch the spec, escalate or retry"
            )

    if missing:
        return f"{len(missing)} widget(s) missing — check renderer + spec"
    if console_errors:
        return "console errors detected — investigate widget implementations"
    return "clean render"
