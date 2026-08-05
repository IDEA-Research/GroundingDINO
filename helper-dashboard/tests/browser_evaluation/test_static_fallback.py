"""Browser evaluation fallback test.

The real Playwright path needs a running frontend + browser. Here we
exercise the static fallback so CI always has something to run.
"""

from app.services.browser_evaluator import BrowserEvaluator, _PlaywrightUnavailable
from app.services.spec_validator import SpecValidator


def test_static_fallback_produces_report(monkeypatch):
    v = SpecValidator()
    spec = v.validate_dashboard(
        {
            "dashboard_id": "demo",
            "title": "Demo",
            "description": "",
            "layout": {"columns": 12, "row_height": 40},
            "variables": [],
            "widgets": [
                {
                    "id": "w1",
                    "type": "line_chart",
                    "title": "x",
                    "description": "",
                    "query": {
                        "source": "prometheus",
                        "promql": "up",
                        "query_type": "instant",
                    },
                    "position": {"x": 0, "y": 0, "w": 6, "h": 6},
                    "encoding": {},
                    "thresholds": [],
                    "options": {},
                }
            ],
            "refresh_interval": "30s",
        }
    )

    evaluator = BrowserEvaluator()

    def raise_unavailable(_spec):
        raise _PlaywrightUnavailable("forced")

    monkeypatch.setattr(evaluator, "_evaluate_with_playwright", raise_unavailable)

    report = evaluator.evaluate(spec)
    assert report.dashboard_id == "demo"
    assert report.page_loaded is False
    assert report.missing_widgets == ["w1"]
    assert "playwright" in report.recommendation.lower()
