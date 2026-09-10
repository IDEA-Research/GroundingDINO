"""bar_gauge widget type: registration + full validate/save/load chain.

Mirrors test_widget_type_gauge.py's minimal enum check, then goes further
(per the bar_gauge task's acceptance criteria) to prove the type is not
just accepted by the enum but actually flows end-to-end through
SpecValidator and SavedDashboardStore — the same round trip a real
"content-aware coach" dashboard save would exercise.
"""

from __future__ import annotations

import pytest

from app.services.saved_dashboard_store import SavedDashboardStore
from app.services.spec_validator import SpecValidationError, SpecValidator
from app.specs.widget_spec import WidgetType


def test_widget_type_includes_bar_gauge():
    assert "bar_gauge" in [member.value for member in WidgetType]


def _bar_gauge_dashboard_payload(display_mode: str = "gradient") -> dict:
    return {
        "dashboard_id": "bar-gauge-demo",
        "title": "Bar Gauge Demo",
        "description": "",
        "layout": {"columns": 12, "row_height": 40},
        "variables": [],
        "widgets": [
            {
                "id": "spo2-bar-gauge",
                "type": "bar_gauge",
                "title": "SpO2",
                "query": {
                    "source": "prometheus",
                    "promql": "max by (patient_id) (patient_spo2)",
                    "query_type": "instant",
                },
                "position": {"x": 0, "y": 0, "w": 6, "h": 4},
                "encoding": {"unit": "%"},
                "thresholds": [
                    {
                        "value": 94,
                        "color": "amber",
                        "direction": "below",
                        "label": "偏低",
                    },
                    {
                        "value": 90,
                        "color": "red",
                        "direction": "below",
                        "label": "危急",
                    },
                ],
                "options": {
                    "min": 0,
                    "max": 100,
                    "display_mode": display_mode,
                    "decimals": 0,
                },
            }
        ],
        "refresh_interval": "30s",
    }


def test_bar_gauge_spec_validates():
    validator = SpecValidator()
    spec = validator.validate_dashboard(_bar_gauge_dashboard_payload())
    widget = spec.widgets[0]
    assert widget.type.value == "bar_gauge"
    assert widget.options["display_mode"] == "gradient"
    assert widget.options["min"] == 0
    assert widget.options["max"] == 100
    assert len(widget.thresholds) == 2


@pytest.mark.parametrize("mode", ["basic", "gradient", "retro-lcd"])
def test_bar_gauge_spec_validates_every_display_mode(mode: str):
    # The validator does not itself constrain display_mode to a literal
    # set (options values are opaque past the key allow-list) — this
    # documents that all three modes the frontend implements pass
    # through unchanged, not silently coerced or dropped.
    validator = SpecValidator()
    spec = validator.validate_dashboard(_bar_gauge_dashboard_payload(mode))
    assert spec.widgets[0].options["display_mode"] == mode


def test_bar_gauge_spec_rejects_unknown_option_key():
    payload = _bar_gauge_dashboard_payload()
    payload["widgets"][0]["options"]["bogus_key"] = 1
    validator = SpecValidator()
    with pytest.raises(SpecValidationError, match="bogus_key"):
        validator.validate_dashboard(payload)


def test_bar_gauge_dashboard_saves_and_loads_back(tmp_path):
    validator = SpecValidator()
    spec = validator.validate_dashboard(_bar_gauge_dashboard_payload())

    store = SavedDashboardStore(root=tmp_path)
    entry = store.save(spec, name="bar gauge demo", description="d", tags=["demo"])
    assert entry.saved_id

    loaded = store.load(entry.saved_id)
    assert loaded is not None
    loaded_widget = loaded.spec.widgets[0]
    assert loaded_widget.type.value == "bar_gauge"
    assert loaded_widget.options["display_mode"] == "gradient"
    assert [t.value for t in loaded_widget.thresholds] == [94, 90]
    assert loaded_widget.encoding.unit == "%"
