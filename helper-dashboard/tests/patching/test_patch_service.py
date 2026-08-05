"""PatchSpec application tests."""

import pytest

from app.services.patch_service import PatchApplicationError, PatchService
from app.services.spec_validator import SpecValidator


def _spec():
    return SpecValidator().validate_dashboard(
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
                    "title": "CPU",
                    "description": "",
                    "query": {
                        "source": "prometheus",
                        "promql": "rate(http_requests_total[5m])",
                        "query_type": "range",
                        "range": "1h",
                        "step": "30s",
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


def _patch(ops):
    return SpecValidator().validate_patch(
        {
            "patch_id": "p1",
            "reason": "test",
            "target_dashboard_id": "demo",
            "created_by": "test",
            "operations": ops,
        }
    )


def test_add_widget():
    spec = _spec()
    patch = _patch(
        [
            {
                "op": "add_widget",
                "widget": {
                    "id": "w2",
                    "type": "stat_card",
                    "title": "Up",
                    "description": "",
                    "query": {
                        "source": "prometheus",
                        "promql": "sum(up)",
                        "query_type": "instant",
                    },
                    "position": {"x": 6, "y": 0, "w": 3, "h": 4},
                    "encoding": {},
                    "thresholds": [],
                    "options": {},
                },
            }
        ]
    )
    new = PatchService().apply(spec, patch)
    assert [w.id for w in new.widgets] == ["w1", "w2"]


def test_remove_widget():
    spec = _spec()
    patch = _patch([{"op": "remove_widget", "widget_id": "w1"}])
    new = PatchService().apply(spec, patch)
    assert new.widgets == []


def test_remove_unknown_widget_raises():
    spec = _spec()
    patch = _patch([{"op": "remove_widget", "widget_id": "nope"}])
    with pytest.raises(PatchApplicationError):
        PatchService().apply(spec, patch)


def test_update_widget_title():
    spec = _spec()
    patch = _patch(
        [{"op": "update_widget", "widget_id": "w1", "fields": {"title": "New"}}]
    )
    new = PatchService().apply(spec, patch)
    assert new.widgets[0].title == "New"


def test_update_widget_rejects_id_change():
    # id/type are not in UPDATE_WIDGET_FIELDS — the validator rejects it.
    with pytest.raises(Exception):
        _patch(
            [{"op": "update_widget", "widget_id": "w1", "fields": {"id": "oops"}}]
        )


def test_wrong_target_dashboard_raises():
    spec = _spec()
    patch = SpecValidator().validate_patch(
        {
            "patch_id": "p1",
            "reason": "test",
            "target_dashboard_id": "other",
            "created_by": "test",
            "operations": [{"op": "remove_widget", "widget_id": "w1"}],
        }
    )
    with pytest.raises(PatchApplicationError):
        PatchService().apply(spec, patch)


def test_patched_dashboard_must_still_validate():
    spec = _spec()
    # Move widget to overflow the layout after patch -> fail validation.
    patch = _patch(
        [
            {
                "op": "update_widget",
                "widget_id": "w1",
                "fields": {"position": {"x": 10, "y": 0, "w": 6, "h": 6}},
            }
        ]
    )
    with pytest.raises(PatchApplicationError):
        PatchService().apply(spec, patch)
