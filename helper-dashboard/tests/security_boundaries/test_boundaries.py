"""Security boundary tests.

These are the highest-priority tests: they assert the invariants
that make the system safe.
"""

import json
from pathlib import Path

import pytest

from app.helper.runtime import BIG_GUY, HELPER_AGENTS, OpenCodeRuntime, RuntimeError_
from app.services.spec_validator import SpecValidationError, SpecValidator


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_big_guy_is_not_a_helper():
    assert BIG_GUY not in HELPER_AGENTS


def test_runtime_rejects_big_guy_as_helper():
    rt = OpenCodeRuntime()
    with pytest.raises(RuntimeError_):
        rt.invoke_helper(BIG_GUY, {"message": "hi"})


def test_opencode_json_marks_big_guy_internal():
    cfg = json.loads((REPO_ROOT / "opencode.json").read_text())
    big = cfg["agents"]["big-guy-developer-agent"]
    assert big.get("internal") is True
    assert big["permissions"]["edit"] is True
    for name in (
        "helper-chat-agent",
        "dashboard-spec-agent",
        "patch-agent",
        "prometheus-agent",
        "browser-eval-agent",
    ):
        perms = cfg["agents"][name]["permissions"]
        assert perms["edit"] is False
        assert perms["shell"] is False


def test_developer_fix_is_developer_only():
    cfg = json.loads((REPO_ROOT / "opencode.json").read_text())
    assert "developer-fix" in cfg["boundaries"]["developer_only"]
    assert "developer-fix" not in cfg["boundaries"]["user_can_invoke"]


def test_dashboardspec_rejects_raw_html():
    v = SpecValidator()
    with pytest.raises(SpecValidationError):
        v.validate_dashboard(
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
                        "raw_html": "<script>alert(1)</script>",
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


def test_dashboardspec_rejects_script_tag_in_description():
    v = SpecValidator()
    with pytest.raises(SpecValidationError):
        v.validate_dashboard(
            {
                "dashboard_id": "demo",
                "title": "demo",
                "description": "safe<script src=x>",
                "layout": {"columns": 12, "row_height": 40},
                "variables": [],
                "widgets": [],
                "refresh_interval": "30s",
            }
        )


def test_patch_rejects_unknown_operation():
    v = SpecValidator()
    with pytest.raises(SpecValidationError):
        v.validate_patch(
            {
                "patch_id": "p1",
                "reason": "x",
                "target_dashboard_id": "demo",
                "created_by": "x",
                "operations": [{"op": "exec", "cmd": "rm -rf /"}],
            }
        )
