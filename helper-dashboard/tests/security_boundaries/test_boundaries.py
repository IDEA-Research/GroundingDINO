"""Security boundary tests.

These are the highest-priority tests: they assert the invariants
that make the system safe.
"""

import json
from pathlib import Path

import pytest

from app.helper.runtime import BIG_GUY, HELPER_AGENTS, HelperCodeRuntime, RuntimeError_
from app.services.spec_validator import SpecValidationError, SpecValidator


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_big_guy_is_not_a_helper():
    assert BIG_GUY not in HELPER_AGENTS


def test_runtime_rejects_big_guy_as_helper():
    rt = HelperCodeRuntime()
    with pytest.raises(RuntimeError_):
        rt.invoke_helper(BIG_GUY, {"message": "hi"})


def _agent_frontmatter(name: str) -> str:
    text = (REPO_ROOT / ".opencode" / "agent" / f"{name}.md").read_text()
    assert text.startswith("---"), f"{name}.md missing frontmatter"
    _, frontmatter, _ = text.split("---", 2)
    return frontmatter


def test_opencode_json_marks_big_guy_internal():
    """517ec41 moved agent permissions from opencode.json's
    non-standard keys into official .opencode/agent/*.md frontmatter;
    opencode.json is now minimal. The INVARIANT is unchanged: Big guy
    is internal (hidden) and the only agent with edit access; Helper
    agents can never edit or shell."""
    cfg = json.loads((REPO_ROOT / "opencode.json").read_text())
    assert "agents" not in cfg  # the old non-standard key must not return

    big = _agent_frontmatter("big-guy-developer-agent")
    assert "hidden: true" in big
    assert "edit: allow" in big

    for name in (
        "helper-chat-agent",
        "dashboard-spec-agent",
        "patch-agent",
        "prometheus-agent",
        "browser-eval-agent",
    ):
        fm = _agent_frontmatter(name)
        assert "edit: deny" in fm, f"{name}: edit must be deny"
        assert "bash: deny" in fm, f"{name}: bash must be deny"
        assert "edit: allow" not in fm
        assert "bash: allow" not in fm


def test_developer_fix_is_developer_only():
    """The command boundary lives in .opencode/PROJECT.md prose and is
    ENFORCED by the runtime gate: developer_fix is never user-facing
    and raises without developer=True."""
    from app.helper.runtime import DEVELOPER_OPERATIONS, USER_OPERATIONS

    assert "developer_fix" in DEVELOPER_OPERATIONS
    assert "developer_fix" not in USER_OPERATIONS

    project = (REPO_ROOT / ".opencode" / "PROJECT.md").read_text()
    assert "Developer-only: developer-fix" in project

    rt = HelperCodeRuntime()
    with pytest.raises(RuntimeError_) as exc:
        rt.invoke_operation("developer_fix", {"instruction": "x"})
    assert "developer-only" in str(exc.value)


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
