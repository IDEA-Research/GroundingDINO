"""bin/opencode auto-fix write allow-list.

Loads the CLI as a module (main() is __main__-guarded) and checks the
developer_fix path gate: the render/product layer is writable, the
clinical anomaly files and the trust-boundary files never are.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path

import pytest


_CLI = Path(__file__).resolve().parents[2] / "bin" / "opencode"


@pytest.fixture(scope="module")
def cli():
    loader = importlib.machinery.SourceFileLoader("_opencode_cli_under_test", str(_CLI))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _p(cli, rel: str) -> Path:
    return (cli._ROOT / rel).resolve()


def test_developer_fix_is_tool_using(cli):
    assert "developer_fix" in cli.TOOL_USE_OPERATIONS
    assert "rescue_extend" in cli.TOOL_USE_OPERATIONS


@pytest.mark.parametrize("rel", [
    "backend/app/services/browser_evaluator.py",
    "frontend/lib/renderer.tsx",
    "frontend/lib/spec-schema.ts",
    "frontend/widget-toolkit/PieChartWidget.tsx",
    "tests/browser_evaluation/test_new.py",
    "tests/review_loop/test_new.py",
    "tests/spec_validation/test_new.py",
])
def test_autofix_allows_render_layer(cli, rel):
    assert cli._path_allowed_for_autofix_write(_p(cli, rel)) is True


@pytest.mark.parametrize("rel", [
    # Clinical safety code — protected paths from CLAUDE.md.
    "backend/app/services/anomaly_core.py",
    "backend/app/services/anomaly_evaluator_service.py",
    "backend/app/services/alert_state_store.py",
    "backend/app/prometheus/neonatal_sim.py",
    "backend/app/prometheus/client.py",
    "backend/app/specs/alert_rule_spec.py",
    "backend/app/api/anomaly.py",
    # Trust boundaries.
    "backend/app/helper/extend_gate.py",
    "backend/app/helper/auto_fix.py",
    "backend/app/helper/runtime.py",
    "bin/opencode",
    ".opencode/agent/big-guy-developer-agent.md",
    ".env",
    # Not in scope at all.
    "backend/app/helper/orchestrator.py",
    "frontend/lib/other-file.ts",
    "docs/SECURITY_BOUNDARIES.md",
])
def test_autofix_denies_everything_else(cli, rel):
    assert cli._path_allowed_for_autofix_write(_p(cli, rel)) is False


def test_write_tool_refuses_clinical_path_for_developer_fix(cli):
    with pytest.raises(cli._ToolError):
        cli._check_write_allowed(
            _p(cli, "backend/app/services/anomaly_core.py"),
            "developer_fix", "",
        )


def test_write_tool_allows_evaluator_for_developer_fix(cli):
    cli._check_write_allowed(
        _p(cli, "backend/app/services/browser_evaluator.py"),
        "developer_fix", "",
    )


def test_extend_allowlist_unchanged_by_autofix_gate(cli):
    # rescue_extend still uses its own six-file list.
    cli._check_write_allowed(
        _p(cli, "backend/app/specs/widget_spec.py"), "rescue_extend", "pie_chart",
    )
    with pytest.raises(cli._ToolError):
        cli._check_write_allowed(
            _p(cli, "backend/app/services/browser_evaluator.py"),
            "rescue_extend", "pie_chart",
        )
