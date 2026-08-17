"""Runtime artifacts (side effects of running code in the worktree)
must be invisible to collect_changes / the copy-back guard."""
from app.helper import devfix_worktree as dw


def test_runtime_artifacts_excluded():
    assert dw._is_runtime_artifact("docs/USER_SESSION_LOG.md")
    assert dw._is_runtime_artifact("backend/app/storage/anomaly_build_audit/2026-06-30.jsonl")
    assert dw._is_runtime_artifact("logs/helpercode-trace.log")
    assert dw._is_runtime_artifact("backend/app/__pycache__/m.cpython-310.pyc")
    assert dw._is_runtime_artifact("tests/spec_validation/__pycache__/t.pyc")
    assert dw._is_runtime_artifact("frontend/tsconfig.tsbuildinfo")


def test_source_paths_are_not_artifacts():
    assert not dw._is_runtime_artifact("frontend/widget-toolkit/GaugeWidget.tsx")
    assert not dw._is_runtime_artifact("tests/spec_validation/test_new.py")
    assert not dw._is_runtime_artifact("backend/app/services/browser_evaluator.py")
    assert not dw._is_runtime_artifact("docs/ARCHITECTURE.md")
