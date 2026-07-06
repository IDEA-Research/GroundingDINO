"""Tests for the M3 tool-using mode in bin/opencode.

The script is loaded as a module so individual helpers (path
allow-list, tool handlers, loop) can be exercised without spawning a
subprocess. We do NOT make real network calls; the loop test
monkey-patches `urllib.request.urlopen` to replay a canned sequence
of LLM responses.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from unittest.mock import MagicMock

import pytest


_CLI_PATH = Path(__file__).resolve().parents[2] / "bin" / "opencode"


@pytest.fixture(scope="module")
def opencode():
    """Import bin/opencode as a Python module.

    The file has no .py suffix, so we use SourceFileLoader directly
    (spec_from_file_location returns None for unknown suffixes).
    """
    loader = SourceFileLoader("_bin_opencode", str(_CLI_PATH))
    spec = spec_from_loader(loader.name, loader)
    assert spec is not None
    mod = module_from_spec(spec)
    loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# M3d: path allowlist
# ---------------------------------------------------------------------------


class TestPathAllowlist:
    def test_validate_rejects_absolute(self, opencode):
        with pytest.raises(opencode._ToolError):
            opencode._validate_safe_relative_path("/etc/passwd")

    def test_validate_rejects_home(self, opencode):
        with pytest.raises(opencode._ToolError):
            opencode._validate_safe_relative_path("~/.ssh/id_rsa")

    def test_validate_rejects_path_traversal(self, opencode):
        with pytest.raises(opencode._ToolError):
            opencode._validate_safe_relative_path("../../etc/passwd")

    def test_validate_rejects_empty(self, opencode):
        with pytest.raises(opencode._ToolError):
            opencode._validate_safe_relative_path("")

    def test_validate_rejects_non_string(self, opencode):
        with pytest.raises(opencode._ToolError):
            opencode._validate_safe_relative_path(None)  # type: ignore[arg-type]

    def test_validate_accepts_relative(self, opencode):
        got = opencode._validate_safe_relative_path("backend/app/specs/widget_spec.py")
        assert isinstance(got, Path)
        assert got.is_absolute()  # resolved to absolute under _ROOT
        assert str(got).endswith("backend/app/specs/widget_spec.py")

    def test_camel_case_conversion(self, opencode):
        assert opencode._camel_case_widget_type("pie_chart") == "PieChart"
        assert opencode._camel_case_widget_type("heatmap") == "Heatmap"
        assert opencode._camel_case_widget_type("multi_word_thing") == "MultiWordThing"

    def test_extend_write_allowlist_size_and_contents(self, opencode):
        paths = opencode._extend_write_allowlist("pie_chart")
        assert len(paths) == 6
        path_strs = [str(p) for p in paths]
        assert any(p.endswith("backend/app/specs/widget_spec.py") for p in path_strs)
        assert any(p.endswith("frontend/widget-toolkit/PieChartWidget.tsx") for p in path_strs)
        assert any(p.endswith("frontend/lib/renderer.tsx") for p in path_strs)
        assert any(p.endswith("frontend/lib/spec-schema.ts") for p in path_strs)
        assert any(p.endswith("tests/spec_validation/test_extend_pie_chart.py") for p in path_strs)

    def test_path_allowed_for_read_includes_backend(self, opencode):
        p = opencode._ROOT / "backend" / "app" / "specs" / "widget_spec.py"
        assert opencode._path_allowed_for_read(p.resolve())

    def test_path_allowed_for_read_rejects_outside_roots(self, opencode):
        # A file at the project root (not under backend/frontend/etc) is denied.
        p = opencode._ROOT / "opencode.json"
        assert not opencode._path_allowed_for_read(p.resolve())

    def test_path_allowed_for_write_only_for_extend_paths(self, opencode):
        p = opencode._ROOT / "backend" / "app" / "specs" / "widget_spec.py"
        assert opencode._path_allowed_for_write(p.resolve(), "pie_chart")
        # Different file under same dir — not allowed
        denied = opencode._ROOT / "backend" / "app" / "specs" / "dashboard_spec.py"
        assert not opencode._path_allowed_for_write(denied.resolve(), "pie_chart")


# ---------------------------------------------------------------------------
# M3b: read_file / write_file / replace_in_file handlers
# ---------------------------------------------------------------------------


class TestReadFileHandler:
    def test_reads_allowed_file(self, opencode):
        # widget_spec.py is under backend/ — allowed for reads.
        content = opencode._tool_read_file({"path": "backend/app/specs/widget_spec.py"})
        assert "WidgetType" in content

    def test_rejects_outside_read_roots(self, opencode):
        # opencode.json is at project root, not under backend/frontend/.opencode/...
        # (read roots include .opencode/ but `opencode.json` is a sibling)
        with pytest.raises(opencode._ToolError) as exc:
            opencode._tool_read_file({"path": "opencode.json"})
        assert "not under the allowed roots" in str(exc.value)

    def test_rejects_traversal(self, opencode):
        with pytest.raises(opencode._ToolError):
            opencode._tool_read_file({"path": "../../etc/passwd"})

    def test_rejects_missing(self, opencode):
        with pytest.raises(opencode._ToolError) as exc:
            opencode._tool_read_file({"path": "backend/app/specs/nope.py"})
        assert "not found" in str(exc.value)


class TestWriteFileHandler:
    def test_rejects_path_outside_allowlist(self, opencode):
        with pytest.raises(opencode._ToolError) as exc:
            opencode._tool_write_file(
                {"path": "backend/app/main.py", "content": "x"},
                "rescue_extend", widget_type="pie_chart",
            )
        assert "not in the extend allow-list" in str(exc.value)

    def test_rejects_non_string_content(self, opencode):
        with pytest.raises(opencode._ToolError):
            opencode._tool_write_file(
                {"path": "frontend/widget-toolkit/PieChartWidget.tsx", "content": 123},
                "rescue_extend", widget_type="pie_chart",
            )

    def test_rejects_oversize_content(self, opencode):
        with pytest.raises(opencode._ToolError) as exc:
            opencode._tool_write_file(
                {"path": "frontend/widget-toolkit/PieChartWidget.tsx",
                 "content": "x" * (opencode._TOOL_FILE_MAX_BYTES + 10)},
                "rescue_extend", widget_type="pie_chart",
            )
        assert "too large" in str(exc.value)

    def test_writes_allowed_path(self, opencode, tmp_path):
        # Write to test_extend_<widget_type>.py — allowed and harmless.
        result = opencode._tool_write_file(
            {"path": "tests/spec_validation/test_extend_pie_chart.py",
             "content": "# auto-generated by M3 test\n"},
            "rescue_extend", widget_type="pie_chart",
        )
        result_dict = json.loads(result)
        assert result_dict["ok"] is True
        # Clean up.
        target = opencode._ROOT / "tests/spec_validation/test_extend_pie_chart.py"
        if target.exists():
            target.unlink()


class TestReplaceInFileHandler:
    def test_rejects_path_outside_allowlist(self, opencode):
        with pytest.raises(opencode._ToolError):
            opencode._tool_replace_in_file(
                {"path": "backend/app/main.py", "old_string": "x", "new_string": "y"},
                "rescue_extend", widget_type="pie_chart",
            )

    def test_rejects_empty_old_string(self, opencode):
        with pytest.raises(opencode._ToolError):
            opencode._tool_replace_in_file(
                {"path": "backend/app/specs/widget_spec.py",
                 "old_string": "", "new_string": "x"},
                "rescue_extend", widget_type="pie_chart",
            )

    def test_rejects_missing_old_string(self, opencode):
        with pytest.raises(opencode._ToolError) as exc:
            opencode._tool_replace_in_file(
                {"path": "backend/app/specs/widget_spec.py",
                 "old_string": "ZZZZZ_NOT_THERE",
                 "new_string": "x"},
                "rescue_extend", widget_type="pie_chart",
            )
        assert "not found" in str(exc.value)


# ---------------------------------------------------------------------------
# M3c: run_command + shell allowlist
# ---------------------------------------------------------------------------


class TestRunCommandHandler:
    def test_rejects_non_allowlisted_command(self, opencode):
        with pytest.raises(opencode._ToolError) as exc:
            opencode._tool_run_command({"cmd": "rm", "args": ["-rf", "/"]})
        assert "not in shell allow-list" in str(exc.value)

    def test_rejects_shell_metacharacters_in_args(self, opencode):
        with pytest.raises(opencode._ToolError) as exc:
            opencode._tool_run_command({"cmd": "pytest", "args": ["foo; rm -rf /"]})
        assert "shell metacharacter" in str(exc.value)

    def test_rejects_pipe_in_args(self, opencode):
        with pytest.raises(opencode._ToolError):
            opencode._tool_run_command({"cmd": "pytest", "args": ["foo|bar"]})

    def test_rejects_backtick_in_args(self, opencode):
        with pytest.raises(opencode._ToolError):
            opencode._tool_run_command({"cmd": "pytest", "args": ["foo`bar`"]})

    def test_rejects_non_list_args(self, opencode):
        with pytest.raises(opencode._ToolError):
            opencode._tool_run_command({"cmd": "pytest", "args": "not a list"})

    def test_runs_allowed_command(self, opencode):
        # Run a trivial python3 -c that just exits.
        result = opencode._tool_run_command({
            "cmd": "python3",
            "args": ["-c", "print('hi')"],
        })
        r = json.loads(result)
        assert r["returncode"] == 0
        assert "hi" in r["stdout"]


# ---------------------------------------------------------------------------
# M3a: tool-use loop integration (mocked HTTP)
# ---------------------------------------------------------------------------


class _FakeHTTPResponse:
    def __init__(self, body: dict):
        self._body = json.dumps(body).encode("utf-8")
    def read(self) -> bytes:
        return self._body
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return None


def _llm_msg(content: str | None = None, tool_calls: list | None = None) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                    "tool_calls": tool_calls,
                }
            }
        ]
    }


def _tool_call(id_: str, name: str, args_dict: dict) -> dict:
    return {
        "id": id_,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args_dict)},
    }


class TestToolLoop:
    def _patch_openrouter(self, opencode, monkeypatch, scripted_responses: list[dict]):
        """Replay a queue of canned OpenRouter responses."""
        queue = list(scripted_responses)
        captured_requests: list[dict] = []

        def fake_urlopen(req, timeout=None, context=None):
            try:
                body = json.loads(req.data.decode("utf-8"))
                captured_requests.append(body)
            except Exception:
                captured_requests.append({})
            if not queue:
                raise AssertionError("test ran out of scripted responses")
            return _FakeHTTPResponse(queue.pop(0))

        monkeypatch.setattr(opencode.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-for-test")
        return captured_requests

    def test_loop_returns_developer_report_on_done(self, opencode, monkeypatch):
        # LLM immediately calls done with a report.
        report = {
            "type": "DeveloperReport",
            "report_id": "rep-test1",
            "summary": "did the thing",
            "actions_taken": ["edited widget_spec.py"],
            "tests_run": ["pytest"],
            "status": "resolved",
        }
        self._patch_openrouter(opencode, monkeypatch, [
            _llm_msg(tool_calls=[_tool_call("c1", "done", {"report": report})]),
        ])

        result = opencode._openrouter_run_with_tools(
            "rescue_extend",
            "big-guy-developer-agent",
            {"extend": {"widget_type": "pie_chart", "rationale": "x"}},
        )
        assert result["type"] == "DeveloperReport"
        assert result["report_id"] == "rep-test1"
        assert result["status"] == "resolved"

    def test_loop_passes_tools_in_first_request(self, opencode, monkeypatch):
        report = {"type": "DeveloperReport", "summary": "x"}
        captured = self._patch_openrouter(opencode, monkeypatch, [
            _llm_msg(tool_calls=[_tool_call("c1", "done", {"report": report})]),
        ])

        opencode._openrouter_run_with_tools(
            "rescue_extend",
            "big-guy-developer-agent",
            {"extend": {"widget_type": "pie_chart", "rationale": "x"}},
        )
        assert len(captured) == 1
        sent = captured[0]
        tool_names = [t["function"]["name"] for t in sent.get("tools") or []]
        assert set(tool_names) == {"read_file", "write_file", "replace_in_file",
                                   "run_command", "done"}

    def test_loop_executes_intermediate_tool_then_done(self, opencode, monkeypatch):
        report = {"type": "DeveloperReport", "summary": "done after read"}
        self._patch_openrouter(opencode, monkeypatch, [
            # Iter 1: LLM asks to read widget_spec.py
            _llm_msg(tool_calls=[_tool_call(
                "c1", "read_file",
                {"path": "backend/app/specs/widget_spec.py"},
            )]),
            # Iter 2: LLM finishes with done()
            _llm_msg(tool_calls=[_tool_call("c2", "done", {"report": report})]),
        ])

        result = opencode._openrouter_run_with_tools(
            "rescue_extend",
            "big-guy-developer-agent",
            {"extend": {"widget_type": "pie_chart", "rationale": "x"}},
        )
        assert result["summary"] == "done after read"

    def test_loop_feeds_tool_error_back_to_llm(self, opencode, monkeypatch):
        # LLM tries to write outside allow-list; tool returns error; LLM
        # then calls done. The loop must not crash on the bad write.
        report = {"type": "DeveloperReport", "summary": "recovered"}
        captured = self._patch_openrouter(opencode, monkeypatch, [
            _llm_msg(tool_calls=[_tool_call(
                "c1", "write_file",
                {"path": "backend/app/main.py", "content": "EVIL"},
            )]),
            _llm_msg(tool_calls=[_tool_call("c2", "done", {"report": report})]),
        ])

        result = opencode._openrouter_run_with_tools(
            "rescue_extend",
            "big-guy-developer-agent",
            {"extend": {"widget_type": "pie_chart", "rationale": "x"}},
        )
        assert result["summary"] == "recovered"
        # The second request must include a tool result message with the
        # error string so the LLM saw it.
        second = captured[1]
        tool_msgs = [m for m in second["messages"] if m.get("role") == "tool"]
        assert any("not in the extend allow-list" in (m.get("content") or "")
                   for m in tool_msgs)
        # The evil write must not have actually happened.
        assert (opencode._ROOT / "backend" / "app" / "main.py").read_text(encoding="utf-8") \
            .find("EVIL") == -1

    def test_loop_rejects_missing_widget_type(self, opencode, monkeypatch):
        # No extend.widget_type → _ProviderError before any HTTP call.
        # Set API key so the failure cannot be from missing creds.
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-for-this-test")
        with pytest.raises(opencode._ProviderError) as exc:
            opencode._openrouter_run_with_tools(
                "rescue_extend",
                "big-guy-developer-agent",
                {"extend": {"rationale": "x"}},
            )
        assert "widget_type" in str(exc.value)

    def test_loop_rejects_invalid_widget_type(self, opencode, monkeypatch):
        # An LLM-poisoned widget_type that somehow bypassed Pydantic.
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
        with pytest.raises(opencode._ProviderError) as exc:
            opencode._openrouter_run_with_tools(
                "rescue_extend",
                "big-guy-developer-agent",
                {"extend": {"widget_type": "../../evil", "rationale": "x"}},
            )
        assert "invalid widget_type" in str(exc.value)

    def test_loop_max_iters_safety(self, opencode, monkeypatch):
        """LLM keeps calling read_file forever; loop must bail."""
        # Build a long script that never calls done.
        forever = [
            _llm_msg(tool_calls=[_tool_call(
                f"c{i}", "read_file",
                {"path": "backend/app/specs/widget_spec.py"},
            )])
            for i in range(opencode._TOOL_LOOP_MAX_ITERS + 5)
        ]
        self._patch_openrouter(opencode, monkeypatch, forever)
        with pytest.raises(opencode._ProviderError) as exc:
            opencode._openrouter_run_with_tools(
                "rescue_extend",
                "big-guy-developer-agent",
                {"extend": {"widget_type": "pie_chart", "rationale": "x"}},
            )
        assert "exceeded" in str(exc.value)


# ---------------------------------------------------------------------------
# bin/opencode-level: rescue_extend in EXPECTED_TYPES / TOOL_USE_OPERATIONS
# ---------------------------------------------------------------------------


def test_expected_types_includes_rescue_extend(opencode):
    assert "rescue_extend" in opencode.EXPECTED_TYPES
    assert opencode.EXPECTED_TYPES["rescue_extend"] == {"DeveloperReport"}


def test_tool_use_operations_contains_rescue_extend(opencode):
    assert opencode.TOOL_USE_OPERATIONS == {"rescue_extend", "developer_fix"}


def test_big_guy_operations_contains_rescue_extend(opencode):
    assert "rescue_extend" in opencode.BIG_GUY_OPERATIONS
