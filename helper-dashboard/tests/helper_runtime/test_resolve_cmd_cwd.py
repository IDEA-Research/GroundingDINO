"""Regression test for the _resolve_cmd cwd-independence fix.

Before the fix, `_resolve_cmd` did `os.path.exists("./bin/opencode")`
against the caller's cwd. When uvicorn launches from backend/, the
relative path resolves to backend/bin/opencode (which doesn't exist)
and the adapter raises `_OpenCodeUnavailable: binary_missing` even
though the subprocess itself would run with cwd=_PROJECT_ROOT where
the binary IS present.

After the fix: relative paths in HELPER_DASHBOARD_OPENCODE_BIN are
resolved against _PROJECT_ROOT regardless of the caller's cwd.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.helper.runtime import _PROJECT_ROOT, _resolve_cmd


BIN_EXPECTED_ABS = (_PROJECT_ROOT / "bin" / "opencode").resolve()


def test_resolve_cmd_finds_bin_when_cwd_is_backend(monkeypatch):
    """Simulate the real failure: uvicorn started from backend/."""
    assert BIN_EXPECTED_ABS.exists(), (
        f"precondition: {BIN_EXPECTED_ABS} must exist"
    )

    monkeypatch.setenv("HELPER_DASHBOARD_OPENCODE_BIN", "./bin/opencode")
    monkeypatch.setenv("HELPER_DASHBOARD_OPENCODE_CMD", json.dumps(["{bin}"]))
    # Move to a directory where "./bin/opencode" does NOT exist.
    monkeypatch.chdir(_PROJECT_ROOT / "backend")
    assert not (Path.cwd() / "bin" / "opencode").exists(), (
        "precondition: backend/bin/opencode must not exist"
    )

    cmd = _resolve_cmd("helper-chat-agent")
    assert cmd is not None, (
        "expected _resolve_cmd to succeed despite caller's cwd "
        "being backend/"
    )
    # The resolved path must be absolute and point at the real file.
    assert os.path.isabs(cmd[0])
    assert Path(cmd[0]).resolve() == BIN_EXPECTED_ABS


def test_resolve_cmd_finds_bin_when_cwd_is_tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("HELPER_DASHBOARD_OPENCODE_BIN", "./bin/opencode")
    monkeypatch.setenv("HELPER_DASHBOARD_OPENCODE_CMD", json.dumps(["{bin}"]))
    monkeypatch.chdir(tmp_path)
    cmd = _resolve_cmd("helper-chat-agent")
    assert cmd is not None
    assert Path(cmd[0]).resolve() == BIN_EXPECTED_ABS


def test_resolve_cmd_rejects_missing_relative_bin(monkeypatch, tmp_path):
    """A relative path that doesn't exist under the project root must
    still return None — we're not loosening validation, we're just
    resolving the path against the right root."""
    monkeypatch.setenv(
        "HELPER_DASHBOARD_OPENCODE_BIN", "./bin/nonexistent-xyz",
    )
    monkeypatch.setenv("HELPER_DASHBOARD_OPENCODE_CMD", json.dumps(["{bin}"]))
    monkeypatch.chdir(tmp_path)
    cmd = _resolve_cmd("helper-chat-agent")
    assert cmd is None


def test_resolve_cmd_accepts_absolute_bin(monkeypatch, tmp_path):
    """Absolute paths should still work unchanged."""
    monkeypatch.setenv(
        "HELPER_DASHBOARD_OPENCODE_BIN", str(BIN_EXPECTED_ABS),
    )
    monkeypatch.setenv("HELPER_DASHBOARD_OPENCODE_CMD", json.dumps(["{bin}"]))
    monkeypatch.chdir(tmp_path)
    cmd = _resolve_cmd("helper-chat-agent")
    assert cmd is not None
    assert Path(cmd[0]).resolve() == BIN_EXPECTED_ABS


def test_resolve_cmd_bare_name_uses_path(monkeypatch, tmp_path):
    """A bare binary name (no slash) still uses PATH via shutil.which."""
    # Create a stub executable in tmp_path, add it to PATH.
    stub = tmp_path / "fake-opencode-test"
    stub.write_text("#!/bin/sh\nexit 0\n")
    stub.chmod(0o755)
    monkeypatch.setenv("HELPER_DASHBOARD_OPENCODE_BIN", "fake-opencode-test")
    monkeypatch.setenv(
        "PATH", f"{tmp_path}:{os.environ.get('PATH', '')}"
    )
    monkeypatch.setenv("HELPER_DASHBOARD_OPENCODE_CMD", json.dumps(["{bin}"]))
    cmd = _resolve_cmd("helper-chat-agent")
    assert cmd is not None
    assert cmd[0] == "fake-opencode-test"  # unresolved; shutil.which confirmed it
