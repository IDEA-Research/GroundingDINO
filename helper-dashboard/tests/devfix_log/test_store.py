"""devfix_log store: lifecycle, rendering rules, cursor reads,
path-traversal guard, retention sweep, status vocabulary + legacy
coercion, redaction, meta harvesting (files/tests), queued hand-off,
events/diff readers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.helper import devfix_log


@pytest.fixture()
def runs_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "runs"
    monkeypatch.setenv("HELPER_DASHBOARD_DEVFIX_RUNS_DIR", str(root))
    return root


def _ev(type_: str, ts: int = 1_755_000_000_000, **part) -> dict:
    return {"type": type_, "timestamp": ts, "sessionID": "s", "part": part}


def _log_lines(logger: devfix_log.RunLogger) -> list[str]:
    return (logger.run_dir / "log.txt").read_text(encoding="utf-8").splitlines()


def _meta(logger: devfix_log.RunLogger) -> dict:
    return json.loads(
        (logger.run_dir / "meta.json").read_text(encoding="utf-8")
    )


def _events_text(logger: devfix_log.RunLogger) -> str:
    return (logger.run_dir / "events.jsonl").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------


def test_start_append_finish_lifecycle(runs_root):
    logger = devfix_log.start_run(
        "T-42/weird chars!", instruction="fix the gauge " * 40, model="m/x",
    )
    # run_id is sanitized and the dir exists with all three files.
    assert (runs_root / logger.run_id).is_dir()
    assert devfix_log._RUN_ID_RE.match(logger.run_id)
    assert logger.run_id.startswith("T-42-weird-chars-")

    meta = _meta(logger)
    assert meta["status"] == "running"
    assert meta["finished_at"] is None
    assert meta["ticket_id"] == "T-42/weird chars!"
    assert meta["model"] == "m/x"
    assert len(meta["instruction_preview"]) <= 200
    # Additive fields exist from the start with neutral defaults.
    assert meta["source"] == "auto"
    assert meta["instruction"].startswith("fix the gauge fix the gauge")
    assert meta["attempt"] == 1 and meta["parent_run_id"] is None
    assert meta["pid"] is None and meta["pgid"] is None
    assert meta["files_read"] == [] and meta["files_changed"] == []
    assert meta["tests"] == {"ran": False, "passed": None, "failed": None, "summary": None}
    assert meta["applied"] == {"status": "n/a", "files": []}
    assert meta["verification"] == {"status": "none"}
    assert meta["report_status"] is None and meta["failure_summary"] is None

    logger.event(_ev("text", text="hello"))
    logger.note("worktree created: /tmp/wt")
    logger.finish("resolved")

    meta = _meta(logger)
    # Report-level "resolved" -> meta vocabulary "success".
    assert meta["status"] == "success"
    assert meta["report_status"] == "resolved"
    assert meta["finished_at"] is not None
    assert meta["error_kind"] is None

    lines = _log_lines(logger)
    assert any("ℹ️ worktree created: /tmp/wt" in line for line in lines)
    assert any("💬 hello" in line for line in lines)
    assert lines[-1].endswith("🏁 run finished: success (resolved)")

    # events.jsonl carries the raw event.
    raw = [json.loads(line) for line in _events_text(logger).splitlines()]
    assert raw == [_ev("text", text="hello")]

    # finish() is idempotent; late appends are ignored.
    logger.finish("failed", error_kind="late")
    logger.event(_ev("text", text="too late"))
    assert _meta(logger)["status"] == "success"


def test_finish_failed_timeout_becomes_timeout_status(runs_root):
    logger = devfix_log.start_run("t", instruction="", model="m")
    logger.finish("failed", error_kind="timeout")
    meta = _meta(logger)
    assert meta["status"] == "timeout"
    assert meta["report_status"] == "timeout"
    assert meta["error_kind"] == "timeout"
    assert meta["failure_summary"]
    assert _log_lines(logger)[-1].endswith("🏁 run finished: timeout")


def test_finish_failed_records_error_kind(runs_root):
    logger = devfix_log.start_run("t", instruction="", model="m")
    logger.event(_ev("error", message="provider exploded"))
    logger.finish("failed", error_kind="non_json")
    meta = _meta(logger)
    assert meta["status"] == "failed"
    assert meta["error_kind"] == "non_json"
    # failure_summary = kind + the last error seen in the stream.
    assert meta["failure_summary"].startswith("non_json")
    assert "provider exploded" in meta["failure_summary"]
    assert _log_lines(logger)[-1].endswith("🏁 run finished: failed (non_json)")


def test_finish_user_cancel_becomes_cancelled(runs_root):
    logger = devfix_log.start_run("t", instruction="", model="m")
    logger.finish("cancelled", error_kind="user_cancel")
    meta = _meta(logger)
    assert meta["status"] == "cancelled"
    assert meta["report_status"] == "cancelled"
    assert meta["error_kind"] == "user_cancel"
    # Legacy spelling maps the same way.
    other = devfix_log.start_run("t2", instruction="", model="m")
    other.finish("failed", error_kind="user_cancel")
    assert _meta(other)["status"] == "cancelled"


def test_rejected_and_needs_context_become_waiting(runs_root):
    a = devfix_log.start_run("a", instruction="", model="m")
    a.finish("rejected")
    b = devfix_log.start_run("b", instruction="", model="m")
    b.finish("needs_context")
    assert _meta(a)["status"] == "waiting" and _meta(a)["report_status"] == "rejected"
    assert _meta(b)["status"] == "waiting" and _meta(b)["report_status"] == "needs_context"
    # Nothing is applied while Big Guy waits on the developer.
    assert _meta(a)["applied"] == {"status": "not_applied", "files": []}
    assert _log_lines(a)[-1].endswith("🏁 run finished: waiting (rejected)")


def test_unknown_status_becomes_failed(runs_root):
    logger = devfix_log.start_run("t", instruction="", model="m")
    logger.finish("nonsense")
    assert _meta(logger)["status"] == "failed"


def test_queued_close_resume_mark_running(runs_root):
    """Manual-queue hand-off: the dir exists at submit time with status
    queued; the runtime later re-opens it and flips it to running."""
    logger = devfix_log.start_run(
        "manual-1", instruction="do x", model="m", status="queued",
        source="manual", title="My task", parent_run_id="p-1", attempt=2,
        safety_flags=["matched pattern 'rm -rf'"],
    )
    meta = _meta(logger)
    assert meta["status"] == "queued"
    assert meta["queued_at"] is not None
    assert meta["source"] == "manual"
    assert meta["title"] == "My task"
    assert meta["parent_run_id"] == "p-1" and meta["attempt"] == 2
    assert meta["safety_flags"] == ["matched pattern 'rm -rf'"]
    logger.close()
    # A closed (not finished) logger drops appends silently.
    logger.event(_ev("text", text="ignored"))
    assert _events_text(logger) == ""

    resumed = devfix_log.resume_run(logger.run_id)
    assert resumed.finished is False
    resumed.mark_running(model="m2")
    assert _meta(logger)["status"] == "running"
    assert _meta(logger)["model"] == "m2"
    resumed.set_process(1234, 5678)
    assert _meta(logger)["pid"] == 1234 and _meta(logger)["pgid"] == 5678
    resumed.event(_ev("text", text="after resume"))
    resumed.finish("resolved")
    assert _meta(logger)["status"] == "success"
    assert any("💬 after resume" in ln for ln in _log_lines(logger))


def test_finalize_run_for_queued_and_noop_when_terminal(runs_root):
    logger = devfix_log.start_run("q", instruction="", model="m", status="queued")
    logger.close()
    assert devfix_log.finalize_run(
        logger.run_id, "cancelled", "user_cancel", failure_summary="cancelled before start",
    ) is True
    meta = devfix_log.read_meta(logger.run_id)
    assert meta["status"] == "cancelled"
    assert meta["failure_summary"] == "cancelled before start"
    # Second finalize is a no-op.
    assert devfix_log.finalize_run(logger.run_id, "failed", "x") is False
    assert devfix_log.read_meta(logger.run_id)["status"] == "cancelled"
    assert devfix_log.finalize_run("no-such-run", "failed") is False


def test_update_meta_and_set_fields_after_finish(runs_root):
    logger = devfix_log.start_run("t", instruction="", model="m")
    logger.finish("resolved")
    # Post-hoc verdict (auto_fix rollback / protected-path guard).
    assert devfix_log.update_meta(logger.run_id, {"status": "failed", "applied": {"status": "violation", "files": ["x.py"]}})
    meta = devfix_log.read_meta(logger.run_id)
    assert meta["status"] == "failed"
    assert meta["applied"]["status"] == "violation"
    logger.set_fields(report_summary="done")
    assert devfix_log.read_meta(logger.run_id)["report_summary"] == "done"
    assert devfix_log.update_meta("../evil", {"x": 1}) is False


# ---------------------------------------------------------------------------
# legacy run dirs
# ---------------------------------------------------------------------------


def _write_legacy(runs_root: Path, run_id: str, status: str, error_kind=None, started="2026-01-01T00:00:00+00:00"):
    d = runs_root / run_id
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({
        "run_id": run_id, "ticket_id": "old", "status": status,
        "started_at": started, "finished_at": started, "error_kind": error_kind,
        "model": "m", "instruction_preview": "old preview",
    }), encoding="utf-8")
    (d / "log.txt").write_text("00:00:00 ℹ️ old\n", encoding="utf-8")


def test_legacy_metas_are_coerced_on_read(runs_root):
    runs_root.mkdir(parents=True)
    _write_legacy(runs_root, "old-resolved", "resolved", started="2026-01-01T00:00:04+00:00")
    _write_legacy(runs_root, "old-rejected", "rejected", started="2026-01-01T00:00:03+00:00")
    _write_legacy(runs_root, "old-timeout", "failed", "timeout", started="2026-01-01T00:00:02+00:00")
    _write_legacy(runs_root, "old-failed", "failed", "non_json", started="2026-01-01T00:00:01+00:00")
    _write_legacy(runs_root, "old-running", "running", started="2026-01-01T00:00:00+00:00")

    by_id = {m["run_id"]: m for m in devfix_log.list_runs()}
    assert by_id["old-resolved"]["status"] == "success"
    assert by_id["old-resolved"]["report_status"] == "resolved"
    assert by_id["old-rejected"]["status"] == "waiting"
    assert by_id["old-timeout"]["status"] == "timeout"
    assert by_id["old-failed"]["status"] == "failed"
    assert by_id["old-running"]["status"] == "running"
    # Additive defaults are filled so the UI gets a stable shape.
    m = by_id["old-resolved"]
    assert m["source"] == "auto"
    assert m["instruction"] == "old preview"
    assert m["files_changed"] == [] and m["applied"]["status"] == "n/a"
    # On-disk file is NOT rewritten.
    raw = json.loads((runs_root / "old-resolved" / "meta.json").read_text(encoding="utf-8"))
    assert raw["status"] == "resolved"
    assert devfix_log.read_run("old-resolved")["meta"]["status"] == "success"


# ---------------------------------------------------------------------------
# redaction
# ---------------------------------------------------------------------------


def test_scrub_text_shapes_and_env_values(monkeypatch):
    monkeypatch.setenv("MY_SERVICE_API_KEY", "supersecretvalue123")
    monkeypatch.setenv("SHORT_TOKEN", "abc")  # too short: never scrubbed as a value
    s = devfix_log.scrub_text(
        "key supersecretvalue123 here; sk-abcdefghijklmnopqrstuvwxyz1234 and "
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig and "
        "https://discord.com/api/webhooks/123456/abcDEF-ghi and "
        "OPENROUTER_API_KEY=sk-live-0123456789abcdef and password: hunter22 and abc stays"
    )
    assert "supersecretvalue123" not in s
    assert "sk-abcdefghijklmnopqrstuvwxyz1234" not in s
    assert "eyJhbGciOiJIUzI1NiJ9" not in s
    assert "discord.com/api/webhooks/123456" not in s
    assert "sk-live-0123456789abcdef" not in s
    assert "hunter22" not in s
    # Names/separators survive so the developer knows WHICH var it was.
    assert "OPENROUTER_API_KEY=[REDACTED]" in s
    assert "password: [REDACTED]" in s
    assert "abc stays" in s
    # Idempotent.
    assert devfix_log.scrub_text(s) == s


def test_scrub_keeps_paths_commands_and_test_output():
    text = (
        "python3 -m pytest tests/spec_validation -q\n"
        "frontend/widget-toolkit/GaugeWidget.tsx:12: error\n"
        "12 passed, 1 failed in 3.21s"
    )
    assert devfix_log.scrub_text(text) == text


def test_planted_secret_never_reaches_disk(runs_root, monkeypatch):
    monkeypatch.setenv("HELPER_DASHBOARD_DEV_TOKEN", "dev-token-value-9f8e7d")
    monkeypatch.setenv("ANOMALY_TEST_WEBHOOK_URL", "https://hooks.example/abc/def/ghi")
    secret = "dev-token-value-9f8e7d"
    logger = devfix_log.start_run(
        "t", instruction=f"use {secret} and sk-plantedsecret000111222", model="m",
        title=f"title {secret}",
    )
    logger.event(_ev("text", text=f"the token is {secret}"))
    logger.event(_ev("tool_use", tool="bash",
                     state={"status": "completed", "title": "cat .env",
                            "input": {"command": "cat .env"},
                            "output": f"OPENROUTER_API_KEY=sk-live-abcdefgh1234\nX={secret}\nWEBHOOK=https://hooks.example/abc/def/ghi"}))
    logger.note(f"note with {secret} and Bearer abcdefghijklmnop")
    logger.set_fields(report_summary=f"summary {secret}")
    logger.save_diff(f"+const t = '{secret}';\n+const k = 'sk-live-abcdefgh1234';\n", [{"path": "x.ts", "additions": 2, "deletions": 0, "status": "modified"}])
    logger.finish("resolved")

    for name in ("events.jsonl", "log.txt", "meta.json", "diff.patch"):
        blob = (logger.run_dir / name).read_text(encoding="utf-8")
        assert secret not in blob, name
        assert "sk-live-abcdefgh1234" not in blob, name
        assert "sk-plantedsecret000111222" not in blob, name
        assert "hooks.example/abc/def/ghi" not in blob, name
        assert "Bearer abcdefghijklmnop" not in blob, name
    meta = _meta(logger)
    assert "[REDACTED]" in meta["instruction"]
    assert "[REDACTED]" in meta["title"]
    # Command/path kept.
    assert "cat .env" in (logger.run_dir / "log.txt").read_text(encoding="utf-8")
    # Structure of the event survived (scrubbed inside, not as a string).
    ev = json.loads(_events_text(logger).splitlines()[1])
    assert ev["part"]["state"]["input"]["command"] == "cat .env"
    assert "[REDACTED]" in ev["part"]["state"]["output"]


# ---------------------------------------------------------------------------
# rendering rules
# ---------------------------------------------------------------------------


def test_reasoning_renders_thought_line(runs_root):
    logger = devfix_log.start_run("t", instruction="", model="m")
    logger.event(_ev("reasoning", text="thinking about widgets"))
    logger.finish("rejected")
    assert any(
        "💭 thinking about widgets" in line for line in _log_lines(logger)
    )


def test_streaming_snapshots_collapse_to_one_line(runs_root):
    """Accumulating re-emits of the same part must render ONCE with the
    final text — by part id when present, by prefix-growth otherwise."""
    logger = devfix_log.start_run("t", instruction="", model="m")
    logger.event(_ev("reasoning", id="p1", text="thin"))
    logger.event(_ev("reasoning", id="p1", text="thinking about widgets"))
    logger.event(_ev("text", text="first part"))
    logger.event(_ev("text", text="first part, then more"))
    logger.event(_ev("text", text="a DIFFERENT second message"))
    logger.finish("rejected")

    lines = _log_lines(logger)
    thoughts = [line for line in lines if "💭" in line]
    assert len(thoughts) == 1
    assert "thinking about widgets" in thoughts[0]
    speech = [line for line in lines if "💬" in line]
    assert len(speech) == 2
    assert "first part, then more" in speech[0]
    assert "a DIFFERENT second message" in speech[1]


def test_tool_use_rendering_icons_and_checkmark(runs_root):
    logger = devfix_log.start_run("t", instruction="", model="m")
    logger.event(_ev("tool_use", tool="read",
                     state={"status": "completed", "title": "backend/app/main.py"}))
    logger.event(_ev("tool_use", tool="edit",
                     state={"status": "completed",
                            "input": {"filePath": "frontend/lib/api.ts"}}))
    logger.event(_ev("tool_use", tool="bash",
                     state={"status": "completed", "title": "pytest -q"}))
    logger.event(_ev("tool_use", tool="webfetch",
                     state={"status": "completed", "title": "example.org"}))
    # pending/running states never render (only terminal ones do).
    logger.event(_ev("tool_use", tool="read", state={"status": "running"}))
    logger.finish("resolved")

    lines = _log_lines(logger)
    assert any("📖 read backend/app/main.py ✓" in line for line in lines)
    assert any("✏️ edit" in line and "✓" in line for line in lines)
    assert any("▶️ bash pytest -q ✓" in line for line in lines)
    assert any("🔧 webfetch example.org ✓" in line for line in lines)
    assert sum("📖" in line for line in lines) == 1  # running state skipped


def test_permission_rejection_renders_blocked_line(runs_root):
    logger = devfix_log.start_run("t", instruction="", model="m")
    logger.event(_ev("tool_use", tool="bash",
                     state={"status": "error", "title": "rm -rf /",
                            "error": "permission denied: rejected by user"}))
    logger.event(_ev("tool_use", tool="edit",
                     state={"status": "error", "title": "x.py",
                            "error": "file does not exist"}))
    logger.finish("rejected")

    lines = _log_lines(logger)
    assert any(
        "⛔ bash rm -rf / → permission rejected" in line for line in lines
    )
    assert any(
        "❌ edit x.py → file does not exist" in line for line in lines
    )


def test_error_event_and_step_markers(runs_root):
    logger = devfix_log.start_run("t", instruction="", model="m")
    logger.event(_ev("step_start"))
    logger.event(_ev("error", message="provider exploded"))
    logger.event(_ev("step_finish"))
    logger.finish("failed", error_kind="nonzero_exit")

    lines = _log_lines(logger)
    assert any("❌ provider exploded" in line for line in lines)
    assert not any("step_start" in line or "step_finish" in line for line in lines)


def test_event_accepts_raw_lines_and_skips_garbage(runs_root):
    logger = devfix_log.start_run("t", instruction="", model="m")
    logger.event(json.dumps(_ev("text", text="from a raw line")))
    logger.event("this is not json {")
    logger.event("")
    logger.event(json.dumps(["not", "a", "dict"]))
    logger.finish("rejected")

    raw = _events_text(logger)
    assert len(raw.splitlines()) == 1
    assert any("💬 from a raw line" in line for line in _log_lines(logger))


def test_timestamps_render_as_hhmmss(runs_root):
    logger = devfix_log.start_run("t", instruction="", model="m")
    logger.event(_ev("tool_use", tool="read", state={"status": "completed"}))
    logger.finish("rejected")
    import re
    for line in _log_lines(logger):
        assert re.match(r"^\d{2}:\d{2}:\d{2} ", line)


# ---------------------------------------------------------------------------
# meta harvesting: files_read / files_changed / tests / verification
# ---------------------------------------------------------------------------


def test_tool_events_harvest_files_and_tests(runs_root):
    logger = devfix_log.start_run("t", instruction="", model="m")
    logger.set_project_dir("/wt/helper-dashboard")
    logger.event(_ev("tool_use", tool="read",
                     state={"status": "completed",
                            "input": {"filePath": "/wt/helper-dashboard/frontend/lib/renderer.tsx"}}))
    logger.event(_ev("tool_use", tool="read",
                     state={"status": "completed",
                            "input": {"filePath": "backend/app/main.py"}}))
    logger.event(_ev("tool_use", tool="edit",
                     state={"status": "completed",
                            "input": {"filePath": "/wt/helper-dashboard/frontend/widget-toolkit/GaugeWidget.tsx"}}))
    logger.event(_ev("tool_use", tool="edit",
                     state={"status": "error", "error": "permission denied",
                            "input": {"filePath": "/wt/helper-dashboard/bin/helpercode"}}))
    logger.event(_ev("tool_use", tool="bash",
                     state={"status": "completed",
                            "input": {"command": "python3 -m pytest tests/spec_validation -q"},
                            "output": "....\n12 passed, 1 skipped in 2.10s",
                            "metadata": {"exit": 0}}))
    logger.event(_ev("tool_use", tool="bash",
                     state={"status": "completed",
                            "input": {"command": "npx tsc --noEmit -p frontend"},
                            "output": "",
                            "metadata": {"exit": 0}}))
    logger.event(_ev("tool_use", tool="bash",
                     state={"status": "completed",
                            "input": {"command": "ls -la"},
                            "output": "not a test"}))
    meta = _meta(logger)
    assert meta["files_read"] == ["frontend/lib/renderer.tsx", "backend/app/main.py"]
    # Failed (permission-denied) edits are NOT changes.
    assert meta["files_changed"] == ["frontend/widget-toolkit/GaugeWidget.tsx"]
    assert meta["tests"]["ran"] is True
    assert meta["tests"]["passed"] == 13  # 12 pytest + 1 clean tsc
    assert meta["tests"]["failed"] == 0
    assert len(meta["tests"]["commands"]) == 2
    logger.finish("resolved")
    assert _meta(logger)["verification"] == {"status": "passed"}


def test_failed_tests_make_verification_failed(runs_root):
    logger = devfix_log.start_run("t", instruction="", model="m")
    logger.event(_ev("tool_use", tool="bash",
                     state={"status": "completed",
                            "input": {"command": "pytest tests/x -q"},
                            "output": "FAILED tests/x.py::t\n3 passed, 2 failed in 1s",
                            "metadata": {"exit": 1}}))
    logger.finish("rejected")
    meta = _meta(logger)
    assert meta["tests"]["passed"] == 3 and meta["tests"]["failed"] == 2
    assert meta["tests"]["summary"] == "3 passed, 2 failed in 1s"
    assert meta["verification"] == {"status": "failed"}


def test_no_tests_means_verification_none(runs_root):
    logger = devfix_log.start_run("t", instruction="", model="m")
    logger.event(_ev("tool_use", tool="edit", state={"status": "completed", "input": {"filePath": "a.ts"}}))
    logger.finish("resolved")
    assert _meta(logger)["verification"] == {"status": "none"}


@pytest.mark.parametrize("command,output,exit_code,expected", [
    ("python3 -m pytest tests -q", "5 passed in 0.5s", 0, (5, 0)),
    ("pytest", "1 failed, 4 passed, 2 errors in 1s", 1, (4, 3)),
    ("npm test", "Tests:       2 failed, 10 passed, 12 total", 1, (10, 2)),
    ("npx tsc --noEmit", "src/a.ts(1,1): error TS2322: x\nFound 1 error.", 2, (None, 1)),
    ("npx tsc --noEmit", "", 0, (1, 0)),
    ("npm run build", "Compiled successfully", 0, (None, None)),
])
def test_parse_test_output(command, output, exit_code, expected):
    parsed = devfix_log.parse_test_output(command, output, exit_code)
    assert (parsed["passed"], parsed["failed"]) == expected


def test_applied_and_diff_round_trip(runs_root):
    logger = devfix_log.start_run("t", instruction="", model="m")
    logger.set_applied("applied", ["frontend/lib/renderer.tsx"])
    logger.finish("resolved")
    # The runtime captures the diff AFTER finish (finally block) — allowed.
    logger.save_diff(
        "diff --git a/frontend/lib/renderer.tsx b/frontend/lib/renderer.tsx\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n",
        [{"path": "frontend/lib/renderer.tsx", "additions": 1, "deletions": 1, "status": "modified"},
         {"path": "frontend/widget-toolkit/New.tsx", "additions": 3, "deletions": 0, "status": "added"}],
    )
    meta = _meta(logger)
    assert meta["applied"] == {"status": "applied", "files": ["frontend/lib/renderer.tsx"]}
    assert meta["files_changed"] == ["frontend/lib/renderer.tsx", "frontend/widget-toolkit/New.tsx"]
    diff = devfix_log.read_diff(logger.run_id)
    assert diff["patch"].startswith("diff --git a/frontend/lib/renderer.tsx")
    assert [f["path"] for f in diff["files"]] == ["frontend/lib/renderer.tsx", "frontend/widget-toolkit/New.tsx"]
    # A run without a diff reads back empty, not 404.
    other = devfix_log.start_run("o", instruction="", model="m")
    other.finish("failed", "x")
    assert devfix_log.read_diff(other.run_id) == {"patch": "", "files": []}


# ---------------------------------------------------------------------------
# read side: list, cursor, events, guards
# ---------------------------------------------------------------------------


def test_list_runs_newest_first_with_limit(runs_root):
    ids = []
    for i in range(4):
        logger = devfix_log.start_run(f"t{i}", instruction="", model="m")
        logger.finish("rejected")
        ids.append(logger.run_id)

    metas = devfix_log.list_runs()
    assert [m["run_id"] for m in metas] == list(reversed(ids))
    assert [m["run_id"] for m in devfix_log.list_runs(limit=2)] == [
        ids[3], ids[2],
    ]


def test_read_run_cursor_tailing(runs_root):
    logger = devfix_log.start_run("t", instruction="", model="m")
    logger.note("one")
    logger.note("two")

    first = devfix_log.read_run(logger.run_id, after=0)
    assert first["meta"]["status"] == "running"
    assert len(first["lines"]) == 2
    assert first["next_cursor"] == 2

    # Nothing new yet: tail from the cursor is empty.
    again = devfix_log.read_run(logger.run_id, after=first["next_cursor"])
    assert again["lines"] == []
    assert again["next_cursor"] == 2

    logger.note("three")
    logger.finish("resolved")
    tail = devfix_log.read_run(logger.run_id, after=first["next_cursor"])
    assert len(tail["lines"]) == 2  # "three" + the finish line
    assert "three" in tail["lines"][0]
    assert tail["next_cursor"] == 4
    assert tail["meta"]["status"] == "success"

    # Negative cursors clamp to 0 instead of wrapping from the end.
    assert len(devfix_log.read_run(logger.run_id, after=-5)["lines"]) == 4


def test_read_events_cursor(runs_root):
    logger = devfix_log.start_run("t", instruction="", model="m")
    logger.event(_ev("step_start"))
    logger.event(_ev("text", text="a"))
    first = devfix_log.read_events(logger.run_id, after=0)
    assert [e["type"] for e in first["events"]] == ["step_start", "text"]
    assert first["next_cursor"] == 2
    logger.event(_ev("tool_use", tool="read", state={"status": "completed"}))
    logger.finish("resolved")
    tail = devfix_log.read_events(logger.run_id, after=first["next_cursor"])
    assert [e["type"] for e in tail["events"]] == ["tool_use"]
    assert tail["next_cursor"] == 3
    with pytest.raises(KeyError):
        devfix_log.read_events("../x")


def test_read_run_guards_path_traversal(runs_root):
    logger = devfix_log.start_run("t", instruction="", model="m")
    logger.finish("resolved")
    for bad in (
        "../../etc/passwd", "..", "a/b", "a\\b", "t.", ".hidden",
        "", "x" * 300, "run;rm",
    ):
        with pytest.raises(KeyError):
            devfix_log.read_run(bad)
        assert devfix_log.read_meta(bad) is None
    with pytest.raises(KeyError):
        devfix_log.read_run("no-such-run-20990101T000000")


# ---------------------------------------------------------------------------
# retention
# ---------------------------------------------------------------------------


def test_retention_sweep_keeps_latest_50(runs_root, monkeypatch):
    for i in range(55):
        logger = devfix_log.start_run(f"r{i}", instruction="", model="m")
        logger.finish("rejected")

    survivors = {d.name for d in runs_root.iterdir() if d.is_dir()}
    assert len(survivors) == 50
    # The newest runs survive, the oldest five were swept.
    metas = devfix_log.list_runs(limit=50)
    assert len(metas) == 50
    assert metas[0]["ticket_id"] == "r54"
    swept = {f"r{i}" for i in range(5)}
    assert not any(m["ticket_id"] in swept for m in metas)
