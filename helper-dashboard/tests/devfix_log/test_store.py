"""devfix_log store: lifecycle, rendering rules, cursor reads,
path-traversal guard, retention sweep."""

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

    logger.event(_ev("text", text="hello"))
    logger.note("worktree created: /tmp/wt")
    logger.finish("resolved")

    meta = _meta(logger)
    assert meta["status"] == "resolved"
    assert meta["finished_at"] is not None
    assert meta["error_kind"] is None

    lines = _log_lines(logger)
    assert any("ℹ️ worktree created: /tmp/wt" in line for line in lines)
    assert any("💬 hello" in line for line in lines)
    assert lines[-1].endswith("🏁 run finished: resolved")

    # events.jsonl carries the raw event.
    raw = [
        json.loads(line)
        for line in (logger.run_dir / "events.jsonl")
        .read_text(encoding="utf-8").splitlines()
    ]
    assert raw == [_ev("text", text="hello")]

    # finish() is idempotent; late appends are ignored.
    logger.finish("failed", error_kind="late")
    logger.event(_ev("text", text="too late"))
    assert _meta(logger)["status"] == "resolved"


def test_finish_failed_records_error_kind(runs_root):
    logger = devfix_log.start_run("t", instruction="", model="m")
    logger.finish("failed", error_kind="timeout")
    meta = _meta(logger)
    assert meta["status"] == "failed"
    assert meta["error_kind"] == "timeout"
    assert _log_lines(logger)[-1].endswith("🏁 run finished: failed (timeout)")


def test_unknown_status_becomes_failed(runs_root):
    logger = devfix_log.start_run("t", instruction="", model="m")
    logger.finish("nonsense")
    assert _meta(logger)["status"] == "failed"


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

    raw = (logger.run_dir / "events.jsonl").read_text(encoding="utf-8")
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
# read side: list, cursor, guards
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
    assert tail["meta"]["status"] == "resolved"

    # Negative cursors clamp to 0 instead of wrapping from the end.
    assert len(devfix_log.read_run(logger.run_id, after=-5)["lines"]) == 4


def test_read_run_guards_path_traversal(runs_root):
    logger = devfix_log.start_run("t", instruction="", model="m")
    logger.finish("resolved")
    for bad in (
        "../../etc/passwd", "..", "a/b", "a\\b", "t.", ".hidden",
        "", "x" * 300, "run;rm",
    ):
        with pytest.raises(KeyError):
            devfix_log.read_run(bad)
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
