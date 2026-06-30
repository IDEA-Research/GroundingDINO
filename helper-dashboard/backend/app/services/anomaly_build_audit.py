"""anomaly_build_audit — structured, fail-closed build/loop audit log.

Every Code -> Observe -> Decide -> Check stage and every significant action
appends one JSONL record to:

    backend/app/storage/anomaly_build_audit/<UTC-date>.jsonl

This mirrors the `extend_audit` convention (one JSON object per line, UTC
ISO timestamps) and keeps its fail-closed posture: an audit-write failure on
a clinical event must SURFACE, not be swallowed. `append()` therefore raises
on I/O failure rather than silently dropping the record.

Secrets discipline: callers must pass already-masked values. `mask_url()` is
provided so a webhook token is never written in full to the audit.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_AUDIT_DIR = (
    Path(__file__).resolve().parent.parent / "storage" / "anomaly_build_audit"
)


def _audit_dir() -> Path:
    d = Path(os.getenv("ANOMALY_BUILD_AUDIT_DIR", str(_AUDIT_DIR)))
    d.mkdir(parents=True, exist_ok=True)
    return d


def mask_url(url: str | None) -> str | None:
    """Mask a webhook URL so the token is never logged in full."""
    if not url:
        return url
    # Keep scheme + host + a short tail hint; drop the secret path/token.
    try:
        scheme, rest = url.split("://", 1)
        host = rest.split("/", 1)[0]
        tail = url[-4:] if len(url) > 4 else "****"
        return f"{scheme}://{host}/****{tail}"
    except Exception:
        return "****"


def append(
    *,
    increment: str,
    stage: str,  # code | observe | decide | check
    action: str,  # read | edit | run | decide
    target: str,
    summary: str,
    reasoning: str,
    evidence: Any = None,
    outcome: str = "",
) -> None:
    """Append one structured audit record. Raises on write failure."""
    record = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "increment": increment,
        "stage": stage,
        "action": action,
        "target": target,
        "summary": summary,
        "reasoning": reasoning,
        "evidence": evidence,
        "outcome": outcome,
    }
    path = _audit_dir() / f"{datetime.now(tz=timezone.utc):%Y-%m-%d}.jsonl"
    line = json.dumps(record, ensure_ascii=False, default=str)
    # Fail closed: any error here propagates to the caller.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())
