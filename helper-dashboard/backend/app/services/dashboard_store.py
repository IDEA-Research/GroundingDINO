"""Local-file JSON store for dashboards, tickets, and evaluation reports.

Files live under `backend/app/storage/`. Each artifact is one file
named by its id. Swap this module out for a real database later.

Drafts are shadowed both in memory (fast, same-process) and on disk
(`_drafts/` directory) so a different uvicorn worker can see them
during the pre-output review loop. Without the disk shadow, the
browser hits worker B for `/api/dashboard/<id>` while worker A
holds the draft in memory and the response is a 404.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..specs import DashboardSpec, BrowserEvaluationReport
from ..specs.developer_ticket import DeveloperTicket


# Defaults for stores constructed without an explicit root. These are
# immutable: a per-instance `root=` (used by tests) must never redirect
# them, or every later default-constructed store in the process leaks
# into that root.
_STORAGE_ROOT = Path(__file__).resolve().parent.parent / "storage"
_DASHBOARDS_DIR = _STORAGE_ROOT / "dashboards"
_TICKETS_DIR = _STORAGE_ROOT / "tickets"
_REPORTS_DIR = _STORAGE_ROOT / "evaluation_reports"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DashboardStore:
    # Class-level in-memory draft shadow. Drafts are dashboards that
    # the review loop is actively rendering in the headless browser
    # but hasn't yet promoted to disk. `load_dashboard` consults this
    # shadow if the file is not found. Drafts are cleared either on
    # promotion (save_dashboard) or on explicit clear_draft.
    _DRAFTS: dict[str, DashboardSpec] = {}

    def __init__(self, root: Path | None = None) -> None:
        self._root = Path(root) if root is not None else _STORAGE_ROOT
        self._dashboards_dir = self._root / "dashboards"
        self._tickets_dir = self._root / "tickets"
        self._reports_dir = self._root / "evaluation_reports"
        self._ensure_dirs()
        self._ensure_drafts_dir()

    def _ensure_dirs(self) -> None:
        for d in (self._dashboards_dir, self._tickets_dir, self._reports_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -- drafts --
    def _drafts_dir(self) -> Path:
        return self._dashboards_dir / "_drafts"

    def _draft_path(self, dashboard_id: str) -> Path:
        return self._drafts_dir() / f"{dashboard_id}.json"

    def _ensure_drafts_dir(self) -> None:
        self._drafts_dir().mkdir(parents=True, exist_ok=True)

    # -- dashboards --
    def save_dashboard(self, spec: DashboardSpec) -> Path:
        self._ensure_dirs()
        path = self._dashboards_dir / f"{spec.dashboard_id}.json"
        path.write_text(json.dumps(spec.model_dump(mode="json", by_alias=True), indent=2))
        # Promoting: clear any draft (memory + disk) for this id.
        DashboardStore._DRAFTS.pop(spec.dashboard_id, None)
        try:
            draft = self._draft_path(spec.dashboard_id)
            if draft.exists():
                draft.unlink()
        except Exception:  # pragma: no cover - best effort
            pass
        return path

    def load_dashboard(self, dashboard_id: str) -> DashboardSpec | None:
        """Resolution order: in-memory draft → on-disk draft → saved disk.

        Drafts MUST win over saved disk so the review loop's
        `/api/dashboard/<id>` serves the version currently being
        reviewed — not a stale prior save of the same id. If the LLM
        reuses an id (very common — `k8s-node-cpu-memory` for k8s
        metrics, `api-gold` for API observability, etc.) the saved
        version would otherwise shadow the new draft and every review
        iteration sees the old DOM while expecting the new spec's
        widget ids → permanent "missing widgets" → permanent rescue
        failure → permanent user-visible clarify message.

        After `clear_draft` (set in review_loop's `finally`), draft
        sources are empty and we fall back to the saved version.
        """
        self._ensure_dirs()
        # 1. In-memory draft (same-process fast path; set_draft writes here)
        cached = DashboardStore._DRAFTS.get(dashboard_id)
        if cached is not None:
            return cached
        # 2. On-disk draft (other uvicorn workers can see this)
        try:
            draft = self._draft_path(dashboard_id)
            if draft.exists():
                return DashboardSpec.model_validate_json(draft.read_text())
        except Exception:  # pragma: no cover - corrupt draft
            pass
        # 3. Saved disk version (no draft active for this id)
        path = self._dashboards_dir / f"{dashboard_id}.json"
        if path.exists():
            return DashboardSpec.model_validate_json(path.read_text())
        return None

    def set_draft(self, spec: DashboardSpec) -> None:
        """Expose a draft to `/api/dashboard/<id>` without persisting it.

        Writes to both the in-memory shadow (fast same-process) and a
        per-id JSON file under `_drafts/` (so a different uvicorn
        worker handling the GET can still see it). Atomic write via
        `tmp → rename` so a partial write never becomes visible.

        Used only by the pre-output review loop. Call `clear_draft`
        to remove.
        """
        DashboardStore._DRAFTS[spec.dashboard_id] = spec
        self._ensure_drafts_dir()
        path = self._draft_path(spec.dashboard_id)
        tmp = path.with_name(f"{spec.dashboard_id}.{uuid.uuid4().hex[:6]}.tmp")
        try:
            tmp.write_text(json.dumps(spec.model_dump(mode="json", by_alias=True), indent=2))
            os.replace(tmp, path)
        except Exception:  # pragma: no cover - best effort
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    def clear_draft(self, dashboard_id: str) -> None:
        DashboardStore._DRAFTS.pop(dashboard_id, None)
        try:
            draft = self._draft_path(dashboard_id)
            if draft.exists():
                draft.unlink()
        except Exception:  # pragma: no cover - best effort
            pass

    def list_dashboard_ids(self) -> list[str]:
        self._ensure_dirs()
        return sorted(p.stem for p in self._dashboards_dir.glob("*.json"))

    # -- tickets --
    def save_ticket(self, ticket: DeveloperTicket) -> Path:
        self._ensure_dirs()
        if ticket.created_at is None:
            ticket = ticket.model_copy(update={"created_at": _now_iso()})
        path = self._tickets_dir / f"{ticket.ticket_id}.json"
        path.write_text(json.dumps(ticket.model_dump(mode="json"), indent=2))
        return path

    def load_ticket(self, ticket_id: str) -> DeveloperTicket | None:
        self._ensure_dirs()
        path = self._tickets_dir / f"{ticket_id}.json"
        if not path.exists():
            return None
        return DeveloperTicket.model_validate_json(path.read_text())

    def list_tickets(self) -> list[dict]:
        self._ensure_dirs()
        out = []
        for p in sorted(self._tickets_dir.glob("*.json")):
            try:
                t = DeveloperTicket.model_validate_json(p.read_text())
                out.append(t.model_dump(mode="json"))
            except Exception:  # pragma: no cover - corrupt file
                continue
        return out

    # -- evaluation reports --
    def save_evaluation(self, report: BrowserEvaluationReport) -> Path:
        self._ensure_dirs()
        if report.created_at is None:
            report = report.model_copy(update={"created_at": _now_iso()})
        stamp = (report.created_at or _now_iso()).replace(":", "-")
        path = self._reports_dir / f"{report.dashboard_id}__{stamp}.json"
        path.write_text(json.dumps(report.model_dump(mode="json"), indent=2))
        return path

    def list_evaluation_reports(self, dashboard_id: str) -> list[dict]:
        self._ensure_dirs()
        out = []
        for p in sorted(self._reports_dir.glob(f"{dashboard_id}__*.json")):
            try:
                out.append(json.loads(p.read_text()))
            except Exception:  # pragma: no cover
                continue
        return out
