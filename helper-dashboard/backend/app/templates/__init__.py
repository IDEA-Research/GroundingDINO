"""Built-in dashboard templates (從範本建立).

Canned, LLM-free DashboardSpec JSON files next to this module. Each
file is a full DashboardSpec whose `dashboard_id` is the template id;
instantiating clones it under a fresh id (see `api/templates.py`).

Rules every template obeys (and tests pin):
- metric names come from the catalog families (patient_*/cerebral_*),
  aggregated `max by (patient_id)` like every clinical query;
- `thresholds: []` everywhere — clinical limits are never invented here;
- widget titles name the measurement only (no time ranges);
- never a stat_card AND a gauge for the same PromQL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DIR = Path(__file__).resolve().parent

# Display order in the 從範本建立 picker.
TEMPLATE_ORDER: tuple[str, ...] = ("hr-spo2", "rso2", "vitals-overview")


def template_path(template_id: str) -> Path:
    return _DIR / f"{template_id}.json"


def list_template_ids() -> list[str]:
    ids = sorted(p.stem for p in _DIR.glob("*.json"))
    ordered = [t for t in TEMPLATE_ORDER if t in ids]
    return ordered + [t for t in ids if t not in ordered]


def load_template_raw(template_id: str) -> dict[str, Any]:
    p = template_path(template_id)
    if not p.is_file():
        raise FileNotFoundError(f"template {template_id!r} not found")
    return json.loads(p.read_text(encoding="utf-8"))


def template_metrics(raw: dict[str, Any]) -> list[str]:
    """Clinical metric names referenced by the template, first-seen order."""
    from ..prometheus.catalog import CLINICAL_METRIC_PREFIXES
    import re

    out: list[str] = []
    for w in raw.get("widgets") or []:
        promqls = []
        if w.get("query"):
            promqls.append(str(w["query"].get("promql", "")))
        for q in w.get("queries") or []:
            promqls.append(str(q.get("promql", "")))
        for p in promqls:
            # Drop `by (patient_id)` groupings so LABEL names (which share
            # the patient_ prefix) are never mistaken for metrics.
            p = re.sub(r"\bby\s*\([^)]*\)", " ", p, flags=re.IGNORECASE)
            for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", p):
                if name.startswith(CLINICAL_METRIC_PREFIXES) and name not in out:
                    out.append(name)
    return out
