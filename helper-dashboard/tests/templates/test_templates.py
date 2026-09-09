"""Built-in templates (從範本建立): the three JSON specs validate, obey
the authoring rules (thresholds [], no time ranges in titles, no
stat_card+gauge twins, catalog metric names, `queries` table joined by
patient_id), and instantiate under a fresh persisted dashboard_id.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from app.api import templates as templates_api
from app.helper.orchestrator import _TITLE_RANGE_SUFFIX_RE
from app.main import app
from app.services.dashboard_store import DashboardStore
from app.services.spec_validator import SpecValidator
from app.templates import list_template_ids, load_template_raw, template_metrics


CATALOG_METRICS = {
    "patient_heart_rate_bpm",
    "patient_spo2_percentage",
    "cerebral_rso2_left_percentage",
    "cerebral_rso2_right_percentage",
}
_RANGE_WORDS = re.compile(r"過去|最近|last|past|\d+\s*(分|小時|h\b|m\b)", re.IGNORECASE)


def test_three_builtin_templates_in_order():
    assert list_template_ids() == ["hr-spo2", "rso2", "vitals-overview"]


@pytest.mark.parametrize("template_id", ["hr-spo2", "rso2", "vitals-overview"])
def test_template_validates_and_obeys_rules(template_id):
    raw = load_template_raw(template_id)
    spec = SpecValidator().validate_dashboard(raw)
    assert spec.dashboard_id == f"tpl-{template_id}"
    assert spec.widgets
    promqls_by_instant_type: dict[str, set[str]] = {}
    for w in spec.widgets:
        assert w.thresholds == [], f"{template_id}/{w.id}: no clinical thresholds may be invented"
        assert not _TITLE_RANGE_SUFFIX_RE.search(w.title), w.title
        assert not _RANGE_WORDS.search(w.title), w.title
        for q in w.all_queries():
            assert q.source.value == "prometheus"
            body = re.sub(r"\bby\s*\([^)]*\)", " ", q.promql)  # drop label groupings
            names = {m for m in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", body) if m.startswith(("patient_", "cerebral_"))}
            assert names and names <= CATALOG_METRICS, q.promql
            assert q.promql.startswith("max by (patient_id)"), q.promql
        if w.type.value in ("stat_card", "gauge"):
            key = " ".join(w.query.promql.split())
            promqls_by_instant_type.setdefault(key, set()).add(w.type.value)
    assert all(types != {"stat_card", "gauge"} for types in promqls_by_instant_type.values())
    assert not any(w.type.value == "gauge" for w in spec.widgets)
    assert template_metrics(raw) and set(template_metrics(raw)) <= CATALOG_METRICS


def test_hr_spo2_composition():
    spec = SpecValidator().validate_dashboard(load_template_raw("hr-spo2"))
    assert spec.title == "心率與血氧監測"
    types = [w.type.value for w in spec.widgets]
    assert types == ["stat_card", "stat_card", "line_chart", "line_chart", "table"]
    table = spec.widgets[-1]
    assert table.title == "各病患一覽"
    assert table.query is None
    assert [q.key for q in table.queries] == ["hr", "spo2"]
    assert [q.label for q in table.queries] == ["心率", "血氧 SpO2"]
    assert [q.unit for q in table.queries] == ["bpm", "%"]
    assert "patient_heart_rate_bpm" in table.queries[0].promql
    assert "patient_spo2_percentage" in table.queries[1].promql
    assert all(q.query_type.value == "instant" for q in table.queries)
    units = {w.id: w.encoding.unit for w in spec.widgets if w.type.value != "table"}
    assert units["hr-stat"] == "bpm" and units["spo2-stat"] == "%"
    lines = [w for w in spec.widgets if w.type.value == "line_chart"]
    assert all(w.query.query_type.value == "range" and w.query.range == "1h" for w in lines)


def test_rso2_and_vitals_metrics():
    rso2 = template_metrics(load_template_raw("rso2"))
    assert rso2 == ["cerebral_rso2_left_percentage", "cerebral_rso2_right_percentage"]
    vitals = template_metrics(load_template_raw("vitals-overview"))
    assert set(vitals) == CATALOG_METRICS
    spec = SpecValidator().validate_dashboard(load_template_raw("vitals-overview"))
    table = spec.widgets[-1]
    assert table.type.value == "table" and len(table.queries) == 4


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@pytest.fixture
def api(monkeypatch, tmp_path):
    monkeypatch.setattr(templates_api, "_dashboards", DashboardStore(root=tmp_path / "store"))
    return TestClient(app)


def test_list_templates_both_forms(api):
    for path in ("/api/templates", "/api/templates/"):
        r = api.get(path)
        assert r.status_code == 200
        items = r.json()["templates"]
        assert [t["id"] for t in items] == ["hr-spo2", "rso2", "vitals-overview"]
        hr = items[0]
        assert hr["name"] == "心率與血氧監測"
        assert hr["widget_count"] == 5
        assert set(hr["metrics"]) == {"patient_heart_rate_bpm", "patient_spo2_percentage"}
        assert "範本" in hr["description"]
        assert set(hr["widget_types"]) == {"stat_card", "line_chart", "table"}


def test_get_single_template_and_404(api):
    r = api.get("/api/templates/rso2")
    assert r.status_code == 200
    assert r.json()["spec"]["title"] == "腦部血氧 rSO2 監測"
    assert api.get("/api/templates/nope").status_code == 404


def test_instantiate_persists_fresh_id(api, tmp_path):
    r = api.post("/api/templates/hr-spo2/instantiate")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["template_id"] == "hr-spo2" and body["name"] == "心率與血氧監測"
    dash = body["dashboard"]
    assert dash["dashboard_id"] != "tpl-hr-spo2"
    assert dash["dashboard_id"].startswith("tpl-hr-spo2-")
    assert dash["title"] == "心率與血氧監測"
    # `query` is null on the queries table in the wire shape
    table = dash["widgets"][-1]
    assert table["query"] is None and len(table["queries"]) == 2
    # Persisted through the working store (chat patches / 儲存 resolve it).
    store = DashboardStore(root=tmp_path / "store")
    assert store.load_dashboard(dash["dashboard_id"]) is not None
    # Every instantiate is a new id.
    r2 = api.post("/api/templates/hr-spo2/instantiate/")
    assert r2.status_code == 200
    assert r2.json()["dashboard"]["dashboard_id"] != dash["dashboard_id"]
    assert api.post("/api/templates/nope/instantiate").status_code == 404
