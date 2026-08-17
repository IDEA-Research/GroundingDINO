"""MetricCatalog — medical vocabulary, TTL cache, explicit failure.

The catalog has NO fallback list: discovery failure raises
`PrometheusUnavailable` so every caller degrades explicitly. Medical
synonym hits survive an empty/partial live catalog (patient_* series
vanish whenever no camera stream runs — they are still the right
answer), reported with ``present=False``.
"""

from __future__ import annotations

from urllib import error as urllib_error

import pytest

from app.prometheus import catalog as catalog_mod
from app.prometheus.catalog import MetricCatalog, resolve_synonyms
from app.prometheus.client import PrometheusUnavailable


# ---------------------------------------------------------------------------
# Synonym resolution (pure, no network)
# ---------------------------------------------------------------------------


def test_chinese_vitals_resolve():
    assert resolve_synonyms("幫我盯心率和血氧") == [
        "patient_heart_rate_bpm",
        "patient_spo2_percentage",
    ]


def test_english_vitals_resolve():
    out = resolve_synonyms("watch heart rate and SpO2 please")
    assert out == ["patient_heart_rate_bpm", "patient_spo2_percentage"]


def test_short_ascii_keys_need_word_boundaries():
    # "hr" must not fire inside "threshold", "pip" not inside "pipeline",
    # "bp" not inside "bpm".
    assert resolve_synonyms("set a threshold on the pipeline bpm") == []
    # ...but they DO fire next to CJK text (plain \b would fail there).
    assert resolve_synonyms("看hr和bp") == [
        "patient_heart_rate_bpm",
        "patient_blood_pressure_systolic_mmhg",
    ]


def test_ventilator_key_does_not_fire_respiration():
    # 呼吸器 (ventilator) contains 呼吸 (respiration) — the longer key
    # consumes the text so the shorter one cannot double-fire.
    assert resolve_synonyms("呼吸器") == ["ventilator_fio2_percentage"]
    assert resolve_synonyms("呼吸") == ["patient_respiration_rate_per_minute"]
    both = resolve_synonyms("盯呼吸和呼吸器")
    assert "patient_respiration_rate_per_minute" in both
    assert "ventilator_fio2_percentage" in both


def test_vitals_trio():
    assert resolve_synonyms("生命徵象") == [
        "patient_heart_rate_bpm",
        "patient_spo2_percentage",
        "patient_respiration_rate_per_minute",
    ]


def test_blood_pressure_sides():
    assert resolve_synonyms("收縮壓") == [
        "patient_blood_pressure_systolic_mmhg"
    ]
    assert resolve_synonyms("舒張壓") == [
        "patient_blood_pressure_diastolic_mmhg"
    ]


def test_cerebral_resolves_both_hemispheres():
    assert resolve_synonyms("腦氧") == [
        "cerebral_rso2_left_percentage",
        "cerebral_rso2_right_percentage",
    ]


def test_ventilator_pressures():
    assert resolve_synonyms("peep") == ["ventilator_peep_cmh2o"]
    assert resolve_synonyms("check PIP") == ["ventilator_pip_cmh2o"]


# ---------------------------------------------------------------------------
# suggest() against a controlled live catalog
# ---------------------------------------------------------------------------


def _catalog_with(monkeypatch, names: list[str]) -> MetricCatalog:
    cat = MetricCatalog(ttl_seconds=1000)
    monkeypatch.setattr(cat, "_fetch_names", lambda: list(names))
    return cat


def test_synonym_hits_survive_catalog_without_patient_metrics(monkeypatch):
    # No stream running: patient_* absent from the live catalog. The
    # synonym answer is still returned, marked present=False.
    cat = _catalog_with(monkeypatch, ["node_cpu_seconds_total", "up"])
    assert cat.suggest("心率") == ["patient_heart_rate_bpm"]
    detailed = cat.suggest_detailed("心率")
    assert detailed[0].metric == "patient_heart_rate_bpm"
    assert detailed[0].present is False
    assert detailed[0].reason == "medical synonym"


def test_synonym_hits_marked_present_when_stream_runs(monkeypatch):
    cat = _catalog_with(
        monkeypatch, ["patient_heart_rate_bpm", "node_cpu_seconds_total"]
    )
    detailed = cat.suggest_detailed("心率")
    assert detailed[0].present is True


def test_catalog_substring_match_after_synonyms(monkeypatch):
    cat = _catalog_with(
        monkeypatch,
        ["node_memory_MemAvailable_bytes", "node_memory_MemTotal_bytes",
         "up"],
    )
    assert cat.suggest("node_memory") == [
        "node_memory_MemAvailable_bytes",
        "node_memory_MemTotal_bytes",
    ]


def test_suggest_dedups_and_limits(monkeypatch):
    cat = _catalog_with(
        monkeypatch, ["patient_heart_rate_bpm", "patient_spo2_percentage"]
    )
    # "心率" synonym resolves patient_heart_rate_bpm; the substring pass
    # must not re-add it.
    out = cat.suggest("心率", limit=1)
    assert out == ["patient_heart_rate_bpm"]


# ---------------------------------------------------------------------------
# TTL cache
# ---------------------------------------------------------------------------


def test_ttl_cache_avoids_refetch_within_ttl(monkeypatch):
    calls = {"n": 0}

    def _fetch():
        calls["n"] += 1
        return ["up"]

    cat = MetricCatalog(ttl_seconds=1000)
    monkeypatch.setattr(cat, "_fetch_names", _fetch)
    assert cat.list_metrics() == ["up"]
    assert cat.list_metrics() == ["up"]
    assert calls["n"] == 1


def test_ttl_zero_refetches_every_call(monkeypatch):
    calls = {"n": 0}

    def _fetch():
        calls["n"] += 1
        return ["up"]

    cat = MetricCatalog(ttl_seconds=0)
    monkeypatch.setattr(cat, "_fetch_names", _fetch)
    cat.list_metrics()
    cat.list_metrics()
    assert calls["n"] == 2


def test_ttl_env_override(monkeypatch):
    monkeypatch.setenv("HELPER_DASHBOARD_CATALOG_TTL_SECONDS", "7.5")
    assert MetricCatalog().ttl_seconds == 7.5
    monkeypatch.setenv("HELPER_DASHBOARD_CATALOG_TTL_SECONDS", "bogus")
    assert MetricCatalog().ttl_seconds == 30.0


# ---------------------------------------------------------------------------
# Explicit failure — no fallback list, ever
# ---------------------------------------------------------------------------


def test_discovery_failure_raises_prometheus_unavailable(monkeypatch):
    # Restore the REAL fetch (the suite-wide conftest fixture stubs it)
    # and break urllib underneath it.
    monkeypatch.setattr(
        MetricCatalog,
        "_fetch_names",
        MetricCatalog._real_fetch_names_for_tests,
    )

    def _boom(*_a, **_k):
        raise urllib_error.URLError("connection refused")

    monkeypatch.setattr(
        catalog_mod.urllib_request, "urlopen", _boom
    )
    cat = MetricCatalog(base_url="http://127.0.0.1:59999")
    with pytest.raises(PrometheusUnavailable) as exc:
        cat.list_metrics()
    assert "metric catalog discovery failed" in str(exc.value)
    # suggest() consults the live catalog → same explicit failure.
    with pytest.raises(PrometheusUnavailable):
        cat.suggest("心率")


def test_fallback_metrics_list_is_gone():
    # The silent FALLBACK_METRICS substitution was deleted on purpose.
    assert not hasattr(catalog_mod, "FALLBACK_METRICS")
