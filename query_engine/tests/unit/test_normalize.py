"""Value cleaning, composites, corrections and primary selection (DESIGN section 4)."""

from datetime import datetime, timezone

import pytest

from nicu_qe.normalize import choose_primary, normalize_medical_values


def test_philips_sample_document():
    mv = {"heart_rate": 168, "respiration_rate": 49, "bp_systolic": None, "bp_diastolic": None, "bp_map": None, "spo2": 92}
    r = normalize_medical_values(mv, None, "monitor")
    assert r.m == {"hr": 168.0, "rr": 49.0, "spo2": 92.0}
    assert r.q == {"hr": ["ok"], "rr": ["ok"], "spo2": ["ok"]}
    assert r.raw_keys == list(mv)
    assert r.unmapped == {} and r.flags == []


def test_off_is_non_numeric():
    r = normalize_medical_values({"PEEP": "OFF", "Leak": "Off", "MV": "---", "VT": ""}, None, "ventilator")
    assert r.m == {}
    for canon, raw in (("peep", "OFF"), ("leak_pct", "Off"), ("mv", "---"), ("vt", "")):
        assert r.q[canon] == ["non_numeric"]
        assert r.raw[canon] == raw


def test_numeric_string():
    r = normalize_medical_values({"peep_meas": "7.3"}, None, "ventilator")
    assert r.m == {"peep": 7.3}
    assert r.q["peep"] == ["ok"]


@pytest.mark.parametrize("raw,expected,flags", [
    (0.21, 21.0, ["unit_normalized"]),
    ("0.21", 21.0, ["unit_normalized"]),
    (40, 40.0, ["ok"]),
    (1.0, 100.0, ["unit_normalized"]),
])
def test_fio2(raw, expected, flags):
    r = normalize_medical_values({"FiO2": raw}, None, "ventilator")
    assert r.m["fio2"] == pytest.approx(expected)
    assert r.q["fio2"] == flags


def test_ie_ratio():
    r = normalize_medical_values({"IERatio": "1:2.0"}, None, "ventilator")
    assert r.m["ie_ratio"] == pytest.approx(0.5)
    assert r.raw["ie_ratio"] == "1:2.0"
    r2 = normalize_medical_values({"IERatio": "1.0:3.3"}, None, "ventilator")
    assert r2.m["ie_ratio"] == pytest.approx(1 / 3.3)


def test_blood_pressure_string_composite():
    r = normalize_medical_values({"blood_pressure": "66/37 (48)"}, None, "monitor")
    assert r.m == {"bp_sys": 66.0, "bp_dia": 37.0, "bp_map": 48.0}
    for k in ("bp_sys", "bp_dia", "bp_map"):
        assert r.q[k] == ["parsed_composite"]
    assert r.raw["blood_pressure"] == "66/37 (48)"


def test_blood_pressure_object_composite():
    r = normalize_medical_values({"blood_pressure": {"systolic": 70, "diastolic": 40, "map": 50}}, None, "monitor")
    assert r.m == {"bp_sys": 70.0, "bp_dia": 40.0, "bp_map": 50.0}
    r2 = normalize_medical_values({"blood_pressure": {"systolic": "71", "diastolic": 41, "mean": 51}}, None, "monitor")
    assert r2.m == {"bp_sys": 71.0, "bp_dia": 41.0, "bp_map": 51.0}


def test_blood_pressure_no_map_and_explicit_wins():
    r = normalize_medical_values({"bp_systolic": 80, "blood_pressure": "66/37"}, None, "monitor")
    assert r.m == {"bp_sys": 80.0, "bp_dia": 37.0}
    assert r.q["bp_sys"] == ["ok"]


def test_blood_pressure_scalar_is_unmapped_not_guessed():
    r = normalize_medical_values({"blood_pressure": 63}, None, "monitor")
    assert r.m == {}
    assert r.unmapped == {"blood_pressure": 63}


def test_out_of_range_keeps_value():
    r = normalize_medical_values({"heart_rate": 999}, None, "monitor")
    assert r.m["hr"] == 999.0
    assert r.q["hr"] == ["out_of_range"]


def test_corrections_overlay_string_and_null():
    r = normalize_medical_values({"peep": 23, "pip": 6.8}, {"peep": "5"}, "ventilator")
    assert r.m == {"peep": 5.0, "pip": 6.8}
    assert r.q["peep"] == ["corrected"]
    r2 = normalize_medical_values({"rso2_left": 76, "rso2_right": 60}, {"rso2_right": "null"}, "invos")
    assert r2.m == {"rso2_left": 76.0}
    assert "rso2_right" not in r2.m
    assert r2.q["rso2_right"] == ["corrected"]


def test_corrections_alias_and_unit_normalization():
    r = normalize_medical_values({"FiO2": 25}, {"FiO2": "0.3"}, "ventilator")
    assert r.m["fio2"] == pytest.approx(30.0)
    assert set(r.q["fio2"]) == {"corrected", "unit_normalized"}


def test_corrections_unmapped_key_flagged():
    r = normalize_medical_values({"pip": 6}, {"mystery": "3"}, "ventilator")
    assert r.unmapped == {"mystery": "3"}
    assert "unmapped_correction" in r.flags


@pytest.mark.parametrize("mv", [{}, [], None])
def test_empty_medical_values(mv):
    r = normalize_medical_values(mv, None, "monitor")
    assert r.m == {}
    assert "empty" in r.flags


def test_unmapped_and_junk():
    r = normalize_medical_values({"heart_rate": 120, "value_1": 5, "weird_key": 9, "MAP": 40}, None, "unknown")
    assert r.m == {"hr": 120.0}
    assert r.unmapped == {"weird_key": 9, "MAP": 40}
    assert "value_1" not in r.unmapped
    assert r.raw_keys == ["heart_rate", "value_1", "weird_key", "MAP"]


def test_alias_conflict_first_numeric_wins():
    r = normalize_medical_values({"HR": "OFF", "heart_rate": 150, "pulse": 152}, None, "monitor")
    assert r.m["hr"] == 150.0
    assert r.q["hr"] == ["alias_conflict"]


def test_bool_is_non_numeric():
    r = normalize_medical_values({"spo2": True}, None, "monitor")
    assert r.m == {}
    assert r.q["spo2"] == ["non_numeric"]


T0 = datetime(2026, 9, 21, 6, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 9, 21, 7, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 9, 21, 8, 0, tzinfo=timezone.utc)


def test_primary_corrected_beats_everything():
    cands = [
        {"is_corrected": False, "success": True, "analyzed_at": T2},
        {"is_corrected": True, "success": False, "analyzed_at": T0},
        {"is_corrected": False, "success": True, "analyzed_at": T1},
    ]
    assert choose_primary(cands) == 1


def test_primary_success_beats_newer_failure():
    cands = [
        {"success": False, "analyzed_at": T2},
        {"success": True, "analyzed_at": T0},
    ]
    assert choose_primary(cands) == 1


def test_primary_latest_analyzed_at():
    cands = [
        {"is_corrected": False, "success": True, "analyzed_at": T0},
        {"is_corrected": False, "success": True, "analyzed_at": T2},
        {"is_corrected": False, "success": True, "analyzed_at": T1},
    ]
    assert choose_primary(cands) == 1


def test_primary_empty_raises():
    with pytest.raises(ValueError):
        choose_primary([])
