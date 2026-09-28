"""Coverage and ambiguity tests for the alias / device-family tables (findings-mongo sections 5 and 6)."""

import copy
import json

import pytest

from nicu_qe import aliases
from nicu_qe.aliases import (
    AliasConfigError,
    AliasRegistry,
    METRIC_ALIASES_PATH,
    DEVICE_FAMILIES_PATH,
    classify_device,
    get_registry,
    normalize_key,
    normalize_key_text,
)

# findings-mongo section 6, "Top 60 medical_values keys", with the family the key occurs in.
TOP60 = [
    ("monitor", "spo2"), ("monitor", "respiration_rate"), ("monitor", "heart_rate"),
    ("monitor", "bp_diastolic"), ("monitor", "bp_systolic"), ("monitor", "bp_map"),
    ("invos", "rso2_left"), ("invos", "rso2_right"), ("invos", "rso2_left_avg"), ("invos", "rso2_right_avg"),
    ("osypka", "svi_meas"), ("osypka", "ci_meas"),
    ("ventilator", "frequency_meas"), ("ventilator", "pip_meas"), ("ventilator", "vt_meas"),
    ("ventilator", "fio2_meas"), ("ventilator", "pmean_meas"), ("ventilator", "peep_meas"),
    ("ventilator", "pip"), ("ventilator", "fio2"), ("ventilator", "peep"), ("ventilator", "map"),
    ("ventilator", "FiO2"), ("ventilator", "PIP"), ("ventilator", "VT"), ("ventilator", "R"),
    ("ventilator", "RR"), ("ventilator", "Cdyn"), ("ventilator", "MV"), ("ventilator", "TC"),
    ("ventilator", "IERatio"), ("ventilator", "MAP"), ("ventilator", "Leak"), ("ventilator", "C20"),
    ("ventilator", "PEEP"), ("ventilator", "Pinsp"), ("ventilator", "Ti"), ("ventilator", "FiO2_set"),
    ("ventilator", "RR_set"), ("monitor", "heart_beat"), ("ventilator", "pmean"),
    ("monitor", "blood_pressure"), ("monitor", "pulse_rate"), ("monitor", "blood_pressure_systolic"),
    ("monitor", "blood_pressure_diastolic"), ("ventilator", "ie_ratio"), ("ventilator", "mv_meas"),
    ("monitor", "pulse"), ("ventilator", "frequency_set"), ("ventilator", "peep_set"),
    ("ventilator", "fio2_set"), ("ventilator", "pinsp_set"), ("ventilator", "ti_set"),
    ("invos", "rSO2_right"), ("ventilator", "cdyn_meas"), ("ventilator", "time_const"),
    ("ventilator", "leak_percent"), ("ventilator", "resistance"), ("ventilator", "compliance_c20"),
    ("monitor", "blood_pressure_mean"),
]

# findings-mongo section 6 family paragraphs (variants beyond the top 60).
VARIANTS = [
    # Philips vitals
    ("monitor", "HR"), ("monitor", "hr"), ("monitor", "hr_bpm"),
    ("monitor", "respiratory_rate"), ("monitor", "resp_rate"), ("monitor", "rr"), ("monitor", "呼吸頻率"),
    ("monitor", "SpO2"), ("monitor", "oxygen_saturation"),
    # blood pressure
    ("monitor", "blood_pressure_map"), ("monitor", "blood_pressure_sys"), ("monitor", "blood_pressure_dia"),
    ("monitor", "blood pressure"), ("monitor", "blood pressure_systolic"),
    ("monitor", "blood pressure_diastolic"), ("monitor", "blood pressure_mean"),
    ("monitor", "nbp_sys"), ("monitor", "nbp_dia"), ("monitor", "nbp_mean"),
    ("monitor", "nibp_sys"), ("monitor", "nibp_dia"), ("monitor", "nibp_mean"),
    ("monitor", "noninvasive_blood_pressure_systolic"), ("monitor", "noninvasive_blood_pressure_diastolic"),
    ("monitor", "noninvasive_blood_pressure_mean"),
    ("monitor", "noninvasive_bp_systolic"), ("monitor", "noninvasive_bp_diastolic"), ("monitor", "noninvasive_bp_mean"),
    ("monitor", "abp_sys"), ("monitor", "abp_dia"), ("monitor", "abp_mean"),
    ("monitor", "systolic_bp"), ("monitor", "diastolic_bp"), ("monitor", "mean_bp"),
    ("monitor", "BP_systolic"), ("monitor", "NBP_sys"), ("monitor", "bp"), ("monitor", "nbp"),
    # INVOS
    ("invos", "rSO2_left"), ("invos", "rSo2_left"), ("invos", "rSo2_right"), ("invos", "L_rSO2"),
    ("invos", "R_rSO2"), ("invos", "left_rso2"), ("invos", "right_rso2"), ("invos", "L"), ("invos", "R"),
    ("invos", "L_value"), ("invos", "R_value"), ("invos", "left_value"), ("invos", "right_value"),
    ("invos", "average_rSO2"), ("invos", "rSO2_average"), ("invos", "rSO2_avg"),
    # Osypka
    ("osypka", "CI"), ("osypka", "CO"), ("osypka", "ICON"), ("osypka", "TFC"), ("osypka", "SVRI"),
    ("osypka", "SVV"), ("osypka", "VIC"), ("osypka", "FTC"), ("osypka", "SI"), ("osypka", "co_meas"),
    ("osypka", "sv_meas"),
    # Ventilators
    ("ventilator", "I:E"), ("ventilator", "f"), ("ventilator", "frequency"), ("ventilator", "Ppeak"),
    ("ventilator", "Plateau"), ("ventilator", "tidal_volume"), ("ventilator", "minute_volume"),
    ("ventilator", "etco2"), ("ventilator", "co2"),
]

JUNK = ["value_1", "setting1", "unknown_1", "other", "possibly peep", "未知數值", "raw_text", "airway_pressure", "unit"]


def test_key_text_normalization_rule():
    assert normalize_key_text("  Blood Pressure-Systolic ") == "blood_pressure_systolic"
    assert normalize_key_text("rSO2_left") == "rso2_left"


@pytest.mark.parametrize("family,key", TOP60)
def test_top60_keys_all_map(family, key):
    assert normalize_key(family, key) is not None, (family, key)


@pytest.mark.parametrize("family,key", VARIANTS)
def test_family_variants_all_map(family, key):
    assert normalize_key(family, key) is not None, (family, key)


def test_top60_list_is_complete():
    assert len(TOP60) == 60
    assert len({k for _, k in TOP60}) == 60


@pytest.mark.parametrize("key", JUNK)
def test_junk_keys_ignored(key):
    reg = get_registry()
    assert reg.is_ignored(key)
    for fam in ("monitor", "invos", "osypka", "ventilator", "unknown"):
        assert normalize_key(fam, key) is None


@pytest.mark.parametrize(
    "family,canon,keys",
    [
        ("monitor", "hr", ["heart_rate", "HR", "heart_beat", "pulse_rate", "pulse", "hr_bpm"]),
        ("monitor", "rr", ["respiration_rate", "respiratory_rate", "resp_rate", "RR", "呼吸頻率"]),
        ("monitor", "spo2", ["spo2", "SpO2", "oxygen_saturation"]),
        ("monitor", "bp_sys", ["bp_systolic", "blood pressure_systolic", "nibp_sys", "NBP_sys", "systolic_bp"]),
        ("monitor", "bp_dia", ["bp_diastolic", "blood_pressure_dia", "nbp_dia", "noninvasive_bp_diastolic"]),
        ("monitor", "bp_map", ["bp_map", "blood_pressure_mean", "mean_bp", "nibp_mean", "MAP"]),
        ("invos", "rso2_left", ["rso2_left", "rSO2_left", "rSo2_left", "L_rSO2", "left_rso2", "L", "L_value"]),
        ("invos", "rso2_right", ["rso2_right", "rSO2_right", "R_rSO2", "right_rso2", "R", "right_value"]),
        ("invos", "rso2_avg", ["average_rSO2", "rSO2_average", "rSO2_avg"]),
        ("ventilator", "pip", ["pip", "PIP", "pip_meas", "Ppeak"]),
        ("ventilator", "peep", ["peep", "PEEP", "peep_meas"]),
        ("ventilator", "pmean", ["pmean", "pmean_meas", "MAP", "map"]),
        ("ventilator", "fio2", ["fio2", "FiO2", "fio2_meas"]),
        ("ventilator", "vt", ["VT", "vt_meas", "tidal_volume"]),
        ("ventilator", "mv", ["MV", "mv_meas", "minute_volume"]),
        ("ventilator", "vent_rr", ["RR", "frequency", "f", "frequency_meas"]),
    ],
)
def test_at_least_three_aliases_each(family, canon, keys):
    assert len(keys) >= 3
    for k in keys:
        assert normalize_key(family, k) == canon, k


def test_family_ambiguity():
    assert normalize_key("monitor", "MAP") == "bp_map"
    assert normalize_key("ventilator", "MAP") == "pmean"
    assert normalize_key("monitor", "RR") == "rr"
    assert normalize_key("ventilator", "RR") == "vent_rr"
    assert normalize_key("unknown", "MAP") is None
    assert normalize_key("invos", "R") == "rso2_right"
    assert normalize_key("ventilator", "R") == "resistance"


def test_unknown_family_only_unambiguous():
    assert normalize_key("unknown", "heart_rate") == "hr"
    assert normalize_key("unknown", "SpO2") == "spo2"
    assert normalize_key("unknown", "respiration_rate") == "rr"
    assert normalize_key("unknown", "rso2_left") == "rso2_left"
    assert normalize_key("unknown", "ci_meas") == "ci"
    assert normalize_key("unknown", "svi_meas") == "svi"
    # ambiguous across families or not in the allowed set
    assert normalize_key("unknown", "RR") is None
    assert normalize_key("unknown", "R") is None
    # contract v2 #14: unambiguous, length >= 3, not denied -> accepted on unknown screens
    assert normalize_key("unknown", "pip") == "pip"
    assert normalize_key("unknown", "blood_pressure") is None
    # WP-0: single-character aliases and bare rr are never accepted for unknown
    assert normalize_key("unknown", "L") is None
    assert normalize_key("unknown", "l") is None
    assert normalize_key("unknown", "rr") is None
    assert normalize_key("unknown", "f") is None
    assert normalize_key("unknown", "map") is None
    assert normalize_key("invos", "L") == "rso2_left"
    assert normalize_key("unknown", "L_rSO2") == "rso2_left"
    # families not in the table are treated as unknown
    assert normalize_key("something_else", "heart_rate") == "hr"
    assert normalize_key("something_else", "MAP") is None


def test_version_attribute():
    assert aliases.version.startswith("metrics:")
    assert get_registry().version == aliases.version


# findings-mongo section 5: detected_model top 15 plus quoted long-tail spellings.
DEVICES = [
    ("unknown", "unknown", "unknown"),
    ("Philips mp30", "monitor", "philips_mp30"),
    ("Medtronic INVOS 5100C", "invos", "invos_5100c"),
    ("Osypka Aesculon Hemodynamic Monitor", "osypka", "osypka_aesculon"),
    ("Dräger Evita V600", "ventilator", "drager_v600"),
    ("Philips IntelliVue MP30", "monitor", "philips_mp30"),
    ("black ASUS computer", "unknown", "unknown"),
    ("Philips IntelliVue MP300 Patient Monitor", "monitor", "philips_mp300"),
    ("Dräger Infinity C500 Ventilator", "ventilator", "drager_c500"),
    ("Philips IntelliVue MP300", "monitor", "philips_mp300"),
    ("Medtronic_INVOS_5100C", "invos", "invos_5100c"),
    ("DRAGER_C500", "ventilator", "drager_c500"),
    ("Dräger Infinity C500", "ventilator", "drager_c500"),
    ("SERVO-U", "ventilator", "servo_u"),
    ("DRAGER_V600", "ventilator", "drager_v600"),
    ("OSYPKA MEDICAL RESCUCION", "osypka", "osypka_other"),
    ("德爾格 Dräger Infinity C500 呼吸機", "ventilator", "drager_c500"),
    ("unKnown", "unknown", "unknown"),
    ("philips MP-30", "monitor", "philips_mp30"),
    ("", "unknown", "unknown"),
    (None, "unknown", "unknown"),
]


@pytest.mark.parametrize("raw,family,model", DEVICES)
def test_classify_device(raw, family, model):
    assert classify_device(raw) == (family, model)


def test_corrected_model_takes_precedence():
    assert classify_device("unknown", "Philips mp30") == ("monitor", "philips_mp30")
    assert classify_device("Philips mp30", "  ") == ("monitor", "philips_mp30")


def _docs():
    with open(METRIC_ALIASES_PATH, encoding="utf-8") as fh:
        m = json.load(fh)
    with open(DEVICE_FAMILIES_PATH, encoding="utf-8") as fh:
        d = json.load(fh)
    return m, d


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m, d: m.pop("version"),
        lambda m, d: m.pop("ignore"),
        lambda m, d: m["metrics"]["hr"].pop("family"),
        lambda m, d: m["metrics"]["hr"].pop("aliases"),
        lambda m, d: m["metrics"]["hr"].pop("range"),
        lambda m, d: m["metrics"]["hr"].__setitem__("range", [5, 1]),
        lambda m, d: m["metrics"]["hr"].__setitem__("family", "bogus"),
        lambda m, d: m["metrics"]["rr"]["aliases"].append("heart_rate"),  # same alias -> 2 canonicals
        lambda m, d: d.pop("families"),
        lambda m, d: d["models"].__setitem__("x", ["("]),
    ],
)
def test_invalid_json_structure_rejected(mutate):
    m, d = _docs()
    m, d = copy.deepcopy(m), copy.deepcopy(d)
    mutate(m, d)
    with pytest.raises(AliasConfigError):
        AliasRegistry(m, d)


def test_ventilator_respiratory_rate_aliases_2026_09_28_2():
    # added from the WP-4 unmapped report: ventilator screens label their rate this way
    assert normalize_key("ventilator", "respiratory_rate") == "vent_rr"
    assert normalize_key("ventilator", "呼吸頻率") == "vent_rr"
    assert normalize_key("monitor", "respiratory_rate") == "rr"
    assert normalize_key("monitor", "呼吸頻率") == "rr"
    # now ambiguous across families -> still unmapped for unknown devices
    assert normalize_key("unknown", "respiratory_rate") is None


def test_unknown_family_rule_v2_14():
    ok = {"bp_systolic": "bp_sys", "bp_diastolic": "bp_dia", "bp_map": "bp_map", "pip": "pip", "PIP": "pip",
          "peep": "peep", "PEEP": "peep", "fio2": "fio2", "FiO2": "fio2", "pmean": "pmean", "vt_meas": "vt",
          "respiration_rate": "rr", "heart_rate": "hr", "L_rSO2": "rso2_left", "ci_meas": "ci"}
    for k, canon in ok.items():
        assert normalize_key("unknown", k) == canon, k
    for k in ("MAP", "map", "RR", "rr", "R", "L", "f", "mv_set", "HR", "VT", "blood_pressure"):
        assert normalize_key("unknown", k) is None, k


def test_osypka_accepts_hr_aliases():
    for k in ("HR", "heart_rate", "hr"):
        assert normalize_key("osypka", k) == "hr"
    assert normalize_key("monitor", "HR") == "hr"
    assert normalize_key("osypka", "heart_beat") is None


def test_unknown_family_deny_is_configurable():
    m, d = _docs()
    m = copy.deepcopy(m)
    m["unknown_family_deny"] = ["rr", "r", "l", "map", "f", "mv_set", "pip"]
    reg = AliasRegistry(m, d)
    assert reg.normalize_key("unknown", "pip") is None
    assert reg.normalize_key("unknown", "peep") == "peep"
