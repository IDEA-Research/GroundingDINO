from __future__ import annotations

import math
import time


def _wave(base: float, amplitude: float, period_seconds: float, offset: float = 0) -> float:
    return base + amplitude * math.sin((time.time() + offset) / period_seconds)


def render_demo_medical_metrics() -> str:
    patients = [
        {
            "patient_id": "demo-patient-1",
            "bed_id": "bed-01",
            "heart_rate": _wave(82, 8, 13),
            "spo2": _wave(97, 1.5, 19),
            "systolic": _wave(116, 7, 23),
            "diastolic": _wave(74, 5, 29),
        },
        {
            "patient_id": "demo-patient-2",
            "bed_id": "bed-02",
            "heart_rate": _wave(105, 14, 11, 7),
            "spo2": _wave(92, 2.8, 17, 5),
            "systolic": _wave(132, 10, 31, 3),
            "diastolic": _wave(86, 6, 37, 9),
        },
    ]

    lines = [
        "# HELP patient_heart_rate_bpm Demo patient heart rate.",
        "# TYPE patient_heart_rate_bpm gauge",
        "# HELP patient_spo2_percent Demo patient oxygen saturation.",
        "# TYPE patient_spo2_percent gauge",
        "# HELP patient_systolic_bp_mmhg Demo patient systolic blood pressure.",
        "# TYPE patient_systolic_bp_mmhg gauge",
        "# HELP patient_diastolic_bp_mmhg Demo patient diastolic blood pressure.",
        "# TYPE patient_diastolic_bp_mmhg gauge",
    ]

    for patient in patients:
        labels = f'patient_id="{patient["patient_id"]}",bed_id="{patient["bed_id"]}"'
        lines.extend(
            [
                f"patient_heart_rate_bpm{{{labels}}} {patient['heart_rate']:.2f}",
                f"patient_spo2_percent{{{labels}}} {patient['spo2']:.2f}",
                f"patient_systolic_bp_mmhg{{{labels}}} {patient['systolic']:.2f}",
                f"patient_diastolic_bp_mmhg{{{labels}}} {patient['diastolic']:.2f}",
            ]
        )

    return "\n".join(lines) + "\n"
