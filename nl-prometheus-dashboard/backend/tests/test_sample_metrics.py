from backend.app.prometheus.sample_metrics import render_demo_medical_metrics


def test_sample_metrics_include_catalog_metric_names():
    metrics = render_demo_medical_metrics()

    assert "patient_heart_rate_bpm" in metrics
    assert "patient_spo2_percent" in metrics
    assert "patient_systolic_bp_mmhg" in metrics
    assert "patient_diastolic_bp_mmhg" in metrics
    assert 'patient_id="demo-patient-1"' in metrics
