from backend.app.prometheus.promql_validator import PromQLValidator
from backend.app.specs.metric_catalog import MetricCatalog
from backend.app.specs.widget_spec import QuerySpec


def test_promql_validator_accepts_catalog_metric():
    catalog = MetricCatalog.load_from_file("configs/metric_catalog.yaml")
    validator = PromQLValidator(catalog)

    result = validator.validate_query_spec(QuerySpec(metric="patient_heart_rate_bpm"))

    assert result.valid


def test_promql_validator_rejects_unknown_metric():
    catalog = MetricCatalog.load_from_file("configs/metric_catalog.yaml")
    validator = PromQLValidator(catalog)

    result = validator.validate_query_spec(QuerySpec(metric="unknown_metric"))

    assert not result.valid
    assert "Metric is not allowed" in result.errors[0]


def test_promql_validator_rejects_global_scan():
    catalog = MetricCatalog.load_from_file("configs/metric_catalog.yaml")
    validator = PromQLValidator(catalog)

    result = validator.validate_query_spec(
        QuerySpec(metric="patient_heart_rate_bpm", promql='{__name__=~".*"}')
    )

    assert not result.valid
    assert any("Global __name__" in error for error in result.errors)


def test_promql_validator_limits_dense_range_queries():
    catalog = MetricCatalog.load_from_file("configs/metric_catalog.yaml")
    validator = PromQLValidator(catalog, max_points=10)

    result = validator.validate_query_spec(
        QuerySpec(metric="patient_heart_rate_bpm", time_range_seconds=300, step_seconds=1)
    )

    assert not result.valid
    assert any("too many points" in error for error in result.errors)

