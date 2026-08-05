from __future__ import annotations

from .client import PrometheusClient


class MetricDiscovery:
    """Thin wrapper for future live discovery.

    The MVP trusts configs/metric_catalog.yaml as the allowlist. This class is
    the extension point for comparing catalog entries with Prometheus metadata.
    """

    def __init__(self, client: PrometheusClient) -> None:
        self.client = client

    async def list_metric_names(self) -> list[str]:
        # TODO: call /api/v1/label/__name__/values and reconcile with catalog.
        return []
