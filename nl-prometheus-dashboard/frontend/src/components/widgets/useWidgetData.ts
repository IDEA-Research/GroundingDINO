import { useEffect, useState } from "react";
import { queryWidget } from "../../api/queryApi";
import type { WidgetQueryResponse, WidgetSpec } from "../../types/dashboard";

export function useWidgetData(
  widget: WidgetSpec,
  variables: Record<string, string>,
  refreshIntervalMs: number
) {
  const [data, setData] = useState<WidgetQueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        setError(null);
        const result = await queryWidget(widget, variables);
        if (!cancelled) {
          setData(result);
          setLoading(false);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Query failed");
          setLoading(false);
        }
      }
    }

    load();
    const timer = window.setInterval(load, refreshIntervalMs || 5000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [widget, variables, refreshIntervalMs]);

  return { data, error, loading };
}

