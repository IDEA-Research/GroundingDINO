import type { WidgetSpec } from "../types/dashboard";
import { LineChartWidget } from "./widgets/LineChartWidget";
import { StatCardWidget } from "./widgets/StatCardWidget";
import { TableWidget } from "./widgets/TableWidget";
import { ThresholdCardWidget } from "./widgets/ThresholdCardWidget";

interface Props {
  widget: WidgetSpec;
  variables: Record<string, string>;
  dashboardRefreshIntervalMs: number;
}

export function WidgetRenderer({ widget, variables, dashboardRefreshIntervalMs }: Props) {
  const refreshIntervalMs = widget.refresh_interval_ms ?? dashboardRefreshIntervalMs ?? 5000;

  if (widget.type === "line_chart") {
    return <LineChartWidget widget={widget} variables={variables} refreshIntervalMs={refreshIntervalMs} />;
  }
  if (widget.type === "stat_card") {
    return <StatCardWidget widget={widget} variables={variables} refreshIntervalMs={refreshIntervalMs} />;
  }
  if (widget.type === "threshold_card") {
    return <ThresholdCardWidget widget={widget} variables={variables} refreshIntervalMs={refreshIntervalMs} />;
  }
  return <TableWidget widget={widget} variables={variables} refreshIntervalMs={refreshIntervalMs} />;
}

