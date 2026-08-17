// Safe, deterministic renderer: DashboardSpec -> React elements drawn
// exclusively from widget-toolkit. Unknown widget types never render.

import type { DashboardSpec, WidgetSpec, WidgetType } from "./spec-schema";
import { WIDGET_TYPES } from "./spec-schema";

import { LineChartWidget } from "@/widget-toolkit/LineChartWidget";
import { StatCardWidget } from "@/widget-toolkit/StatCardWidget";
import { GaugeWidget } from "@/widget-toolkit/GaugeWidget";
import { TableWidget } from "@/widget-toolkit/TableWidget";
import { AlertListWidget } from "@/widget-toolkit/AlertListWidget";
import { PieChartWidget } from "@/widget-toolkit/PieChartWidget";
import { BarChartWidget } from "@/widget-toolkit/BarChartWidget";
import { HeatmapWidget } from "@/widget-toolkit/HeatmapWidget";
import { DecisionFlowWidget } from "@/widget-toolkit/DecisionFlowWidget";

export interface RenderProblem {
  widget_id: string;
  reason: string;
}

export function isSupportedType(t: string): t is WidgetType {
  return (WIDGET_TYPES as string[]).includes(t);
}

export function renderWidget(widget: WidgetSpec): JSX.Element | null {
  switch (widget.type) {
    case "line_chart":
      return <LineChartWidget widget={widget} />;
    case "stat_card":
      return <StatCardWidget widget={widget} />;
    case "gauge":
      return <GaugeWidget widget={widget} />;
    case "table":
      return <TableWidget widget={widget} />;
    case "alert_list":
      return <AlertListWidget widget={widget} />;
    case "pie_chart":
      return <PieChartWidget widget={widget} />;
    case "bar_chart":
      return <BarChartWidget widget={widget} />;
    case "heatmap":
      return <HeatmapWidget widget={widget} />;
    case "decision_flow":
      return <DecisionFlowWidget widget={widget} />;
    default:
      // Defense in depth — validator already rejected this. Never
      // interpret as code, just skip.
      return null;
  }
}

export function collectRenderProblems(spec: DashboardSpec): RenderProblem[] {
  const problems: RenderProblem[] = [];
  for (const w of spec.widgets) {
    if (!isSupportedType(w.type)) {
      problems.push({
        widget_id: w.id,
        reason: `unsupported widget type: ${w.type}`,
      });
    }
  }
  return problems;
}
