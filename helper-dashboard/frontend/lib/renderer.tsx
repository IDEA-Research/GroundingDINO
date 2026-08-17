"use client";

// Safe, deterministic renderer: DashboardSpec -> React elements drawn
// exclusively from widget-toolkit. Unknown widget types never render.
//
// Every query-driven widget is wrapped in QueryGate: the gate owns the
// single fetch for the widget and renders the shared loading / NO-DATA /
// ERROR states, so no widget can silently show a blank (or fabricated)
// panel. Widgets themselves are purely presentational and receive the
// already-fetched WidgetData.

import type { DashboardSpec, WidgetSpec, WidgetType } from "./spec-schema";
import { WIDGET_TYPES } from "./spec-schema";
import { useQueryData, type WidgetData } from "./useQueryData";

import { LineChartWidget } from "@/widget-toolkit/LineChartWidget";
import { StatCardWidget } from "@/widget-toolkit/StatCardWidget";
import { GaugeWidget } from "@/widget-toolkit/GaugeWidget";
import { TableWidget } from "@/widget-toolkit/TableWidget";
import { AlertListWidget } from "@/widget-toolkit/AlertListWidget";
import { PieChartWidget } from "@/widget-toolkit/PieChartWidget";
import { BarChartWidget } from "@/widget-toolkit/BarChartWidget";
import { HeatmapWidget } from "@/widget-toolkit/HeatmapWidget";
import { DecisionFlowWidget } from "@/widget-toolkit/DecisionFlowWidget";
import { DataStateNotice } from "@/widget-toolkit/DataStateNotice";
import { WidgetFrame } from "@/widget-toolkit/WidgetFrame";

export interface RenderProblem {
  widget_id: string;
  reason: string;
}

export function isSupportedType(t: string): t is WidgetType {
  return (WIDGET_TYPES as string[]).includes(t);
}

// Runs the widget's query exactly once and gates rendering on its status.
// `renderEmptyAsContent` lets a widget present "empty" itself when empty is
// a meaningful GOOD state (alert_list: no firing alerts is calm, not grey).
function QueryGate({
  widget,
  renderEmptyAsContent = false,
  children,
}: {
  widget: WidgetSpec;
  renderEmptyAsContent?: boolean;
  children: (data: WidgetData) => JSX.Element;
}) {
  const data = useQueryData(widget.query);
  const passThrough =
    data.status === "ok" ||
    (renderEmptyAsContent && data.status === "empty");
  if (!passThrough) {
    return (
      <WidgetFrame widget={widget}>
        <DataStateNotice data={data} promql={widget.query.promql} />
      </WidgetFrame>
    );
  }
  if (data.nSeries > 1) {
    // The backend collapsed several series into one by MEAN (free-text
    // device_model labels split one vital into parallel series). Say so
    // — an averaged value must never pass for a raw reading.
    return (
      <div className="relative h-full">
        {children(data)}
        <div
          data-annotation="n-series"
          className="pointer-events-none absolute bottom-1 right-2 rounded bg-slate-800/80 px-1.5 py-0.5 text-[10px] text-slate-400"
          title="Multiple Prometheus series were averaged into this value"
        >
          {data.nSeries} series (averaged)
        </div>
      </div>
    );
  }
  return children(data);
}

export function renderWidget(widget: WidgetSpec): JSX.Element | null {
  switch (widget.type) {
    case "line_chart":
      return (
        <QueryGate widget={widget}>
          {(data) => <LineChartWidget widget={widget} data={data} />}
        </QueryGate>
      );
    case "stat_card":
      return (
        <QueryGate widget={widget}>
          {(data) => <StatCardWidget widget={widget} data={data} />}
        </QueryGate>
      );
    case "gauge":
      return (
        <QueryGate widget={widget}>
          {(data) => <GaugeWidget widget={widget} data={data} />}
        </QueryGate>
      );
    case "table":
      return (
        <QueryGate widget={widget}>
          {(data) => <TableWidget widget={widget} data={data} />}
        </QueryGate>
      );
    case "alert_list":
      return (
        <QueryGate widget={widget} renderEmptyAsContent>
          {(data) => <AlertListWidget widget={widget} data={data} />}
        </QueryGate>
      );
    case "pie_chart":
      return (
        <QueryGate widget={widget}>
          {(data) => <PieChartWidget widget={widget} data={data} />}
        </QueryGate>
      );
    case "bar_chart":
      return (
        <QueryGate widget={widget}>
          {(data) => <BarChartWidget widget={widget} data={data} />}
        </QueryGate>
      );
    case "heatmap":
      return (
        <QueryGate widget={widget}>
          {(data) => <HeatmapWidget widget={widget} data={data} />}
        </QueryGate>
      );
    case "decision_flow":
      // Not query-driven: renders real alert events from the anomaly store.
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
