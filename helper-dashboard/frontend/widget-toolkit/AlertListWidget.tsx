"use client";

import React from "react";

import type { WidgetSpec } from "@/lib/spec-schema";
import type { WidgetData } from "@/lib/useQueryData";
import { WidgetFrame } from "./WidgetFrame";

// Renders REAL alerts from the widget's own query — the natural query is
// ALERTS{alertstate="firing"}, whose instant-vector rows carry
// alertname/severity/instance labels (surfaced by the backend in `table`).
// There is no hardcoded alert list: zero rows is rendered as an explicit,
// calm "no active alerts" state (a GOOD state — the query succeeded and
// nothing is firing); loading/error states are handled by the renderer's
// QueryGate before this component is reached.
export function AlertListWidget({
  widget,
  data,
}: {
  widget: WidgetSpec;
  data: WidgetData;
}) {
  const filter = (widget.options as any)?.severity_filter as
    | string
    | undefined;

  const alerts = data.table
    .map((row) => ({
      name:
        row.metric.alertname ??
        row.metric.__name__ ??
        "unnamed alert",
      severity: row.metric.severity ?? "warning",
      instance:
        row.metric.instance ?? row.metric.patient_id ?? row.metric.job ?? "",
    }))
    .filter((a) => (filter ? a.severity === filter : true));

  return (
    <WidgetFrame widget={widget}>
      {alerts.length === 0 ? (
        <div
          data-state="no-alerts"
          className="flex h-full items-center justify-center text-xs text-slate-400"
        >
          目前沒有作用中的警報 / No active alerts
        </div>
      ) : (
        <ul className="h-full overflow-auto text-xs">
          {alerts.map((a, i) => (
            <li
              key={i}
              className="flex items-center justify-between border-b border-slate-800 px-2 py-1.5"
            >
              <div>
                <span
                  className={
                    "mr-2 inline-block rounded px-1.5 py-0.5 text-[10px] uppercase " +
                    (a.severity === "critical"
                      ? "bg-red-700 text-red-100"
                      : "bg-amber-700 text-amber-100")
                  }
                >
                  {a.severity}
                </span>
                <span className="font-medium text-slate-100">{a.name}</span>
              </div>
              <span className="text-slate-400">{a.instance}</span>
            </li>
          ))}
        </ul>
      )}
    </WidgetFrame>
  );
}
