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
//
// Visual grammar per IEC 60601-1-8: high priority = red, medium = amber,
// informational = blue-gray; entries sort priority-first. Per LD-7 every
// alerting surface carries the non-suppressible decision-support footer.

type Priority = 0 | 1 | 2; // 0 = high, 1 = medium, 2 = info

function priorityOf(severity: string): Priority {
  const s = severity.toLowerCase();
  if (/critical|emergency|fatal|page|high/.test(s)) return 0;
  if (/warning|warn|medium|error|major/.test(s)) return 1;
  return 2;
}

const PRIORITY_STYLE: Record<
  Priority,
  { border: string; badge: string }
> = {
  0: {
    border: "border-l-alarm-high",
    badge: "bg-alarm-high/15 text-alarm-high",
  },
  1: {
    border: "border-l-alarm-med",
    badge: "bg-alarm-med/15 text-alarm-med",
  },
  2: {
    border: "border-l-alarm-info",
    badge: "bg-alarm-info/15 text-alarm-info",
  },
};

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
    .map((row, i) => ({
      name:
        row.metric.alertname ??
        row.metric.__name__ ??
        "unnamed alert",
      severity: row.metric.severity ?? "warning",
      instance:
        row.metric.instance ?? row.metric.patient_id ?? row.metric.job ?? "",
      order: i,
    }))
    .filter((a) => (filter ? a.severity === filter : true))
    // Priority first; within a priority keep arrival order (the instant
    // vector carries no timestamps, so source order is the recency proxy).
    .sort((a, b) =>
      priorityOf(a.severity) !== priorityOf(b.severity)
        ? priorityOf(a.severity) - priorityOf(b.severity)
        : a.order - b.order,
    );

  return (
    <WidgetFrame widget={widget}>
      <div className="flex h-full flex-col">
        {alerts.length === 0 ? (
          <div
            data-state="no-alerts"
            className="flex flex-1 items-center justify-center gap-2 text-xs text-ink-faint"
          >
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-ok" />
            目前沒有作用中的警報
          </div>
        ) : (
          <ul className="min-h-0 flex-1 overflow-auto text-xs">
            {alerts.map((a, i) => {
              const st = PRIORITY_STYLE[priorityOf(a.severity)];
              return (
                <li
                  key={i}
                  className={
                    "flex items-center justify-between gap-2 border-b border-l-2 border-b-edge px-2 py-1.5 " +
                    st.border
                  }
                >
                  <div className="min-w-0">
                    <span
                      className={
                        "mr-2 inline-block rounded px-1.5 py-0.5 text-2xs font-semibold uppercase " +
                        st.badge
                      }
                    >
                      {a.severity}
                    </span>
                    <span className="font-medium text-ink">{a.name}</span>
                  </div>
                  <span className="truncate font-mono text-2xs text-ink-faint">
                    {a.instance}
                  </span>
                </li>
              );
            })}
          </ul>
        )}
        {/* LD-7: non-suppressible on every alerting surface. */}
        <div
          data-non-diagnostic="true"
          className="mt-1 shrink-0 border-t border-edge pt-1 text-2xs italic text-ink-faint"
        >
          決策支援,非診斷
        </div>
      </div>
    </WidgetFrame>
  );
}
