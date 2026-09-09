"use client";

import React from "react";

import type { WidgetSpec } from "@/lib/spec-schema";
import type { WidgetData } from "@/lib/useQueryData";
import { NonDiagnosticFooter, WidgetFrame, frameStatusProps } from "./WidgetFrame";

// Renders REAL alerts from the widget's own query — the natural query is
// ALERTS{alertstate="firing"}, whose instant-vector rows carry
// alertname/severity/instance labels (surfaced by the backend in `table`).
// There is no hardcoded alert list: zero rows is rendered as an explicit,
// calm "no active alerts" state (a GOOD state — the query succeeded and
// nothing is firing); loading/error states are handled by the renderer's
// QueryGate before this component is reached.
//
// Status grammar: high priority = danger (red), medium = warn (amber),
// informational = info (blue-gray), always paired with a TEXT badge;
// entries sort priority-first. Per LD-7 every alerting surface carries the
// non-suppressible decision-support footer.

type Priority = 0 | 1 | 2; // 0 = high, 1 = medium, 2 = info

function priorityOf(severity: string): Priority {
  const s = severity.toLowerCase();
  if (/critical|emergency|fatal|page|high/.test(s)) return 0;
  if (/warning|warn|medium|error|major/.test(s)) return 1;
  return 2;
}

const PRIORITY_STYLE: Record<
  Priority,
  { border: string; badge: string; text: string }
> = {
  0: {
    border: "border-l-danger",
    badge: "bg-danger/10 text-danger border-danger/30",
    text: "高",
  },
  1: {
    border: "border-l-warn",
    badge: "bg-warn/10 text-warn border-warn/30",
    text: "中",
  },
  2: {
    border: "border-l-info",
    badge: "bg-info/10 text-info border-info/30",
    text: "資訊",
  },
};

export function AlertListWidget({
  widget,
  data,
}: {
  widget: WidgetSpec;
  data: WidgetData;
}) {
  const filter = (widget.options as { severity_filter?: unknown } | undefined)
    ?.severity_filter as string | undefined;

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
    <WidgetFrame
      widget={widget}
      {...frameStatusProps(data)}
      footer={<NonDiagnosticFooter />}
    >
      <div className="flex h-full flex-col">
        {alerts.length === 0 ? (
          <div
            data-state="no-alerts"
            className="flex flex-1 items-center justify-center gap-2 text-sm text-ink-dim"
          >
            <span
              aria-hidden="true"
              className="inline-block h-2 w-2 rounded-full bg-ok"
            />
            目前沒有作用中的警報
          </div>
        ) : (
          <ul className="min-h-0 flex-1 overflow-auto text-sm">
            {alerts.map((a, i) => {
              const st = PRIORITY_STYLE[priorityOf(a.severity)];
              return (
                <li
                  key={i}
                  className={
                    "flex items-center justify-between gap-2 border-b border-l-2 border-b-edge px-2 py-2 " +
                    st.border
                  }
                >
                  <div className="flex min-w-0 items-center gap-2">
                    <span
                      className={
                        "inline-flex shrink-0 items-center gap-1 rounded border px-1.5 py-0.5 text-xs font-medium " +
                        st.badge
                      }
                    >
                      {st.text}
                      <span className="font-mono text-2xs uppercase">{a.severity}</span>
                    </span>
                    <span className="truncate font-medium text-ink">{a.name}</span>
                  </div>
                  {a.instance ? (
                    <span className="truncate font-mono text-xs text-ink-faint">
                      {a.instance}
                    </span>
                  ) : null}
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </WidgetFrame>
  );
}
