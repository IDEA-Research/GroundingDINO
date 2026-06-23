"use client";

import React from "react";

import type { WidgetSpec } from "@/lib/spec-schema";
import { WidgetFrame } from "./WidgetFrame";

// Mock alerts for MVP — a real pipeline will hydrate from Prometheus
// `ALERTS` via the backend.
const MOCK_ALERTS = [
  { name: "HighCPU", severity: "warning", instance: "node-1" },
  { name: "LowDiskSpace", severity: "critical", instance: "node-3" },
  { name: "HighLatency", severity: "warning", instance: "api-2" },
];

export function AlertListWidget({ widget }: { widget: WidgetSpec }) {
  const filter = (widget.options as any)?.severity_filter as
    | string
    | undefined;
  const alerts = filter
    ? MOCK_ALERTS.filter((a) => a.severity === filter)
    : MOCK_ALERTS;

  return (
    <WidgetFrame widget={widget} badge="mock">
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
        {alerts.length === 0 ? (
          <li className="px-2 py-4 text-center text-slate-500">no alerts</li>
        ) : null}
      </ul>
    </WidgetFrame>
  );
}
