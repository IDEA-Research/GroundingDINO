"use client";

import React from "react";

import type { WidgetSpec } from "@/lib/spec-schema";
import { useQueryData } from "@/lib/useQueryData";
import { WidgetFrame } from "./WidgetFrame";

export function StatCardWidget({ widget }: { widget: WidgetSpec }) {
  const data = useQueryData(widget.query);
  const unit = widget.encoding?.unit || "";
  const value = data.latest ?? 0;

  const color = pickThresholdColor(value, widget.thresholds);

  return (
    <WidgetFrame widget={widget} badge={data.source === "mock" ? "mock" : null}>
      <div className="flex h-full items-center justify-center">
        <div className="text-center">
          <div
            className="text-4xl font-bold tabular-nums"
            style={{ color }}
          >
            {formatNumber(value)}
            {unit ? (
              <span className="ml-1 text-base text-slate-400">{unit}</span>
            ) : null}
          </div>
        </div>
      </div>
    </WidgetFrame>
  );
}

function formatNumber(v: number): string {
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

function pickThresholdColor(
  value: number,
  thresholds?: WidgetSpec["thresholds"],
): string {
  if (!thresholds || thresholds.length === 0) return "#e2e8f0";
  const sorted = [...thresholds].sort((a, b) => a.value - b.value);
  let color = "#e2e8f0";
  for (const t of sorted) {
    if (value >= t.value) color = t.color;
  }
  return color;
}
