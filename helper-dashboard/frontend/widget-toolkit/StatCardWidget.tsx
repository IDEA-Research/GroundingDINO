"use client";

import React from "react";

import {
  CHART_COLORS,
  alarmColorForSpec,
  vitalColorFor,
} from "@/lib/chart-theme";
import type { WidgetSpec } from "@/lib/spec-schema";
import type { WidgetData } from "@/lib/useQueryData";
import { WidgetFrame } from "./WidgetFrame";

// Bedside-monitor stat card: one dominant numeral, readable at 3 meters,
// in the vital's conventional color. A threshold breach recolors the
// NUMERAL with the normalized IEC alarm color — never the card background.
export function StatCardWidget({
  widget,
  data,
}: {
  widget: WidgetSpec;
  data: WidgetData;
}) {
  const unit = widget.encoding?.unit || "";
  const value = data.latest ?? 0;

  const alarm = activeAlarmColor(value, widget.thresholds);
  const color =
    alarm ??
    widget.encoding?.color ??
    vitalColorFor(widget.query.promql, widget.title) ??
    CHART_COLORS.ink;

  return (
    <WidgetFrame widget={widget}>
      <div className="flex h-full items-center justify-center">
        <div className="flex items-baseline gap-2">
          <span
            className="stat-numeral font-mono tabular-nums"
            style={{ color }}
          >
            {formatNumber(value)}
          </span>
          {unit ? (
            <span className="text-sm font-medium text-ink-dim">{unit}</span>
          ) : null}
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

// Same activation rule as before (highest threshold <= value wins), but the
// spec's free-form color is normalized onto the alarm tokens. Returns null
// when no threshold is active — the numeral keeps its normal vital color.
function activeAlarmColor(
  value: number,
  thresholds?: WidgetSpec["thresholds"],
): string | null {
  if (!thresholds || thresholds.length === 0) return null;
  const sorted = [...thresholds].sort((a, b) => a.value - b.value);
  let hit: (typeof sorted)[number] | null = null;
  for (const t of sorted) {
    if (value >= t.value) hit = t;
  }
  if (!hit) return null;
  return alarmColorForSpec(hit.color) ?? CHART_COLORS.alarmInfo;
}
