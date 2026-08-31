"use client";

import React from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  CHART_PALETTE,
  axisTick,
  gridStroke,
  tooltipContentStyle,
  tooltipLabelStyle,
  vitalColorFor,
} from "@/lib/chart-theme";
import type { WidgetSpec } from "@/lib/spec-schema";
import type { WidgetData } from "@/lib/useQueryData";
import { WidgetFrame } from "./WidgetFrame";

// X values stay numeric (epoch ms), never pre-formatted strings: every
// chart pins the IDENTICAL time window from the response's end and
// range_seconds (contract C5), so two charts with different data
// coverage still paint the same window at the same pixel width. All
// line charts share syncId="dash" for a common crosshair/tooltip
// (syncMethod="value" keys on the x value, correct for series of
// different lengths).

function formatTick(ms: number, longRange: boolean): string {
  const d = new Date(ms);
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  if (longRange) {
    const mo = String(d.getMonth() + 1).padStart(2, "0");
    const da = String(d.getDate()).padStart(2, "0");
    return `${mo}/${da} ${hh}:${mm}`;
  }
  return `${hh}:${mm}`;
}

export function LineChartWidget({
  widget,
  data,
}: {
  widget: WidgetSpec;
  data: WidgetData;
}) {
  const rows = data.points.map((p) => ({
    t: p.t * 1000,
    v: Number(p.v.toFixed(2)),
  }));

  // Pin the x-domain from the window the backend ACTUALLY used. Until
  // the response carries end/range_seconds (older backend), fall back
  // to the data extent so the chart still renders.
  const hasWindow =
    data.end !== null && data.rangeSeconds !== null && data.rangeSeconds > 0;
  const longRange = (data.rangeSeconds ?? 0) >= 86400;
  const domain: [number | string, number | string] = hasWindow
    ? [
        ((data.end as number) - (data.rangeSeconds as number)) * 1000,
        (data.end as number) * 1000,
      ]
    : ["dataMin", "dataMax"];

  // Spec override wins; otherwise the vital's conventional monitor color
  // (HR green / SpO2 cyan / rSO2 violet); otherwise the palette lead.
  const stroke =
    widget.encoding?.color ||
    vitalColorFor(widget.query.promql, widget.title) ||
    CHART_PALETTE[0];

  return (
    <WidgetFrame widget={widget} unit={widget.encoding?.unit}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={rows}
          syncId="dash"
          syncMethod="value"
          margin={{ top: 8, right: 8, bottom: 0, left: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
          <XAxis
            dataKey="t"
            type="number"
            domain={domain}
            allowDataOverflow
            tickFormatter={(v) => formatTick(Number(v), longRange)}
            tick={axisTick}
          />
          <YAxis tick={axisTick} />
          <Tooltip
            labelFormatter={(v) => formatTick(Number(v), longRange)}
            contentStyle={tooltipContentStyle}
            labelStyle={tooltipLabelStyle}
          />
          <Line
            type="monotone"
            dataKey="v"
            stroke={stroke}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </WidgetFrame>
  );
}
