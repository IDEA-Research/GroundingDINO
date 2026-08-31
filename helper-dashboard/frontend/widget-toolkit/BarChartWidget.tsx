"use client";

import React from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  CHART_COLORS,
  axisTick,
  gridStroke,
  seriesColor,
  tooltipContentStyle,
  tooltipLabelStyle,
} from "@/lib/chart-theme";
import type { WidgetSpec } from "@/lib/spec-schema";
import type { WidgetData } from "@/lib/useQueryData";
import { WidgetFrame } from "./WidgetFrame";

interface BarRow {
  label: string;
  value: number;
}

function buildRows(
  table: { metric: Record<string, string>; value: number }[],
  decimals: number,
): BarRow[] {
  return table.map((r, i) => {
    const label =
      Object.entries(r.metric).find(([k]) => k !== "job")?.[1] ??
      `bar-${i}`;
    return {
      label,
      value: Number(r.value.toFixed(decimals)),
    };
  });
}

const valueLabelStyle = {
  fill: CHART_COLORS.inkDim,
  fontSize: 12,
};

export function BarChartWidget({
  widget,
  data,
}: {
  widget: WidgetSpec;
  data: WidgetData;
}) {
  const opts = (widget.options ?? {}) as Record<string, unknown>;
  const horizontal = opts.horizontal === true;
  const showGrid = opts.show_grid !== false;
  const showValues = opts.show_values === true;
  const decimals = typeof opts.decimals === "number" ? opts.decimals : 2;

  const rows = buildRows(data.table, decimals);

  // A single-category bar chart may pin one color via the spec; otherwise
  // bars cycle the shared (never-red/yellow) palette.
  const specColor = widget.encoding?.color || null;
  const fillFor = (i: number) => specColor ?? seriesColor(i);

  return (
    <WidgetFrame widget={widget} unit={widget.encoding?.unit}>
      <ResponsiveContainer width="100%" height="100%">
        {horizontal ? (
          <BarChart
            data={rows}
            layout="vertical"
            margin={{ top: 8, right: 16, bottom: 0, left: 0 }}
          >
            {showGrid && (
              <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
            )}
            <YAxis
              dataKey="label"
              type="category"
              tick={axisTick}
              width={80}
            />
            <XAxis type="number" tick={axisTick} />
            <Tooltip
              contentStyle={tooltipContentStyle}
              labelStyle={tooltipLabelStyle}
            />
            <Bar
              dataKey="value"
              isAnimationActive={false}
              label={
                showValues
                  ? { position: "right", ...valueLabelStyle }
                  : false
              }
            >
              {rows.map((_, i) => (
                <Cell key={`cell-${i}`} fill={fillFor(i)} />
              ))}
            </Bar>
          </BarChart>
        ) : (
          <BarChart
            data={rows}
            margin={{ top: 8, right: 8, bottom: 0, left: 0 }}
          >
            {showGrid && (
              <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
            )}
            <XAxis dataKey="label" tick={axisTick} />
            <YAxis tick={axisTick} />
            <Tooltip
              contentStyle={tooltipContentStyle}
              labelStyle={tooltipLabelStyle}
            />
            <Bar
              dataKey="value"
              isAnimationActive={false}
              label={
                showValues
                  ? { position: "top", ...valueLabelStyle }
                  : false
              }
            >
              {rows.map((_, i) => (
                <Cell key={`cell-${i}`} fill={fillFor(i)} />
              ))}
            </Bar>
          </BarChart>
        )}
      </ResponsiveContainer>
    </WidgetFrame>
  );
}
