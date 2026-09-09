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

import { seriesColor, useChartTheme } from "@/lib/chart-theme";
import type { WidgetSpec } from "@/lib/spec-schema";
import type { TableRow, WidgetData } from "@/lib/useQueryData";
import { WidgetFrame, frameStatusProps } from "./WidgetFrame";

interface BarRow {
  label: string;
  // null = missing (Recharts leaves the bar out; never drawn as 0).
  value: number | null;
}

function buildRows(table: TableRow[], decimals: number): BarRow[] {
  return table.map((r, i) => {
    const label =
      Object.entries(r.metric).find(([k]) => k !== "job")?.[1] ??
      `bar-${i}`;
    return {
      label,
      value:
        r.value === null || !Number.isFinite(r.value)
          ? null
          : Number(r.value.toFixed(decimals)),
    };
  });
}

export function BarChartWidget({
  widget,
  data,
}: {
  widget: WidgetSpec;
  data: WidgetData;
}) {
  const theme = useChartTheme();
  const opts = (widget.options ?? {}) as Record<string, unknown>;
  const horizontal = opts.horizontal === true;
  const showGrid = opts.show_grid !== false;
  const showValues = opts.show_values === true;
  const decimals = typeof opts.decimals === "number" ? opts.decimals : 2;
  const unit = widget.encoding?.unit || "";

  const rows = buildRows(data.table, decimals);

  // A single-category bar chart may pin one color via the spec; otherwise
  // bars cycle the shared (never-red/amber) data palette.
  const specColor = widget.encoding?.color || null;
  const fillFor = (i: number) => specColor ?? seriesColor(i, theme.colors);

  const valueLabelStyle = { fill: theme.colors.inkDim, fontSize: 12 };
  const tooltipFormatter = (value: unknown) => [
    typeof value === "number" ? `${value}${unit ? ` ${unit}` : ""}` : "—",
    widget.encoding?.legend || widget.title,
  ];

  return (
    <WidgetFrame
      widget={widget}
      unit={unit || null}
      {...frameStatusProps(data)}
    >
      <ResponsiveContainer width="100%" height="100%">
        {horizontal ? (
          <BarChart
            data={rows}
            layout="vertical"
            margin={{ top: 8, right: 16, bottom: 0, left: 0 }}
          >
            {showGrid && (
              <CartesianGrid strokeDasharray="3 3" stroke={theme.gridStroke} />
            )}
            <YAxis
              dataKey="label"
              type="category"
              tick={theme.axisTick}
              axisLine={theme.axisLine}
              tickLine={false}
              width={88}
            />
            <XAxis
              type="number"
              tick={theme.axisTick}
              axisLine={theme.axisLine}
              tickLine={false}
            />
            <Tooltip
              formatter={tooltipFormatter as never}
              contentStyle={theme.tooltipContentStyle}
              labelStyle={theme.tooltipLabelStyle}
              cursor={{ fill: theme.colors.surface3, fillOpacity: 0.5 }}
            />
            <Bar
              dataKey="value"
              name={widget.encoding?.legend || widget.title}
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
              <CartesianGrid strokeDasharray="3 3" stroke={theme.gridStroke} />
            )}
            <XAxis
              dataKey="label"
              tick={theme.axisTick}
              axisLine={theme.axisLine}
              tickLine={false}
            />
            <YAxis
              tick={theme.axisTick}
              axisLine={theme.axisLine}
              tickLine={false}
            />
            <Tooltip
              formatter={tooltipFormatter as never}
              contentStyle={theme.tooltipContentStyle}
              labelStyle={theme.tooltipLabelStyle}
              cursor={{ fill: theme.colors.surface3, fillOpacity: 0.5 }}
            />
            <Bar
              dataKey="value"
              name={widget.encoding?.legend || widget.title}
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
