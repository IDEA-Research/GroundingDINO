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

import type { WidgetSpec } from "@/lib/spec-schema";
import { useQueryData } from "@/lib/useQueryData";
import { WidgetFrame } from "./WidgetFrame";

// Colour palette — same deterministic set used by PieChartWidget.
const PALETTE = [
  "#38bdf8", // sky-400
  "#f472b6", // pink-400
  "#a78bfa", // violet-400
  "#34d399", // emerald-400
  "#fbbf24", // amber-400
  "#fb923c", // orange-400
  "#60a5fa", // blue-400
  "#e879f9", // fuchsia-400
  "#2dd4bf", // teal-400
  "#f87171", // red-400
];

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

export function BarChartWidget({ widget }: { widget: WidgetSpec }) {
  const data = useQueryData(widget.query);
  const opts = (widget.options ?? {}) as Record<string, unknown>;
  const horizontal = opts.horizontal === true;
  const showGrid = opts.show_grid !== false;
  const showValues = opts.show_values === true;
  const decimals = typeof opts.decimals === "number" ? opts.decimals : 2;

  const rows = buildRows(data.table, decimals);

  const barColor = widget.encoding?.color || "#38bdf8";

  return (
    <WidgetFrame widget={widget} badge={data.source === "mock" ? "mock" : null}>
      <ResponsiveContainer width="100%" height="100%">
        {horizontal ? (
          <BarChart
            data={rows}
            layout="vertical"
            margin={{ top: 8, right: 16, bottom: 0, left: 0 }}
          >
            {showGrid && (
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            )}
            <YAxis
              dataKey="label"
              type="category"
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              width={80}
            />
            <XAxis
              type="number"
              tick={{ fontSize: 10, fill: "#94a3b8" }}
            />
            <Tooltip
              contentStyle={{
                background: "#0f172a",
                border: "1px solid #1e293b",
                color: "#e2e8f0",
              }}
            />
            <Bar
              dataKey="value"
              isAnimationActive={false}
              label={
                showValues
                  ? { position: "right", fill: "#94a3b8", fontSize: 10 }
                  : false
              }
            >
              {rows.map((_, i) => (
                <Cell
                  key={`cell-${i}`}
                  fill={PALETTE[i % PALETTE.length]}
                />
              ))}
            </Bar>
          </BarChart>
        ) : (
          <BarChart
            data={rows}
            margin={{ top: 8, right: 8, bottom: 0, left: 0 }}
          >
            {showGrid && (
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            )}
            <XAxis
              dataKey="label"
              tick={{ fontSize: 10, fill: "#94a3b8" }}
            />
            <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} />
            <Tooltip
              contentStyle={{
                background: "#0f172a",
                border: "1px solid #1e293b",
                color: "#e2e8f0",
              }}
            />
            <Bar
              dataKey="value"
              isAnimationActive={false}
              label={
                showValues
                  ? { position: "top", fill: "#94a3b8", fontSize: 10 }
                  : false
              }
            >
              {rows.map((_, i) => (
                <Cell
                  key={`cell-${i}`}
                  fill={PALETTE[i % PALETTE.length]}
                />
              ))}
            </Bar>
          </BarChart>
        )}
      </ResponsiveContainer>
    </WidgetFrame>
  );
}
