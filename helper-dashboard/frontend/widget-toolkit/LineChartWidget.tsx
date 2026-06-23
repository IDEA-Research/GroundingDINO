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

import type { WidgetSpec } from "@/lib/spec-schema";
import { useQueryData } from "@/lib/useQueryData";
import { WidgetFrame } from "./WidgetFrame";

export function LineChartWidget({ widget }: { widget: WidgetSpec }) {
  const data = useQueryData(widget.query);
  const rows = data.points.map((p) => ({
    t: new Date(p.t * 1000).toLocaleTimeString(),
    v: Number(p.v.toFixed(2)),
  }));

  return (
    <WidgetFrame widget={widget} badge={data.source === "mock" ? "mock" : null}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey="t" tick={{ fontSize: 10, fill: "#94a3b8" }} />
          <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} />
          <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", color: "#e2e8f0" }} />
          <Line
            type="monotone"
            dataKey="v"
            stroke={widget.encoding?.color || "#38bdf8"}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </WidgetFrame>
  );
}
