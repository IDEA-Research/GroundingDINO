"use client";

import React, { useState } from "react";
import {
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Sector,
  Tooltip,
} from "recharts";

import { seriesColor, useChartTheme, type ChartColors } from "@/lib/chart-theme";
import type { WidgetSpec } from "@/lib/spec-schema";
import type { TableRow, WidgetData } from "@/lib/useQueryData";
import { WidgetFrame, frameStatusProps } from "./WidgetFrame";

interface Slice {
  label: string;
  // Signed raw value — shown in the tooltip exactly as the backend gave it.
  value: number;
  // abs(value) — this is what actually drives arc size (Recharts' dataKey),
  // matching the old hand-rolled renderer where angles were built from
  // Math.abs(r.value), never the signed value.
  magnitude: number;
  pct: number;
  color: string;
}

function buildSlices(rows: TableRow[], colors: ChartColors): Slice[] {
  // Missing values contribute nothing — they are never treated as 0.
  const present = rows.filter(
    (r): r is TableRow & { value: number } =>
      r.value !== null && Number.isFinite(r.value),
  );
  const total = present.reduce((s, r) => s + Math.abs(r.value), 0);
  if (total === 0) return [];

  return present.map((r, i) => {
    const magnitude = Math.abs(r.value);
    // Use the first non-"job" metric label as the slice label.
    const label =
      Object.entries(r.metric).find(([k]) => k !== "job")?.[1] ??
      `slice-${i}`;
    return {
      label,
      value: r.value,
      magnitude,
      pct: magnitude / total,
      color: seriesColor(i, colors),
    };
  });
}

// Percentage label centered inside each slice (donut: mid-ring; pie: 60%
// of the radius) — same 5% visibility threshold as the previous hand-rolled
// arc renderer, same "readable on top of a saturated fill" color pick.
function renderPercentLabel(
  isDonut: boolean,
  textColor: string,
) {
  return (props: {
    cx: number;
    cy: number;
    midAngle: number;
    innerRadius: number;
    outerRadius: number;
    percent: number;
  }) => {
    const { cx, cy, midAngle, innerRadius, outerRadius, percent } = props;
    if (!percent || percent <= 0.05) return null;
    const RADIAN = Math.PI / 180;
    const r = isDonut
      ? (innerRadius + outerRadius) / 2
      : outerRadius * 0.6;
    const x = cx + r * Math.cos(-midAngle * RADIAN);
    const y = cy + r * Math.sin(-midAngle * RADIAN);
    return (
      <text
        x={x}
        y={y}
        textAnchor="middle"
        dominantBaseline="central"
        fill={textColor}
        fontSize={12}
        fontWeight={600}
      >
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    );
  };
}

// Recharts' classic "active shape" hover recipe: the hovered slice grows a
// few px past its resting outerRadius so the highlight reads as an actual
// shape change, not just a tooltip appearing.
function renderActiveShape(props: {
  cx: number;
  cy: number;
  innerRadius: number;
  outerRadius: number;
  startAngle: number;
  endAngle: number;
  fill: string;
}) {
  const { cx, cy, innerRadius, outerRadius, startAngle, endAngle, fill } =
    props;
  return (
    <Sector
      cx={cx}
      cy={cy}
      innerRadius={innerRadius}
      outerRadius={outerRadius + 6}
      startAngle={startAngle}
      endAngle={endAngle}
      fill={fill}
    />
  );
}

export function PieChartWidget({
  widget,
  data,
}: {
  widget: WidgetSpec;
  data: WidgetData;
}) {
  const theme = useChartTheme();
  const opts = (widget.options ?? {}) as Record<string, unknown>;
  const showLabels = opts.show_labels !== false;
  const showLegend = opts.show_legend !== false;
  const isDonut = opts.donut === true;

  const slices = buildSlices(data.table, theme.colors);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);

  const labelColor =
    theme.name === "dark" ? theme.colors.canvas : theme.colors.surface;

  const tooltipFormatter = (value: unknown, _name: unknown, entry: unknown) => {
    const payload = (entry as { payload?: Slice })?.payload;
    const raw = typeof value === "number" ? value : payload?.value;
    const pct = payload?.pct ?? 0;
    return [
      `${typeof raw === "number" ? raw.toFixed(2) : "—"} (${(pct * 100).toFixed(1)}%)`,
      payload?.label ?? widget.title,
    ];
  };

  const legendFormatter = (value: unknown, entry: unknown) => {
    const payload = (entry as { payload?: Slice })?.payload;
    const pct = payload?.pct ?? 0;
    return `${value} ${(pct * 100).toFixed(0)}%`;
  };

  return (
    <WidgetFrame
      widget={widget}
      unit={widget.encoding?.unit}
      {...frameStatusProps(data)}
    >
      {slices.length === 0 ? (
        <div className="flex h-full items-center justify-center text-sm text-ink-faint">
          尚無資料
        </div>
      ) : (
        <div
          role="img"
          aria-label={widget.title}
          className="h-full w-full"
        >
          <ResponsiveContainer width="100%" height="100%">
            <PieChart margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
              <Tooltip
                formatter={tooltipFormatter as never}
                contentStyle={theme.tooltipContentStyle}
                labelStyle={theme.tooltipLabelStyle}
              />
              {showLegend && (
                <Legend
                  layout="vertical"
                  verticalAlign="middle"
                  align="right"
                  iconType="circle"
                  iconSize={8}
                  formatter={legendFormatter as never}
                  wrapperStyle={{
                    ...theme.legendStyle,
                    maxHeight: "100%",
                    overflowY: "auto",
                  }}
                />
              )}
              <Pie
                data={slices}
                dataKey="magnitude"
                nameKey="label"
                cx="50%"
                cy="50%"
                innerRadius={isDonut ? "60%" : 0}
                outerRadius="80%"
                startAngle={90}
                endAngle={-270}
                isAnimationActive={false}
                label={showLabels ? renderPercentLabel(isDonut, labelColor) : false}
                labelLine={false}
                activeIndex={activeIndex ?? undefined}
                activeShape={renderActiveShape as never}
                onMouseEnter={(_, index) => setActiveIndex(index)}
                onMouseLeave={() => setActiveIndex(null)}
              >
                {slices.map((s, i) => (
                  <Cell
                    key={`cell-${i}`}
                    fill={s.color}
                    stroke={theme.colors.surface}
                    strokeWidth={1}
                  />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        </div>
      )}
    </WidgetFrame>
  );
}
