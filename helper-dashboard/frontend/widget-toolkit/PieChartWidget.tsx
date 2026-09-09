"use client";

import React from "react";

import { seriesColor, useChartTheme, type ChartColors } from "@/lib/chart-theme";
import type { WidgetSpec } from "@/lib/spec-schema";
import type { TableRow, WidgetData } from "@/lib/useQueryData";
import { WidgetFrame, frameStatusProps } from "./WidgetFrame";

interface Slice {
  label: string;
  value: number;
  pct: number;
  startAngle: number;
  endAngle: number;
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

  let cursor = 0;
  return present.map((r, i) => {
    const pct = Math.abs(r.value) / total;
    const startAngle = cursor;
    const endAngle = cursor + pct * 360;
    cursor = endAngle;
    // Use the first non-"job" metric label as the slice label.
    const label =
      Object.entries(r.metric).find(([k]) => k !== "job")?.[1] ??
      `slice-${i}`;
    return {
      label,
      value: r.value,
      pct,
      startAngle,
      endAngle,
      color: seriesColor(i, colors),
    };
  });
}

function polarToCartesian(cx: number, cy: number, r: number, deg: number) {
  const rad = ((deg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function arcPath(
  cx: number,
  cy: number,
  r: number,
  startDeg: number,
  endDeg: number,
): string {
  // Full circle edge case
  if (endDeg - startDeg >= 359.999) {
    const half = (startDeg + endDeg) / 2;
    return arcPath(cx, cy, r, startDeg, half) + " " + arcPath(cx, cy, r, half, endDeg);
  }
  const start = polarToCartesian(cx, cy, r, startDeg);
  const end = polarToCartesian(cx, cy, r, endDeg);
  const largeArc = endDeg - startDeg > 180 ? 1 : 0;
  return `M ${cx} ${cy} L ${start.x.toFixed(2)} ${start.y.toFixed(2)} A ${r} ${r} 0 ${largeArc} 1 ${end.x.toFixed(2)} ${end.y.toFixed(2)} Z`;
}

function donutArcPath(
  cx: number,
  cy: number,
  outerR: number,
  innerR: number,
  startDeg: number,
  endDeg: number,
): string {
  if (endDeg - startDeg >= 359.999) {
    const half = (startDeg + endDeg) / 2;
    return (
      donutArcPath(cx, cy, outerR, innerR, startDeg, half) +
      " " +
      donutArcPath(cx, cy, outerR, innerR, half, endDeg)
    );
  }
  const oStart = polarToCartesian(cx, cy, outerR, startDeg);
  const oEnd = polarToCartesian(cx, cy, outerR, endDeg);
  const iStart = polarToCartesian(cx, cy, innerR, startDeg);
  const iEnd = polarToCartesian(cx, cy, innerR, endDeg);
  const largeArc = endDeg - startDeg > 180 ? 1 : 0;
  return [
    `M ${oStart.x.toFixed(2)} ${oStart.y.toFixed(2)}`,
    `A ${outerR} ${outerR} 0 ${largeArc} 1 ${oEnd.x.toFixed(2)} ${oEnd.y.toFixed(2)}`,
    `L ${iEnd.x.toFixed(2)} ${iEnd.y.toFixed(2)}`,
    `A ${innerR} ${innerR} 0 ${largeArc} 0 ${iStart.x.toFixed(2)} ${iStart.y.toFixed(2)}`,
    "Z",
  ].join(" ");
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

  const cx = 100;
  const cy = 100;
  const outerR = 80;
  const innerR = 48;

  return (
    <WidgetFrame
      widget={widget}
      unit={widget.encoding?.unit}
      {...frameStatusProps(data)}
    >
      <div className="flex h-full items-center justify-center gap-4">
        <svg
          viewBox="0 0 200 200"
          className="h-full max-h-48 w-auto flex-shrink-0"
          role="img"
          aria-label={widget.title}
        >
          {slices.length === 0 ? (
            <text
              x={cx}
              y={cy}
              textAnchor="middle"
              fill={theme.colors.inkFaint}
              fontSize={14}
            >
              尚無資料
            </text>
          ) : (
            slices.map((s, i) => (
              <path
                key={i}
                d={
                  isDonut
                    ? donutArcPath(cx, cy, outerR, innerR, s.startAngle, s.endAngle)
                    : arcPath(cx, cy, outerR, s.startAngle, s.endAngle)
                }
                fill={s.color}
                stroke={theme.colors.surface}
                strokeWidth={1}
              >
                <title>
                  {s.label}: {s.value.toFixed(2)} ({(s.pct * 100).toFixed(1)}%)
                </title>
              </path>
            ))
          )}
          {showLabels &&
            slices
              .filter((s) => s.pct > 0.05)
              .map((s, i) => {
                const mid = (s.startAngle + s.endAngle) / 2;
                const labelR = isDonut ? (outerR + innerR) / 2 : outerR * 0.6;
                const pos = polarToCartesian(cx, cy, labelR, mid);
                return (
                  <text
                    key={`lbl-${i}`}
                    x={pos.x}
                    y={pos.y}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fill={theme.name === "dark" ? theme.colors.canvas : theme.colors.surface}
                    fontSize={12}
                    fontWeight={600}
                  >
                    {(s.pct * 100).toFixed(0)}%
                  </text>
                );
              })}
        </svg>
        {showLegend && slices.length > 0 && (
          <ul className="flex flex-col gap-1 overflow-auto text-sm">
            {slices.map((s, i) => (
              <li key={i} className="flex items-center gap-1.5">
                <span
                  aria-hidden="true"
                  className="inline-block h-2.5 w-2.5 flex-shrink-0 rounded-sm"
                  style={{ backgroundColor: s.color }}
                />
                <span className="max-w-[140px] truncate text-ink-dim" title={s.label}>
                  {s.label}
                </span>
                <span className="ml-auto font-mono text-xs tabular-nums text-ink-faint">
                  {(s.pct * 100).toFixed(0)}%
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </WidgetFrame>
  );
}
