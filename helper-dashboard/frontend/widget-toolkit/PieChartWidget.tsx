"use client";

import React from "react";

import type { WidgetSpec } from "@/lib/spec-schema";
import { useQueryData } from "@/lib/useQueryData";
import { WidgetFrame } from "./WidgetFrame";

// Colour palette for slices — deterministic, no external deps.
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

interface Slice {
  label: string;
  value: number;
  pct: number;
  startAngle: number;
  endAngle: number;
  color: string;
}

function buildSlices(
  rows: { metric: Record<string, string>; value: number }[],
): Slice[] {
  const total = rows.reduce((s, r) => s + Math.abs(r.value), 0);
  if (total === 0) return [];

  let cursor = 0;
  return rows.map((r, i) => {
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
      color: PALETTE[i % PALETTE.length],
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

export function PieChartWidget({ widget }: { widget: WidgetSpec }) {
  const data = useQueryData(widget.query);
  const opts = (widget.options ?? {}) as Record<string, unknown>;
  const showLabels = opts.show_labels !== false;
  const showLegend = opts.show_legend !== false;
  const isDonut = opts.donut === true;

  const slices = buildSlices(data.table);

  const cx = 100;
  const cy = 100;
  const outerR = 80;
  const innerR = 48;

  return (
    <WidgetFrame widget={widget} badge={data.source === "mock" ? "mock" : null}>
      <div className="flex h-full items-center justify-center gap-4">
        <svg viewBox="0 0 200 200" className="h-full max-h-48 w-auto flex-shrink-0">
          {slices.length === 0 ? (
            <text
              x={cx}
              y={cy}
              textAnchor="middle"
              fill="#64748b"
              fontSize={14}
            >
              no data
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
                stroke="#0f172a"
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
                    fill="#e2e8f0"
                    fontSize={10}
                    fontWeight={600}
                  >
                    {(s.pct * 100).toFixed(0)}%
                  </text>
                );
              })}
        </svg>
        {showLegend && slices.length > 0 && (
          <div className="flex flex-col gap-1 overflow-auto text-xs">
            {slices.map((s, i) => (
              <div key={i} className="flex items-center gap-1.5">
                <span
                  className="inline-block h-2.5 w-2.5 flex-shrink-0 rounded-sm"
                  style={{ backgroundColor: s.color }}
                />
                <span className="text-slate-300 truncate max-w-[120px]">
                  {s.label}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </WidgetFrame>
  );
}
