"use client";

import React from "react";

import {
  CHART_COLORS,
  CHART_PALETTE,
  FONT_MONO,
  alarmColorForSpec,
  vitalColorFor,
} from "@/lib/chart-theme";
import type { WidgetSpec } from "@/lib/spec-schema";
import type { WidgetData } from "@/lib/useQueryData";
import { WidgetFrame } from "./WidgetFrame";

// Half-circle gauge. Spec thresholds render as colored arc bands on the
// outer rim (colors normalized onto the IEC alarm tokens); the value arc
// and numeral take the active band's color when a threshold is breached,
// otherwise the vital/spec color. min/max/threshold VALUES stay entirely
// spec-driven — no clinical limits are hardcoded here.
export function GaugeWidget({
  widget,
  data,
}: {
  widget: WidgetSpec;
  data: WidgetData;
}) {
  const raw = data.latest ?? 0;
  const min = Number((widget.options as any)?.min ?? 0);
  const max = Number((widget.options as any)?.max ?? 100);
  const span = max - min;
  const frac = (v: number): number =>
    Math.min(1, Math.max(0, span === 0 ? 0 : (v - min) / span));
  const pct = frac(raw);
  const angle = -90 + 180 * pct; // half-circle gauge

  const thresholds = [...(widget.thresholds ?? [])].sort(
    (a, b) => a.value - b.value,
  );

  // Outer-rim bands: below the first threshold is neutral; each threshold
  // opens a band in its normalized alarm color up to the next threshold.
  const bands: { from: number; to: number; color: string }[] = [];
  let cursor = 0;
  let cursorColor: string = CHART_COLORS.edgeStrong;
  for (const t of thresholds) {
    const f = frac(t.value);
    if (f > cursor) bands.push({ from: cursor, to: f, color: cursorColor });
    cursor = Math.max(cursor, f);
    cursorColor = alarmColorForSpec(t.color) ?? CHART_COLORS.alarmInfo;
  }
  if (cursor < 1) bands.push({ from: cursor, to: 1, color: cursorColor });

  // Active band color for the current value (last threshold <= value).
  let active: string | null = null;
  for (const t of thresholds) {
    if (raw >= t.value) active = alarmColorForSpec(t.color);
  }

  const normalColor =
    widget.encoding?.color ||
    vitalColorFor(widget.query.promql, widget.title) ||
    CHART_PALETTE[1];
  const valueColor = active ?? normalColor;

  return (
    <WidgetFrame widget={widget}>
      <div className="flex h-full items-center justify-center">
        <svg viewBox="-110 -104 220 152" className="h-full max-h-40 w-full">
          {/* threshold bands, outer rim */}
          {bands.map((b, i) => (
            <path
              key={`band-${i}`}
              d={arcSeg(98, b.from, b.to)}
              stroke={b.color}
              strokeWidth={5}
              fill="none"
            />
          ))}
          {/* track */}
          <path
            d="M -90 0 A 90 90 0 0 1 90 0"
            stroke={CHART_COLORS.surface3}
            strokeWidth={14}
            fill="none"
          />
          {/* value arc */}
          <path
            d={arcSeg(90, 0, pct)}
            stroke={valueColor}
            strokeWidth={14}
            fill="none"
            strokeLinecap="round"
          />
          <g transform={`rotate(${angle})`}>
            <line
              x1={0}
              y1={0}
              x2={0}
              y2={-72}
              stroke={CHART_COLORS.ink}
              strokeWidth={2}
            />
            <circle r={5} fill={CHART_COLORS.ink} />
          </g>
          <text
            x={0}
            y={22}
            textAnchor="middle"
            fill={valueColor}
            fontSize={22}
            fontWeight={700}
            fontFamily={FONT_MONO}
            style={{ fontVariantNumeric: "tabular-nums" }}
          >
            {raw.toFixed(1)}
          </text>
          {widget.encoding?.unit ? (
            <text
              x={0}
              y={40}
              textAnchor="middle"
              fill={CHART_COLORS.inkDim}
              fontSize={12}
              fontFamily={FONT_MONO}
            >
              {widget.encoding.unit}
            </text>
          ) : null}
        </svg>
      </div>
    </WidgetFrame>
  );
}

// Arc segment on the half circle, from/to as 0..1 fractions of the sweep.
function arcSeg(r: number, from: number, to: number): string {
  const a0 = -Math.PI + Math.PI * Math.min(from, to);
  const a1 = -Math.PI + Math.PI * Math.max(from, to);
  const x0 = r * Math.cos(a0);
  const y0 = r * Math.sin(a0);
  const x1 = r * Math.cos(a1);
  const y1 = r * Math.sin(a1);
  const large = Math.abs(to - from) > 0.5 ? 1 : 0;
  return `M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
}
