"use client";

import React from "react";

import {
  FONT_MONO,
  colorForSeverity,
  severityForSpec,
  useChartTheme,
  vitalColorFor,
  type ChartColors,
} from "@/lib/chart-theme";
import { sampleTsOf } from "@/lib/data-status";
import {
  widgetQuery,
  type ThresholdDirection,
  type WidgetSpec,
  type WidgetThreshold,
} from "@/lib/spec-schema";
import {
  resolveThresholdDirection,
  sortedForDirection,
  thresholdCaption,
} from "@/lib/threshold";
import { activeThresholdViaEngine } from "@/lib/threshold-adapter";
import type { WidgetData } from "@/lib/useQueryData";
import {
  NonDiagnosticFooter,
  SeverityIcon,
  WidgetFrame,
  fmtClock,
  formatValue,
  frameStatusProps,
} from "./WidgetFrame";

// Grafana-style horizontal bar gauge. Data interface mirrors GaugeWidget's
// (min/max in options, thresholds the shared WidgetThreshold[] ladder,
// value from the widget's single query) so the two stay interchangeable
// from a spec-authoring point of view. "Which band is active" is always
// decided by activeThresholdViaEngine — this file invents no coloring
// rule of its own, in any of the three modes.
//
// options.display_mode (default "basic"):
//   - "basic": one solid fill color = the CURRENTLY active threshold's
//     color (identical rule to Gauge/StatCard).
//   - "gradient": the fill is a continuous SVG linearGradient anchored at
//     each threshold's VALUE position (own severity color), flat-
//     extrapolated past the first/last one. A from-scratch, dependency-
//     free interpolation — deliberately does NOT import
//     lib/color-scale.ts, which may not be committed on this branch yet
//     (a parallel agent may be adding it). SVG's native <stop>
//     interpolation does the actual color blending.
//   - "retro-lcd": the bar is chopped into LCD_CELL_COUNT fixed cells;
//     each LIT cell is colored by activeThresholdViaEngine evaluated at
//     that cell's own position along [min,max] (so a bar spanning
//     several bands visibly changes color at the band boundary), unlit
//     cells use a dim track color.
//
// Horizontal orientation only, per the task's "Basic first" scope.
export type BarGaugeDisplayMode = "basic" | "gradient" | "retro-lcd";

const LCD_CELL_COUNT = 20;

export function BarGaugeWidget({
  widget,
  data,
}: {
  widget: WidgetSpec;
  data: WidgetData;
}) {
  const theme = useChartTheme();
  const query = widgetQuery(widget);
  const opts = (widget.options ?? {}) as Record<string, unknown>;
  const min = Number(opts.min ?? 0);
  const max = Number(opts.max ?? 100);
  const span = max - min;
  const frac = (v: number): number =>
    Math.min(1, Math.max(0, span === 0 ? 0 : (v - min) / span));

  const raw = data.latest;
  const pct = raw === null ? 0 : frac(raw);

  const thresholds = widget.thresholds ?? [];
  const direction = resolveThresholdDirection(thresholds);
  const active = activeThresholdViaEngine(raw, thresholds, direction);

  const normalColor =
    widget.encoding?.color ||
    vitalColorFor(query.promql, widget.title, theme.colors) ||
    theme.colors.data2;
  const valueColor = active
    ? colorForSeverity(active.severity, theme.colors)
    : normalColor;

  const unit = widget.encoding?.unit || "";
  const decimals =
    typeof opts.decimals === "number" ? (opts.decimals as number) : 1;
  const modeOpt = opts.display_mode;
  const displayMode: BarGaugeDisplayMode =
    modeOpt === "gradient" || modeOpt === "retro-lcd" ? modeOpt : "basic";

  const barLabel =
    raw === null
      ? `${widget.title}:無資料`
      : `${widget.title}:${formatValue(raw, decimals)}${unit ? " " + unit : ""}`;

  let bar: React.ReactNode;
  if (displayMode === "gradient") {
    const gradientId = `bar-gauge-gradient-${widget.id}`;
    const stops = gradientStops(thresholds, direction, frac, normalColor, theme.colors);
    bar = (
      <svg
        viewBox="0 0 100 10"
        preserveAspectRatio="none"
        className="h-6 w-full overflow-hidden rounded-sm"
        role="img"
        aria-label={barLabel}
      >
        <defs>
          <linearGradient
            id={gradientId}
            gradientUnits="userSpaceOnUse"
            x1="0"
            y1="0"
            x2="100"
            y2="0"
          >
            {stops.map((s, i) => (
              <stop key={i} offset={s.offset} stopColor={s.color} />
            ))}
          </linearGradient>
        </defs>
        <rect x={0} y={0} width={100} height={10} fill={theme.colors.surface3} />
        {pct > 0 ? (
          <rect x={0} y={0} width={pct * 100} height={10} fill={`url(#${gradientId})`} />
        ) : null}
      </svg>
    );
  } else if (displayMode === "retro-lcd") {
    const numLit = Math.min(LCD_CELL_COUNT, Math.round(pct * LCD_CELL_COUNT));
    bar = (
      <div className="flex h-6 w-full gap-0.5" role="img" aria-label={barLabel}>
        {Array.from({ length: LCD_CELL_COUNT }, (_, i) => {
          const lit = i < numLit;
          let cellColor = theme.colors.surface3;
          if (lit) {
            const cellValue = min + ((i + 0.5) / LCD_CELL_COUNT) * span;
            const cellActive = activeThresholdViaEngine(cellValue, thresholds, direction);
            cellColor = cellActive
              ? colorForSeverity(cellActive.severity, theme.colors)
              : normalColor;
          }
          return (
            <div
              key={i}
              className="h-full flex-1 rounded-[1px]"
              style={{ backgroundColor: cellColor }}
            />
          );
        })}
      </div>
    );
  } else {
    bar = (
      <div
        className="relative h-6 w-full overflow-hidden rounded-sm"
        style={{ backgroundColor: theme.colors.surface3 }}
        role="img"
        aria-label={barLabel}
      >
        {pct > 0 ? (
          <div
            className="h-full rounded-sm"
            style={{ width: `${pct * 100}%`, backgroundColor: valueColor }}
          />
        ) : null}
      </div>
    );
  }

  const sampleTs = sampleTsOf(data);
  const legend = sortedForDirection(thresholds, direction);

  return (
    <WidgetFrame
      widget={widget}
      {...frameStatusProps(data)}
      footer={thresholds.length > 0 ? <NonDiagnosticFooter /> : null}
    >
      <div className="flex h-full flex-col justify-center gap-1.5">
        <div className="flex items-baseline justify-between gap-2">
          <span
            className="text-2xl font-bold tabular-nums"
            style={{
              color: raw === null ? theme.colors.inkFaint : valueColor,
              fontFamily: FONT_MONO,
            }}
          >
            {raw === null ? "—" : formatValue(raw, decimals)}
          </span>
          {unit ? (
            <span
              className="text-xs text-ink-faint"
              style={{ fontFamily: FONT_MONO }}
            >
              {unit}
            </span>
          ) : null}
        </div>
        {bar}
        <div
          className="flex items-center justify-between text-[11px] text-ink-faint"
          style={{ fontFamily: FONT_MONO }}
        >
          <span>{min}</span>
          <span>{max}</span>
        </div>
        {active ? (
          <div
            className="inline-flex items-center gap-1.5 text-sm font-medium"
            style={{ color: valueColor }}
          >
            <SeverityIcon severity={active.severity} />
            <span>{active.label}</span>
          </div>
        ) : raw === null && data.status !== "loading" ? (
          <div className="text-sm text-ink-dim">{data.reason ?? "尚無資料"}</div>
        ) : null}
        {legend.length > 0 ? (
          <ul className="flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-ink-faint">
            {legend.map((t, i) => (
              <li key={i} className="inline-flex items-center gap-1">
                <span
                  aria-hidden="true"
                  className="inline-block h-2 w-2 rounded-full"
                  style={{
                    backgroundColor: colorForSeverity(
                      severityForSpec(t.color) ?? "info",
                      theme.colors,
                    ),
                  }}
                />
                <span className="font-mono tabular-nums">
                  {thresholdCaption(t, direction, unit)}
                </span>
              </li>
            ))}
          </ul>
        ) : null}
        <div className="text-xs text-ink-faint">
          <span className="font-mono tabular-nums">
            {sampleTs !== null ? `最後量測 ${fmtClock(sampleTs)}` : ""}
          </span>
        </div>
      </div>
    </WidgetFrame>
  );
}

// Continuous colour stops for "gradient" mode: one stop per threshold at
// its own VALUE position (its own severity colour), plus a flat
// extrapolation at the far end of the range using the widget's normal
// (non-alarmed) colour on the SAFE side and the nearest threshold's
// colour on the ALARMED side. `frac` maps a raw value to 0..1 the same
// way the bar fill itself does, so a stop always lines up with the pixel
// where its threshold actually fires.
function gradientStops(
  thresholds: WidgetThreshold[],
  direction: ThresholdDirection,
  frac: (v: number) => number,
  normalColor: string,
  colors: ChartColors,
): { offset: string; color: string }[] {
  const valid = thresholds.filter((t) => Number.isFinite(t.value));
  if (valid.length === 0) {
    return [
      { offset: "0%", color: normalColor },
      { offset: "100%", color: normalColor },
    ];
  }
  const byValue = [...valid].sort((a, b) => a.value - b.value);
  const colorOf = (t: WidgetThreshold) =>
    colorForSeverity(severityForSpec(t.color) ?? "info", colors);
  const pctOf = (v: number) => `${(frac(v) * 100).toFixed(2)}%`;

  const stops: { offset: string; color: string }[] = [];
  stops.push({
    offset: "0%",
    color: direction === "below" ? colorOf(byValue[0]) : normalColor,
  });
  for (const t of byValue) {
    stops.push({ offset: pctOf(t.value), color: colorOf(t) });
  }
  stops.push({
    offset: "100%",
    color:
      direction === "below" ? normalColor : colorOf(byValue[byValue.length - 1]),
  });
  return stops;
}
