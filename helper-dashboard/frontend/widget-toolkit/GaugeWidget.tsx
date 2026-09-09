"use client";

import React from "react";

import {
  FONT_MONO,
  colorForSeverity,
  useChartTheme,
  vitalColorFor,
} from "@/lib/chart-theme";
import { sampleTsOf } from "@/lib/data-status";
import { widgetQuery, type WidgetSpec } from "@/lib/spec-schema";
import {
  activeThreshold,
  resolveThresholdDirection,
  sortedForDirection,
  thresholdBands,
  thresholdCaption,
} from "@/lib/threshold";
import type { WidgetData } from "@/lib/useQueryData";
import {
  NonDiagnosticFooter,
  SeverityIcon,
  WidgetFrame,
  fmtClock,
  formatValue,
  frameStatusProps,
} from "./WidgetFrame";

// Half-circle gauge. Spec thresholds render as labelled rim bands honoring
// the effective DIRECTION (above: bands open upward from each threshold;
// below: bands open downward), so a low-is-bad ladder such as SpO2
// [94 amber, 90 red] colors the LOW end. The numeral and value arc take the
// active band's status color and the state is also written as text + icon.
// min/max/threshold VALUES stay entirely spec-driven — nothing clinical is
// hardcoded here.
//
// `compact` = folded beneath a stat card (PairedStatGauge): no header, no
// text tiers, just the arc — the stat card already carries title/state.
export function GaugeWidget({
  widget,
  data,
  compact = false,
}: {
  widget: WidgetSpec;
  data: WidgetData;
  compact?: boolean;
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
  const angle = -90 + 180 * pct; // half-circle gauge

  const thresholds = widget.thresholds ?? [];
  const direction = resolveThresholdDirection(thresholds);
  const bands = thresholdBands(thresholds, direction, frac);
  const active = activeThreshold(raw, thresholds, direction);

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

  const svg = (
    <svg
      viewBox="-110 -104 220 152"
      className={compact ? "h-full w-full" : "h-full max-h-40 w-full"}
      role="img"
      aria-label={
        raw === null
          ? `${widget.title}:無資料`
          : `${widget.title}:${formatValue(raw, decimals)}${unit ? " " + unit : ""}`
      }
    >
      {/* threshold bands, outer rim */}
      {bands.map((b, i) => (
        <path
          key={`band-${i}`}
          d={arcSeg(98, b.from, b.to)}
          stroke={
            b.severity ? colorForSeverity(b.severity, theme.colors) : theme.colors.edgeStrong
          }
          strokeWidth={5}
          fill="none"
        />
      ))}
      {/* track */}
      <path
        d="M -90 0 A 90 90 0 0 1 90 0"
        stroke={theme.colors.surface3}
        strokeWidth={14}
        fill="none"
      />
      {/* value arc */}
      {raw !== null && pct > 0 ? (
        <path
          d={arcSeg(90, 0, pct)}
          stroke={valueColor}
          strokeWidth={14}
          fill="none"
          strokeLinecap="round"
        />
      ) : null}
      {raw !== null ? (
        <g transform={`rotate(${angle})`}>
          <line
            x1={0}
            y1={0}
            x2={0}
            y2={-72}
            stroke={theme.colors.ink}
            strokeWidth={2}
          />
          <circle r={5} fill={theme.colors.ink} />
        </g>
      ) : null}
      <text
        x={0}
        y={compact ? 18 : 22}
        textAnchor="middle"
        fill={raw === null ? theme.colors.inkFaint : valueColor}
        fontSize={compact ? 26 : 30}
        fontWeight={700}
        fontFamily={FONT_MONO}
        style={{ fontVariantNumeric: "tabular-nums" }}
      >
        {raw === null ? "—" : formatValue(raw, decimals)}
      </text>
      {!compact && unit ? (
        <text
          x={0}
          y={42}
          textAnchor="middle"
          fill={theme.colors.inkDim}
          fontSize={12}
          fontFamily={FONT_MONO}
        >
          {unit}
        </text>
      ) : null}
      {!compact ? (
        <>
          <text
            x={-90}
            y={18}
            textAnchor="middle"
            fill={theme.colors.inkFaint}
            fontSize={11}
            fontFamily={FONT_MONO}
          >
            {min}
          </text>
          <text
            x={90}
            y={18}
            textAnchor="middle"
            fill={theme.colors.inkFaint}
            fontSize={11}
            fontFamily={FONT_MONO}
          >
            {max}
          </text>
        </>
      ) : null}
    </svg>
  );

  if (compact) {
    return (
      <WidgetFrame widget={widget} compact {...frameStatusProps(data)}>
        <div className="flex h-full items-center justify-center">{svg}</div>
      </WidgetFrame>
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
      <div className="flex h-full flex-col items-center justify-center gap-1">
        <div className="min-h-0 w-full flex-1">{svg}</div>
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
          <ul className="flex flex-wrap justify-center gap-x-3 gap-y-0.5 text-xs text-ink-faint">
            {legend.map((t, i) => (
              <li key={i} className="inline-flex items-center gap-1">
                <span
                  aria-hidden="true"
                  className="inline-block h-2 w-2 rounded-full"
                  style={{
                    backgroundColor: colorForSeverity(
                      (active && active.threshold === t
                        ? active.severity
                        : bandSeverity(bands, t)) ?? "info",
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

function bandSeverity(
  bands: ReturnType<typeof thresholdBands>,
  t: WidgetSpec["thresholds"] extends (infer T)[] | undefined ? T : never,
) {
  const b = bands.find((x) => x.threshold === t);
  return b?.severity ?? null;
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
