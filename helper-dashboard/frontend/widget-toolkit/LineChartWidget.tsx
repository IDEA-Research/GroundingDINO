"use client";

import React, { useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  FONT_MONO,
  colorForSeverity,
  severityForSpec,
  useChartTheme,
  vitalColorFor,
} from "@/lib/chart-theme";
import { widgetQuery, type WidgetSpec } from "@/lib/spec-schema";
import {
  resolveThresholdDirection,
  sortedForDirection,
  thresholdCaption,
} from "@/lib/threshold";
import { useEffectiveRange } from "@/lib/time-range";
import { useQueryFilters, type WidgetData } from "@/lib/useQueryData";
import {
  NonDiagnosticFooter,
  WidgetFrame,
  frameStatusProps,
} from "./WidgetFrame";

// X values stay numeric (epoch ms), never pre-formatted strings: every
// chart pins the IDENTICAL time window from the response's end and
// range_seconds (contract C5), so two charts with different data
// coverage still paint the same window at the same pixel width. All
// line charts share syncId="dash" for a common crosshair/tooltip.
//
// Gaps: the backend emits `v: null` where samples are farther apart than
// 2 x step; connectNulls stays FALSE so a gap is drawn as a gap.
// Thresholds: one dashed ReferenceLine per spec threshold, colored by its
// normalized severity and labelled (label + ≥/≤ value + unit).

function formatTick(ms: number, longRange: boolean): string {
  const d = new Date(ms);
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  if (longRange) {
    const mo = String(d.getMonth() + 1).padStart(2, "0");
    const da = String(d.getDate()).padStart(2, "0");
    return `${mo}/${da} ${hh}:${mm}`;
  }
  return `${hh}:${mm}`;
}

function formatTooltipTime(ms: number, longRange: boolean): string {
  const d = new Date(ms);
  const ss = String(d.getSeconds()).padStart(2, "0");
  return `${formatTick(ms, longRange)}:${ss}`;
}

const canCopy =
  typeof navigator !== "undefined" && !!navigator.clipboard?.writeText;

// Click a point to freeze it (survives the mouse leaving the chart, unlike
// Recharts' own hover-only Tooltip) — rendered via WidgetFrame's headerExtra
// slot so it sits with unit/badge instead of overlaying the plot area.
function PinnedValueBadge({
  t,
  v,
  unit,
  longRange,
  onClear,
}: {
  t: number;
  v: number | null;
  unit: string;
  longRange: boolean;
  onClear: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const text = `${formatTooltipTime(t, longRange)}  ${
    v === null ? "—" : `${v}${unit ? ` ${unit}` : ""}`
  }`;

  return (
    <span className="inline-flex items-center gap-1 rounded-md border border-edge-strong bg-surface-2 py-0.5 pl-2 pr-1 font-mono text-xs tabular-nums text-ink">
      <span className="text-ink-faint">已釘選</span>
      {text}
      {canCopy ? (
        <button
          type="button"
          title="複製數值"
          aria-label="複製釘選數值"
          onClick={async () => {
            try {
              await navigator.clipboard.writeText(text);
              setCopied(true);
              setTimeout(() => setCopied(false), 1200);
            } catch {
              // Clipboard can be unavailable (permissions, non-secure
              // context) — the badge still shows the value either way.
            }
          }}
          className="rounded px-1 text-ink-faint hover:bg-surface-3 hover:text-ink"
        >
          {copied ? "已複製" : "複製"}
        </button>
      ) : null}
      <button
        type="button"
        title="取消釘選"
        aria-label="取消釘選"
        onClick={onClear}
        className="rounded px-1 text-ink-faint hover:bg-surface-3 hover:text-ink"
      >
        ×
      </button>
    </span>
  );
}

export function LineChartWidget({
  widget,
  data,
}: {
  widget: WidgetSpec;
  data: WidgetData;
}) {
  const theme = useChartTheme();
  const filters = useQueryFilters();
  const query = widgetQuery(widget);
  const range = useEffectiveRange(widget.query ?? null);
  const opts = (widget.options ?? {}) as Record<string, unknown>;
  const showGrid = opts.show_grid !== false;
  const showLegend = opts.show_legend === true;
  const unit = widget.encoding?.unit || "";

  const rows = data.points.map((p) => ({
    t: p.t * 1000,
    v: p.v === null ? null : Number(p.v.toFixed(2)),
  }));

  // Pin the x-domain from the window the backend ACTUALLY used. Until
  // the response carries end/range_seconds (older backend), fall back
  // to the data extent so the chart still renders.
  const hasWindow =
    data.end !== null && data.rangeSeconds !== null && data.rangeSeconds > 0;
  const longRange = (data.rangeSeconds ?? 0) >= 86400;
  const domain: [number | string, number | string] = hasWindow
    ? [
        ((data.end as number) - (data.rangeSeconds as number)) * 1000,
        (data.end as number) * 1000,
      ]
    : ["dataMin", "dataMax"];

  // Spec override wins; otherwise the vital's data color; otherwise data-1.
  const stroke =
    widget.encoding?.color ||
    vitalColorFor(query.promql, widget.title, theme.colors) ||
    theme.palette[0];

  const seriesName =
    widget.encoding?.legend ||
    (filters?.patient_id
      ? `${widget.title} · ${filters.patient_id}`
      : widget.title);

  const thresholds = widget.thresholds ?? [];
  const direction = resolveThresholdDirection(thresholds);
  const refLines = sortedForDirection(thresholds, direction);

  const [pinned, setPinned] = useState<{ t: number; v: number | null } | null>(
    null,
  );

  return (
    <WidgetFrame
      widget={widget}
      unit={unit || null}
      badge={range?.label ?? null}
      headerExtra={
        pinned ? (
          <PinnedValueBadge
            t={pinned.t}
            v={pinned.v}
            unit={unit}
            longRange={longRange}
            onClear={() => setPinned(null)}
          />
        ) : null
      }
      {...frameStatusProps(data)}
      footer={thresholds.length > 0 ? <NonDiagnosticFooter /> : null}
    >
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={rows}
          syncId="dash"
          syncMethod="value"
          margin={{ top: 8, right: 12, bottom: 0, left: 0 }}
          onClick={(e) => {
            const label = e?.activeLabel;
            if (label === undefined || label === null) return;
            const payloadValue = e?.activePayload?.[0]?.value;
            setPinned({
              t: Number(label),
              v: typeof payloadValue === "number" ? payloadValue : null,
            });
          }}
        >
          {showGrid ? (
            <CartesianGrid strokeDasharray="3 3" stroke={theme.gridStroke} />
          ) : null}
          <XAxis
            dataKey="t"
            type="number"
            domain={domain}
            allowDataOverflow
            tickFormatter={(v) => formatTick(Number(v), longRange)}
            tick={theme.axisTick}
            axisLine={theme.axisLine}
            tickLine={false}
            minTickGap={24}
          />
          <YAxis
            tick={theme.axisTick}
            axisLine={theme.axisLine}
            tickLine={false}
            width={unit ? 56 : 44}
            label={
              unit
                ? {
                    value: unit,
                    angle: -90,
                    position: "insideLeft",
                    offset: 10,
                    style: {
                      fill: theme.colors.inkFaint,
                      fontSize: 12,
                      fontFamily: FONT_MONO,
                      textAnchor: "middle",
                    },
                  }
                : undefined
            }
          />
          <Tooltip
            labelFormatter={(v) => formatTooltipTime(Number(v), longRange)}
            formatter={(value) => [
              typeof value === "number"
                ? `${value}${unit ? ` ${unit}` : ""}`
                : "—",
              seriesName,
            ]}
            contentStyle={theme.tooltipContentStyle}
            labelStyle={theme.tooltipLabelStyle}
          />
          {showLegend ? (
            <Legend
              iconType="plainline"
              wrapperStyle={theme.legendStyle}
            />
          ) : null}
          {refLines.map((t, i) => {
            const sev = severityForSpec(t.color) ?? "info";
            const color = colorForSeverity(sev, theme.colors);
            return (
              <ReferenceLine
                key={`thr-${i}`}
                y={t.value}
                stroke={color}
                strokeDasharray="4 4"
                strokeWidth={1.5}
                ifOverflow="extendDomain"
                label={{
                  value: thresholdCaption(t, direction, unit),
                  position: "insideTopRight",
                  fill: color,
                  fontSize: 12,
                  fontFamily: FONT_MONO,
                }}
              />
            );
          })}
          <Line
            type="monotone"
            dataKey="v"
            name={seriesName}
            stroke={stroke}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 3 }}
            connectNulls={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </WidgetFrame>
  );
}
