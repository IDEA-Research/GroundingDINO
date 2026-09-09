"use client";

import React from "react";

import {
  colorForSeverity,
  mixHex,
  useChartTheme,
  vitalColorFor,
} from "@/lib/chart-theme";
import { sampleTsOf } from "@/lib/data-status";
import { widgetQuery, type WidgetSpec } from "@/lib/spec-schema";
import { activeThreshold, resolveThresholdDirection } from "@/lib/threshold";
import { useEffectiveRange } from "@/lib/time-range";
import {
  REASON_TEXT,
  useQueryFilters,
  type WidgetData,
} from "@/lib/useQueryData";
import {
  NonDiagnosticFooter,
  SeverityIcon,
  WidgetFrame,
  fmtClock,
  fmtClockMs,
  formatValue,
  frameStatusProps,
} from "./WidgetFrame";

// Stat card, four tiers: 名稱 (frame header) -> 數值 (「—」 when null) -> 單位
// -> 資料時間/狀態 (最後量測 HH:mm:ss or the reason, 多位病患平均 note).
// A threshold breach shows TEXT + icon in the status color (never color
// alone) and recolors the numeral; a normal reading uses the vital's data
// color (HR data-1 / SpO2 data-2 / rSO2 data-3). Widgets with thresholds
// carry the LD-7 footer.
export function StatCardWidget({
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

  const unit = widget.encoding?.unit || "";
  const decimals = (widget.options as { decimals?: unknown } | undefined)
    ?.decimals;
  const value = data.latest;
  const thresholds = widget.thresholds ?? [];
  const direction = resolveThresholdDirection(thresholds);
  const active = activeThreshold(value, thresholds, direction);

  const color = active
    ? colorForSeverity(active.severity, theme.colors)
    : widget.encoding?.color ||
      vitalColorFor(query.promql, widget.title, theme.colors) ||
      theme.colors.ink;

  const sampleTs = sampleTsOf(data);
  const averaged = data.nSeries > 1 && !filters?.patient_id;

  // Optional Grafana BigValue-style "background" color mode (opt-in via
  // spec, default off): a LOW-saturation wash toward the active severity
  // color, only while a threshold is actually breached — a normal reading
  // stays neutral so the wash reads as "something needs attention", not
  // decoration. Text/icon still carry the same information (color is never
  // the only cue).
  const colorMode = (widget.options as { color_mode?: unknown } | undefined)
    ?.color_mode;
  const washStyle =
    active && colorMode === "background"
      ? {
          backgroundColor: mixHex(theme.colors.surface, color, 0.12),
          borderColor: mixHex(theme.colors.edge, color, 0.4),
        }
      : undefined;

  return (
    <WidgetFrame
      widget={widget}
      badge={range?.label ?? null}
      style={washStyle}
      {...frameStatusProps(data)}
      footer={thresholds.length > 0 ? <NonDiagnosticFooter /> : null}
    >
      <div className="flex h-full flex-col justify-center gap-1.5">
        <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
          <span
            className="stat-numeral font-mono"
            style={{ color: value === null ? theme.colors.inkFaint : color }}
          >
            {value === null
              ? "—"
              : formatValue(value, typeof decimals === "number" ? decimals : null)}
          </span>
          {unit ? (
            <span className="text-sm font-medium text-ink-dim">{unit}</span>
          ) : null}
        </div>

        {active ? (
          <div
            className="inline-flex items-center gap-1.5 text-sm font-medium"
            style={{ color }}
          >
            <SeverityIcon severity={active.severity} />
            <span>{active.label}</span>
          </div>
        ) : null}

        {value === null && data.status !== "loading" ? (
          <div className="text-sm text-ink-dim">
            {data.reason ?? REASON_TEXT.empty}
          </div>
        ) : null}

        <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-ink-faint">
          <span className="font-mono tabular-nums">
            {sampleTs !== null
              ? `最後量測 ${fmtClock(sampleTs)}`
              : data.lastSuccessAt !== null
                ? `最後查詢 ${fmtClockMs(data.lastSuccessAt)}`
                : "尚無量測時間"}
          </span>
          {averaged ? <span>多位病患平均</span> : null}
          {data.filtersApplied?.patient_id ? (
            <span>病患 {data.filtersApplied.patient_id}</span>
          ) : null}
        </div>
      </div>
    </WidgetFrame>
  );
}
