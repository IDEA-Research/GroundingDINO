"use client";

import React from "react";

import { heatmapScale, scaleColorAt, scaleTextColorAt, useChartTheme } from "@/lib/chart-theme";
import {
  colorScaleAt,
  normalize,
  sampleScale,
  statusColorScale,
  textColorForScale,
} from "@/lib/color-scale";
import type { WidgetSpec } from "@/lib/spec-schema";
import type { TableRow, WidgetData } from "@/lib/useQueryData";
import { WidgetFrame, frameStatusProps } from "./WidgetFrame";

// Color scale: the default (and only undocumented/no-option case) is the
// project's semantic severity ramp — ok (green) -> warn (amber) -> danger
// (red) — built from lib/color-scale.ts's continuous HSL interpolation
// (see that module for why HSL beats a raw RGB blend for this ramp).
// The legacy named ramps ("viridis" | "warm" | "cool") documented in
// backend/app/specs/widget_schema_doc.py still resolve through
// lib/chart-theme.ts's heatmapScale for backward compatibility with any
// spec that explicitly asks for one of them.
const LEGACY_SCALE_NAMES = new Set(["viridis", "warm", "cool"]);

// ---------------------------------------------------------------------------
// Build a 2D grid from the table data.
// Each table row becomes one cell; we derive x/y from metric labels.
// ---------------------------------------------------------------------------

interface HeatCell {
  x: string;
  y: string;
  value: number | null;
}

function buildGrid(table: TableRow[]): {
  cells: HeatCell[];
  xLabels: string[];
  yLabels: string[];
} {
  const cells: HeatCell[] = [];
  const xSet = new Set<string>();
  const ySet = new Set<string>();

  for (const row of table) {
    const keys = Object.keys(row.metric).filter((k) => k !== "job");
    // Use first non-job key as y-axis, second (or index) as x-axis.
    const yVal = keys.length > 0 ? row.metric[keys[0]] : "row";
    const xVal = keys.length > 1 ? row.metric[keys[1]] : String(cells.length);
    xSet.add(xVal);
    ySet.add(yVal);
    cells.push({ x: xVal, y: yVal, value: row.value });
  }

  return {
    cells,
    xLabels: Array.from(xSet),
    yLabels: Array.from(ySet),
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function HeatmapWidget({
  widget,
  data,
}: {
  widget: WidgetSpec;
  data: WidgetData;
}) {
  const theme = useChartTheme();
  const opts = (widget.options ?? {}) as Record<string, unknown>;
  const decimals = typeof opts.decimals === "number" ? opts.decimals : 1;
  const xLabel = typeof opts.x_label === "string" ? opts.x_label : "";
  const yLabel = typeof opts.y_label === "string" ? opts.y_label : "";

  const colorScaleName =
    typeof opts.color_scale === "string" ? opts.color_scale : undefined;
  const legacyScale =
    colorScaleName && LEGACY_SCALE_NAMES.has(colorScaleName)
      ? heatmapScale(colorScaleName, theme.colors)
      : null;
  const statusScale = statusColorScale(theme.colors);

  const colorAt = (t: number): string =>
    legacyScale ? scaleColorAt(legacyScale, t) : colorScaleAt(statusScale, t);
  const textColorAt = (t: number): string =>
    legacyScale
      ? scaleTextColorAt(legacyScale, t, theme.colors)
      : textColorForScale(statusScale, t, theme.colors.ink);
  const legendStops = legacyScale ?? sampleScale(statusScale, 12);

  const { cells, xLabels, yLabels } = buildGrid(data.table);

  // Value range for normalisation — missing cells are excluded, never 0.
  const values = cells
    .map((c) => c.value)
    .filter((v): v is number => v !== null && Number.isFinite(v));
  const minV = values.length ? Math.min(...values) : 0;
  const maxV = values.length ? Math.max(...values) : 1;

  // Build a lookup map for O(1) cell access.
  const lookup = new Map<string, HeatCell>();
  for (const c of cells) lookup.set(`${c.y}|${c.x}`, c);

  const cellW = xLabels.length ? `${Math.floor(100 / xLabels.length)}%` : "40px";

  return (
    <WidgetFrame
      widget={widget}
      unit={widget.encoding?.unit}
      {...frameStatusProps(data)}
    >
      <div className="flex h-full flex-col overflow-auto text-sm">
        {/* Grid */}
        <div className="flex-1 overflow-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <th className="p-1 text-left font-medium text-ink-dim">
                  {yLabel || ""}
                </th>
                {xLabels.map((x) => (
                  <th
                    key={x}
                    className="p-1 text-center font-medium text-ink-dim"
                    style={{ width: cellW }}
                  >
                    {x}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {yLabels.map((y) => (
                <tr key={y}>
                  <td className="whitespace-nowrap p-1 text-ink-dim">{y}</td>
                  {xLabels.map((x) => {
                    const cell = lookup.get(`${y}|${x}`);
                    const v = cell?.value ?? null;
                    if (v === null) {
                      return (
                        <td
                          key={x}
                          data-missing="true"
                          className="p-1 text-center text-ink-faint"
                          title={`${y} / ${x}: 無資料`}
                        >
                          —
                        </td>
                      );
                    }
                    const t = normalize(v, minV, maxV);
                    return (
                      <td
                        key={x}
                        className="p-1 text-center transition-colors"
                        style={{ backgroundColor: colorAt(t) }}
                        title={`${y} / ${x}: ${v.toFixed(decimals)}`}
                      >
                        <span
                          className="font-mono text-sm tabular-nums"
                          style={{ color: textColorAt(t) }}
                        >
                          {v.toFixed(decimals)}
                        </span>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {/* Axis label */}
        {xLabel && (
          <div className="mt-1 text-center text-xs text-ink-faint">{xLabel}</div>
        )}
        {/* Compact color legend: min → max across the active scale. */}
        <div className="mt-1 flex shrink-0 items-center gap-2 text-xs text-ink-faint">
          <span className="font-mono tabular-nums">{minV.toFixed(decimals)}</span>
          <span
            className="h-2 min-w-0 flex-1 rounded-sm"
            style={{
              background: `linear-gradient(to right, ${legendStops.join(", ")})`,
            }}
            aria-hidden="true"
          />
          <span className="font-mono tabular-nums">{maxV.toFixed(decimals)}</span>
        </div>
      </div>
    </WidgetFrame>
  );
}
