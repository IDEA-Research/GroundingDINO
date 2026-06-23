"use client";

import React from "react";

import type { WidgetSpec } from "@/lib/spec-schema";
import { useQueryData } from "@/lib/useQueryData";
import { WidgetFrame } from "./WidgetFrame";

// ---------------------------------------------------------------------------
// Colour scales — small built-in set, no external deps.
// ---------------------------------------------------------------------------

const SCALES: Record<string, string[]> = {
  warm: ["#1e293b", "#854d0e", "#ca8a04", "#facc15", "#fef08a"],
  cool: ["#1e293b", "#1e3a5f", "#2563eb", "#60a5fa", "#bfdbfe"],
  viridis: ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
};

function pickScale(name?: string): string[] {
  if (name && SCALES[name]) return SCALES[name];
  return SCALES.warm;
}

function interpolateColor(scale: string[], t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  const idx = clamped * (scale.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, scale.length - 1);
  const frac = idx - lo;
  return blendHex(scale[lo], scale[hi], frac);
}

function blendHex(a: string, b: string, t: number): string {
  const [r1, g1, b1] = hexToRgb(a);
  const [r2, g2, b2] = hexToRgb(b);
  const r = Math.round(r1 + (r2 - r1) * t);
  const g = Math.round(g1 + (g2 - g1) * t);
  const bl = Math.round(b1 + (b2 - b1) * t);
  return `rgb(${r},${g},${bl})`;
}

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.substring(0, 2), 16),
    parseInt(h.substring(2, 4), 16),
    parseInt(h.substring(4, 6), 16),
  ];
}

// ---------------------------------------------------------------------------
// Build a 2D grid from the mock table data.
// Each table row becomes one cell; we derive x/y from metric labels.
// ---------------------------------------------------------------------------

interface HeatCell {
  x: string;
  y: string;
  value: number;
}

function buildGrid(
  table: { metric: Record<string, string>; value: number }[],
): { cells: HeatCell[]; xLabels: string[]; yLabels: string[] } {
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

export function HeatmapWidget({ widget }: { widget: WidgetSpec }) {
  const data = useQueryData(widget.query);
  const opts = (widget.options ?? {}) as Record<string, unknown>;
  const decimals = typeof opts.decimals === "number" ? opts.decimals : 1;
  const xLabel = typeof opts.x_label === "string" ? opts.x_label : "";
  const yLabel = typeof opts.y_label === "string" ? opts.y_label : "";
  const scale = pickScale(
    typeof opts.color_scale === "string" ? opts.color_scale : undefined,
  );

  const { cells, xLabels, yLabels } = buildGrid(data.table);

  // Compute value range for normalisation.
  const values = cells.map((c) => c.value);
  const minV = values.length ? Math.min(...values) : 0;
  const maxV = values.length ? Math.max(...values) : 1;
  const range = maxV - minV || 1;

  // Build a lookup map for O(1) cell access.
  const lookup = new Map<string, HeatCell>();
  for (const c of cells) lookup.set(`${c.y}|${c.x}`, c);

  const cellW = xLabels.length ? `${Math.floor(100 / xLabels.length)}%` : "40px";

  return (
    <WidgetFrame widget={widget} badge={data.source === "mock" ? "mock" : null}>
      <div className="flex h-full flex-col overflow-auto text-[10px]">
        {/* Grid */}
        <div className="flex-1 overflow-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <th className="p-1 text-left text-slate-400">
                  {yLabel || ""}
                </th>
                {xLabels.map((x) => (
                  <th
                    key={x}
                    className="p-1 text-center text-slate-400"
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
                  <td className="whitespace-nowrap p-1 text-slate-300">
                    {y}
                  </td>
                  {xLabels.map((x) => {
                    const cell = lookup.get(`${y}|${x}`);
                    const v = cell?.value ?? 0;
                    const t = (v - minV) / range;
                    const bg = interpolateColor(scale, t);
                    return (
                      <td
                        key={x}
                        className="p-1 text-center"
                        style={{ backgroundColor: bg }}
                        title={`${y} / ${x}: ${v.toFixed(decimals)}`}
                      >
                        <span className="text-[9px] text-white/80">
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
        {/* Axis labels */}
        {xLabel && (
          <div className="mt-1 text-center text-[10px] text-slate-400">
            {xLabel}
          </div>
        )}
      </div>
    </WidgetFrame>
  );
}
