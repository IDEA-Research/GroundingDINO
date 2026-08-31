"use client";

import React from "react";

import { useTimeRange, type RangeExpr } from "@/lib/time-range";

// Compact global time-range picker. A pure VIEW-LAYER control: it flips
// the TimeRangeProvider's override, never the DashboardSpec. 自動 =
// every widget uses its own spec range/step; a preset = all time-series
// widgets query the override window.

function fmtClock(epochSeconds: number): string {
  const d = new Date(epochSeconds * 1000);
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

export function TimeRangePicker() {
  const { override, setOverride, lastUpdated, presets } = useTimeRange();

  const isActive = (expr: RangeExpr | null): boolean =>
    expr === null
      ? override.mode === "auto"
      : override.mode === "override" && override.expr === expr;

  const btnClass = (active: boolean): string =>
    "rounded px-2 py-0.5 text-xs transition-colors " +
    (active
      ? "bg-surface-3 text-ink ring-1 ring-ring/60"
      : "text-ink-dim hover:bg-surface-2 hover:text-ink");

  return (
    <div className="flex flex-wrap items-center gap-1">
      <div className="flex items-center gap-0.5 rounded-md border border-edge bg-surface p-0.5">
        <button
          type="button"
          aria-pressed={isActive(null)}
          className={btnClass(isActive(null))}
          onClick={() => setOverride({ mode: "auto" })}
        >
          自動
        </button>
        {presets.map((p) => (
          <button
            key={p.expr}
            type="button"
            aria-pressed={isActive(p.expr)}
            className={btnClass(isActive(p.expr))}
            onClick={() => setOverride({ mode: "override", expr: p.expr })}
          >
            {p.label}
          </button>
        ))}
      </div>
      <span className="ml-1 flex items-center gap-1.5 font-mono text-xs tabular-nums text-ink-faint">
        <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-ok" />
        {lastUpdated !== null
          ? `最後更新 ${fmtClock(lastUpdated)}`
          : "最後更新 —"}
      </span>
    </div>
  );
}
