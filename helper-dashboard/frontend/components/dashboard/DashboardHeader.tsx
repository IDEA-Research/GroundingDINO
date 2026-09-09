"use client";

import React, { type RefObject } from "react";

import { Badge } from "@/components/ui";
import {
  TimeRangeControls,
  focusTimeRangeControls,
} from "@/components/dashboard/TimeRangeControls";
import { widgetQuery, type Capabilities, type DashboardSpec } from "@/lib/spec-schema";
import { useTimeRange } from "@/lib/time-range";
import type { BackendStatusState } from "@/lib/useBackendStatus";

import { DataStatusStrip, type DatasourceState } from "./DataStatusStrip";
import { PatientSelector, type PatientFilterState } from "./PatientSelector";

// Top of the main column on `/`: title + description, 病患 selector, time
// range + auto-refresh controls (C's TimeRangeControls), the effective
// range readout, the optional 「顯示所有元件」 toggle (only when the spec
// has a stat_card/gauge pair that DashboardPreview would fold), and the
// DataStatusStrip. Must render inside a TimeRangeProvider.

// True when a gauge duplicates a stat_card's PromQL — the case
// DashboardPreview pairs by default.
export function hasStatGaugePairs(spec: DashboardSpec | null): boolean {
  if (!spec) return false;
  const norm = (s: string | null | undefined) => (s ?? "").replace(/\s+/g, " ").trim();
  const stats = new Set<string>();
  const gauges = new Set<string>();
  for (const w of spec.widgets) {
    const q = norm(widgetQuery(w).promql);
    if (!q) continue;
    if (w.type === "stat_card") stats.add(q);
    if (w.type === "gauge") gauges.add(q);
  }
  for (const q of gauges) if (stats.has(q)) return true;
  return false;
}

export function DashboardHeader({
  spec,
  savedName,
  patient,
  showAllWidgets,
  onToggleShowAll,
  backend,
  datasource,
  capabilities,
  onOpenDataSource,
  timeRangeRef,
}: {
  spec: DashboardSpec;
  // Library entry name when the spec was loaded from the library.
  savedName: string | null;
  patient: PatientFilterState;
  showAllWidgets: boolean;
  onToggleShowAll: (v: boolean) => void;
  backend: BackendStatusState;
  datasource: DatasourceState;
  capabilities: Capabilities | null;
  onOpenDataSource: () => void;
  // Focus target for 「調整時間範圍」.
  timeRangeRef: RefObject<HTMLDivElement>;
}) {
  const tr = useTimeRange();
  const pairs = hasStatGaugePairs(spec);
  const rangeText = tr.range.mode === "override" ? tr.range.label : "依各元件設定";

  return (
    <header className="shrink-0 border-b border-edge bg-canvas">
      <div className="flex flex-col gap-3 px-3 pt-3 pb-3 lg:px-4">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="min-w-0 break-words text-lg font-semibold leading-tight text-ink">
              {spec.title}
            </h1>
            {savedName && savedName !== spec.title ? (
              <span className="truncate text-sm text-ink-dim">（{savedName}）</span>
            ) : null}
            {patient.multiPatientNote ? <Badge tone="info">顯示多位病患平均</Badge> : null}
          </div>
          {spec.description ? (
            <p className="mt-0.5 max-w-3xl text-sm leading-relaxed text-ink-dim">
              {spec.description}
            </p>
          ) : null}
        </div>

        <div className="flex flex-col gap-3 md:flex-row md:flex-wrap md:items-end">
          <PatientSelector state={patient} />

          <div
            ref={timeRangeRef}
            tabIndex={-1}
            aria-label="時間範圍與自動更新"
            className="min-w-0 rounded-md focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
          >
            <TimeRangeControls />
          </div>

          <div className="flex flex-wrap items-center gap-3 text-sm text-ink-dim md:ml-auto">
            <span className="flex items-center gap-1.5 whitespace-nowrap">
              目前範圍
              <Badge tone="info">{rangeText}</Badge>
            </span>
            {pairs ? (
              <label className="flex min-h-[40px] cursor-pointer items-center gap-2 whitespace-nowrap text-ink lg:min-h-0">
                <input
                  type="checkbox"
                  checked={showAllWidgets}
                  onChange={(e) => onToggleShowAll(e.target.checked)}
                  className="h-4 w-4 accent-[rgb(var(--accent))]"
                />
                顯示所有元件
              </label>
            ) : null}
          </div>
        </div>
      </div>

      <DataStatusStrip
        backend={backend}
        datasource={datasource}
        onOpenDataSource={onOpenDataSource}
        onFocusTimeRange={() => {
          const root = timeRangeRef.current;
          root?.scrollIntoView?.({ block: "nearest" });
          // C's helper focuses the active preset; fall back to the wrapper.
          focusTimeRangeControls();
          if (root && !root.contains(document.activeElement)) {
            const target =
              root.querySelector<HTMLElement>('[role="radio"][aria-checked="true"]') ??
              root.querySelector<HTMLElement>('[role="radio"], button, input, select');
            (target ?? root).focus();
          }
        }}
      />
    </header>
  );
}
