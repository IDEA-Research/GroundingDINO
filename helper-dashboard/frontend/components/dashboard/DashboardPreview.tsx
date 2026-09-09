"use client";

import React, { useMemo } from "react";

import { Badge } from "@/components/ui/Badge";
import { Disclosure } from "@/components/ui/Disclosure";
import { useDevMode } from "@/lib/dev-mode";
import { collectRenderProblems, renderWidget } from "@/lib/renderer";
import type {
  DashboardSpec,
  QueryFilters,
  ReviewTrailEvent,
  WidgetSpec,
} from "@/lib/spec-schema";
import { widgetQuery } from "@/lib/spec-schema";
import { TimeRangeProvider, useHasTimeRangeProvider } from "@/lib/time-range";
import { QueryFiltersProvider } from "@/lib/useQueryData";
import { PairedStatGauge } from "@/widget-toolkit/PairedStatGauge";
import { TimeRangeControls } from "@/components/dashboard/TimeRangeControls";

// Renders a DashboardSpec as the responsive widget grid.
//
//   >= lg : the spec grid (layout.columns x row_height). zoneOrder=false
//           (evaluator route) places every widget at its spec x/y/w/h;
//           zoneOrder=true ("/") keeps w/h but auto-flows in zone order:
//           stat_card/gauge -> charts -> table/alert_list/decision_flow,
//           (y, x) within a zone.
//   <  lg : x/y ignored, 2 columns, stat_card/gauge span 1, everything else
//           spans the row, auto row heights (CSS in globals.css .dash-grid).
//
// pairDuplicates folds a gauge whose normalized PromQL equals a stat card's
// into ONE cell (PairedStatGauge) — both widgets keep their own
// data-widget-id roots, so the evaluator still counts both.
//
// A TimeRangeProvider is self-provided ONLY when none exists above (the
// headless /dashboard/[id] route); on "/" the shell mounts it so the
// DashboardHeader controls and the widgets share one clock.

export interface DashboardPreviewProps {
  spec: DashboardSpec | null;
  reviewing?: boolean;
  // Real pre-output review transcript for THIS spec (home page passes it;
  // the headless evaluator route has none). Drives the review badge —
  // there is no unconditional badge.
  reviewTrail?: ReviewTrailEvent[];
  // Enforce zone order (true on "/", false on the evaluator route).
  zoneOrder?: boolean;
  // Fold gauge+stat_card duplicates into one cell (default on; the
  // evaluator route passes false so spec placement stays 1:1).
  pairDuplicates?: boolean;
  // Patient / device filters applied to every widget query.
  filters?: QueryFilters | null;
  // Render the compact title/description/TimeRangeControls header. Off by
  // default — on "/" the DashboardHeader owns that row.
  showHeader?: boolean;
  className?: string;
}

type Cell =
  | { kind: "single"; widget: WidgetSpec }
  | { kind: "pair"; stat: WidgetSpec; gauge: WidgetSpec };

function zoneRank(type: string): number {
  switch (type) {
    case "stat_card":
    case "gauge":
      return 0;
    case "line_chart":
    case "bar_chart":
    case "heatmap":
    case "pie_chart":
      return 1;
    default:
      return 2;
  }
}

function normalizePromql(s: string | null | undefined): string {
  return (s ?? "").replace(/\s+/g, " ").trim().toLowerCase();
}

function byPosition(a: WidgetSpec, b: WidgetSpec): number {
  return a.position.y - b.position.y || a.position.x - b.position.x;
}

export function buildCells(
  widgets: WidgetSpec[],
  opts: { zoneOrder: boolean; pairDuplicates: boolean },
): Cell[] {
  const sorted = [...widgets].sort((a, b) =>
    opts.zoneOrder
      ? zoneRank(a.type) - zoneRank(b.type) || byPosition(a, b)
      : byPosition(a, b),
  );

  const pairedGaugeByStat = new Map<string, WidgetSpec>();
  const consumedGauges = new Set<string>();
  if (opts.pairDuplicates) {
    const stats = sorted.filter((w) => w.type === "stat_card");
    for (const gauge of sorted) {
      if (gauge.type !== "gauge") continue;
      const key = normalizePromql(widgetQuery(gauge).promql);
      if (!key) continue;
      const stat = stats.find(
        (s) =>
          !pairedGaugeByStat.has(s.id) &&
          normalizePromql(widgetQuery(s).promql) === key,
      );
      if (stat) {
        pairedGaugeByStat.set(stat.id, gauge);
        consumedGauges.add(gauge.id);
      }
    }
  }

  const cells: Cell[] = [];
  for (const w of sorted) {
    if (consumedGauges.has(w.id)) continue;
    const gauge = pairedGaugeByStat.get(w.id);
    if (gauge) cells.push({ kind: "pair", stat: w, gauge });
    else cells.push({ kind: "single", widget: w });
  }
  return cells;
}

// Rows the compact gauge (h-16 + gap) adds beneath a paired stat card.
function pairExtraRows(rowHeight: number): number {
  return Math.max(1, Math.ceil((64 + 12 + 8) / Math.max(10, rowHeight)));
}

function cellStyle(
  primary: WidgetSpec,
  columns: number,
  extraRows: number,
): React.CSSProperties {
  const x = Math.max(0, Math.min(primary.position.x, columns - 1));
  const w = Math.max(1, Math.min(primary.position.w, columns - x));
  const y = Math.max(0, primary.position.y);
  const h = Math.max(1, primary.position.h) + extraRows;
  return {
    ["--gc" as string]: `${x + 1} / span ${w}`,
    ["--gr" as string]: `${y + 1} / span ${h}`,
    ["--gw" as string]: String(w),
    ["--gh" as string]: String(h),
  } as React.CSSProperties;
}

export function DashboardPreview(props: DashboardPreviewProps) {
  const hasProvider = useHasTimeRangeProvider();
  const content = <DashboardPreviewBody {...props} />;
  if (hasProvider || !props.spec) return content;
  return (
    <TimeRangeProvider refreshInterval={props.spec.refresh_interval}>
      {content}
    </TimeRangeProvider>
  );
}

function DashboardPreviewBody({
  spec,
  reviewing = false,
  reviewTrail,
  zoneOrder = false,
  pairDuplicates = true,
  filters = null,
  showHeader = false,
  className,
}: DashboardPreviewProps) {
  const { unlocked: devUnlocked } = useDevMode();

  const columns = spec?.layout?.columns || 12;
  const rowHeight = spec?.layout?.row_height || 40;

  const cells = useMemo(
    () => (spec ? buildCells(spec.widgets, { zoneOrder, pairDuplicates }) : []),
    [spec, zoneOrder, pairDuplicates],
  );

  if (!spec) {
    if (!reviewing) return null;
    return (
      <div className="flex h-full items-center justify-center px-4 text-center text-sm text-ink-dim">
        正在審查渲染結果…
      </div>
    );
  }

  const problems = collectRenderProblems(spec);

  // Show the badge ONLY when the review loop actually ran and its latest
  // review verdict was "approve" (backend ReviewDecision literals are
  // approve | patch | escalate). No trail, or a non-approve last verdict,
  // renders nothing.
  const lastReview = [...(reviewTrail ?? [])]
    .reverse()
    .find((e) => e.stage === "review" && typeof e.decision === "string");
  const reviewApproved = lastReview?.decision === "approve";

  const extraRows = pairExtraRows(rowHeight);

  return (
    <QueryFiltersProvider filters={filters}>
      <div className={["relative flex h-full min-h-0 flex-col", className].filter(Boolean).join(" ")}>
        {showHeader ? (
          <div className="shrink-0 border-b border-edge px-3 py-2">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
              <div className="flex min-w-0 flex-wrap items-center gap-2">
                <h1 className="truncate text-base font-semibold text-ink">
                  {spec.title}
                </h1>
                {reviewApproved ? <Badge tone="ok">已審查</Badge> : null}
              </div>
              <div className="ml-auto">
                <TimeRangeControls compact />
              </div>
            </div>
            {spec.description ? (
              <p className="mt-0.5 text-sm text-ink-dim">{spec.description}</p>
            ) : null}
          </div>
        ) : null}

        {problems.length > 0 ? (
          <div
            role="alert"
            className="shrink-0 border-b border-warn/40 bg-warn/10 px-3 py-2 text-sm text-ink"
          >
            <span className="font-medium text-warn">
              {problems.length} 個元件無法顯示
            </span>
            <span className="ml-2 text-ink-dim">請向 Helper 說明需要調整的元件。</span>
            {devUnlocked ? (
              <Disclosure summary="技術細節" className="mt-1">
                <ul className="list-disc pl-4 text-xs text-ink-dim">
                  {problems.map((p) => (
                    <li key={p.widget_id}>
                      <code className="font-mono">{p.widget_id}</code> — {p.reason}
                    </li>
                  ))}
                </ul>
              </Disclosure>
            ) : null}
          </div>
        ) : null}

        <div className="min-h-0 flex-1 overflow-auto p-3">
          <div
            className={zoneOrder ? "dash-grid dash-grid--flow" : "dash-grid"}
            style={
              {
                ["--dash-cols" as string]: String(columns),
                ["--dash-row-h" as string]: `${rowHeight}px`,
              } as React.CSSProperties
            }
          >
            {cells.map((cell) =>
              cell.kind === "pair" ? (
                <div
                  key={`${cell.stat.id}+${cell.gauge.id}`}
                  className="dash-cell"
                  data-widget-type="pair"
                  style={cellStyle(cell.stat, columns, extraRows)}
                >
                  <PairedStatGauge
                    stat={renderWidget(cell.stat)}
                    gauge={renderWidget(cell.gauge, { compact: true })}
                  />
                </div>
              ) : (
                <div
                  key={cell.widget.id}
                  className="dash-cell"
                  data-widget-type={cell.widget.type}
                  style={cellStyle(cell.widget, columns, 0)}
                >
                  {renderWidget(cell.widget)}
                </div>
              ),
            )}
          </div>
        </div>

        {reviewing ? (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-canvas/60">
            <div
              role="status"
              className="rounded-lg border border-edge bg-surface px-4 py-3 text-sm text-ink shadow-pop"
            >
              <div className="mb-0.5 font-medium">輸出前審查進行中…</div>
              <div className="text-sm text-ink-dim">
                Helper 正在檢查渲染後的儀表板。
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </QueryFiltersProvider>
  );
}
