"use client";

import React, { useEffect, useMemo, useState } from "react";
import type { Layout } from "react-grid-layout";

import { Badge } from "@/components/ui/Badge";
import { Disclosure } from "@/components/ui/Disclosure";
import { CrosshairProvider } from "@/lib/crosshair";
import { useDevMode } from "@/lib/dev-mode";
import { collectRenderProblems, renderWidget } from "@/lib/renderer";
import type {
  DashboardSpec,
  QueryFilters,
  ReviewTrailEvent,
  WidgetPosition,
  WidgetSpec,
} from "@/lib/spec-schema";
import { widgetQuery } from "@/lib/spec-schema";
import { TimeRangeProvider, useHasTimeRangeProvider } from "@/lib/time-range";
import { QueryFiltersProvider } from "@/lib/useQueryData";
import { PairedStatGauge } from "@/widget-toolkit/PairedStatGauge";
import { WidgetActionsProvider } from "@/widget-toolkit/WidgetFrame";
import { TimeRangeControls } from "@/components/dashboard/TimeRangeControls";
import { WidgetEditDrawer } from "@/components/dashboard/WidgetEditDrawer";
import {
  DashboardGrid,
  GRID_COLUMNS,
  type GridPanel,
} from "@/components/dashboard/DashboardGrid";

// Renders a DashboardSpec as the widget grid (Phase 2 — DashboardGrid.tsx /
// react-grid-layout, replacing the old static CSS grid).
//
//   >= 768px : every widget/pair renders as a react-grid-layout panel at
//              its real x/y/w/h, draggable and resizable when `editable`.
//              react-grid-layout's native grid is GRID_COLUMNS (24) wide,
//              matching Grafana's GRID_COLUMN_COUNT; a dashboard authored
//              on a narrower `layout.columns` (existing specs are 12) has
//              its x/w scaled up on read so proportions are unchanged —
//              see `scale` below. `editable` (true only on "/") is the
//              ONLY thing that turns dragging on; the headless evaluator
//              route (dashboard/[id]) renders the identical geometry
//              frozen.
//   <  768px : DashboardGrid drops react-grid-layout and stacks every
//              panel full-width in one column (our own choice, not a
//              claimed Grafana breakpoint) — see DashboardGrid.tsx.
//
// pairDuplicates folds a gauge whose normalized PromQL equals a stat card's
// into ONE panel (PairedStatGauge) — both widgets keep their own
// data-widget-id roots, so the evaluator still counts both. Only the
// stat card gets its own react-grid-layout panel; the folded gauge's
// position is basis-migrated (see onLayoutSettled) but never itself laid
// out, same as before Phase 2.
//
// `zoneOrder` no longer changes ANY panel's on-screen position — that was
// the old CSS `grid-auto-flow: dense` behavior, which is fundamentally
// incompatible with drag-to-position persistence (it recomputed every
// panel's slot from DOM order on every render, ignoring stored x/y).
// react-grid-layout requires — and this Phase 2 conversion is built
// around — every panel's position being real, explicit, and durable, the
// same model Grafana uses. `zoneOrder` is kept only to order pairing/tab
// order (stat_card/gauge first) via buildCells' sort; it has no layout effect.
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
  // Kept for pairing/tab order only — see the module comment above.
  zoneOrder?: boolean;
  // Fold gauge+stat_card duplicates into one cell (default on; the
  // evaluator route passes false so spec placement stays 1:1).
  pairDuplicates?: boolean;
  // Patient / device filters applied to every widget query.
  filters?: QueryFilters | null;
  // Render the compact title/description/TimeRangeControls header. Off by
  // default — on "/" the DashboardHeader owns that row.
  showHeader?: boolean;
  // Enables react-grid-layout drag/resize (Phase 2). Default off — only
  // "/" passes true; the headless evaluator route stays frozen.
  editable?: boolean;
  // Fires after a drag/resize settles, with a FULLY updated DashboardSpec
  // (every widget's position — including panels not directly dragged —
  // re-expressed on the GRID_COLUMNS basis; see onLayoutSettled). The
  // caller owns persistence (setSpec + whatever backend write applies).
  onLayoutChange?: (next: DashboardSpec) => void;
  className?: string;
}

export type Cell =
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

// A dashboard's `layout.columns` (existing specs: 12) expressed on
// DashboardGrid's fixed GRID_COLUMNS (24) canvas. x/w scale by this
// factor so a panel keeps the same fraction of the row's width it had
// before Phase 2; y/h are row counts and are never scaled.
function columnScale(columns: number): number {
  return GRID_COLUMNS / Math.max(1, columns);
}

// One panel's react-grid-layout geometry from its real spec position —
// scaled to GRID_COLUMNS and clamped the same way the old CSS-grid
// cellStyle() clamped (x within bounds, w never overflowing the grid).
function toGridGeometry(
  position: WidgetPosition,
  scale: number,
  extraRows: number,
): { x: number; y: number; w: number; h: number } {
  const x = Math.max(0, Math.min(GRID_COLUMNS - 1, Math.round(position.x * scale)));
  const w = Math.max(1, Math.min(GRID_COLUMNS - x, Math.round(position.w * scale)));
  const y = Math.max(0, position.y);
  const h = Math.max(1, position.h) + extraRows;
  return { x, y, w, h };
}

export function DashboardPreview(props: DashboardPreviewProps) {
  const hasProvider = useHasTimeRangeProvider();
  // CrosshairProvider scopes every panel rendered for THIS dashboard (line
  // charts today; any future synced widget for free) under one syncId —
  // mounted here, the single shared entry point both "/" and
  // /dashboard/[id] render through, so neither call site has to remember
  // to wrap it itself.
  const content = (
    <CrosshairProvider dashboardId={props.spec?.dashboard_id ?? null}>
      <DashboardPreviewBody {...props} />
    </CrosshairProvider>
  );
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
  editable = false,
  onLayoutChange,
  className,
}: DashboardPreviewProps) {
  const { unlocked: devUnlocked } = useDevMode();

  // Phase 5: WidgetFrame's hover-menu 編輯/全螢幕檢視 callbacks (see
  // WidgetActionsProvider in widget-toolkit/WidgetFrame.tsx). Only wired
  // up in editable mode ("/") — the headless /dashboard/[id] evaluator
  // route stays frozen, exactly like drag/resize.
  const [editingWidgetId, setEditingWidgetId] = useState<string | null>(null);
  const [viewingWidgetId, setViewingWidgetId] = useState<string | null>(null);

  const editingWidget = useMemo(
    () => (spec ? spec.widgets.find((w) => w.id === editingWidgetId) ?? null : null),
    [spec, editingWidgetId],
  );
  const viewingWidget = useMemo(
    () => (spec ? spec.widgets.find((w) => w.id === viewingWidgetId) ?? null : null),
    [spec, viewingWidgetId],
  );

  const widgetActions = useMemo(
    () =>
      editable
        ? {
            onEdit: (id: string) => setEditingWidgetId(id),
            onView: (id: string) => setViewingWidgetId(id),
          }
        : null,
    [editable],
  );

  // The edit drawer hands back one fully-edited WidgetSpec; folding it
  // into the full DashboardSpec and persisting it is onLayoutChange's job
  // already (setSpec + patchSaved position-only PATCH when saved) — same
  // call site DashboardGrid's drag/resize uses, no new save path.
  const handleWidgetSaved = (nextWidget: WidgetSpec) => {
    if (!spec) return;
    const nextWidgets = spec.widgets.map((w) =>
      w.id === nextWidget.id ? nextWidget : w,
    );
    onLayoutChange?.({ ...spec, widgets: nextWidgets });
    setEditingWidgetId(null);
  };

  const columns = spec?.layout?.columns || 12;
  const rowHeight = spec?.layout?.row_height || 40;
  const scale = columnScale(columns);

  const cells = useMemo(
    () => (spec ? buildCells(spec.widgets, { zoneOrder, pairDuplicates }) : []),
    [spec, zoneOrder, pairDuplicates],
  );

  const extraRows = pairExtraRows(rowHeight);

  const panels: GridPanel[] = useMemo(
    () =>
      cells.map((cell) =>
        cell.kind === "pair"
          ? {
              id: cell.stat.id,
              type: "pair",
              ...toGridGeometry(cell.stat.position, scale, extraRows),
              content: (
                <PairedStatGauge
                  stat={renderWidget(cell.stat)}
                  gauge={renderWidget(cell.gauge, { compact: true })}
                />
              ),
            }
          : {
              id: cell.widget.id,
              type: cell.widget.type,
              ...toGridGeometry(cell.widget.position, scale, 0),
              content: renderWidget(cell.widget),
            },
      ),
    [cells, scale, extraRows],
  );

  // Fires once a drag or resize completes. Rebuilds the FULL widget list
  // on the GRID_COLUMNS basis — not just the panel that moved — so a
  // paired gauge (never its own react-grid-layout panel; see the module
  // comment) stays correctly proportioned if pairing is later turned off
  // (「顯示全部」), and so `layout.columns` can move to GRID_COLUMNS in
  // the same, single, atomic update.
  const handleLayoutSettled = (layout: Layout[]) => {
    if (!spec || !onLayoutChange) return;
    const byId = new Map(layout.map((item) => [item.i, item]));
    const pairExtraById = new Map<string, number>();
    for (const cell of cells) {
      if (cell.kind === "pair") pairExtraById.set(cell.stat.id, extraRows);
    }
    const nextWidgets = spec.widgets.map((w) => {
      const item = byId.get(w.id);
      if (item) {
        const extra = pairExtraById.get(w.id) ?? 0;
        return {
          ...w,
          position: {
            x: item.x,
            y: item.y,
            w: item.w,
            h: Math.max(1, item.h - extra),
          },
        };
      }
      // Not directly represented on screen (a folded pair gauge) — only
      // migrate its coordinate basis, never move it.
      return { ...w, position: toGridGeometry(w.position, scale, 0) };
    });
    onLayoutChange({
      ...spec,
      layout: { ...spec.layout, columns: GRID_COLUMNS },
      widgets: nextWidgets,
    });
  };

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

  return (
    <QueryFiltersProvider filters={filters}>
      <WidgetActionsProvider actions={widgetActions}>
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
          <DashboardGrid
            panels={panels}
            rowHeight={rowHeight}
            editable={editable}
            onLayoutSettled={handleLayoutSettled}
          />
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

      {editingWidget ? (
        <WidgetEditDrawer
          widget={editingWidget}
          onClose={() => setEditingWidgetId(null)}
          onSave={handleWidgetSaved}
        />
      ) : null}

      {viewingWidget ? (
        <WidgetViewModal widget={viewingWidget} onClose={() => setViewingWidgetId(null)} />
      ) : null}
      </WidgetActionsProvider>
    </QueryFiltersProvider>
  );
}

// Fullscreen 「全螢幕檢視」 for WidgetFrame's onView — a plain centered
// overlay reusing renderWidget's own dispatch (same as the edit drawer's
// preview), not a second rendering path. Escape or a backdrop click
// closes it.
function WidgetViewModal({
  widget,
  onClose,
}: {
  widget: WidgetSpec;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  // Same id-collision reasoning as WidgetEditDrawer's preview.
  const previewWidget = useMemo<WidgetSpec>(
    () => ({ ...widget, id: `${widget.id}__fullscreen` }),
    [widget],
  );

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`${widget.title} 全螢幕檢視`}
      className="fixed inset-0 z-50 flex items-center justify-center bg-canvas/70 p-6"
      onClick={onClose}
    >
      <div
        className="flex h-full max-h-[85vh] w-full max-w-4xl flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {renderWidget(previewWidget)}
      </div>
    </div>
  );
}
