"use client";

import "react-grid-layout/css/styles.css";
import "react-resizable/css/styles.css";

import React, { useEffect, useState } from "react";
import GridLayout, { WidthProvider, type Layout } from "react-grid-layout";

// Phase 2 layout engine: a thin, domain-agnostic wrapper around
// react-grid-layout. It knows nothing about WidgetSpec/DashboardSpec —
// DashboardPreview.tsx owns the spec<->24-column scaling and the
// pairing (stat_card+gauge) composition; this component only ever sees
// {id, x, y, w, h, content} panels already expressed on the GRID_COLUMNS
// basis, and reports back react-grid-layout's own Layout[] verbatim.
const GridLayoutAuto = WidthProvider(GridLayout);

// Grafana's GRID_COLUMN_COUNT. The backend's WidgetPosition schema
// already bounds x <= 23 / w <= 24 (specs/widget_spec.py) — this is the
// first renderer to actually use that full width; dashboards authored
// against a narrower `layout.columns` are scaled up on read so their
// proportions are unchanged (see DashboardPreview.tsx).
export const GRID_COLUMNS = 24;

// Below this width DashboardGrid drops react-grid-layout entirely and
// renders a plain single-column stack. An app-level design choice (not
// a Grafana breakpoint) — dragging on a touch-width screen is out of
// scope for Phase 2.
const NARROW_BREAKPOINT_PX = 768;

export interface GridPanel {
  id: string;
  type: string;
  // Already on the GRID_COLUMNS basis.
  x: number;
  y: number;
  w: number;
  h: number;
  content: React.ReactNode;
}

export interface DashboardGridProps {
  panels: GridPanel[];
  rowHeight: number;
  // false on the headless evaluator route (dashboard/[id]) — renders
  // the same geometry, but frozen (no drag/resize affordance).
  editable: boolean;
  // Fires with react-grid-layout's full layout (every panel, still on
  // the GRID_COLUMNS basis) once a drag or resize completes.
  onLayoutSettled?: (layout: Layout[]) => void;
  className?: string;
}

function useIsNarrow(breakpointPx: number): boolean {
  // Starts false (matches SSR) and corrects itself post-mount so the
  // server/first-client render agree and hydration never mismatches;
  // real devices under the breakpoint get one desktop-shaped paint
  // before this flips, which is an acceptable trade for that safety.
  const [narrow, setNarrow] = useState(false);
  useEffect(() => {
    const mql = window.matchMedia(`(max-width: ${breakpointPx - 0.02}px)`);
    const update = () => setNarrow(mql.matches);
    update();
    mql.addEventListener("change", update);
    return () => mql.removeEventListener("change", update);
  }, [breakpointPx]);
  return narrow;
}

export function DashboardGrid({
  panels,
  rowHeight,
  editable,
  onLayoutSettled,
  className,
}: DashboardGridProps) {
  const narrow = useIsNarrow(NARROW_BREAKPOINT_PX);

  if (narrow) {
    // Single-column, full-width stack. y/h keep each panel's row count
    // (and therefore its height relative to the others) unchanged —
    // only x/w collapse to "full width, one column".
    const ordered = [...panels].sort((a, b) => a.y - b.y || a.x - b.x);
    return (
      <div className={["dash-stack", className].filter(Boolean).join(" ")}>
        {ordered.map((p) => (
          <div
            key={p.id}
            className="dash-cell"
            data-widget-type={p.type}
            style={{ height: `${p.h * rowHeight}px` }}
          >
            {p.content}
          </div>
        ))}
      </div>
    );
  }

  const layout: Layout[] = panels.map((p) => ({
    i: p.id,
    x: p.x,
    y: p.y,
    w: p.w,
    h: p.h,
  }));

  const handleSettled = (next: Layout[]) => onLayoutSettled?.(next);

  return (
    <GridLayoutAuto
      className={[
        "dash-rgl",
        editable ? "dash-rgl--editable" : null,
        className,
      ]
        .filter(Boolean)
        .join(" ")}
      layout={layout}
      cols={GRID_COLUMNS}
      rowHeight={rowHeight}
      margin={[12, 12]}
      containerPadding={[0, 0]}
      draggableHandle=".dash-drag-handle"
      compactType={null}
      preventCollision
      isBounded
      isDraggable={editable}
      isResizable={editable}
      onDragStop={handleSettled}
      onResizeStop={handleSettled}
      useCSSTransforms
    >
      {panels.map((p) => (
        <div key={p.id} className="dash-cell" data-widget-type={p.type}>
          {p.content}
        </div>
      ))}
    </GridLayoutAuto>
  );
}
