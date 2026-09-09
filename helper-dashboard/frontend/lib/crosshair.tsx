"use client";

import React, { createContext, useContext, useMemo, useState } from "react";

// CrosshairProvider — dashboard-wide cross-panel cursor sync, aligned with
// Grafana Dashboard Settings > General > "Graph tooltip":
//   - "off":              every panel's crosshair/tooltip is independent.
//   - "shared-crosshair":  every time-series panel's cursor line moves
//                          together, but only the panel under the pointer
//                          shows a value tooltip box.
//   - "shared-tooltip":    every panel's cursor line moves together AND
//                          every panel shows its own value tooltip box at
//                          the synced position.
//
// Mechanism note (read before wiring Phase 4): recharts' own
//   <LineChart syncId="..." syncMethod="value">
// already synchronizes the hover/cursor state across every chart sharing
// the same syncId — LineChartWidget.tsx already does exactly this today,
// with syncId HARDCODED to the literal "dash". That existing behavior
// (synced cursor + each chart's own full Tooltip box) already IS
// "shared-tooltip" — it is the cheap, already-proven case, not the
// expensive one. Getting "shared-crosshair" (cursor line only, no value
// box on panels you are not pointing at) needs one extra, well-known
// recharts recipe: give <Tooltip> a `content={() => null}` renderer while
// leaving its `cursor` prop alone — recharts renders the synced cursor
// line independently of what `content` returns, so this suppresses only
// the floating value box. `crosshairTooltipContentProps()` below produces
// that prop. This recipe has NOT been visually verified in a live browser
// as part of this change (this environment is a headless SSH box with no
// browser) — it is a standard, documented recharts pattern rather than a
// fragile hack, but Phase 4 should eyeball it once LineChartWidget/
// BarChartWidget actually consume this provider.
//
// This file only builds the Context/Provider + helpers. Wiring
// LineChartWidget.tsx / BarChartWidget.tsx onto it is Phase 4 — see the
// integration sketch at the bottom of this file.

export type CrosshairMode = "off" | "shared-crosshair" | "shared-tooltip";

export interface CrosshairContextValue {
  mode: CrosshairMode;
  setMode: (mode: CrosshairMode) => void;
  // undefined in "off" mode: pass straight through to recharts'
  // <LineChart syncId={...}> — omitting the prop is how recharts disables
  // cross-chart sync, so callers never need an if/else of their own.
  syncId: string | undefined;
}

const CrosshairContext = createContext<CrosshairContextValue | null>(null);

const SYNC_ID_PREFIX = "hd-dashboard-crosshair";

export interface CrosshairProviderProps {
  children: React.ReactNode;
  // Scopes the syncId to one dashboard instance. Only matters when more
  // than one dashboard can be mounted on the same page at once (e.g. a
  // live view next to a template preview) so their crosshairs never sync
  // with each other; the common single-dashboard-per-page case can omit
  // it entirely.
  dashboardId?: string | null;
  defaultMode?: CrosshairMode;
}

export function CrosshairProvider({
  children,
  dashboardId,
  defaultMode = "shared-tooltip",
}: CrosshairProviderProps) {
  const [mode, setMode] = useState<CrosshairMode>(defaultMode);
  const id = dashboardId ? `${SYNC_ID_PREFIX}-${dashboardId}` : SYNC_ID_PREFIX;

  const value = useMemo<CrosshairContextValue>(
    () => ({
      mode,
      setMode,
      syncId: mode === "off" ? undefined : id,
    }),
    [mode, id],
  );

  return (
    <CrosshairContext.Provider value={value}>
      {children}
    </CrosshairContext.Provider>
  );
}

// Mirrors lib/time-range.tsx's FALLBACK pattern: a chart rendered outside
// any CrosshairProvider (e.g. a Storybook-style isolated preview) just
// never syncs, instead of throwing.
const FALLBACK: CrosshairContextValue = {
  mode: "off",
  setMode: () => {},
  syncId: undefined,
};

export function useCrosshair(): CrosshairContextValue {
  return useContext(CrosshairContext) ?? FALLBACK;
}

// Convenience for the common case — a chart usually only needs the id.
export function useCrosshairSyncId(): string | undefined {
  return useCrosshair().syncId;
}

// Spread onto recharts' <Tooltip .../> in each time-series widget. See the
// mechanism note above for why this is the only piece "shared-crosshair"
// needs beyond the shared syncId.
export function crosshairTooltipContentProps(
  mode: CrosshairMode,
): { content?: () => null } {
  if (mode === "shared-crosshair") return { content: () => null };
  return {};
}

// ---------------------------------------------------------------------------
// Phase 4 integration sketch for LineChartWidget.tsx (NOT applied in this
// phase — widget render logic is out of scope here). Today's code has:
//
//   <LineChart data={rows} syncId="dash" syncMethod="value" ...>
//     ...
//     <Tooltip
//       labelFormatter={...}
//       formatter={...}
//       contentStyle={theme.tooltipContentStyle}
//       labelStyle={theme.tooltipLabelStyle}
//     />
//
// becomes:
//
//   const { mode, syncId } = useCrosshair();
//   ...
//   <LineChart data={rows} syncId={syncId} syncMethod="value" ...>
//     ...
//     <Tooltip
//       labelFormatter={...}
//       formatter={...}
//       contentStyle={theme.tooltipContentStyle}
//       labelStyle={theme.tooltipLabelStyle}
//       {...crosshairTooltipContentProps(mode)}
//     />
//
// and the dashboard shell (wherever DashboardGrid/DashboardPreview is
// mounted, once per rendered dashboard) wraps its panels with:
//
//   <CrosshairProvider dashboardId={dashboard.id}>
//     <DashboardGrid ... />
//   </CrosshairProvider>
//
// BarChartWidget.tsx (no syncId today) picks up the same two lines to join
// the same synced group. A widget that should NEVER participate (e.g. a
// modal preview rendered outside the dashboard's provider) needs no
// special-casing — useCrosshair() falls back to "off" with no provider
// mounted.
