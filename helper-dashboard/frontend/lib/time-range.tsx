"use client";

// Shared refresh clock + global time-range selection (view-layer only).
//
// One provider owns a single interval driven by the spec's
// refresh_interval. Each tick computes endEpoch = floor(now) exactly ONCE
// and bumps a tick counter, so every widget refetches against the same end
// (contract C5 — the backend floors it to each widget's step).
//
// Two SEPARATE controls (spec 五.3):
//   - range: single-select. "spec" (依儀表板設定) = every widget uses its own
//     spec range/step; a preset = every range widget queries that window.
//     The DashboardSpec is never mutated.
//   - autoRefresh: on/off + interval readout + refreshNow(). Turning it off
//     stops the interval; refreshNow() bumps the clock immediately.
//
// lastSuccessAt / inFlightCount are fed by useQueryData through trackFetch
// so 最後查詢 reflects data ARRIVAL, never the scheduled tick.

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import type { QuerySpec } from "./spec-schema";

export type RangeExpr =
  | "spec"
  | "now-5m"
  | "now-15m"
  | "now-1h"
  | "now-6h"
  | "now-24h"
  | "now-72h";

export interface RangePreset {
  expr: RangeExpr;
  label: string;
  // null for "spec" (each widget keeps its own window).
  rangeSeconds: number | null;
  // Client-computed step for the override window. null = keep the
  // widget's own spec step (fallback 30s) — used for ranges <= 1h.
  stepSeconds: number | null;
}

export const RANGE_PRESETS: RangePreset[] = [
  { expr: "spec", label: "依儀表板設定", rangeSeconds: null, stepSeconds: null },
  { expr: "now-5m", label: "5分", rangeSeconds: 300, stepSeconds: null },
  { expr: "now-15m", label: "15分", rangeSeconds: 900, stepSeconds: null },
  { expr: "now-1h", label: "1小時", rangeSeconds: 3600, stepSeconds: null },
  { expr: "now-6h", label: "6小時", rangeSeconds: 21600, stepSeconds: 120 },
  { expr: "now-24h", label: "24小時", rangeSeconds: 86400, stepSeconds: 300 },
  { expr: "now-72h", label: "72小時", rangeSeconds: 259200, stepSeconds: 900 },
];

export const MIN_REFRESH_SECONDS = 5;

// Parse a duration expression ('30s' / '5m' / '1h' / '2d' / '500ms') to
// seconds. null when unparseable.
export function parseDurationExpr(
  s: string | null | undefined,
): number | null {
  if (!s) return null;
  const m = /^(\d+(?:\.\d+)?)\s*(ms|s|m|h|d|w)$/i.exec(s.trim());
  if (!m) return null;
  const n = parseFloat(m[1]);
  if (!Number.isFinite(n) || n <= 0) return null;
  switch (m[2].toLowerCase()) {
    case "ms":
      return n / 1000;
    case "s":
      return n;
    case "m":
      return n * 60;
    case "h":
      return n * 3600;
    case "d":
      return n * 86400;
    case "w":
      return n * 604800;
    default:
      return null;
  }
}

// Spec refresh_interval -> seconds. Accepts ms (backend clamps to >= 5s
// too); anything unparseable falls back to 30s. Never below 5s.
export function parseRefreshInterval(s: string | null | undefined): number {
  const secs = parseDurationExpr(s);
  if (secs === null) return 30;
  return Math.max(MIN_REFRESH_SECONDS, Math.round(secs));
}

// Human label for a window length: 300 -> 5分, 3600 -> 1小時, 259200 -> 72小時.
export function formatDurationLabel(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) {
    return "—";
  }
  const s = Math.round(seconds);
  if (s < 60) return `${s}秒`;
  if (s < 3600) {
    const m = Math.floor(s / 60);
    const r = s % 60;
    return r ? `${m}分${r}秒` : `${m}分`;
  }
  if (s <= 72 * 3600) {
    const h = Math.floor(s / 3600);
    const r = Math.round((s % 3600) / 60);
    return r ? `${h}小時${r}分` : `${h}小時`;
  }
  const d = s / 86400;
  return Number.isInteger(d) ? `${d}天` : `${Math.round(d * 10) / 10}天`;
}

export interface EffectiveRange {
  mode: "spec" | "override";
  expr: RangeExpr;
  // null in spec mode (each widget decides).
  seconds: number | null;
  label: string;
}

export interface AutoRefreshState {
  enabled: boolean;
  intervalSeconds: number;
  setEnabled: (enabled: boolean) => void;
}

export interface TimeRangeContextValue {
  // Increments on every refresh beat, range change and refreshNow().
  tick: number;
  // Epoch seconds, computed once per tick and shared by every widget.
  // null only when no provider is mounted (then no "end" is sent).
  endEpoch: number | null;
  range: EffectiveRange;
  setRange: (exprOrSpec: RangeExpr | "spec") => void;
  presets: RangePreset[];
  autoRefresh: AutoRefreshState;
  refreshNow: () => void;
  // Widget fetches currently in flight.
  inFlightCount: number;
  // Epoch ms of the most recent successful widget response.
  lastSuccessAt: number | null;
  // Resolved override window; both null in spec mode. overrideStepSeconds
  // null = keep the widget's spec step (or 30s).
  overrideRangeSeconds: number | null;
  overrideStepSeconds: number | null;
  // Fed by useQueryData — not for UI code.
  trackFetch: { start: () => void; finish: (ok: boolean) => void };
  hasProvider: boolean;
}

const noop = () => {};

const SPEC_RANGE: EffectiveRange = {
  mode: "spec",
  expr: "spec",
  seconds: null,
  label: "依儀表板設定",
};

// Safe fallback for trees without a provider: single fetch, no "end" sent,
// spec range, no auto refresh.
const FALLBACK: TimeRangeContextValue = {
  tick: 0,
  endEpoch: null,
  range: SPEC_RANGE,
  setRange: noop,
  presets: RANGE_PRESETS,
  autoRefresh: { enabled: false, intervalSeconds: 30, setEnabled: noop },
  refreshNow: noop,
  inFlightCount: 0,
  lastSuccessAt: null,
  overrideRangeSeconds: null,
  overrideStepSeconds: null,
  trackFetch: { start: noop, finish: noop },
  hasProvider: false,
};

const TimeRangeContext = createContext<TimeRangeContextValue | null>(null);

export function useTimeRange(): TimeRangeContextValue {
  return useContext(TimeRangeContext) ?? FALLBACK;
}

// True when a TimeRangeProvider is mounted above the caller. DashboardPreview
// uses this to self-provide one on the headless evaluator route only.
export function useHasTimeRangeProvider(): boolean {
  return useContext(TimeRangeContext) !== null;
}

function presetFor(expr: RangeExpr): RangePreset {
  return RANGE_PRESETS.find((p) => p.expr === expr) ?? RANGE_PRESETS[0];
}

export function TimeRangeProvider({
  refreshInterval,
  children,
}: {
  refreshInterval?: string | null;
  children: React.ReactNode;
}) {
  const intervalSeconds = useMemo(
    () => parseRefreshInterval(refreshInterval),
    [refreshInterval],
  );

  const [clock, setClock] = useState<{ tick: number; endEpoch: number }>(
    () => ({ tick: 0, endEpoch: Math.floor(Date.now() / 1000) }),
  );
  const [expr, setExpr] = useState<RangeExpr>("spec");
  const [enabled, setEnabled] = useState(true);
  const [inFlightCount, setInFlight] = useState(0);
  const [lastSuccessAt, setLastSuccessAt] = useState<number | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const bump = useCallback(() => {
    // Computed ONCE per tick; every widget shares this end.
    const endEpoch = Math.floor(Date.now() / 1000);
    setClock((prev) => ({ tick: prev.tick + 1, endEpoch }));
  }, []);

  useEffect(() => {
    if (!enabled) return;
    const id = setInterval(bump, intervalSeconds * 1000);
    return () => clearInterval(id);
  }, [enabled, intervalSeconds, bump]);

  const setRange = useCallback(
    (next: RangeExpr | "spec") => {
      setExpr(presetFor(next).expr);
      // A window change is an immediate refresh against one fresh end.
      bump();
    },
    [bump],
  );

  const trackFetch = useMemo(
    () => ({
      start: () => {
        if (mountedRef.current) setInFlight((n) => n + 1);
      },
      finish: (ok: boolean) => {
        if (!mountedRef.current) return;
        setInFlight((n) => Math.max(0, n - 1));
        if (ok) setLastSuccessAt(Date.now());
      },
    }),
    [],
  );

  const value = useMemo<TimeRangeContextValue>(() => {
    const preset = presetFor(expr);
    const range: EffectiveRange =
      preset.expr === "spec"
        ? SPEC_RANGE
        : {
            mode: "override",
            expr: preset.expr,
            seconds: preset.rangeSeconds,
            label: preset.label,
          };
    return {
      tick: clock.tick,
      endEpoch: clock.endEpoch,
      range,
      setRange,
      presets: RANGE_PRESETS,
      autoRefresh: { enabled, intervalSeconds, setEnabled },
      refreshNow: bump,
      inFlightCount,
      lastSuccessAt,
      overrideRangeSeconds: preset.expr === "spec" ? null : preset.rangeSeconds,
      overrideStepSeconds: preset.expr === "spec" ? null : preset.stepSeconds,
      trackFetch,
      hasProvider: true,
    };
  }, [
    clock,
    expr,
    setRange,
    enabled,
    intervalSeconds,
    bump,
    inFlightCount,
    lastSuccessAt,
    trackFetch,
  ]);

  return (
    <TimeRangeContext.Provider value={value}>
      {children}
    </TimeRangeContext.Provider>
  );
}

// The window a RANGE widget is effectively showing: the override when one
// is active, else the widget's own spec range. null for instant queries
// (no window to badge).
export function effectiveRangeFor(
  range: EffectiveRange,
  query: QuerySpec | null | undefined,
): EffectiveRange | null {
  if (!query || query.query_type !== "range") return null;
  if (range.mode === "override") return range;
  const secs = parseDurationExpr(query.range) ?? 3600;
  return {
    mode: "spec",
    expr: "spec",
    seconds: secs,
    label: formatDurationLabel(secs),
  };
}

export function useEffectiveRange(
  query: QuerySpec | null | undefined,
): EffectiveRange | null {
  const { range } = useTimeRange();
  return useMemo(() => effectiveRangeFor(range, query), [range, query]);
}
