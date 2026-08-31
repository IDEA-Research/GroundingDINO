"use client";

// Shared refresh clock + global time-range override (view-layer only).
//
// One provider (mounted inside DashboardPreview so both the home page and
// the headless /dashboard/[id] evaluator route are covered) owns a single
// setInterval driven by the spec's refresh_interval. Each tick computes
// endEpoch = Math.floor(Date.now() / 1000) exactly ONCE and bumps a tick
// counter, so every widget refetches against the same end (contract C5 —
// the backend floors it to each widget's step).
//
// The range override NEVER mutates the DashboardSpec: mode "auto" means
// each widget uses its own spec range/step; mode "override" means all
// time-series widgets query with the override window and a client-computed
// step. JsonInspector, saved entries, and chat patches keep operating on
// the untouched spec.

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

export type RangeExpr =
  | "now-5m"
  | "now-15m"
  | "now-1h"
  | "now-6h"
  | "now-24h"
  | "now-72h";

export type RangeOverride =
  | { mode: "auto" }
  | { mode: "override"; expr: RangeExpr };

export interface RangePreset {
  expr: RangeExpr;
  label: string;
  rangeSeconds: number;
  // Client-computed step for the override window. null = keep the
  // widget's own spec step (fallback 30s) — used for ranges <= 1h.
  stepSeconds: number | null;
}

export const RANGE_PRESETS: RangePreset[] = [
  { expr: "now-5m", label: "5分", rangeSeconds: 300, stepSeconds: null },
  { expr: "now-15m", label: "15分", rangeSeconds: 900, stepSeconds: null },
  { expr: "now-1h", label: "1小時", rangeSeconds: 3600, stepSeconds: null },
  { expr: "now-6h", label: "6小時", rangeSeconds: 21600, stepSeconds: 120 },
  { expr: "now-24h", label: "24小時", rangeSeconds: 86400, stepSeconds: 300 },
  { expr: "now-72h", label: "72小時", rangeSeconds: 259200, stepSeconds: 900 },
];

// Parse a spec refresh_interval ('30s' / '5m' / '1h' pattern) to seconds.
// Anything unparseable falls back to the backend default of 30s.
export function parseRefreshInterval(s: string | null | undefined): number {
  if (!s) return 30;
  const m = /^(\d+)(s|m|h)$/.exec(s.trim());
  if (!m) return 30;
  const n = parseInt(m[1], 10);
  if (!Number.isFinite(n) || n <= 0) return 30;
  if (m[2] === "s") return n;
  if (m[2] === "m") return n * 60;
  return n * 3600;
}

export interface TimeRangeContextValue {
  // Increments on every refresh beat (and on override changes).
  tick: number;
  // Epoch seconds, computed once per tick and shared by every widget.
  // null only when no provider is mounted (then no "end" is sent).
  endEpoch: number | null;
  refreshSeconds: number;
  // Epoch seconds of the latest tick — drives the 最後更新 readout.
  lastUpdated: number | null;
  override: RangeOverride;
  setOverride: (o: RangeOverride) => void;
  // Resolved override window; both null when mode === "auto".
  // overrideStepSeconds null = keep the widget's spec step (or 30s).
  overrideRangeSeconds: number | null;
  overrideStepSeconds: number | null;
  presets: RangePreset[];
}

const noop = () => {};

// Safe fallback for trees without a provider: behaves like the pre-refresh
// world (single fetch, no "end" sent, no override).
const FALLBACK: TimeRangeContextValue = {
  tick: 0,
  endEpoch: null,
  refreshSeconds: 30,
  lastUpdated: null,
  override: { mode: "auto" },
  setOverride: noop,
  overrideRangeSeconds: null,
  overrideStepSeconds: null,
  presets: RANGE_PRESETS,
};

const TimeRangeContext = createContext<TimeRangeContextValue | null>(null);

export function useTimeRange(): TimeRangeContextValue {
  return useContext(TimeRangeContext) ?? FALLBACK;
}

export function TimeRangeProvider({
  refreshInterval,
  children,
}: {
  refreshInterval?: string | null;
  children: React.ReactNode;
}) {
  const refreshSeconds = useMemo(
    () => parseRefreshInterval(refreshInterval),
    [refreshInterval],
  );

  const [clock, setClock] = useState<{ tick: number; endEpoch: number }>(
    () => ({ tick: 0, endEpoch: Math.floor(Date.now() / 1000) }),
  );
  const [override, setOverrideState] = useState<RangeOverride>({
    mode: "auto",
  });

  useEffect(() => {
    const id = setInterval(() => {
      // Computed ONCE per tick; every widget shares this end.
      const endEpoch = Math.floor(Date.now() / 1000);
      setClock((prev) => ({ tick: prev.tick + 1, endEpoch }));
    }, refreshSeconds * 1000);
    return () => clearInterval(id);
  }, [refreshSeconds]);

  const setOverride = useCallback((o: RangeOverride) => {
    setOverrideState(o);
    // A window change is an immediate refresh: bump the shared clock so
    // every widget refetches against one fresh end.
    const endEpoch = Math.floor(Date.now() / 1000);
    setClock((prev) => ({ tick: prev.tick + 1, endEpoch }));
  }, []);

  const value = useMemo<TimeRangeContextValue>(() => {
    const preset =
      override.mode === "override"
        ? RANGE_PRESETS.find((p) => p.expr === override.expr) ?? null
        : null;
    return {
      tick: clock.tick,
      endEpoch: clock.endEpoch,
      refreshSeconds,
      lastUpdated: clock.endEpoch,
      override,
      setOverride,
      overrideRangeSeconds: preset ? preset.rangeSeconds : null,
      overrideStepSeconds: preset ? preset.stepSeconds : null,
      presets: RANGE_PRESETS,
    };
  }, [clock, refreshSeconds, override, setOverride]);

  return (
    <TimeRangeContext.Provider value={value}>
      {children}
    </TimeRangeContext.Provider>
  );
}
