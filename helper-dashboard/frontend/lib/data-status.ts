"use client";

// Dashboard-level data status store. Every query-driven widget registers
// its own {status, lastSuccessAt, sampleTs} (renderer QueryGate does this
// via useRegisterDataStatus); the DataStatusStrip reads ONE aggregate:
// receiving / lastQueryAt / lastSampleAt / worstState / counts. A widget
// that unmounts unregisters, so switching dashboards never leaves ghosts.

import { useEffect, useSyncExternalStore } from "react";

import type { QueryStatus, WidgetData } from "./useQueryData";

export interface WidgetStatusEntry {
  id: string;
  status: QueryStatus;
  // Client arrival time (epoch ms) of the last successful response.
  lastSuccessAt: number | null;
  // Newest measurement behind the widget (epoch seconds), if known.
  sampleTs: number | null;
  simulated?: boolean;
  simScenario?: string | null;
  reason?: string | null;
  budgetSeconds?: number | null;
}

export type WorstState = QueryStatus | "idle";

export interface DashboardDataStatus {
  // At least one widget currently holds fresh, real data (status "ok").
  receiving: boolean;
  // max lastSuccessAt over widgets (epoch ms), null when nothing succeeded.
  lastQueryAt: number | null;
  // max sampleTs over widgets (epoch seconds), null when unknown.
  lastSampleAt: number | null;
  worstState: WorstState;
  // Reason text of the worst widget (e.g. 更新失敗 / 來源中斷), if any.
  worstReason: string | null;
  counts: Record<QueryStatus, number>;
  total: number;
  anySimulated: boolean;
  simScenario: string | null;
  // Largest freshness budget reported by any widget (seconds).
  budgetSeconds: number | null;
}

const SEVERITY: Record<QueryStatus, number> = {
  ok: 0,
  loading: 1,
  empty: 2,
  stale: 3,
  error: 4,
  offline: 5,
};

function emptyCounts(): Record<QueryStatus, number> {
  return { loading: 0, ok: 0, stale: 0, empty: 0, offline: 0, error: 0 };
}

const EMPTY: DashboardDataStatus = {
  receiving: false,
  lastQueryAt: null,
  lastSampleAt: null,
  worstState: "idle",
  worstReason: null,
  counts: emptyCounts(),
  total: 0,
  anySimulated: false,
  simScenario: null,
  budgetSeconds: null,
};

const entries = new Map<string, WidgetStatusEntry>();
const listeners = new Set<() => void>();
let snapshot: DashboardDataStatus = EMPTY;
let dirty = false;

function compute(): DashboardDataStatus {
  if (entries.size === 0) return EMPTY;
  const counts = emptyCounts();
  let lastQueryAt: number | null = null;
  let lastSampleAt: number | null = null;
  let worst: WidgetStatusEntry | null = null;
  let anySimulated = false;
  let simScenario: string | null = null;
  let budget: number | null = null;
  entries.forEach((e) => {
    counts[e.status] += 1;
    if (e.lastSuccessAt !== null) {
      lastQueryAt = lastQueryAt === null ? e.lastSuccessAt : Math.max(lastQueryAt, e.lastSuccessAt);
    }
    if (e.sampleTs !== null) {
      lastSampleAt = lastSampleAt === null ? e.sampleTs : Math.max(lastSampleAt, e.sampleTs);
    }
    if (!worst || SEVERITY[e.status] > SEVERITY[worst.status]) worst = e;
    if (e.simulated) {
      anySimulated = true;
      if (!simScenario && e.simScenario) simScenario = e.simScenario;
    }
    if (typeof e.budgetSeconds === "number") {
      budget = budget === null ? e.budgetSeconds : Math.max(budget, e.budgetSeconds);
    }
  });
  const w = worst as WidgetStatusEntry | null;
  return {
    receiving: counts.ok > 0,
    lastQueryAt,
    lastSampleAt,
    worstState: w ? w.status : "idle",
    worstReason: w?.reason ?? null,
    counts,
    total: entries.size,
    anySimulated,
    simScenario,
    budgetSeconds: budget,
  };
}

function emit() {
  dirty = true;
  listeners.forEach((l) => l());
}

function sameEntry(a: WidgetStatusEntry, b: WidgetStatusEntry): boolean {
  return (
    a.status === b.status &&
    a.lastSuccessAt === b.lastSuccessAt &&
    a.sampleTs === b.sampleTs &&
    Boolean(a.simulated) === Boolean(b.simulated) &&
    (a.simScenario ?? null) === (b.simScenario ?? null) &&
    (a.reason ?? null) === (b.reason ?? null) &&
    (a.budgetSeconds ?? null) === (b.budgetSeconds ?? null)
  );
}

export function registerWidgetStatus(entry: WidgetStatusEntry): void {
  const prev = entries.get(entry.id);
  if (prev && sameEntry(prev, entry)) return;
  entries.set(entry.id, entry);
  emit();
}

export function unregisterWidgetStatus(id: string): void {
  if (entries.delete(id)) emit();
}

export function resetDataStatusStore(): void {
  if (entries.size === 0) return;
  entries.clear();
  emit();
}

export function getDashboardDataStatus(): DashboardDataStatus {
  if (dirty) {
    snapshot = compute();
    dirty = false;
  }
  return snapshot;
}

export function subscribeDashboardDataStatus(cb: () => void): () => void {
  listeners.add(cb);
  return () => {
    listeners.delete(cb);
  };
}

function getServerSnapshot(): DashboardDataStatus {
  return EMPTY;
}

// Aggregate for the DataStatusStrip / header. Re-renders only when a
// widget's registered status actually changes.
export function useDashboardDataStatus(): DashboardDataStatus {
  return useSyncExternalStore(
    subscribeDashboardDataStatus,
    getDashboardDataStatus,
    getServerSnapshot,
  );
}

// Newest measurement behind a WidgetData (epoch seconds), if known.
export function sampleTsOf(data: WidgetData): number | null {
  if (data.freshness && typeof data.freshness.sample_ts === "number") {
    return data.freshness.sample_ts;
  }
  for (let i = data.points.length - 1; i >= 0; i -= 1) {
    const p = data.points[i];
    if (p.v !== null) return p.t;
  }
  return null;
}

// Called by the renderer's QueryGate for every query-driven widget.
export function useRegisterDataStatus(id: string, data: WidgetData): void {
  const sampleTs = sampleTsOf(data);
  const budget = data.freshness?.budget_seconds ?? null;
  useEffect(() => {
    registerWidgetStatus({
      id,
      status: data.status,
      lastSuccessAt: data.lastSuccessAt,
      sampleTs,
      simulated: data.simulated,
      simScenario: data.simScenario,
      reason: data.reason,
      budgetSeconds: budget,
    });
  }, [
    id,
    data.status,
    data.lastSuccessAt,
    sampleTs,
    data.simulated,
    data.simScenario,
    data.reason,
    budget,
  ]);
  useEffect(() => {
    return () => unregisterWidgetStatus(id);
  }, [id]);
}
