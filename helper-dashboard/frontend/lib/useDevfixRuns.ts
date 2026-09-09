"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { api, describeApiError, isApiError } from "./api";
import {
  DEV_RUN_STATUS_LABELS,
  normalizeDevRunStatus,
  type DevRunMeta,
  type DevRunStatus,
} from "./spec-schema";

// Polls GET /api/devfix/runs (dev-gated) and derives what the shell needs:
//  - `active`  : the run that is queued / running / waiting right now
//  - `current` : `active`, or a run that finished < 10 min ago — what the
//                top-bar 開發者 button and the 目前任務 card show. Anything
//                older is HISTORY and never colors the button (待命).
// Polling only happens while `enabled` (developer unlocked) and the tab
// is visible; without a token the backend answers 403 and nothing is shown.

export type { DevRunMeta, DevRunStatus };
// Legacy aliases kept for older imports.
export type DevfixRunMeta = DevRunMeta;
export type DevfixRunStatus = DevRunStatus;

export const LIST_POLL_MS = 3000;
export const EVENTS_POLL_MS = 2000;
// A run that finished less than this long ago still counts as "current".
export const CURRENT_GRACE_MS = 10 * 60 * 1000;

export type RunStatusOrIdle = DevRunStatus | "idle";
export type StatusTone = "neutral" | "ok" | "warn" | "danger" | "info";

export function runStatus(meta: DevRunMeta | null | undefined): RunStatusOrIdle {
  if (!meta) return "idle";
  const s = normalizeDevRunStatus(meta.status);
  // An unknown on-disk status is a finished run whose outcome we cannot
  // read — show it as 失敗 rather than pretending it is idle.
  return s === "idle" ? "failed" : s;
}

export function runStatusLabel(status: RunStatusOrIdle): string {
  return DEV_RUN_STATUS_LABELS[status] ?? status;
}

export function runStatusTone(status: RunStatusOrIdle): StatusTone {
  switch (status) {
    case "running":
      return "warn";
    case "queued":
      return "info";
    case "waiting":
      return "warn";
    case "success":
      return "ok";
    case "failed":
    case "timeout":
      return "danger";
    case "cancelled":
    case "idle":
    default:
      return "neutral";
  }
}

// In flight on the backend (has a live process or a queue slot). `waiting`
// is TERMINAL server-side (Big Guy stopped and asked the developer;
// finished_at is set) — it needs attention, but nothing is running.
export function isActiveStatus(status: RunStatusOrIdle): boolean {
  return status === "queued" || status === "running";
}

export function needsAttention(status: RunStatusOrIdle): boolean {
  return status === "waiting";
}

export function isTerminalStatus(status: RunStatusOrIdle): boolean {
  return (
    status === "success" ||
    status === "waiting" ||
    status === "failed" ||
    status === "cancelled" ||
    status === "timeout"
  );
}

export function canRetry(status: RunStatusOrIdle): boolean {
  return status === "failed" || status === "timeout" || status === "cancelled";
}

// Backend answers 409 not_active for anything else.
export function canCancel(status: RunStatusOrIdle): boolean {
  return status === "queued" || status === "running";
}

function parseTs(s: string | null | undefined): number | null {
  if (!s) return null;
  const t = Date.parse(s);
  return Number.isFinite(t) ? t : null;
}

export function runStartedAt(meta: DevRunMeta): number | null {
  return parseTs(meta.started_at) ?? parseTs(meta.queued_at);
}

export function runFinishedAt(meta: DevRunMeta): number | null {
  return parseTs(meta.finished_at);
}

// "1分05秒" style elapsed time; ticks while the run is unfinished.
export function elapsedLabel(meta: DevRunMeta, now: number = Date.now()): string {
  const start = runStartedAt(meta);
  if (start === null) return "—";
  const end = runFinishedAt(meta) ?? now;
  const s = Math.max(0, Math.floor((end - start) / 1000));
  const m = Math.floor(s / 60);
  const h = Math.floor(m / 60);
  if (h > 0) return `${h}時${String(m % 60).padStart(2, "0")}分`;
  if (m > 0) return `${m}分${String(s % 60).padStart(2, "0")}秒`;
  return `${s}秒`;
}

// The run to show as 目前任務 (null = 待命): a queued/running run, else the
// newest run that finished (incl. `waiting`) within the grace window. A
// days-old failure or needs_context never colors the button.
export function currentRun(runs: DevRunMeta[], now: number = Date.now()): DevRunMeta | null {
  const active = runs.find((r) => isActiveStatus(runStatus(r)));
  if (active) return active;
  let best: DevRunMeta | null = null;
  let bestT = -Infinity;
  for (const r of runs) {
    const t = runFinishedAt(r);
    if (t !== null && now - t < CURRENT_GRACE_MS && t > bestT) {
      best = r;
      bestT = t;
    }
  }
  return best;
}

export interface DevfixRunsState {
  runs: DevRunMeta[];
  // Queued/running/waiting run, if any.
  active: DevRunMeta | null;
  // Active run, or one finished < 10 min ago. null = 待命.
  current: DevRunMeta | null;
  // True once the first successful list response arrived.
  loaded: boolean;
  // 繁中 reason when the list cannot be read (403 = token rejected).
  error: string | null;
  forbidden: boolean;
  refresh: () => void;
}

export function useDevfixRuns(enabled: boolean = true): DevfixRunsState {
  const [runs, setRuns] = useState<DevRunMeta[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [forbidden, setForbidden] = useState(false);
  const [, setClock] = useState(0);
  const cancelled = useRef(false);

  const poll = useCallback(async () => {
    if (
      typeof document !== "undefined" &&
      document.visibilityState === "hidden"
    ) {
      return;
    }
    try {
      const data = await api.devfixRuns();
      if (cancelled.current) return;
      setRuns(Array.isArray(data.runs) ? data.runs : []);
      setLoaded(true);
      setError(null);
      setForbidden(false);
    } catch (e) {
      if (cancelled.current) return;
      setError(describeApiError(e));
      setForbidden(isApiError(e) && e.status === 403);
    }
  }, []);

  useEffect(() => {
    cancelled.current = false;
    if (!enabled) {
      setRuns([]);
      setLoaded(false);
      setError(null);
      setForbidden(false);
      return;
    }
    poll();
    const t = setInterval(poll, LIST_POLL_MS);
    return () => {
      cancelled.current = true;
      clearInterval(t);
    };
  }, [enabled, poll]);

  // Re-evaluate `current` once a minute so a finished run drops back to
  // 待命 after the grace window even when the list itself is unchanged.
  useEffect(() => {
    if (!enabled) return;
    const t = setInterval(() => setClock((n) => n + 1), 60 * 1000);
    return () => clearInterval(t);
  }, [enabled]);

  const active = runs.find((r) => isActiveStatus(runStatus(r))) ?? null;
  const current = currentRun(runs);

  return { runs, active, current, loaded, error, forbidden, refresh: poll };
}
