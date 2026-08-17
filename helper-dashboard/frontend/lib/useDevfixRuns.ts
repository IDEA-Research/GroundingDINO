"use client";

import { useEffect, useRef, useState } from "react";

export type DevfixRunStatus =
  | "running"
  | "resolved"
  | "rejected"
  | "needs_context"
  | "failed";

export interface DevfixRunMeta {
  run_id: string;
  ticket_id: string;
  status: DevfixRunStatus;
  started_at: string;
  finished_at: string | null;
  error_kind?: string | null;
  model: string;
  instruction_preview: string;
}

export interface DevfixReadResponse {
  meta: DevfixRunMeta;
  lines: string[];
  next_cursor: number;
}

const LIST_POLL_MS = 3000;
// A run that finished less than this long ago still counts as "active"
// so its outcome stays on screen instead of vanishing mid-read.
const ACTIVE_GRACE_MS = 10 * 60 * 1000;

export interface DevfixRunsState {
  runs: DevfixRunMeta[];
  /** Latest run while it is running or freshly finished (<10 min). */
  active: DevfixRunMeta | null;
}

export function useDevfixRuns(): DevfixRunsState {
  const [runs, setRuns] = useState<DevfixRunMeta[]>([]);
  const cancelled = useRef(false);

  useEffect(() => {
    cancelled.current = false;

    async function poll() {
      // Pause polling while the tab is hidden; the next visible tick
      // catches up.
      if (
        typeof document !== "undefined" &&
        document.visibilityState === "hidden"
      ) {
        return;
      }
      try {
        const res = await fetch("/api/devfix/runs", { cache: "no-store" });
        if (!res.ok || cancelled.current) return;
        const data = (await res.json()) as { runs?: DevfixRunMeta[] };
        if (!cancelled.current) setRuns(data.runs ?? []);
      } catch {
        // Backend unreachable — keep the last known state; the status
        // dot simply stops updating until the backend is back.
      }
    }

    poll();
    const t = setInterval(poll, LIST_POLL_MS);
    return () => {
      cancelled.current = true;
      clearInterval(t);
    };
  }, []);

  const latest = runs.length > 0 ? runs[0] : null;
  const finishedAt = latest?.finished_at ? Date.parse(latest.finished_at) : NaN;
  const active =
    latest &&
    (latest.status === "running" ||
      (Number.isFinite(finishedAt) && Date.now() - finishedAt < ACTIVE_GRACE_MS))
      ? latest
      : null;

  return { runs, active };
}
