"use client";

import React, { useEffect, useRef, useState } from "react";

import {
  useDevfixRuns,
  type DevfixReadResponse,
  type DevfixRunMeta,
  type DevfixRunStatus,
} from "@/lib/useDevfixRuns";

// Live monitor for devfix ("Big Guy") runs: when the dashboard needs
// Big Guy's support to fix its own code, this panel tails the run log
// so the user can see what he thinks and does, right now. Fixed
// bottom-right so it never fights the dashboard grid; collapsed to an
// unobtrusive status dot when nothing is happening.

const TAIL_POLL_MS = 2000;

function chipFor(status: DevfixRunStatus): { label: string; cls: string } {
  switch (status) {
    case "running":
      return {
        label: "running",
        cls: "animate-pulse border-amber-500/60 bg-amber-500/10 text-amber-300",
      };
    case "resolved":
      return {
        label: "resolved",
        cls: "border-emerald-500/60 bg-emerald-500/10 text-emerald-300",
      };
    case "needs_context":
      return {
        label: "needs context",
        cls: "border-slate-500/60 bg-slate-500/10 text-slate-300",
      };
    default: // rejected | failed
      return {
        label: status,
        cls: "border-red-500/60 bg-red-500/10 text-red-300",
      };
  }
}

function dotFor(status: DevfixRunStatus | null): string {
  switch (status) {
    case "running":
      return "bg-amber-400 animate-pulse";
    case "resolved":
      return "bg-emerald-500";
    case "needs_context":
      return "bg-slate-500";
    case "rejected":
    case "failed":
      return "bg-red-500";
    default:
      return "bg-slate-600";
  }
}

function elapsedLabel(meta: DevfixRunMeta): string {
  const start = Date.parse(meta.started_at);
  if (!Number.isFinite(start)) return "";
  const end = meta.finished_at ? Date.parse(meta.finished_at) : Date.now();
  const s = Math.max(0, Math.floor((end - start) / 1000));
  const m = Math.floor(s / 60);
  return m > 0 ? `${m}m ${s % 60}s` : `${s}s`;
}

export function DevFixMonitor() {
  const { runs, active } = useDevfixRuns();
  const [open, setOpen] = useState(false);
  const [tailMeta, setTailMeta] = useState<DevfixRunMeta | null>(null);
  const [lines, setLines] = useState<string[]>([]);
  const cursor = useRef(0);
  const tailedRunId = useRef<string | null>(null);
  const lastRunningId = useRef<string | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [, setClock] = useState(0);

  // The panel shows the active run; with none active, the most recent
  // finished one (so clicking the dot reveals what happened last).
  const shown = active ?? (runs.length > 0 ? runs[0] : null);
  const shownRunId = shown?.run_id ?? null;
  const display =
    tailMeta && shown && tailMeta.run_id === shown.run_id ? tailMeta : shown;
  const isRunning = display?.status === "running";

  // Auto-expand the moment a run transitions to running.
  useEffect(() => {
    const id = active && active.status === "running" ? active.run_id : null;
    if (id && lastRunningId.current !== id) setOpen(true);
    lastRunningId.current = id;
  }, [active]);

  // Tail the shown run's log while the panel is open; polling stops on
  // its own once the run reports a terminal status (the final poll
  // already carries the remaining lines).
  useEffect(() => {
    if (!open || !shownRunId) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    if (tailedRunId.current !== shownRunId) {
      tailedRunId.current = shownRunId;
      cursor.current = 0;
      setLines([]);
      setTailMeta(null);
    }

    async function tail() {
      try {
        const res = await fetch(
          `/api/devfix/runs/${encodeURIComponent(shownRunId as string)}` +
            `?after=${cursor.current}`,
          { cache: "no-store" },
        );
        if (!res.ok || stopped) return;
        const data = (await res.json()) as DevfixReadResponse;
        if (stopped) return;
        cursor.current = data.next_cursor;
        if (data.lines.length > 0) {
          setLines((prev) => [...prev, ...data.lines]);
        }
        setTailMeta(data.meta);
        if (data.meta.status === "running") {
          timer = setTimeout(tail, TAIL_POLL_MS);
        }
      } catch {
        if (!stopped) timer = setTimeout(tail, TAIL_POLL_MS);
      }
    }

    tail();
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [open, shownRunId]);

  // Auto-scroll the log to the newest line.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines]);

  // 1s re-render while running so the elapsed label ticks.
  useEffect(() => {
    if (!isRunning) return;
    const t = setInterval(() => setClock((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, [isRunning]);

  if (!display) {
    return (
      <div className="fixed bottom-3 right-3 z-50">
        <div
          className="flex items-center gap-1.5 rounded-full border border-slate-800 bg-slate-900/80 px-2.5 py-1 text-[10px] text-slate-500"
          title="Big Guy 待命中 — 尚無 devfix 紀錄"
        >
          <span className="h-1.5 w-1.5 rounded-full bg-slate-600" />
          Big Guy
        </div>
      </div>
    );
  }

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="fixed bottom-3 right-3 z-50 flex items-center gap-1.5 rounded-full border border-slate-700 bg-slate-900/90 px-2.5 py-1 text-[10px] text-slate-400 hover:border-slate-500 hover:text-slate-200"
        title={`Big Guy · ${display.ticket_id} · ${display.status}`}
      >
        <span
          className={`h-1.5 w-1.5 rounded-full ${dotFor(display.status)}`}
        />
        Big Guy
      </button>
    );
  }

  const chip = chipFor(display.status);

  return (
    <div className="fixed bottom-3 right-3 z-50 flex max-h-[60vh] w-[30rem] max-w-[calc(100vw-1.5rem)] flex-col overflow-hidden rounded-lg border border-slate-700 bg-slate-950/95 shadow-xl shadow-black/50">
      <div className="flex items-center gap-2 border-b border-slate-800 bg-slate-900/80 px-3 py-2">
        <span className="min-w-0 truncate text-xs font-semibold text-slate-200">
          🔧 {isRunning ? "Big Guy 工作中" : "Big Guy"} · {display.ticket_id} ·{" "}
          {elapsedLabel(display)}
        </span>
        <span
          className={`shrink-0 whitespace-nowrap rounded-full border px-1.5 py-0.5 text-[10px] ${chip.cls}`}
        >
          {chip.label}
        </span>
        <button
          type="button"
          onClick={() => setOpen(false)}
          className="ml-auto shrink-0 rounded px-1.5 text-xs text-slate-500 hover:bg-slate-800 hover:text-slate-200"
          title="收合"
        >
          —
        </button>
      </div>
      {display.instruction_preview ? (
        <div className="truncate border-b border-slate-800 px-3 py-1.5 text-[10px] text-slate-500">
          {display.instruction_preview}
        </div>
      ) : null}
      <div
        ref={scrollRef}
        className="min-h-[6rem] flex-1 overflow-y-auto px-3 py-2 font-mono text-[11px] leading-relaxed text-slate-300"
      >
        {lines.length === 0 ? (
          <div className="text-slate-600">等待日誌輸出… / waiting for log…</div>
        ) : (
          lines.map((line, i) => (
            <div key={i} className="whitespace-pre-wrap break-words">
              {line}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
