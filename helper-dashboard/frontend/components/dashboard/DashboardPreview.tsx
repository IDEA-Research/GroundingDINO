"use client";

import React from "react";

import type { DashboardSpec } from "@/lib/spec-schema";
import { collectRenderProblems, renderWidget } from "@/lib/renderer";

export function DashboardPreview({
  spec,
  reviewing = false,
}: {
  spec: DashboardSpec | null;
  reviewing?: boolean;
}) {
  if (!spec) {
    return (
      <div className="flex h-full items-center justify-center text-slate-500">
        {reviewing ? "Reviewing rendered dashboard…" : "No dashboard yet — ask Helper to build one."}
      </div>
    );
  }

  const problems = collectRenderProblems(spec);
  const columns = spec.layout?.columns || 12;
  const rowHeight = spec.layout?.row_height || 40;

  return (
    <div className="relative flex h-full flex-col">
      <div className="border-b border-slate-800 px-3 py-2">
        <div className="flex items-center gap-2">
          <div className="text-sm font-semibold">{spec.title}</div>
          <span className="rounded bg-emerald-700 px-1.5 py-0.5 text-[10px] uppercase text-emerald-100">
            reviewed
          </span>
        </div>
        {spec.description ? (
          <div className="text-xs text-slate-400">{spec.description}</div>
        ) : null}
        <div className="mt-1 text-[10px] uppercase tracking-wide text-slate-500">
          refresh {spec.refresh_interval} · {spec.widgets.length} widget
          {spec.widgets.length === 1 ? "" : "s"}
        </div>
      </div>

      {problems.length > 0 ? (
        <div className="border-b border-rose-900 bg-rose-950/40 px-3 py-2 text-xs text-rose-200">
          {problems.length} render problem
          {problems.length === 1 ? "" : "s"}:
          <ul className="list-disc pl-4">
            {problems.map((p) => (
              <li key={p.widget_id}>
                <code>{p.widget_id}</code> — {p.reason}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className="flex-1 overflow-auto p-3">
        <div
          className="grid gap-3"
          style={{
            gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
            gridAutoRows: `${rowHeight}px`,
          }}
        >
          {spec.widgets.map((w) => (
            <div
              key={w.id}
              style={{
                gridColumn: `${w.position.x + 1} / span ${w.position.w}`,
                gridRow: `${w.position.y + 1} / span ${w.position.h}`,
              }}
            >
              {renderWidget(w)}
            </div>
          ))}
        </div>
      </div>

      {reviewing ? (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-slate-900/60 backdrop-blur-[1px]">
          <div className="rounded-lg bg-slate-800/90 px-4 py-3 text-sm text-slate-200 shadow-lg">
            <div className="mb-0.5 font-medium">Pre-output review running…</div>
            <div className="text-xs text-slate-400">
              Helper and Big guy are checking the rendered dashboard before it's returned.
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
