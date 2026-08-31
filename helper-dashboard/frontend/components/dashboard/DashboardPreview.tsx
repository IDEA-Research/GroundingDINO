"use client";

import React from "react";

import type { DashboardSpec, ReviewTrailEvent } from "@/lib/spec-schema";
import { collectRenderProblems, renderWidget } from "@/lib/renderer";
import { TimeRangeProvider } from "@/lib/time-range";
import { TimeRangePicker } from "@/components/dashboard/TimeRangePicker";

export function DashboardPreview({
  spec,
  reviewing = false,
  reviewTrail,
}: {
  spec: DashboardSpec | null;
  reviewing?: boolean;
  // Real pre-output review transcript for THIS spec (home page passes it;
  // the headless evaluator route has none). Drives the review badge —
  // there is no unconditional badge.
  reviewTrail?: ReviewTrailEvent[];
}) {
  if (!spec) {
    return (
      <div className="flex h-full items-center justify-center px-4 text-center text-sm text-ink-faint">
        {reviewing
          ? "正在審查渲染結果…"
          : "尚無儀表板 — 請向 Helper 描述你想要的監控畫面。"}
      </div>
    );
  }

  const problems = collectRenderProblems(spec);
  const columns = spec.layout?.columns || 12;
  const rowHeight = spec.layout?.row_height || 40;

  // Show the badge ONLY when the review loop actually ran and its latest
  // review verdict was "approve" (backend ReviewDecision literals are
  // approve | patch | escalate). No trail, or a non-approve last verdict,
  // renders nothing.
  const lastReview = [...(reviewTrail ?? [])]
    .reverse()
    .find((e) => e.stage === "review" && typeof e.decision === "string");
  const reviewApproved = lastReview?.decision === "approve";

  // TimeRangeProvider lives HERE (not in a page) so both the home page
  // and the headless /dashboard/[id] evaluator route get the shared
  // refresh clock and the view-layer range override.
  return (
    <TimeRangeProvider refreshInterval={spec.refresh_interval}>
    <div className="relative flex h-full flex-col">
      <div className="border-b border-edge px-3 py-2">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <div className="text-sm font-semibold text-ink">{spec.title}</div>
          {reviewApproved ? (
            <span className="rounded border border-ok/40 bg-ok/10 px-1.5 py-0.5 text-2xs uppercase tracking-wide text-ok">
              已審查
            </span>
          ) : null}
          <div className="ml-auto">
            <TimeRangePicker />
          </div>
        </div>
        {spec.description ? (
          <div className="text-xs text-ink-faint">{spec.description}</div>
        ) : null}
        <div className="mt-1 font-mono text-2xs uppercase tracking-wide text-ink-faint">
          更新 {spec.refresh_interval} · {spec.widgets.length} 個元件
        </div>
      </div>

      {problems.length > 0 ? (
        <div className="border-b border-alarm-high/40 bg-alarm-high/10 px-3 py-2 text-xs text-alarm-high">
          {problems.length} 個渲染問題:
          <ul className="list-disc pl-4">
            {problems.map((p) => (
              <li key={p.widget_id}>
                <code className="font-mono">{p.widget_id}</code> — {p.reason}
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
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-canvas/60 backdrop-blur-[1px]">
          <div className="rounded-lg border border-edge bg-surface-2/95 px-4 py-3 text-sm text-ink shadow-lg">
            <div className="mb-0.5 font-medium">輸出前審查進行中…</div>
            <div className="text-xs text-ink-faint">
              Helper 與 Big guy 正在檢查渲染後的儀表板。
            </div>
          </div>
        </div>
      ) : null}
    </div>
    </TimeRangeProvider>
  );
}
