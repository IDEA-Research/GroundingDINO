"use client";

import React, { useState } from "react";

import { api } from "@/lib/api";
import type {
  BrowserEvaluationReport,
  DashboardSpec,
  PatchSpec,
  ReviewTrailEvent,
} from "@/lib/spec-schema";

type Tab = "spec" | "patch" | "evaluation" | "review";

// Developer panel: raw spec/patch/review/evaluation views in 12px mono.
// On narrow screens it opens as an overlay behind the 開發者 toggle —
// `onClose` renders the overlay's close button.
export function JsonInspector({
  spec,
  lastPatch,
  warnings,
  reviewTrail,
  onClose,
}: {
  spec: DashboardSpec | null;
  lastPatch: PatchSpec | null;
  warnings: string[];
  reviewTrail: ReviewTrailEvent[];
  onClose?: () => void;
}) {
  const [tab, setTab] = useState<Tab>("spec");
  const [evalReport, setEvalReport] = useState<BrowserEvaluationReport | null>(
    null,
  );
  const [evaluating, setEvaluating] = useState(false);
  const [evalError, setEvalError] = useState<string | null>(null);

  async function runEvaluate() {
    if (!spec) return;
    setEvaluating(true);
    setEvalError(null);
    try {
      const r = await api.evaluate(spec.dashboard_id);
      setEvalReport(r.report);
      setTab("evaluation");
    } catch (e) {
      setEvalError(e instanceof Error ? e.message : String(e));
    } finally {
      setEvaluating(false);
    }
  }

  return (
    <div className="flex h-full flex-col bg-canvas">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-edge px-3 py-2 text-xs">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-2xs uppercase tracking-wide text-ink-faint">
            開發者
          </span>
          <TabBtn active={tab === "spec"} onClick={() => setTab("spec")}>
            規格
          </TabBtn>
          <TabBtn active={tab === "patch"} onClick={() => setTab("patch")}>
            補丁
          </TabBtn>
          <TabBtn active={tab === "review"} onClick={() => setTab("review")}>
            審查
          </TabBtn>
          <TabBtn
            active={tab === "evaluation"}
            onClick={() => setTab("evaluation")}
          >
            評估
          </TabBtn>
        </div>
        <div className="flex items-center gap-1.5">
          <button
            disabled={!spec || evaluating}
            onClick={runEvaluate}
            className="rounded border border-edge-strong bg-surface-3 px-2 py-1 text-xs font-medium text-ink hover:bg-surface-2 disabled:opacity-40"
          >
            {evaluating ? "評估中…" : "瀏覽器評估"}
          </button>
          {onClose ? (
            <button
              type="button"
              onClick={onClose}
              className="rounded border border-edge-strong px-2 py-1 text-xs text-ink-dim hover:bg-surface-2 hover:text-ink lg:hidden"
            >
              關閉
            </button>
          ) : null}
        </div>
      </div>

      {warnings.length > 0 ? <WarningsBanner warnings={warnings} /> : null}

      <div className="flex-1 overflow-auto p-3 font-mono text-xs">
        {tab === "spec" ? (
          <pre className="whitespace-pre-wrap break-all text-ink-dim">
            {spec ? JSON.stringify(spec, null, 2) : "// 尚無儀表板"}
          </pre>
        ) : null}
        {tab === "patch" ? (
          <pre className="whitespace-pre-wrap break-all text-ink-dim">
            {lastPatch ? JSON.stringify(lastPatch, null, 2) : "// 尚無補丁"}
          </pre>
        ) : null}
        {tab === "review" ? (
          <ReviewTrail trail={reviewTrail} />
        ) : null}
        {tab === "evaluation" ? (
          <div>
            {evalError ? (
              <div className="mb-2 text-alarm-high">{evalError}</div>
            ) : null}
            <pre className="whitespace-pre-wrap break-all text-ink-dim">
              {evalReport
                ? JSON.stringify(evalReport, null, 2)
                : "// 尚無評估報告"}
            </pre>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function ReviewTrail({ trail }: { trail: ReviewTrailEvent[] }) {
  if (!trail || trail.length === 0) {
    return (
      <div className="text-ink-faint">
        {"// 審查軌跡會在 Helper／Big guy 檢查儀表板後顯示"}
      </div>
    );
  }
  return (
    <ol className="space-y-1.5">
      {trail.map((e, i) => (
        <li
          key={i}
          className="rounded border border-edge bg-surface-2/60 p-2"
        >
          <div className="flex flex-wrap items-center gap-1.5 text-2xs uppercase tracking-wide">
            <span className={`rounded px-1.5 py-0.5 ${stageColor(e.stage)}`}>
              {e.stage}
            </span>
            {typeof e.attempt === "number" ? (
              <span className="text-ink-faint">第 {e.attempt} 次</span>
            ) : null}
            {e.decision ? (
              <span className="text-ok">decision: {e.decision}</span>
            ) : null}
            {e.kind ? (
              <span className="text-vital-spo2">kind: {e.kind}</span>
            ) : null}
            {e.to ? (
              <span className="text-chart-3">→ {e.to}</span>
            ) : null}
            {typeof e.page_loaded === "boolean" ? (
              <span
                className={e.page_loaded ? "text-ok" : "text-alarm-high"}
              >
                page_loaded: {String(e.page_loaded)}
              </span>
            ) : null}
            {typeof e.console_errors === "number" ? (
              <span className="text-alarm-med">
                console: {e.console_errors}
              </span>
            ) : null}
            {e.missing && e.missing.length > 0 ? (
              <span className="text-alarm-med">
                missing: {e.missing.join(", ")}
              </span>
            ) : null}
          </div>
          {e.rationale ? (
            <div className="mt-1 text-ink-dim">{e.rationale}</div>
          ) : null}
          {e.error ? (
            <div className="mt-1 text-alarm-high">{e.error}</div>
          ) : null}
        </li>
      ))}
    </ol>
  );
}

function stageColor(stage: string): string {
  switch (stage) {
    case "render":
      return "bg-surface-3 text-ink-dim";
    case "review":
      return "bg-chart-4/20 text-chart-4";
    case "patch_failed":
    case "patch_validation_failed":
    case "rescue_patch_failed":
    case "rescue_patch_validation_failed":
    case "rescue_ticket_validation_failed":
    case "rescue_runtime_error":
    case "failed":
      return "bg-alarm-high/20 text-alarm-high";
    case "escalate":
      return "bg-chart-3/20 text-chart-3";
    case "rescue":
    case "rescue_confirm":
    case "rescue_unknown_kind":
      return "bg-chart-6/20 text-chart-6";
    case "fallback_ask_user":
      return "bg-alarm-med/20 text-alarm-med";
    default:
      return "bg-surface-3 text-ink-dim";
  }
}

/**
 * Friendly summary of a single warning. The full text (often a long
 * pydantic dump like "rescue ticket parse failed: 7 validation errors
 * for DeveloperTicket source_agent Field required ...") goes into the
 * collapsible Details below. The summary is a single human-readable
 * line based on pattern matching the warning's prefix.
 */
function summarizeWarning(raw: string): string {
  const s = raw.trim();
  if (/rescue ticket parse failed/i.test(s)) {
    return "內部警告:rescue agent 回傳的 ticket 格式錯誤,使用者流程已自動恢復。";
  }
  if (/rescue_review failed/i.test(s) || /rescue runtime/i.test(s)) {
    return "內部警告:rescue 步驟發生執行期問題。";
  }
  if (/review_rendered failed/i.test(s)) {
    return "內部警告:審查步驟發生執行期問題。";
  }
  if (/spec validation/i.test(s) || /validation error/i.test(s)) {
    return "內部警告:驗證器拒絕了自動補丁,協調器已帶回饋重試。";
  }
  // Fallback: first 120 chars; the toggle reveals the rest.
  return s.length > 120 ? s.slice(0, 117) + "…" : s;
}

function WarningsBanner({ warnings }: { warnings: string[] }) {
  const [open, setOpen] = React.useState(false);
  return (
    <div className="border-b border-alarm-med/40 bg-alarm-med/10 px-3 py-1.5 text-xs text-alarm-med">
      <div className="flex items-center justify-between gap-2">
        <div className="min-w-0 truncate" title={warnings.join("\n")}>
          {warnings.length === 1
            ? summarizeWarning(warnings[0])
            : `${warnings.length} 則內部警告 — ${summarizeWarning(warnings[0])}`}
        </div>
        <button
          onClick={() => setOpen((v) => !v)}
          className="shrink-0 rounded border border-alarm-med/40 px-1.5 py-0.5 text-2xs uppercase tracking-wide text-alarm-med hover:bg-alarm-med/20"
          aria-expanded={open}
        >
          {open ? "隱藏詳情" : "詳情"}
        </button>
      </div>
      {open ? (
        <ul className="mt-1.5 space-y-1 border-t border-alarm-med/30 pt-1.5 text-xs text-alarm-med/90">
          {warnings.map((w, i) => (
            <li key={i} className="whitespace-pre-wrap break-all font-mono">
              {w}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function TabBtn({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={
        "rounded px-2 py-1 transition-colors " +
        (active
          ? "bg-surface-3 text-ink ring-1 ring-ring/60"
          : "text-ink-dim hover:bg-surface-2 hover:text-ink")
      }
    >
      {children}
    </button>
  );
}
