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

export function JsonInspector({
  spec,
  lastPatch,
  warnings,
  reviewTrail,
}: {
  spec: DashboardSpec | null;
  lastPatch: PatchSpec | null;
  warnings: string[];
  reviewTrail: ReviewTrailEvent[];
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
    <div className="flex h-full flex-col bg-panel2">
      <div className="flex items-center justify-between border-b border-slate-800 px-3 py-2 text-xs">
        <div className="flex gap-2">
          <TabBtn active={tab === "spec"} onClick={() => setTab("spec")}>
            Spec
          </TabBtn>
          <TabBtn active={tab === "patch"} onClick={() => setTab("patch")}>
            Last patch
          </TabBtn>
          <TabBtn active={tab === "review"} onClick={() => setTab("review")}>
            Review
          </TabBtn>
          <TabBtn
            active={tab === "evaluation"}
            onClick={() => setTab("evaluation")}
          >
            Evaluation
          </TabBtn>
        </div>
        <button
          disabled={!spec || evaluating}
          onClick={runEvaluate}
          className="rounded bg-slate-700 px-2 py-1 text-[11px] font-medium text-slate-100 disabled:opacity-40"
        >
          {evaluating ? "Evaluating…" : "Run browser evaluation"}
        </button>
      </div>

      {warnings.length > 0 ? <WarningsBanner warnings={warnings} /> : null}

      <div className="flex-1 overflow-auto p-3 text-xs">
        {tab === "spec" ? (
          <pre className="whitespace-pre-wrap break-all text-slate-300">
            {spec ? JSON.stringify(spec, null, 2) : "// no dashboard yet"}
          </pre>
        ) : null}
        {tab === "patch" ? (
          <pre className="whitespace-pre-wrap break-all text-slate-300">
            {lastPatch ? JSON.stringify(lastPatch, null, 2) : "// no patch yet"}
          </pre>
        ) : null}
        {tab === "review" ? (
          <ReviewTrail trail={reviewTrail} />
        ) : null}
        {tab === "evaluation" ? (
          <div>
            {evalError ? (
              <div className="mb-2 text-rose-300">{evalError}</div>
            ) : null}
            <pre className="whitespace-pre-wrap break-all text-slate-300">
              {evalReport
                ? JSON.stringify(evalReport, null, 2)
                : "// no evaluation report yet"}
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
      <div className="text-slate-500">
        {"// review trail appears here after Helper/Big guy check the dashboard"}
      </div>
    );
  }
  return (
    <ol className="space-y-1.5">
      {trail.map((e, i) => (
        <li
          key={i}
          className="rounded border border-slate-800 bg-slate-900/50 p-2"
        >
          <div className="flex flex-wrap items-center gap-1.5 text-[10px] uppercase tracking-wide">
            <span className={`rounded px-1.5 py-0.5 ${stageColor(e.stage)}`}>
              {e.stage}
            </span>
            {typeof e.attempt === "number" ? (
              <span className="text-slate-400">attempt {e.attempt}</span>
            ) : null}
            {e.decision ? (
              <span className="text-emerald-300">decision: {e.decision}</span>
            ) : null}
            {e.kind ? (
              <span className="text-sky-300">kind: {e.kind}</span>
            ) : null}
            {e.to ? (
              <span className="text-violet-300">→ {e.to}</span>
            ) : null}
            {typeof e.page_loaded === "boolean" ? (
              <span
                className={e.page_loaded ? "text-emerald-300" : "text-rose-300"}
              >
                page_loaded: {String(e.page_loaded)}
              </span>
            ) : null}
            {typeof e.console_errors === "number" ? (
              <span className="text-amber-300">
                console: {e.console_errors}
              </span>
            ) : null}
            {e.missing && e.missing.length > 0 ? (
              <span className="text-amber-300">
                missing: {e.missing.join(", ")}
              </span>
            ) : null}
          </div>
          {e.rationale ? (
            <div className="mt-1 text-slate-300">{e.rationale}</div>
          ) : null}
          {e.error ? (
            <div className="mt-1 text-rose-300">{e.error}</div>
          ) : null}
        </li>
      ))}
    </ol>
  );
}

function stageColor(stage: string): string {
  switch (stage) {
    case "render":
      return "bg-slate-700 text-slate-100";
    case "review":
      return "bg-sky-700 text-sky-100";
    case "patch_failed":
    case "patch_validation_failed":
    case "rescue_patch_failed":
    case "rescue_patch_validation_failed":
    case "rescue_ticket_validation_failed":
    case "rescue_runtime_error":
    case "failed":
      return "bg-rose-800 text-rose-100";
    case "escalate":
      return "bg-violet-700 text-violet-100";
    case "rescue":
    case "rescue_confirm":
    case "rescue_unknown_kind":
      return "bg-fuchsia-700 text-fuchsia-100";
    case "fallback_ask_user":
      return "bg-amber-700 text-amber-100";
    default:
      return "bg-slate-700 text-slate-100";
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
    return "Internal warning: rescue agent returned a malformed ticket. The user-facing flow recovered automatically.";
  }
  if (/rescue_review failed/i.test(s) || /rescue runtime/i.test(s)) {
    return "Internal warning: rescue step had a runtime issue.";
  }
  if (/review_rendered failed/i.test(s)) {
    return "Internal warning: review step had a runtime issue.";
  }
  if (/spec validation/i.test(s) || /validation error/i.test(s)) {
    return "Internal warning: validator rejected an automated patch; the orchestrator retried with feedback.";
  }
  // Fallback: first 120 chars; the toggle reveals the rest.
  return s.length > 120 ? s.slice(0, 117) + "…" : s;
}

function WarningsBanner({ warnings }: { warnings: string[] }) {
  const [open, setOpen] = React.useState(false);
  return (
    <div className="border-b border-amber-900 bg-amber-950/40 px-3 py-1.5 text-[11px] text-amber-200">
      <div className="flex items-center justify-between gap-2">
        <div className="min-w-0 truncate" title={warnings.join("\n")}>
          {warnings.length === 1
            ? summarizeWarning(warnings[0])
            : `${warnings.length} internal warnings — ${summarizeWarning(warnings[0])}`}
        </div>
        <button
          onClick={() => setOpen((v) => !v)}
          className="shrink-0 rounded border border-amber-800 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-amber-100 hover:bg-amber-900/40"
          aria-expanded={open}
        >
          {open ? "Hide details" : "Details"}
        </button>
      </div>
      {open ? (
        <ul className="mt-1.5 space-y-1 border-t border-amber-900 pt-1.5 text-[11px] text-amber-100/90">
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
        "rounded px-2 py-1 " +
        (active
          ? "bg-sky-700 text-white"
          : "bg-slate-800 text-slate-300 hover:bg-slate-700")
      }
    >
      {children}
    </button>
  );
}
