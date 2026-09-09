"use client";

import React, { useState } from "react";

import { Badge, Button, Disclosure } from "@/components/ui";
import { api, describeApiError } from "@/lib/api";
import type {
  BrowserEvaluationReport,
  DashboardSpec,
  PatchSpec,
  ReviewTrailEvent,
} from "@/lib/spec-schema";

// Developer-only views, hosted as the 規格 / 補丁 / 審查 / 評估 tabs of the
// DevWorkspaceDrawer (which owns the Tabs control). Raw JSON, review
// trails and the browser evaluator never render in the user UI; the
// 瀏覽器評估 button sends the developer token (api.evaluate).

export type DevTab = "spec" | "patch" | "review" | "evaluation";

export function JsonInspector({
  tab,
  spec,
  lastPatch,
  warnings,
  reviewTrail,
}: {
  tab: DevTab;
  spec: DashboardSpec | null;
  lastPatch: PatchSpec | null;
  warnings: string[];
  reviewTrail: ReviewTrailEvent[];
}) {
  return (
    <div className="flex h-full min-h-0 flex-col">
      {tab === "spec" ? <SpecView spec={spec} /> : null}
      {tab === "patch" ? <PatchView patch={lastPatch} /> : null}
      {tab === "review" ? <ReviewView warnings={warnings} trail={reviewTrail} /> : null}
      {tab === "evaluation" ? <EvaluationView spec={spec} /> : null}
    </div>
  );
}

function Pre({ children }: { children: React.ReactNode }) {
  return (
    <pre className="min-h-0 flex-1 overflow-auto whitespace-pre-wrap break-all rounded-md border border-edge bg-canvas p-3 font-mono text-xs leading-relaxed text-ink-dim">
      {children}
    </pre>
  );
}

function SpecView({ spec }: { spec: DashboardSpec | null }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2 p-3">
      <div className="flex flex-wrap items-center gap-2 text-xs text-ink-dim">
        <span className="font-medium text-ink">DashboardSpec</span>
        {spec ? (
          <>
            <span className="font-mono">{spec.dashboard_id}</span>
            <span>· {spec.widgets.length} widgets</span>
            <span>· refresh {spec.refresh_interval}</span>
          </>
        ) : (
          <span>尚無儀表板</span>
        )}
      </div>
      <Pre>{spec ? JSON.stringify(spec, null, 2) : "// 尚無儀表板"}</Pre>
    </div>
  );
}

function PatchView({ patch }: { patch: PatchSpec | null }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2 p-3">
      <div className="flex flex-wrap items-center gap-2 text-xs text-ink-dim">
        <span className="font-medium text-ink">最近一次補丁</span>
        {patch ? (
          <>
            <span className="font-mono">{patch.patch_id}</span>
            <span>· {patch.operations.length} ops</span>
            <span>· by {patch.created_by}</span>
          </>
        ) : (
          <span>尚無補丁</span>
        )}
      </div>
      {patch?.reason ? <div className="text-sm text-ink">{patch.reason}</div> : null}
      <Pre>{patch ? JSON.stringify(patch, null, 2) : "// 尚無補丁"}</Pre>
    </div>
  );
}

function ReviewView({ warnings, trail }: { warnings: string[]; trail: ReviewTrailEvent[] }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-3">
      {warnings.length > 0 ? <WarningsBanner warnings={warnings} /> : null}
      <ReviewTrail trail={trail} />
    </div>
  );
}

function EvaluationView({ spec }: { spec: DashboardSpec | null }) {
  const [report, setReport] = useState<BrowserEvaluationReport | null>(null);
  const [evaluating, setEvaluating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    if (!spec) return;
    setEvaluating(true);
    setError(null);
    try {
      const r = await api.evaluate(spec.dashboard_id);
      setReport(r.report);
    } catch (e) {
      setError(e instanceof Error ? e.message : describeApiError(e));
    } finally {
      setEvaluating(false);
    }
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-3">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          type="button"
          variant="primary"
          size="sm"
          disabled={!spec}
          loading={evaluating}
          onClick={run}
        >
          瀏覽器評估
        </Button>
        <span className="text-xs text-ink-dim">
          {spec
            ? `以 Playwright 開啟 /dashboard/${spec.dashboard_id} 並檢查每個元件是否渲染。`
            : "需要先開啟一個儀表板。"}
        </span>
      </div>
      {error ? (
        <div className="rounded-md border border-danger/40 bg-danger/10 p-2 font-mono text-xs text-danger">
          {error}
        </div>
      ) : null}
      {report ? (
        <div className="space-y-2 text-sm">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone={report.page_loaded ? "ok" : "danger"}>
              {report.page_loaded ? "page loaded" : "page failed"}
            </Badge>
            <Badge tone={report.missing_widgets.length === 0 ? "ok" : "danger"}>
              {report.widgets_rendered.length} rendered · {report.missing_widgets.length} missing
            </Badge>
            <Badge tone={report.console_errors.length === 0 ? "ok" : "warn"}>
              console {report.console_errors.length}
            </Badge>
            <Badge tone={report.layout_errors.length === 0 ? "ok" : "warn"}>
              layout {report.layout_errors.length}
            </Badge>
            <Badge tone={report.prometheus_errors.length === 0 ? "ok" : "warn"}>
              prometheus {report.prometheus_errors.length}
            </Badge>
          </div>
          <div className="text-ink">{report.recommendation}</div>
          {report.missing_widgets.length > 0 ? (
            <div className="font-mono text-xs text-danger">
              missing: {report.missing_widgets.join(", ")}
            </div>
          ) : null}
          {report.screenshot_path ? (
            <div className="font-mono text-xs text-ink-dim">{report.screenshot_path}</div>
          ) : null}
          <Disclosure summary="完整報告 JSON">
            <Pre>{JSON.stringify(report, null, 2)}</Pre>
          </Disclosure>
        </div>
      ) : (
        <div className="text-xs text-ink-dim">尚無評估報告。</div>
      )}
    </div>
  );
}

function ReviewTrail({ trail }: { trail: ReviewTrailEvent[] }) {
  if (!trail || trail.length === 0) {
    return (
      <div className="text-xs text-ink-dim">
        審查軌跡會在 Helper／Big Guy 檢查儀表板後顯示。
      </div>
    );
  }
  return (
    <ol className="space-y-1.5">
      {trail.map((e, i) => (
        <li key={i} className="rounded-md border border-edge bg-surface p-2">
          <div className="flex flex-wrap items-center gap-1.5 font-mono text-xs">
            <Badge tone={stageTone(e.stage)}>{e.stage}</Badge>
            {typeof e.attempt === "number" ? (
              <span className="text-ink-dim">attempt {e.attempt}</span>
            ) : null}
            {e.decision ? <span className="text-ok">decision: {e.decision}</span> : null}
            {e.kind ? <span className="text-accent">kind: {e.kind}</span> : null}
            {e.to ? <span className="text-info">→ {e.to}</span> : null}
            {typeof e.page_loaded === "boolean" ? (
              <span className={e.page_loaded ? "text-ok" : "text-danger"}>
                page_loaded: {String(e.page_loaded)}
              </span>
            ) : null}
            {typeof e.console_errors === "number" ? (
              <span className="text-warn">console: {e.console_errors}</span>
            ) : null}
            {e.missing && e.missing.length > 0 ? (
              <span className="text-warn">missing: {e.missing.join(", ")}</span>
            ) : null}
          </div>
          {e.rationale ? <div className="mt-1 text-sm text-ink-dim">{e.rationale}</div> : null}
          {e.error ? (
            <div className="mt-1 break-all font-mono text-xs text-danger">{e.error}</div>
          ) : null}
        </li>
      ))}
    </ol>
  );
}

function stageTone(stage: string): "neutral" | "ok" | "warn" | "danger" | "info" | "accent" {
  switch (stage) {
    case "review":
      return "info";
    case "patch_failed":
    case "patch_validation_failed":
    case "rescue_patch_failed":
    case "rescue_patch_validation_failed":
    case "rescue_ticket_validation_failed":
    case "rescue_runtime_error":
    case "failed":
      return "danger";
    case "escalate":
      return "accent";
    case "rescue":
    case "rescue_confirm":
    case "rescue_unknown_kind":
    case "fallback_ask_user":
      return "warn";
    default:
      return "neutral";
  }
}

/**
 * Friendly summary of a single warning. The full text (often a long
 * pydantic dump) is available in the Disclosure below.
 */
function summarizeWarning(raw: string): string {
  const s = raw.trim();
  if (/rescue ticket parse failed/i.test(s)) {
    return "內部警告：rescue agent 回傳的 ticket 格式錯誤，使用者流程已自動恢復。";
  }
  if (/rescue_review failed/i.test(s) || /rescue runtime/i.test(s)) {
    return "內部警告：rescue 步驟發生執行期問題。";
  }
  if (/review_rendered failed/i.test(s)) {
    return "內部警告：審查步驟發生執行期問題。";
  }
  if (/spec validation/i.test(s) || /validation error/i.test(s)) {
    return "內部警告：驗證器拒絕了自動補丁，協調器已帶回饋重試。";
  }
  return s.length > 120 ? s.slice(0, 117) + "…" : s;
}

function WarningsBanner({ warnings }: { warnings: string[] }) {
  return (
    <div className="rounded-md border border-warn/40 bg-warn/10 p-2 text-sm text-ink">
      <div className="flex items-center gap-2">
        <Badge tone="warn">{warnings.length} 則內部警告</Badge>
        <span className="min-w-0 flex-1 truncate" title={warnings.join("\n")}>
          {summarizeWarning(warnings[0])}
        </span>
      </div>
      <Disclosure summary="詳情">
        <ul className="mt-1.5 space-y-1 border-t border-warn/30 pt-1.5">
          {warnings.map((w, i) => (
            <li key={i} className="whitespace-pre-wrap break-all font-mono text-xs text-ink-dim">
              {w}
            </li>
          ))}
        </ul>
      </Disclosure>
    </div>
  );
}
