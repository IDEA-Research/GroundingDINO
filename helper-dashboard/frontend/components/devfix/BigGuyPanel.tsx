"use client";

import React, { useEffect, useState } from "react";

import {
  Badge,
  Button,
  Disclosure,
  Field,
  Input,
  Skeleton,
  Tabs,
  Textarea,
  fieldDescribedBy,
  useToast,
} from "@/components/ui";
import { api, describeApiError } from "@/lib/api";
import type { Capabilities, DevRunMeta, DevfixConfig } from "@/lib/spec-schema";
import {
  canCancel,
  canRetry,
  elapsedLabel,
  isActiveStatus,
  runStatus,
  runStatusLabel,
  runStatusTone,
  type DevfixRunsState,
} from "@/lib/useDevfixRuns";

import { RunDiffView } from "./RunDiffView";
import { RunFilesView } from "./RunFilesView";
import { RunLogView } from "./RunLogView";

// Big Guy — local development workspace tab.
//
//   任務輸入  -> 執行前檢查 (GET /devfix/config) -> 「在本機執行」
//   目前任務  (status chip · elapsed · progress · 取消)   vs   歷史任務 (collapsed)
//   任務詳情  = 紀錄 / 檔案 / 差異, 檢查結果, and the three explicit chips
//              修改 / 驗證 / 套用 (a change is not a success).
//
// Opening the panel never starts anything: only the button does, and the
// backend still enforces the token + kill switch on that request. Old
// failures live under 歷史任務 so they never read as a live outage.

const INSTRUCTION_MAX = 2000;

const LOCAL_RUNS_REASON: Record<string, string> = {
  local_runs_disabled: "伺服器未設定 HELPER_DASHBOARD_DEVFIX_LOCAL_RUNS=1",
  mock_mode: "HELPER_DASHBOARD_HELPERCODE 為 mock 模式",
  runtime_not_opencode_cli: "HELPER_DASHBOARD_DEVFIX_RUNTIME 不是 opencode-cli",
  binary_missing: "找不到 opencode 執行檔",
};

function localRunsReasonText(code: string | null | undefined): string | null {
  if (!code) return null;
  return LOCAL_RUNS_REASON[code] ?? code;
}

function runTitle(meta: DevRunMeta): string {
  return meta.title?.trim() || meta.instruction_preview?.trim() || meta.ticket_id;
}

function fmtWhen(iso: string | null | undefined): string {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return iso;
  const d = new Date(t);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getMonth() + 1)}/${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function OutcomeChips({ meta }: { meta: DevRunMeta }) {
  const changed = meta.files_changed?.length ?? 0;
  const v = meta.verification?.status ?? "none";
  const a = meta.applied?.status ?? (changed > 0 ? "not_applied" : "n/a");
  const verify =
    v === "passed"
      ? { tone: "ok" as const, label: "驗證：通過" }
      : v === "failed"
        ? { tone: "danger" as const, label: "驗證：失敗" }
        : { tone: "neutral" as const, label: "驗證：未驗證" };
  const applied =
    a === "applied"
      ? { tone: "ok" as const, label: "套用：已套用" }
      : a === "violation"
        ? { tone: "danger" as const, label: "套用：違反允許清單" }
        : a === "n/a"
          ? { tone: "neutral" as const, label: "套用：不適用" }
          : { tone: "neutral" as const, label: "套用：未套用" };
  return (
    <div className="flex flex-wrap gap-1.5" aria-label="結果狀態">
      <Badge tone={changed > 0 ? "accent" : "neutral"}>
        修改：{changed > 0 ? `${changed} 個檔案` : "無"}
      </Badge>
      <Badge tone={verify.tone}>{verify.label}</Badge>
      <Badge tone={applied.tone}>{applied.label}</Badge>
    </div>
  );
}

function TestsLine({ meta }: { meta: DevRunMeta }) {
  const t = meta.tests;
  if (!t || !t.ran) {
    return <div className="text-xs text-ink-dim">檢查結果：未執行測試</div>;
  }
  const failed = t.failed ?? 0;
  return (
    <div className={"text-xs " + (failed > 0 ? "text-danger" : "text-ink-dim")}>
      檢查結果：通過 {t.passed ?? "—"} · 失敗 {t.failed ?? "—"}
      {t.summary ? <span className="ml-1 font-mono">{t.summary}</span> : null}
    </div>
  );
}

function RunCard({
  meta,
  now,
  selected,
  isCurrent,
  progress,
  busy,
  onSelect,
  onCancel,
  onRetry,
}: {
  meta: DevRunMeta;
  now: number;
  selected: boolean;
  isCurrent: boolean;
  progress: string | null;
  busy: "cancel" | "retry" | null;
  onSelect: () => void;
  onCancel: () => void;
  onRetry: () => void;
}) {
  const st = runStatus(meta);
  const title = runTitle(meta);
  const failedLike = st === "failed" || st === "timeout";
  return (
    <article
      aria-current={selected ? "true" : undefined}
      className={
        "rounded-lg border bg-canvas p-3 " +
        (selected ? "border-accent/60" : "border-edge")
      }
    >
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone={runStatusTone(st)} dot>
          {runStatusLabel(st)}
        </Badge>
        <Badge tone="neutral">{meta.source === "manual" ? "手動" : "自動"}</Badge>
        {typeof meta.attempt === "number" && meta.attempt > 1 ? (
          <Badge tone="neutral">第 {meta.attempt} 次</Badge>
        ) : null}
        {isCurrent ? null : <span className="text-xs text-ink-faint">歷史</span>}
        <Badge tone="neutral" className="ml-auto font-mono tabular-nums">
          {elapsedLabel(meta, now)}
        </Badge>
      </div>
      <div className="mt-1.5 break-words text-sm font-medium text-ink">{title}</div>
      <div className="mt-0.5 break-all font-mono text-xs text-ink-faint">
        {meta.run_id} · {fmtWhen(meta.started_at ?? meta.queued_at)}
        {meta.model ? ` · ${meta.model}` : ""}
      </div>
      {meta.instruction && meta.instruction.trim() !== title ? (
        <Disclosure summary="任務說明全文">
          <pre className="mt-1 whitespace-pre-wrap break-words font-mono text-xs text-ink-dim">
            {meta.instruction}
          </pre>
        </Disclosure>
      ) : null}
      {isActiveStatus(st) && progress ? (
        <div className="mt-1.5 text-xs text-ink-dim">
          進度：<span className="text-ink">{progress}</span>
        </div>
      ) : null}
      {meta.report_summary ? (
        <div className="mt-1.5 rounded-md border border-edge bg-surface p-2 text-xs text-ink">
          <span className="text-ink-dim">Big Guy 回報：</span>
          {meta.report_summary}
        </div>
      ) : null}
      {st === "waiting" ? (
        <div className="mt-1.5 rounded-md border border-warn/40 bg-warn/10 p-2 text-xs text-ink">
          Big Guy 需要開發者處理：{meta.failure_summary || meta.report_status || "請查看紀錄中的回覆"}。
        </div>
      ) : meta.failure_summary ? (
        <div className="mt-1.5 rounded-md border border-danger/40 bg-danger/10 p-2 text-xs text-danger">
          {meta.failure_summary}
          {meta.error_kind ? <span className="ml-1 font-mono">({meta.error_kind})</span> : null}
        </div>
      ) : failedLike && meta.error_kind ? (
        <div className="mt-1.5 rounded-md border border-danger/40 bg-danger/10 p-2 font-mono text-xs text-danger">
          {meta.error_kind}
        </div>
      ) : null}
      <div className="mt-2 space-y-1.5">
        <OutcomeChips meta={meta} />
        <TestsLine meta={meta} />
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {canCancel(st) ? (
          <Button type="button" variant="danger" size="sm" loading={busy === "cancel"} onClick={onCancel}>
            取消
          </Button>
        ) : null}
        {canRetry(st) ? (
          <Button type="button" variant="secondary" size="sm" loading={busy === "retry"} onClick={onRetry}>
            重新執行
          </Button>
        ) : null}
        <Button type="button" variant={selected ? "ghost" : "secondary"} size="sm" onClick={onSelect} aria-pressed={selected}>
          {selected ? "檢視中" : "檢視詳情"}
        </Button>
      </div>
    </article>
  );
}

export function BigGuyPanel({
  capabilities,
  runs,
  config,
  configError,
  configLoading,
  onRefreshConfig,
}: {
  capabilities: Capabilities | null;
  runs: DevfixRunsState;
  config: DevfixConfig | null;
  configError: string | null;
  configLoading: boolean;
  onRefreshConfig: () => void;
}) {
  const toast = useToast();
  const [text, setText] = useState("");
  const [title, setTitle] = useState("");
  const [starting, setStarting] = useState(false);
  const [busyRun, setBusyRun] = useState<{ id: string; op: "cancel" | "retry" } | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detailTab, setDetailTab] = useState<string>("log");
  const [progress, setProgress] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());

  const current = runs.current;
  const history = runs.runs.filter((r) => r.run_id !== current?.run_id);
  const selectedMeta =
    runs.runs.find((r) => r.run_id === (selectedId ?? current?.run_id)) ?? current ?? null;
  const selectedIsCurrent = Boolean(selectedMeta && current && selectedMeta.run_id === current.run_id);

  // Elapsed labels tick while something is in flight (keyed by run_id so
  // the 3 s list poll does not recreate the interval each time).
  const activeRunId = runs.active?.run_id ?? null;
  useEffect(() => {
    if (!activeRunId) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [activeRunId]);
  useEffect(() => {
    setNow(Date.now());
  }, [runs.runs]);
  // Progress text belongs to the CURRENT run only.
  useEffect(() => {
    if (!selectedIsCurrent) setProgress(null);
  }, [selectedIsCurrent]);

  const localRuns = capabilities?.devfix_local_runs === true;
  const quota = config?.daily_quota;
  const quotaExhausted = Boolean(quota && quota.limit > 0 && quota.used >= quota.limit);
  const tooLong = text.length > INSTRUCTION_MAX;
  const pendingCount = config?.queue?.pending_run_ids.length ?? null;
  const maxPending = config?.queue?.max_pending ?? null;

  let blockReason: string | null = null;
  if (!localRuns || (config && !config.local_runs_enabled)) {
    const why = localRunsReasonText(config?.local_runs_reason);
    blockReason = why
      ? `本機執行層未啟用：${why}。`
      : "本機執行層未啟用：伺服器需要 HELPER_DASHBOARD_DEVFIX_LOCAL_RUNS=1、HELPER_DASHBOARD_DEVFIX_RUNTIME=opencode-cli、非 mock 模式，且 opencode 執行檔存在。";
  } else if (quotaExhausted && quota) {
    blockReason = `今日配額已用完（${quota.used}/${quota.limit}）。`;
  } else if (tooLong) {
    blockReason = `任務說明超過 ${INSTRUCTION_MAX} 字。`;
  } else if (!text.trim()) {
    blockReason = "請先輸入任務說明。";
  }

  async function start() {
    const instruction = text.trim();
    if (!instruction || blockReason || starting) return;
    setStarting(true);
    try {
      const r = await api.devfixStart({
        instruction,
        title: title.trim() ? title.trim() : undefined,
      });
      toast.show({
        title: r.status === "running" ? "任務已開始執行" : "任務已排入佇列",
        description: r.run_id,
        tone: "ok",
        durationMs: 4000,
      });
      setText("");
      setTitle("");
      setSelectedId(r.run_id);
      setDetailTab("log");
      runs.refresh();
    } catch (e) {
      toast.show({
        title: "無法啟動任務",
        description: e instanceof Error ? e.message : describeApiError(e),
        tone: "danger",
        durationMs: 8000,
      });
    } finally {
      setStarting(false);
    }
  }

  async function cancel(meta: DevRunMeta) {
    setBusyRun({ id: meta.run_id, op: "cancel" });
    try {
      const r = await api.devfixCancel(meta.run_id);
      toast.show({
        title: r.status === "cancelled" ? "已從佇列移除" : "已送出取消",
        description:
          r.status === "cancelled"
            ? runTitle(meta)
            : `${runTitle(meta)} — 正在結束程序，最多約 8 秒後標示為已取消。`,
        tone: "neutral",
        durationMs: 5000,
      });
      runs.refresh();
    } catch (e) {
      toast.show({
        title: "取消失敗",
        description: e instanceof Error ? e.message : describeApiError(e),
        tone: "danger",
        durationMs: 8000,
      });
    } finally {
      setBusyRun(null);
    }
  }

  async function retry(meta: DevRunMeta) {
    setBusyRun({ id: meta.run_id, op: "retry" });
    try {
      const r = await api.devfixRetry(meta.run_id);
      toast.show({
        title: "已重新排入",
        description: `${runTitle(meta)} → ${r.run_id}`,
        tone: "ok",
        durationMs: 4000,
      });
      setSelectedId(r.run_id);
      setDetailTab("log");
      runs.refresh();
    } catch (e) {
      toast.show({
        title: "無法重新執行",
        description: e instanceof Error ? e.message : describeApiError(e),
        tone: "danger",
        durationMs: 8000,
      });
    } finally {
      setBusyRun(null);
    }
  }

  const busyFor = (meta: DevRunMeta) =>
    busyRun && busyRun.id === meta.run_id ? busyRun.op : null;

  return (
    <div className="space-y-4 p-3 text-sm">
      {runs.error ? (
        <div className="rounded-md border border-danger/40 bg-danger/10 p-2 text-xs text-danger">
          無法讀取任務清單：{runs.error}
        </div>
      ) : null}

      {/* 任務輸入 */}
      <section aria-labelledby="hd-bg-task" className="rounded-lg border border-edge bg-canvas p-3">
        <h3 id="hd-bg-task" className="mb-2 text-sm font-semibold text-ink">
          任務輸入
        </h3>
        <div className="space-y-3">
          <Field
            label="任務說明"
            htmlFor="hd-bg-instruction"
            hint={`描述要 Big Guy 在本機檢查或修改的內容（最多 ${INSTRUCTION_MAX} 字）。只會送出這段文字；不接受指令、路徑或模型設定。`}
            error={tooLong ? `已超過 ${INSTRUCTION_MAX} 字（目前 ${text.length}）` : undefined}
            required
          >
            <Textarea
              id="hd-bg-instruction"
              aria-describedby={fieldDescribedBy("hd-bg-instruction", { hint: true, error: tooLong })}
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={4}
              invalid={tooLong}
              placeholder="例如：折線圖門檻線的高於/低於方向判斷不一致，請檢查 LineChartWidget 並補上測試。"
            />
          </Field>
          <Field label="標題（選填）" htmlFor="hd-bg-title">
            <Input
              id="hd-bg-title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              maxLength={120}
              autoComplete="off"
              placeholder="顯示在任務清單中的簡短名稱"
            />
          </Field>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              type="button"
              variant="primary"
              size="md"
              loading={starting}
              disabled={blockReason !== null}
              onClick={start}
            >
              在本機執行
            </Button>
            {blockReason ? (
              <span className="text-xs text-ink-dim">{blockReason}</span>
            ) : runs.active ? (
              <span className="text-xs text-ink-dim">目前有任務執行或排隊中，新任務會排隊等待。</span>
            ) : null}
          </div>
          <p className="text-xs text-ink-faint">
            開啟此面板不會自動啟動任何任務；只有按下「在本機執行」才會開始本機開發流程。
          </p>
        </div>
      </section>

      {/* 執行前檢查 */}
      <section aria-labelledby="hd-bg-preflight" className="rounded-lg border border-edge bg-canvas p-3">
        <div className="mb-2 flex items-center justify-between gap-2">
          <h3 id="hd-bg-preflight" className="text-sm font-semibold text-ink">
            執行前檢查
          </h3>
          <Button type="button" variant="ghost" size="sm" loading={configLoading} onClick={onRefreshConfig}>
            重新讀取
          </Button>
        </div>
        {configError ? (
          <div className="mb-2 rounded-md border border-danger/40 bg-danger/10 p-2 font-mono text-xs text-danger">
            {configError}
          </div>
        ) : null}
        {config ? (
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-xs">
            <dt className="text-ink-dim">目標專案</dt>
            <dd className="break-all font-mono text-ink">{config.project_root}</dd>
            <dt className="text-ink-dim">工作目錄</dt>
            <dd className="break-all font-mono text-ink">{config.worktree_parent}</dd>
            <dt className="text-ink-dim">模型</dt>
            <dd className="font-mono text-ink">{config.model}</dd>
            <dt className="text-ink-dim">逾時</dt>
            <dd className="text-ink">{config.timeout_seconds} 秒</dd>
            <dt className="text-ink-dim">執行層</dt>
            <dd className="text-ink">
              <span className="font-mono">{config.devfix_runtime}</span> · 模式{" "}
              <span className="font-mono">{config.runtime_mode}</span> · opencode{" "}
              {config.bin_exists ? "可用" : "找不到"}
              <span className="ml-1 break-all font-mono text-ink-faint">({config.bin})</span>
            </dd>
            <dt className="text-ink-dim">本機執行層狀態</dt>
            <dd>
              <Badge tone={config.local_runs_enabled && localRuns ? "ok" : "danger"} dot>
                {config.local_runs_enabled && localRuns ? "已啟用" : "未啟用"}
              </Badge>
            </dd>
            <dt className="text-ink-dim">自動修復</dt>
            <dd className="text-ink">
              {config.auto_fix_enabled
                ? "啟用（聊天工單會自動排程，清單中標示為「自動」）"
                : "停用"}
            </dd>
            <dt className="text-ink-dim">今日配額</dt>
            <dd className={"tabular-nums " + (quotaExhausted ? "text-danger" : "text-ink")}>
              {config.daily_quota.used}/{config.daily_quota.limit}
            </dd>
            {pendingCount !== null ? (
              <>
                <dt className="text-ink-dim">佇列</dt>
                <dd className="tabular-nums text-ink">
                  {config?.queue?.active_run_id ? "1 個執行中 · " : ""}
                  {pendingCount} 個待執行{maxPending !== null ? `（上限 ${maxPending}）` : ""}
                </dd>
              </>
            ) : null}
          </dl>
        ) : configLoading ? (
          <div className="space-y-1.5" aria-busy="true">
            <Skeleton className="h-4 w-3/4 rounded" />
            <Skeleton className="h-4 w-2/3 rounded" />
            <Skeleton className="h-4 w-1/2 rounded" />
          </div>
        ) : (
          <div className="text-xs text-ink-dim">尚未讀取設定。</div>
        )}
      </section>

      {/* 目前任務 */}
      <section aria-labelledby="hd-bg-current" className="space-y-2">
        <h3 id="hd-bg-current" className="text-sm font-semibold text-ink">
          目前任務
        </h3>
        {current ? (
          <RunCard
            meta={current}
            now={now}
            selected={selectedMeta?.run_id === current.run_id}
            isCurrent
            progress={progress}
            busy={busyFor(current)}
            onSelect={() => setSelectedId(current.run_id)}
            onCancel={() => cancel(current)}
            onRetry={() => retry(current)}
          />
        ) : (
          <div className="flex flex-wrap items-center gap-2 rounded-lg border border-dashed border-edge-strong p-3 text-ink-dim">
            <Badge tone="neutral" dot>
              待命
            </Badge>
            <span>沒有進行中或剛完成的任務。</span>
            {!runs.loaded && !runs.error ? <span className="text-xs">（讀取中…）</span> : null}
          </div>
        )}
      </section>

      {/* 歷史任務 */}
      <section aria-labelledby="hd-bg-history">
        <Disclosure
          summary={
            <span id="hd-bg-history" className="text-sm font-semibold text-ink">
              歷史任務（{history.length}）
            </span>
          }
        >
          <p className="mb-2 mt-1 text-xs text-ink-faint">
            歷史紀錄僅供回顧；舊的失敗不代表目前網站或任務有問題。
          </p>
          {history.length === 0 ? (
            <div className="text-xs text-ink-dim">尚無歷史任務。</div>
          ) : (
            <ul className="space-y-2">
              {history.map((m) => (
                <li key={m.run_id}>
                  <RunCard
                    meta={m}
                    now={now}
                    selected={selectedMeta?.run_id === m.run_id}
                    isCurrent={false}
                    progress={null}
                    busy={busyFor(m)}
                    onSelect={() => setSelectedId(m.run_id)}
                    onCancel={() => cancel(m)}
                    onRetry={() => retry(m)}
                  />
                </li>
              ))}
            </ul>
          )}
        </Disclosure>
      </section>

      {/* 任務詳情 */}
      {selectedMeta ? (
        <section aria-labelledby="hd-bg-detail" className="rounded-lg border border-edge bg-canvas p-3">
          <h3 id="hd-bg-detail" className="mb-2 flex flex-wrap items-center gap-2 text-sm font-semibold text-ink">
            任務詳情
            <span className="min-w-0 truncate font-normal text-ink-dim">{runTitle(selectedMeta)}</span>
            <Badge tone={runStatusTone(runStatus(selectedMeta))} dot>
              {runStatusLabel(runStatus(selectedMeta))}
            </Badge>
          </h3>
          <Tabs
            tabs={[
              { id: "log", label: "紀錄" },
              {
                id: "files",
                label: "檔案",
                badge: (selectedMeta.files_changed?.length ?? 0) > 0 ? selectedMeta.files_changed?.length : undefined,
              },
              { id: "diff", label: "差異" },
            ]}
            value={detailTab}
            onChange={setDetailTab}
          >
            {(id) =>
              id === "log" ? (
                <RunLogView
                  key={selectedMeta.run_id}
                  runId={selectedMeta.run_id}
                  status={runStatus(selectedMeta)}
                  onLatest={selectedIsCurrent ? setProgress : undefined}
                />
              ) : id === "files" ? (
                <RunFilesView meta={selectedMeta} />
              ) : (
                <RunDiffView
                  runId={selectedMeta.run_id}
                  active={isActiveStatus(runStatus(selectedMeta))}
                />
              )
            }
          </Tabs>
        </section>
      ) : null}
    </div>
  );
}
