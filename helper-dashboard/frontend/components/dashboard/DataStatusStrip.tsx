"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";

import { Badge, Button } from "@/components/ui";
import { api, describeApiError } from "@/lib/api";
import { useDashboardDataStatus } from "@/lib/data-status";
import type { DatasourceStatus } from "@/lib/spec-schema";
import { useTimeRange } from "@/lib/time-range";
import type { BackendStatusState } from "@/lib/useBackendStatus";

// One dashboard-level status line answering 「資料是否仍在更新?」:
//   伺服器連線 · 資料來源 · 資料接收 · 最後查詢 · 最後量測
// plus ONE worst-state message with its next step (重新載入 / 檢查資料來源 /
// 調整時間範圍). Per-card notices stay terse; the full copy lives here so
// a dashboard of empty cards does not repeat the same paragraph 8 times.
// "Query succeeded" is never presented as "data received": 資料接收 comes
// from widgets that hold real data, 最後量測 from sample timestamps.

type Tone = "ok" | "warn" | "danger" | "neutral" | "info";

// HH:mm:ss for an epoch in seconds OR milliseconds (tolerant on purpose:
// the store reports ms for arrival times and s for sample timestamps).
export function formatClock(ts: number | null | undefined): string {
  if (ts === null || ts === undefined || !Number.isFinite(ts) || ts <= 0) return "—";
  const ms = ts > 1e12 ? ts : ts * 1000;
  const d = new Date(ms);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

export function formatAge(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "—";
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} 秒前`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} 分前`;
  const h = Math.floor(m / 60);
  if (h < 48) return `${h} 小時前`;
  return `${Math.floor(h / 24)} 天前`;
}

export interface DatasourceState {
  status: DatasourceStatus | null;
  error: string | null;
  loading: boolean;
  lastFetched: number | null;
  refresh: () => void;
}

// Polls GET /api/datasource/status (public, booleans/timestamps only).
export function useDatasourceStatus(intervalMs = 30000, enabled = true): DatasourceState {
  const [status, setStatus] = useState<DatasourceStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastFetched, setLastFetched] = useState<number | null>(null);
  const cancelled = useRef(false);

  const refresh = useCallback(async () => {
    if (typeof document !== "undefined" && document.visibilityState === "hidden") return;
    try {
      const s = await api.datasourceStatus();
      if (cancelled.current) return;
      setStatus(s);
      setError(null);
    } catch (e) {
      if (cancelled.current) return;
      setError(describeApiError(e));
    } finally {
      if (!cancelled.current) {
        setLoading(false);
        setLastFetched(Date.now());
      }
    }
  }, []);

  useEffect(() => {
    cancelled.current = false;
    if (!enabled) return;
    refresh();
    const t = setInterval(refresh, intervalMs);
    return () => {
      cancelled.current = true;
      clearInterval(t);
    };
  }, [enabled, intervalMs, refresh]);

  return { status, error, loading, lastFetched, refresh };
}

// Plain-language summary of the datasource hops.
export function datasourceSummary(ds: DatasourceState): { label: string; tone: Tone } {
  if (!ds.status) {
    if (ds.loading) return { label: "檢查中…", tone: "neutral" };
    return { label: "無法取得狀態", tone: "warn" };
  }
  const s = ds.status;
  if (s.simulation?.active) return { label: "模擬資料", tone: "warn" };
  if (!s.prometheus.reachable) return { label: "Prometheus 無法連線", tone: "danger" };
  if (s.exporter.health === "down") return { label: "量測輸出端離線", tone: "danger" };
  if (s.exporter.health === "unknown") return { label: "輸出端狀態未知", tone: "warn" };
  if (s.clinical_metrics_present.length === 0) return { label: "尚無臨床量測指標", tone: "warn" };
  return { label: "正常", tone: "ok" };
}

const DOT: Record<Tone, string> = {
  ok: "bg-ok",
  warn: "bg-warn",
  danger: "bg-danger",
  neutral: "bg-ink-faint",
  info: "bg-info",
};

function Item({
  label,
  value,
  tone = "neutral",
  attrs,
}: {
  label: string;
  value: string;
  tone?: Tone;
  attrs?: Record<string, string>;
}) {
  return (
    <div className="flex items-center gap-1.5 whitespace-nowrap">
      <dt className="flex items-center gap-1.5 text-ink-dim">
        <span
          aria-hidden="true"
          {...attrs}
          className={`inline-block h-2 w-2 shrink-0 rounded-full ${DOT[tone]}`}
        />
        {label}
      </dt>
      <dd className="font-medium tabular-nums text-ink">{value}</dd>
    </div>
  );
}

type Action = "reload" | "datasource" | "timerange";

interface Notice {
  badge: string;
  tone: Tone;
  text: string;
  actions: Action[];
}

export function DataStatusStrip({
  backend,
  datasource,
  onOpenDataSource,
  onFocusTimeRange,
}: {
  backend: BackendStatusState;
  datasource: DatasourceState;
  onOpenDataSource: () => void;
  onFocusTimeRange: () => void;
}) {
  const tr = useTimeRange();
  const agg = useDashboardDataStatus();

  const backendItem: { value: string; tone: Tone } =
    backend.status === "up"
      ? { value: "正常", tone: "ok" }
      : backend.status === "down"
        ? { value: "中斷", tone: "danger" }
        : { value: "檢查中", tone: "neutral" };

  const dsSummary = datasourceSummary(datasource);

  const receiving: { value: string; tone: Tone } =
    agg.total === 0
      ? { value: "—", tone: "neutral" }
      : agg.receiving
        ? { value: "接收中", tone: "ok" }
        : { value: "未接收", tone: "warn" };

  const lastQuery = agg.lastQueryAt ?? tr.lastSuccessAt;
  const lastSample = agg.lastSampleAt ?? datasource.status?.last_sample_ts ?? null;

  let notice: Notice | null = null;
  if (backend.status === "down") {
    notice = {
      badge: "伺服器中斷",
      tone: "danger",
      text: "無法連線到伺服器，儀表板暫時無法更新。",
      actions: ["reload"],
    };
  } else if (agg.total > 0) {
    switch (agg.worstState) {
      case "offline":
        notice = {
          badge: "來源中斷",
          tone: "danger",
          text: "資料來源中斷：無法從 Prometheus 或量測輸出端取得資料。",
          actions: ["datasource", "reload"],
        };
        break;
      case "error":
        notice = {
          badge: "查詢失敗",
          tone: "danger",
          text: `部分元件查詢失敗${agg.worstReason ? `（${agg.worstReason}）` : ""}，請重新載入；若持續發生請檢查資料來源。`,
          actions: ["reload", "datasource"],
        };
        break;
      case "stale":
        notice = {
          badge: "資料過期",
          tone: "warn",
          text: "資料已過期：最後量測時間超過更新預算，量測可能已停止接收。",
          actions: ["datasource", "reload"],
        };
        break;
      case "empty":
        notice = {
          badge: "尚無資料",
          tone: "warn",
          text: "目前時間範圍內沒有量測資料，可調整時間範圍或檢查資料來源。",
          actions: ["timerange", "datasource"],
        };
        break;
      case "loading":
        notice = { badge: "載入中", tone: "neutral", text: "正在載入資料…", actions: [] };
        break;
      default:
        notice = null;
    }
  }

  const okCount = agg.counts.ok;

  return (
    <div
      data-data-status={agg.worstState}
      aria-label="資料狀態"
      className="flex flex-col gap-2 border-b border-edge bg-surface px-3 py-2 text-sm"
    >
      <dl className="flex flex-wrap items-center gap-x-4 gap-y-1">
        <Item
          label="伺服器連線"
          value={backendItem.value}
          tone={backendItem.tone}
          attrs={{ "data-backend-status": backend.status }}
        />
        <Item label="資料來源" value={dsSummary.label} tone={dsSummary.tone} />
        <Item label="資料接收" value={receiving.value} tone={receiving.tone} />
        <Item label="最後查詢" value={formatClock(lastQuery)} tone={lastQuery ? "neutral" : "neutral"} />
        <Item label="最後量測" value={formatClock(lastSample)} tone="neutral" />
        {tr.inFlightCount > 0 ? (
          <span className="flex items-center gap-1 text-xs text-ink-dim">
            <span
              aria-hidden="true"
              className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-accent motion-reduce:animate-none"
            />
            更新中
          </span>
        ) : null}
      </dl>

      {notice ? (
        <div
          role="status"
          aria-live="polite"
          className="flex flex-wrap items-center gap-2 rounded-md border border-edge bg-canvas px-2 py-1.5"
        >
          <Badge tone={notice.tone}>{notice.badge}</Badge>
          {/* basis-full: on narrow screens the badge + counter (fixed
              width) leave only a sliver of the flex-wrap row for this
              span; flex-1/min-w-0 alone lets it shrink into that
              sliver instead of wrapping to its own line, so long CJK
              text renders one character per line. Force a full-width
              line below sm, revert to the original flexible behavior
              at sm+ where there is room. */}
          <span className="min-w-0 basis-full text-ink sm:basis-0 sm:flex-1">{notice.text}</span>
          {agg.total > 0 ? (
            <span className="text-xs text-ink-dim">
              {okCount}/{agg.total} 個元件有資料
            </span>
          ) : null}
          {notice.actions.length > 0 ? (
            <div className="flex flex-wrap gap-1.5">
              {notice.actions.map((a) =>
                a === "reload" ? (
                  <Button key={a} type="button" variant="secondary" size="sm" onClick={() => tr.refreshNow()}>
                    重新載入
                  </Button>
                ) : a === "datasource" ? (
                  <Button key={a} type="button" variant="secondary" size="sm" onClick={onOpenDataSource}>
                    檢查資料來源
                  </Button>
                ) : (
                  <Button key={a} type="button" variant="secondary" size="sm" onClick={onFocusTimeRange}>
                    調整時間範圍
                  </Button>
                ),
              )}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
