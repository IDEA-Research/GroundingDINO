"use client";

import React from "react";

import { Button } from "@/components/ui/Button";
import { Disclosure } from "@/components/ui/Disclosure";
import { Skeleton } from "@/components/ui/Skeleton";
import { cx } from "@/components/ui/cx";
import { useDevMode } from "@/lib/dev-mode";
import { REASON_TEXT, type QueryStatus, type WidgetData } from "@/lib/useQueryData";
import { SeverityIcon } from "./WidgetFrame";

// Shared, unmissable non-ok states for every query-driven widget.
//
// A clinical dashboard must never render silence: "no data", "offline",
// "stale" and "error" are explicit, visible states — never a blank panel
// and never fabricated values. (Metrics once went stale for 11 days
// without anyone noticing, because "no data" rendered as nothing.)
//
// Two variants:
//   compact — inside a card: 「—」 + a short reason (尚無資料 / 來源中斷 /
//             資料過期 / 查詢失敗 / 載入中). No long copy per card.
//   full    — ONE dashboard-level notice (DataStatusStrip): copy + actions
//             (重新載入 / 檢查資料來源 / 調整時間範圍).
// PromQL / raw errors only appear inside a 「技術細節」 disclosure and only
// when the developer workspace is unlocked.

export interface DataStateActions {
  onReload?: () => void;
  onCheckSource?: () => void;
  onAdjustRange?: () => void;
}

export type NoticeState = Exclude<QueryStatus, "ok">;

interface Copy {
  title: string;
  body: string;
  severity: "danger" | "warn" | "ok" | "info";
  actions: (keyof DataStateActions)[];
}

const COPY: Record<NoticeState, Copy> = {
  loading: {
    title: REASON_TEXT.loading,
    body: "正在向資料來源查詢。",
    severity: "info",
    actions: [],
  },
  empty: {
    title: REASON_TEXT.empty,
    body: "查詢成功,但目前的時間範圍與篩選條件下沒有任何量測值。",
    severity: "info",
    actions: ["onAdjustRange", "onCheckSource"],
  },
  offline: {
    title: REASON_TEXT.offline,
    body: "無法連線至資料來源,顯示的內容可能不是最新狀態。",
    severity: "danger",
    actions: ["onCheckSource", "onReload"],
  },
  stale: {
    title: REASON_TEXT.stale,
    body: "最後量測已超過新鮮度預算,畫面顯示的是最後一次取得的數值。",
    severity: "warn",
    actions: ["onReload", "onCheckSource"],
  },
  error: {
    title: REASON_TEXT.error,
    body: "查詢未能完成,請重新載入;若持續發生,請檢查資料來源。",
    severity: "danger",
    actions: ["onReload", "onCheckSource"],
  },
};

const ACTION_LABEL: Record<keyof DataStateActions, string> = {
  onReload: "重新載入",
  onCheckSource: "檢查資料來源",
  onAdjustRange: "調整時間範圍",
};

const SEVERITY_TEXT: Record<Copy["severity"], string> = {
  danger: "text-danger",
  warn: "text-warn",
  ok: "text-ok",
  info: "text-ink-dim",
};

export function noticeState(data: WidgetData): NoticeState {
  return data.status === "ok" ? "stale" : data.status;
}

export interface DataStateNoticeProps {
  data: WidgetData;
  promql?: string | null;
  variant?: "compact" | "full";
  actions?: DataStateActions;
  // Override the reason line (compact) / title (full).
  title?: string;
  className?: string;
}

function TechnicalDetails({
  data,
  promql,
}: {
  data: WidgetData;
  promql?: string | null;
}) {
  const { unlocked } = useDevMode();
  if (!unlocked) return null;
  if (!promql && !data.errorDetail) return null;
  return (
    <Disclosure summary="技術細節" className="mt-1 text-left">
      <div className="space-y-1 text-xs">
        {data.errorDetail ? (
          <div className="break-words font-mono text-ink-dim">
            {data.errorStatus ? `HTTP ${data.errorStatus} · ` : ""}
            {data.errorDetail}
          </div>
        ) : null}
        {promql ? (
          <code className="block break-all font-mono text-ink-faint">{promql}</code>
        ) : null}
      </div>
    </Disclosure>
  );
}

export function DataStateNotice({
  data,
  promql,
  variant = "compact",
  actions,
  title,
  className,
}: DataStateNoticeProps) {
  const state = noticeState(data);
  const copy = COPY[state];
  const reason = title ?? data.reason ?? copy.title;

  if (variant === "compact") {
    if (state === "loading") {
      return (
        <div
          data-state="loading"
          role="status"
          aria-label="載入中"
          className={cx(
            "flex h-full flex-col justify-center gap-2 px-1",
            className,
          )}
        >
          <Skeleton className="h-8 w-2/3" />
          <Skeleton className="h-3.5 w-1/3" />
          <span className="text-xs text-ink-faint">載入中</span>
        </div>
      );
    }
    return (
      <div
        data-state={state}
        role={state === "offline" || state === "error" ? "status" : undefined}
        className={cx(
          "flex h-full flex-col items-center justify-center gap-1 text-center",
          className,
        )}
      >
        <span
          aria-hidden="true"
          className="stat-numeral font-mono text-ink-faint"
        >
          —
        </span>
        <span
          className={cx(
            "inline-flex items-center gap-1.5 text-sm font-medium",
            SEVERITY_TEXT[copy.severity],
          )}
        >
          <SeverityIcon severity={copy.severity} />
          {reason}
        </span>
        <TechnicalDetails data={data} promql={promql} />
      </div>
    );
  }

  // full variant
  const available = copy.actions.filter((k) => actions && actions[k]);
  return (
    <div
      data-state={state}
      role={state === "offline" || state === "error" ? "alert" : "status"}
      className={cx(
        "flex flex-col gap-2 rounded-lg border border-edge bg-surface p-3 text-sm text-ink shadow-card sm:flex-row sm:items-start sm:justify-between",
        className,
      )}
    >
      <div className="min-w-0 flex-1">
        <div
          className={cx(
            "inline-flex items-center gap-1.5 font-semibold",
            SEVERITY_TEXT[copy.severity],
          )}
        >
          <SeverityIcon severity={copy.severity} />
          {reason}
        </div>
        <p className="mt-0.5 text-sm text-ink-dim">{copy.body}</p>
        <TechnicalDetails data={data} promql={promql} />
      </div>
      {available.length > 0 ? (
        <div className="flex shrink-0 flex-wrap gap-2">
          {available.map((k, i) => (
            <Button
              key={k}
              size="sm"
              variant={i === 0 ? "primary" : "secondary"}
              onClick={actions?.[k]}
            >
              {ACTION_LABEL[k]}
            </Button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
