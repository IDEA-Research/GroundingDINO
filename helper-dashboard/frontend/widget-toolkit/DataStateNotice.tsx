"use client";

import React from "react";

import type { WidgetData } from "@/lib/useQueryData";

// Shared, unmissable non-ok states for every query-driven widget.
//
// A clinical dashboard must never render silence: "no data" and "error"
// are rendered as explicit, visible states — never a blank panel and
// never fabricated values. (Metrics once went stale for 11 days without
// anyone noticing, because "no data" rendered as nothing.)
export function DataStateNotice({
  data,
  promql,
}: {
  data: WidgetData;
  promql: string;
}) {
  if (data.status === "loading") {
    return (
      <div
        data-state="loading"
        className="flex h-full items-center justify-center text-xs text-ink-faint"
      >
        載入中…
      </div>
    );
  }

  if (data.status === "error") {
    return (
      <div
        data-state="error"
        className="flex h-full flex-col items-center justify-center gap-1 rounded border-2 border-alarm-high bg-alarm-high/10 p-2 text-center"
      >
        <div className="text-sm font-bold tracking-wide text-alarm-high">
          查詢錯誤
        </div>
        <div className="max-w-full break-words text-xs text-ink">
          {data.errorDetail || "unknown error"}
        </div>
        <code className="max-w-full truncate font-mono text-xs text-alarm-high/80">
          {promql}
        </code>
      </div>
    );
  }

  // status === "empty" — the query succeeded but returned nothing.
  return (
    <div
      data-state="empty"
      className="flex h-full flex-col items-center justify-center gap-1 rounded border-2 border-dashed border-edge-strong bg-surface-2/60 p-2 text-center"
    >
      <div className="text-sm font-bold uppercase tracking-wide text-ink">
        尚無資料 NO DATA
      </div>
      <div className="text-xs text-ink-faint">
        查詢成功但沒有任何資料點
      </div>
      <code className="max-w-full truncate font-mono text-xs text-ink-faint">
        {promql}
      </code>
    </div>
  );
}
