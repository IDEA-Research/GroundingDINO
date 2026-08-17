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
        className="flex h-full items-center justify-center text-xs text-slate-500"
      >
        載入中… / loading…
      </div>
    );
  }

  if (data.status === "error") {
    return (
      <div
        data-state="error"
        className="flex h-full flex-col items-center justify-center gap-1 rounded border-2 border-red-600 bg-red-950/50 p-2 text-center"
      >
        <div className="text-sm font-bold uppercase tracking-wide text-red-300">
          資料錯誤 / DATA ERROR
        </div>
        <div className="max-w-full break-words text-xs text-red-200">
          {data.errorDetail || "unknown error"}
        </div>
        <code className="max-w-full truncate text-[10px] text-red-400/80">
          {promql}
        </code>
      </div>
    );
  }

  // status === "empty" — the query succeeded but returned nothing.
  return (
    <div
      data-state="empty"
      className="flex h-full flex-col items-center justify-center gap-1 rounded border-2 border-dashed border-slate-500 bg-slate-900/60 p-2 text-center"
    >
      <div className="text-sm font-bold uppercase tracking-wide text-slate-300">
        無資料 / NO DATA
      </div>
      <div className="text-xs text-slate-400">
        查詢成功但沒有任何資料點 / query returned zero series
      </div>
      <code className="max-w-full truncate text-[10px] text-slate-500">
        {promql}
      </code>
    </div>
  );
}
