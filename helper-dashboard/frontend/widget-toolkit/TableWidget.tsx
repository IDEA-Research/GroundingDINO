"use client";

import React from "react";

import type { WidgetSpec } from "@/lib/spec-schema";
import type { WidgetData } from "@/lib/useQueryData";
import { WidgetFrame } from "./WidgetFrame";

export function TableWidget({
  widget,
  data,
}: {
  widget: WidgetSpec;
  data: WidgetData;
}) {
  const rowLimit = Number((widget.options as any)?.row_limit ?? 20);
  const rows = data.table.slice(0, rowLimit);

  const columns: string[] = Array.from(
    rows.reduce((acc, r) => {
      for (const k of Object.keys(r.metric)) acc.add(k);
      return acc;
    }, new Set<string>()),
  );

  return (
    <WidgetFrame widget={widget} unit={widget.encoding?.unit}>
      <div className="h-full overflow-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-edge text-ink-faint">
              {columns.map((c) => (
                <th key={c} className="px-2 py-1 font-medium">
                  {c}
                </th>
              ))}
              <th className="px-2 py-1 font-medium">value</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="border-b border-edge/60 hover:bg-surface-2/60">
                {columns.map((c) => (
                  <td key={c} className="px-2 py-1 text-ink-dim">
                    {r.metric[c] ?? ""}
                  </td>
                ))}
                <td className="px-2 py-1 font-mono tabular-nums text-ink">
                  {r.value.toFixed(2)}
                </td>
              </tr>
            ))}
            {rows.length === 0 ? (
              <tr>
                <td
                  colSpan={columns.length + 1}
                  className="px-2 py-4 text-center text-ink-faint"
                >
                  沒有資料列
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </WidgetFrame>
  );
}
