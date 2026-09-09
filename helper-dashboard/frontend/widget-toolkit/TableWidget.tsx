"use client";

import React, { useId, useMemo, useState } from "react";

import { Button } from "@/components/ui/Button";
import { cx } from "@/components/ui/cx";
import { inputClassName } from "@/components/ui/Field";
import { sampleTsOf } from "@/lib/data-status";
import type { WidgetSpec } from "@/lib/spec-schema";
import type { WidgetData } from "@/lib/useQueryData";
import {
  WidgetFrame,
  fmtClock,
  formatValue,
  frameStatusProps,
} from "./WidgetFrame";

// Table. Columns come from `columns` (multi-query tables joined by
// patient_id) or from the single query's labels + one 數值 column. Headers
// are 繁中 (patient_id -> 病患, device_model -> 裝置, value -> 數值), units
// live in the header, a 量測時間 column appears when rows carry sample
// timestamps, null renders 「—」 with data-missing (never 0). Click a header
// to sort; a filter box appears above 8 rows. On phones only the primary
// columns show and each row has a 「展開」 toggle for the rest.

const LABEL_HEADERS: Record<string, string> = {
  patient_id: "病患",
  device_model: "裝置",
  instance: "節點",
  job: "工作",
  __name__: "指標",
  alertname: "警報",
  severity: "嚴重度",
  alertstate: "狀態",
};

const LABEL_PRIORITY = ["patient_id", "device_model", "alertname", "severity"];

interface Col {
  key: string;
  header: string;
  unit: string | null;
  kind: "label" | "value" | "time";
}

interface Cell {
  text: string;
  num: number | null;
  missing: boolean;
}

interface Row {
  id: string;
  cells: Record<string, Cell>;
}

function orderLabelKeys(keys: Set<string>): string[] {
  const all = Array.from(keys);
  const prioritized = LABEL_PRIORITY.filter((k) => keys.has(k));
  const rest = all.filter((k) => !LABEL_PRIORITY.includes(k)).sort();
  return [...prioritized, ...rest];
}

function labelCell(v: string | undefined): Cell {
  const text = v ?? "";
  return { text: text || "—", num: null, missing: !text };
}

function numberCell(v: number | null | undefined, decimals: number | null): Cell {
  if (v === null || v === undefined || !Number.isFinite(v)) {
    return { text: "—", num: null, missing: true };
  }
  return { text: formatValue(v, decimals), num: v, missing: false };
}

function timeCell(ts: number | null | undefined): Cell {
  if (ts === null || ts === undefined || !Number.isFinite(ts)) {
    return { text: "—", num: null, missing: true };
  }
  return { text: fmtClock(ts), num: ts, missing: false };
}

function buildModel(
  widget: WidgetSpec,
  data: WidgetData,
  decimals: number | null,
): { cols: Col[]; rows: Row[] } {
  const unit = widget.encoding?.unit || null;

  if (data.joined.length > 0 && data.columns.length > 0) {
    const labelKeys = new Set<string>();
    for (const r of data.joined) for (const k of Object.keys(r.labels ?? {})) labelKeys.add(k);
    const labels = orderLabelKeys(labelKeys);
    const cols: Col[] = [
      ...labels.map<Col>((k) => ({
        key: `l:${k}`,
        header: LABEL_HEADERS[k] ?? k,
        unit: null,
        kind: "label",
      })),
      ...data.columns.map<Col>((c) => ({
        key: `v:${c.key}`,
        header: c.label || c.key,
        unit: c.unit ?? null,
        kind: "value",
      })),
      { key: "t", header: "量測時間", unit: null, kind: "time" },
    ];
    const rows: Row[] = data.joined.map((r, i) => {
      const cells: Record<string, Cell> = {};
      for (const k of labels) cells[`l:${k}`] = labelCell(r.labels?.[k]);
      let latest: number | null = null;
      for (const c of data.columns) {
        cells[`v:${c.key}`] = numberCell(r.values?.[c.key], decimals);
        const ts = r.ts?.[c.key];
        if (typeof ts === "number" && Number.isFinite(ts)) {
          latest = latest === null ? ts : Math.max(latest, ts);
        }
      }
      cells.t = timeCell(latest);
      return { id: `${i}`, cells };
    });
    return { cols, rows };
  }

  const labelKeys = new Set<string>();
  for (const r of data.table) for (const k of Object.keys(r.metric)) labelKeys.add(k);
  const labels = orderLabelKeys(labelKeys);
  const anyTs = data.table.some(
    (r) => typeof r.ts === "number" && Number.isFinite(r.ts as number),
  );
  const cols: Col[] = [
    ...labels.map<Col>((k) => ({
      key: `l:${k}`,
      header: LABEL_HEADERS[k] ?? k,
      unit: null,
      kind: "label",
    })),
    { key: "v", header: "數值", unit, kind: "value" },
    ...(anyTs
      ? [{ key: "t", header: "量測時間", unit: null, kind: "time" } as Col]
      : []),
  ];
  const rows: Row[] = data.table.map((r, i) => {
    const cells: Record<string, Cell> = {};
    for (const k of labels) cells[`l:${k}`] = labelCell(r.metric[k]);
    cells.v = numberCell(r.value, decimals);
    if (anyTs) cells.t = timeCell(r.ts ?? null);
    return { id: `${i}`, cells };
  });
  return { cols, rows };
}

type SortDir = "asc" | "desc";

export function TableWidget({
  widget,
  data,
}: {
  widget: WidgetSpec;
  data: WidgetData;
}) {
  const opts = (widget.options ?? {}) as Record<string, unknown>;
  const rowLimit = Math.max(1, Number(opts.row_limit ?? 20) || 20);
  const decimals = typeof opts.decimals === "number" ? (opts.decimals as number) : null;
  const filterId = useId();

  const { cols, rows } = useMemo(
    () => buildModel(widget, data, decimals),
    [widget, data, decimals],
  );

  // Default sort from options.sort_by / sort_dir (label key or column key).
  const defaultSort = useMemo<{ key: string; dir: SortDir } | null>(() => {
    const by = typeof opts.sort_by === "string" ? (opts.sort_by as string) : null;
    if (!by) return null;
    const dir: SortDir = opts.sort_dir === "asc" ? "asc" : "desc";
    const col =
      cols.find((c) => c.key === by) ??
      cols.find((c) => c.key === `l:${by}` || c.key === `v:${by}`) ??
      (by === "value" ? cols.find((c) => c.kind === "value") : undefined);
    return col ? { key: col.key, dir } : null;
  }, [opts.sort_by, opts.sort_dir, cols]);

  const [sort, setSort] = useState<{ key: string; dir: SortDir } | null>(null);
  const [filter, setFilter] = useState("");
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const effectiveSort = sort ?? defaultSort;

  const visibleRows = useMemo(() => {
    let out = rows;
    const q = filter.trim().toLowerCase();
    if (q) {
      out = out.filter((r) =>
        Object.values(r.cells).some((c) => c.text.toLowerCase().includes(q)),
      );
    }
    if (effectiveSort) {
      const { key, dir } = effectiveSort;
      const sign = dir === "asc" ? 1 : -1;
      out = [...out].sort((a, b) => {
        const ca = a.cells[key];
        const cb = b.cells[key];
        if (!ca || !cb) return 0;
        // Missing values always sink to the bottom.
        if (ca.missing !== cb.missing) return ca.missing ? 1 : -1;
        if (ca.num !== null && cb.num !== null) return (ca.num - cb.num) * sign;
        return ca.text.localeCompare(cb.text, "zh-Hant") * sign;
      });
    }
    return out.slice(0, rowLimit);
  }, [rows, filter, effectiveSort, rowLimit]);

  const toggleSort = (key: string) => {
    setSort((prev) => {
      if (!prev || prev.key !== key) {
        const col = cols.find((c) => c.key === key);
        return { key, dir: col?.kind === "label" ? "asc" : "desc" };
      }
      return { key, dir: prev.dir === "asc" ? "desc" : "asc" };
    });
  };

  // Phones: first label column + first value column stay; the rest fold.
  const firstLabel = cols.find((c) => c.kind === "label")?.key;
  const firstValue = cols.find((c) => c.kind === "value")?.key;
  const primary = new Set([firstLabel, firstValue].filter(Boolean) as string[]);
  const hasSecondary = cols.some((c) => !primary.has(c.key));

  const sampleTs = sampleTsOf(data);
  const showFilter = rows.length > 8;

  return (
    <WidgetFrame
      widget={widget}
      unit={data.joined.length > 0 ? null : widget.encoding?.unit}
      {...frameStatusProps(data)}
    >
      <div className="flex h-full flex-col">
        {showFilter ? (
          <div className="mb-2 flex shrink-0 items-center gap-2">
            <label htmlFor={filterId} className="text-sm text-ink-dim">
              篩選
            </label>
            <input
              id={filterId}
              type="search"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="輸入病患、裝置或數值…"
              className={cx(inputClassName, "max-w-xs")}
            />
          </div>
        ) : null}
        <div className="min-h-0 flex-1 overflow-auto">
          <table className="w-full border-collapse text-left text-sm">
            <thead className="sticky top-0 bg-surface">
              <tr className="border-b border-edge text-ink-dim">
                {cols.map((c) => {
                  const active = effectiveSort?.key === c.key;
                  const ariaSort = active
                    ? effectiveSort?.dir === "asc"
                      ? "ascending"
                      : "descending"
                    : "none";
                  return (
                    <th
                      key={c.key}
                      scope="col"
                      aria-sort={ariaSort}
                      className={cx(
                        "px-2 py-1.5 font-medium",
                        c.kind !== "label" && "text-right",
                        !primary.has(c.key) && "hidden sm:table-cell",
                      )}
                    >
                      <button
                        type="button"
                        onClick={() => toggleSort(c.key)}
                        className={cx(
                          "inline-flex min-h-8 items-center gap-1 rounded px-1 hover:text-ink",
                          c.kind !== "label" && "flex-row-reverse",
                          active && "text-ink",
                        )}
                      >
                        <span>
                          {c.header}
                          {c.unit ? (
                            <span className="ml-1 font-mono text-xs text-ink-faint">
                              ({c.unit})
                            </span>
                          ) : null}
                        </span>
                        <span aria-hidden="true" className="text-xs">
                          {active ? (effectiveSort?.dir === "asc" ? "▲" : "▼") : "↕"}
                        </span>
                      </button>
                    </th>
                  );
                })}
                {hasSecondary ? (
                  <th scope="col" className="w-16 sm:hidden">
                    <span className="sr-only">更多欄位</span>
                  </th>
                ) : null}
              </tr>
            </thead>
            <tbody>
              {visibleRows.map((r) => {
                const isOpen = Boolean(expanded[r.id]);
                return (
                  <React.Fragment key={r.id}>
                    <tr className="border-b border-edge/70 hover:bg-surface-2/60">
                      {cols.map((c) => {
                        const cell = r.cells[c.key];
                        return (
                          <td
                            key={c.key}
                            data-missing={cell?.missing ? "true" : undefined}
                            className={cx(
                              "px-2 py-1.5 align-top",
                              c.kind === "label" ? "text-ink-dim" : "text-right font-mono tabular-nums text-ink",
                              cell?.missing && "text-ink-faint",
                              !primary.has(c.key) && "hidden sm:table-cell",
                            )}
                          >
                            {cell?.text ?? "—"}
                          </td>
                        );
                      })}
                      {hasSecondary ? (
                        <td className="px-1 py-1 text-right sm:hidden">
                          <Button
                            size="sm"
                            variant="ghost"
                            aria-expanded={isOpen}
                            onClick={() =>
                              setExpanded((e) => ({ ...e, [r.id]: !isOpen }))
                            }
                          >
                            {isOpen ? "收合" : "展開"}
                          </Button>
                        </td>
                      ) : null}
                    </tr>
                    {hasSecondary && isOpen ? (
                      <tr className="border-b border-edge/70 bg-surface-2/40 sm:hidden">
                        <td colSpan={cols.length + 1} className="px-2 py-2">
                          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-sm">
                            {cols
                              .filter((c) => !primary.has(c.key))
                              .map((c) => {
                                const cell = r.cells[c.key];
                                return (
                                  <React.Fragment key={c.key}>
                                    <dt className="text-ink-dim">
                                      {c.header}
                                      {c.unit ? ` (${c.unit})` : ""}
                                    </dt>
                                    <dd
                                      data-missing={cell?.missing ? "true" : undefined}
                                      className={cx(
                                        "font-mono tabular-nums",
                                        cell?.missing ? "text-ink-faint" : "text-ink",
                                      )}
                                    >
                                      {cell?.text ?? "—"}
                                    </dd>
                                  </React.Fragment>
                                );
                              })}
                          </dl>
                        </td>
                      </tr>
                    ) : null}
                  </React.Fragment>
                );
              })}
              {visibleRows.length === 0 ? (
                <tr>
                  <td
                    colSpan={cols.length + (hasSecondary ? 1 : 0)}
                    className="px-2 py-6 text-center text-sm text-ink-faint"
                  >
                    {filter.trim() ? "沒有符合篩選條件的資料列" : "沒有資料列"}
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
        <div className="mt-1 flex shrink-0 flex-wrap items-center justify-between gap-x-2 text-xs text-ink-faint">
          <span>
            {rows.length > visibleRows.length
              ? `顯示 ${visibleRows.length} / ${rows.length} 列`
              : `${rows.length} 列`}
          </span>
          {sampleTs !== null && !cols.some((c) => c.kind === "time") ? (
            <span className="font-mono tabular-nums">最後量測 {fmtClock(sampleTs)}</span>
          ) : null}
        </div>
      </div>
    </WidgetFrame>
  );
}
