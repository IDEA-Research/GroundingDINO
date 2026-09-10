"use client";

import React, { useMemo, useState } from "react";

import {
  Button,
  Drawer,
  Field,
  Input,
  Select,
  Tabs,
  Textarea,
  type TabItem,
} from "@/components/ui";
import { renderWidget } from "@/lib/renderer";
import type {
  QuerySpec,
  ThresholdDirection,
  WidgetSpec,
  WidgetThreshold,
} from "@/lib/spec-schema";

// Panel edit drawer (Phase 5): the real destination for WidgetFrame's
// "編輯" hover-menu item (see WidgetActionsProvider in WidgetFrame.tsx).
// Built entirely from existing pieces — the shared Drawer/Tabs shell,
// lib/renderer.tsx's own renderWidget() dispatch for the live preview —
// nothing here re-implements widget rendering or invents a new drawer
// chrome.
//
// Editing happens on a local, disconnected clone (`draft`); nothing is
// written back to the caller's spec until 儲存 fires `onSave(draft)`. The
// caller (DashboardPreview.tsx) owns turning that into a full
// DashboardSpec and persisting it (see its handleWidgetSaved).

export interface WidgetEditDrawerProps {
  widget: WidgetSpec;
  onClose: () => void;
  onSave: (next: WidgetSpec) => void;
}

type EditTabId = "panel" | "thresholds" | "visual" | "source";

// Canonical, backend-pattern-safe (`^[a-zA-Z]{3,20}$`) color words that
// lib/chart-theme.ts's severityForSpec() already recognizes by name
// (red-family -> danger, amber/orange/yellow -> warn, green-family -> ok)
// — a quick-pick, not the only allowed values (the free-text field next
// to it accepts any word or hex the backend pattern allows).
const SEMANTIC_COLOR_PRESETS: { value: string; label: string }[] = [
  { value: "green", label: "綠 · 正常 (ok)" },
  { value: "amber", label: "琥珀 · 警示 (warn)" },
  { value: "red", label: "紅 · 危險 (danger)" },
];

function cloneWidget(w: WidgetSpec): WidgetSpec {
  return JSON.parse(JSON.stringify(w)) as WidgetSpec;
}

function isHex6(s: string): boolean {
  return /^#[0-9a-fA-F]{6}$/.test(s.trim());
}

export function WidgetEditDrawer({ widget, onClose, onSave }: WidgetEditDrawerProps) {
  const [draft, setDraft] = useState<WidgetSpec>(() => cloneWidget(widget));
  const [tab, setTab] = useState<EditTabId>("panel");

  // Suffixed id: the live grid panel for this widget is very likely still
  // mounted behind the drawer, and lib/data-status.ts's dashboard-status
  // registry is a single module-level Map keyed by widget id — sharing
  // the real id would make closing the drawer unregister the LIVE
  // widget's status entry too (its own effect only re-registers on its
  // next data change). A distinct id sidesteps that; nothing else keys
  // off a widget's id in a way this preview relies on.
  const previewWidget = useMemo<WidgetSpec>(
    () => ({ ...draft, id: `${draft.id}__edit-preview` }),
    [draft],
  );

  function updateThreshold(index: number, patch: Partial<WidgetThreshold>) {
    setDraft((prev) => {
      const next = [...(prev.thresholds ?? [])];
      next[index] = { ...next[index], ...patch };
      return { ...prev, thresholds: next };
    });
  }

  function removeThreshold(index: number) {
    setDraft((prev) => ({
      ...prev,
      thresholds: (prev.thresholds ?? []).filter((_, i) => i !== index),
    }));
  }

  function addThreshold() {
    setDraft((prev) => ({
      ...prev,
      thresholds: [...(prev.thresholds ?? []), { value: 0, color: "amber", label: "" }],
    }));
  }

  function updateOption(key: string, value: unknown) {
    setDraft((prev) => ({
      ...prev,
      options: { ...(prev.options ?? {}), [key]: value },
    }));
  }

  const tabs: TabItem[] = [
    { id: "panel", label: "Panel options" },
    {
      id: "thresholds",
      label: "Thresholds",
      badge: draft.thresholds && draft.thresholds.length > 0 ? draft.thresholds.length : undefined,
    },
    { id: "visual", label: "視覺化設定" },
    { id: "source", label: "資料來源" },
  ];

  return (
    <Drawer
      id="hd-widget-edit-drawer"
      open
      onClose={onClose}
      side="right"
      title={`編輯面板：${widget.title}`}
      width={520}
      resizable
      bodyClassName="flex flex-col"
      footer={
        <div className="flex items-center justify-end gap-2">
          <Button type="button" variant="ghost" size="md" onClick={onClose}>
            取消
          </Button>
          <Button
            type="button"
            variant="primary"
            size="md"
            disabled={!draft.title.trim()}
            onClick={() => onSave(draft)}
          >
            儲存
          </Button>
        </div>
      }
    >
      <div className="flex min-h-0 flex-1 flex-col gap-3 p-3">
        <div>
          <div className="mb-1 text-xs font-medium text-ink-faint">即時預覽</div>
          <div className="h-56 shrink-0 overflow-hidden rounded-lg border border-edge bg-canvas p-2">
            {renderWidget(previewWidget) ?? (
              <div className="flex h-full items-center justify-center text-sm text-ink-faint">
                無法預覽此類型
              </div>
            )}
          </div>
        </div>

        <Tabs
          className="min-h-0 flex-1"
          panelClassName="min-h-0 flex-1 overflow-auto pt-3"
          aria-label="面板編輯設定"
          tabs={tabs}
          value={tab}
          onChange={(id) => setTab(id as EditTabId)}
        >
          {(id) => {
            if (id === "panel") {
              return (
                <div className="flex flex-col gap-3">
                  <Field label="標題" htmlFor="hd-widget-title" required>
                    <Input
                      id="hd-widget-title"
                      value={draft.title}
                      onChange={(e) =>
                        setDraft((prev) => ({ ...prev, title: e.target.value }))
                      }
                    />
                  </Field>
                  <Field
                    label="描述"
                    htmlFor="hd-widget-desc"
                    hint="顯示在標題旁說明圖示的 hover 內容，選填。"
                  >
                    <Textarea
                      id="hd-widget-desc"
                      rows={3}
                      value={draft.description ?? ""}
                      onChange={(e) =>
                        setDraft((prev) => ({ ...prev, description: e.target.value }))
                      }
                    />
                  </Field>
                </div>
              );
            }
            if (id === "thresholds") {
              return (
                <ThresholdsPane
                  thresholds={draft.thresholds ?? []}
                  onUpdate={updateThreshold}
                  onRemove={removeThreshold}
                  onAdd={addThreshold}
                />
              );
            }
            if (id === "visual") {
              return <VisualPane widget={draft} onUpdateOption={updateOption} />;
            }
            return <DataSourcePane widget={draft} />;
          }}
        </Tabs>
      </div>
    </Drawer>
  );
}

function ThresholdsPane({
  thresholds,
  onUpdate,
  onRemove,
  onAdd,
}: {
  thresholds: WidgetThreshold[];
  onUpdate: (index: number, patch: Partial<WidgetThreshold>) => void;
  onRemove: (index: number) => void;
  onAdd: () => void;
}) {
  return (
    <div className="flex flex-col gap-3">
      {thresholds.length === 0 ? (
        <p className="text-sm text-ink-dim">尚未設定門檻值。</p>
      ) : null}
      {thresholds.map((t, i) => (
        <div key={i} className="rounded-md border border-edge p-2">
          <div className="flex flex-wrap items-end gap-2">
            <Field label="數值" htmlFor={`hd-th-value-${i}`} className="w-24">
              <Input
                id={`hd-th-value-${i}`}
                type="number"
                value={Number.isFinite(t.value) ? t.value : 0}
                onChange={(e) =>
                  onUpdate(i, {
                    value: e.target.value === "" ? 0 : Number(e.target.value),
                  })
                }
              />
            </Field>
            <Field label="標籤（選填）" htmlFor={`hd-th-label-${i}`} className="min-w-[8rem] flex-1">
              <Input
                id={`hd-th-label-${i}`}
                value={t.label ?? ""}
                onChange={(e) => onUpdate(i, { label: e.target.value })}
              />
            </Field>
            <Field label="方向（選填）" htmlFor={`hd-th-dir-${i}`} className="w-32">
              <Select
                id={`hd-th-dir-${i}`}
                value={t.direction ?? ""}
                onChange={(e) =>
                  onUpdate(i, {
                    direction: (e.target.value || null) as ThresholdDirection | null,
                  })
                }
              >
                <option value="">自動判斷</option>
                <option value="above">高於（above）</option>
                <option value="below">低於（below）</option>
              </Select>
            </Field>
            <Button
              type="button"
              variant="danger"
              size="sm"
              onClick={() => onRemove(i)}
            >
              刪除
            </Button>
          </div>
          <div className="mt-2 flex flex-wrap items-end gap-2">
            <Field label="快速選色" htmlFor={`hd-th-preset-${i}`} className="w-40">
              <Select
                id={`hd-th-preset-${i}`}
                value=""
                onChange={(e) => {
                  if (e.target.value) onUpdate(i, { color: e.target.value });
                }}
              >
                <option value="">— 選擇 —</option>
                {SEMANTIC_COLOR_PRESETS.map((c) => (
                  <option key={c.value} value={c.value}>
                    {c.label}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="顏色字串" htmlFor={`hd-th-color-${i}`} className="min-w-[10rem] flex-1">
              <div className="flex items-center gap-2">
                <Input
                  id={`hd-th-color-${i}`}
                  value={t.color}
                  onChange={(e) => onUpdate(i, { color: e.target.value })}
                  placeholder="red / amber / #dc2626"
                />
                <input
                  type="color"
                  aria-label="以色盤選擇顏色"
                  className="h-9 w-9 shrink-0 rounded-md border border-edge-strong bg-surface p-0"
                  value={isHex6(t.color) ? t.color : "#dc2626"}
                  onChange={(e) => onUpdate(i, { color: e.target.value })}
                />
              </div>
            </Field>
          </div>
        </div>
      ))}
      <Button type="button" variant="secondary" size="sm" onClick={onAdd} className="self-start">
        + 新增門檻
      </Button>
    </div>
  );
}

// stat_card's color_mode and bar_gauge's display_mode — the two options
// this phase implements, confirmed by reading StatCardWidget.tsx /
// BarGaugeWidget.tsx rather than guessed. GaugeWidget.tsx does NOT read
// any display_mode option today (only bar_gauge does), so "gauge" falls
// through to the generic "no extra settings" message below instead of a
// fabricated control.
function VisualPane({
  widget,
  onUpdateOption,
}: {
  widget: WidgetSpec;
  onUpdateOption: (key: string, value: unknown) => void;
}) {
  const opts = (widget.options ?? {}) as Record<string, unknown>;

  if (widget.type === "stat_card") {
    const colorMode = typeof opts.color_mode === "string" ? opts.color_mode : "none";
    return (
      <div className="flex flex-col gap-3">
        <Field
          label="數值上色模式（color_mode）"
          htmlFor="hd-widget-colormode"
          hint='目前 StatCardWidget 只在 color_mode = "background" 時，於門檻觸發時加上卡片背景色；「無」與「數值」目前渲染效果相同——數值文字本身一律依門檻狀態上色，不受此選項影響。'
        >
          <Select
            id="hd-widget-colormode"
            value={colorMode}
            onChange={(e) =>
              onUpdateOption(
                "color_mode",
                e.target.value === "none" ? undefined : e.target.value,
              )
            }
          >
            <option value="none">無（None）</option>
            <option value="value">數值（Value）</option>
            <option value="background">背景（Background）</option>
          </Select>
        </Field>
      </div>
    );
  }

  if (widget.type === "bar_gauge") {
    const mode =
      opts.display_mode === "gradient" || opts.display_mode === "retro-lcd"
        ? (opts.display_mode as string)
        : "basic";
    return (
      <div className="flex flex-col gap-3">
        <Field label="顯示模式（display_mode）" htmlFor="hd-widget-displaymode">
          <Select
            id="hd-widget-displaymode"
            value={mode}
            onChange={(e) => onUpdateOption("display_mode", e.target.value)}
          >
            <option value="basic">單色（basic）</option>
            <option value="gradient">漸層（gradient）</option>
            <option value="retro-lcd">LCD 分格（retro-lcd）</option>
          </Select>
        </Field>
      </div>
    );
  }

  return (
    <p className="text-sm text-ink-dim">
      「{widget.type}」類型目前沒有額外的視覺化設定。
    </p>
  );
}

// Deliberately read-only: there is no existing API or type that
// enumerates "which fields/metrics this widget could query" (QuerySpec is
// a free-form PromQL string, not a field picker), so a dropdown here
// would have to invent an unverified field list. Honesty over a fake
// affordance — see the task's own instruction on this point.
function DataSourcePane({ widget }: { widget: WidgetSpec }) {
  const queries: QuerySpec[] =
    widget.queries && widget.queries.length > 0
      ? widget.queries
      : widget.query
        ? [widget.query]
        : [];

  return (
    <div className="flex flex-col gap-3 text-sm">
      <p className="text-ink-dim">
        資料來源（來源、PromQL、時間窗）目前僅唯讀顯示。這個 widget 可選欄位並沒有現成的列舉
        API 或型別可以枚舉——為避免顯示一份不保證正確的欄位清單，這裡不提供下拉選單。如需變更
        資料來源，請透過 Helper 對話讓它重新產生查詢規格。
      </p>
      {queries.length === 0 ? (
        <p className="text-ink-faint">此面板沒有查詢設定（例如決策流程圖）。</p>
      ) : (
        <div className="flex flex-col gap-2">
          {queries.map((q, i) => (
            <dl
              key={i}
              className="grid grid-cols-[5rem_1fr] gap-x-2 gap-y-1 rounded-md border border-edge bg-surface-2 p-2 font-mono text-xs"
            >
              <dt className="text-ink-faint">source</dt>
              <dd className="break-all">{q.source}</dd>
              <dt className="text-ink-faint">query_type</dt>
              <dd className="break-all">{q.query_type}</dd>
              <dt className="text-ink-faint">promql</dt>
              <dd className="break-all">{q.promql || "(空)"}</dd>
              {q.range ? (
                <>
                  <dt className="text-ink-faint">range</dt>
                  <dd className="break-all">{q.range}</dd>
                </>
              ) : null}
              {q.step ? (
                <>
                  <dt className="text-ink-faint">step</dt>
                  <dd className="break-all">{q.step}</dd>
                </>
              ) : null}
              {q.key ? (
                <>
                  <dt className="text-ink-faint">key</dt>
                  <dd className="break-all">{q.key}</dd>
                </>
              ) : null}
              {q.label ? (
                <>
                  <dt className="text-ink-faint">label</dt>
                  <dd className="break-all">{q.label}</dd>
                </>
              ) : null}
              {q.unit ? (
                <>
                  <dt className="text-ink-faint">unit</dt>
                  <dd className="break-all">{q.unit}</dd>
                </>
              ) : null}
            </dl>
          ))}
        </div>
      )}
    </div>
  );
}
