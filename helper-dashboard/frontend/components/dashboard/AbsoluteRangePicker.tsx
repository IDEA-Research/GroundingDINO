"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui/Button";
import { cx } from "@/components/ui/cx";
import { Field, Input } from "@/components/ui/Field";
import { useFocusTrap } from "@/components/chat/HelperDrawer";
import {
  formatAbsoluteInstant,
  formatAbsoluteRangeLabel,
  useTimeRange,
  type AbsoluteRange,
  type RangeExpr,
} from "@/lib/time-range";

// Grafana-style split popover for the time-range button's "自訂範圍…" entry:
// left = the same quick-range list already in the pill row (so the popover
// is a complete picker on its own, not just the leftover custom form),
// right = an absolute from/to form with a draggable timeline bar above it.
//
// The bar's track is a DYNAMIC window (3x the current selection, centered),
// not tied to any "when did data start" backend query — this stays a pure
// view-layer convenience and avoids a new API dependency (spec 全域限制).

function toLocalInputValue(epochSeconds: number): string {
  const d = new Date(epochSeconds * 1000);
  const pad = (n: number) => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
    `T${pad(d.getHours())}:${pad(d.getMinutes())}`
  );
}

function fromLocalInputValue(value: string): number | null {
  if (!value) return null;
  const ms = new Date(value).getTime();
  if (!Number.isFinite(ms)) return null;
  return Math.floor(ms / 1000);
}

const MIN_WINDOW_SECONDS = 60;

export function AbsoluteRangePicker({
  open,
  onClose,
  triggerRef,
}: {
  open: boolean;
  onClose: () => void;
  triggerRef: React.RefObject<HTMLElement>;
}) {
  const { range, setRange, setCustomRange, recentCustomRanges, presets } =
    useTimeRange();
  const popRef = useRef<HTMLDivElement>(null);
  useFocusTrap(popRef, open, onClose);

  const nowSec = () => Math.floor(Date.now() / 1000);

  const initial = useMemo((): AbsoluteRange => {
    if (range.expr === "custom" && range.from && range.to) {
      return { from: range.from, to: range.to };
    }
    const to = nowSec();
    const width = range.seconds && range.seconds > 0 ? range.seconds : 3600;
    return { from: to - width, to };
    // Only recompute when the popover (re)opens, not on every context tick.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const [draft, setDraft] = useState<AbsoluteRange>(initial);
  useEffect(() => {
    if (open) setDraft(initial);
  }, [open, initial]);

  const error =
    draft.to - draft.from < MIN_WINDOW_SECONDS
      ? "結束時間必須晚於起始時間至少 1 分鐘"
      : draft.to > nowSec() + 5
        ? "結束時間不能晚於現在"
        : null;

  const apply = () => {
    if (error) return;
    setCustomRange(draft.from, draft.to);
    onClose();
  };

  // Click-outside close (same convention as DashboardPickerMenu/SaveButton).
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node;
      if (popRef.current?.contains(t)) return;
      if (triggerRef.current?.contains(t)) return;
      onClose();
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open, onClose, triggerRef]);

  if (!open) return null;

  return (
    <div
      ref={popRef}
      role="dialog"
      aria-label="自訂時間範圍"
      className="absolute right-0 top-full z-30 mt-1 flex w-[min(92vw,640px)] flex-col overflow-hidden rounded-lg border border-edge bg-surface shadow-pop sm:flex-row"
    >
      <div className="flex flex-col gap-0.5 border-b border-edge p-2 sm:w-40 sm:border-b-0 sm:border-r">
        {presets.map((p) => (
          <button
            key={p.expr}
            type="button"
            onClick={() => {
              setRange(p.expr as RangeExpr);
              onClose();
            }}
            className={cx(
              "min-h-8 rounded px-2.5 text-left text-sm transition-colors",
              range.expr === p.expr
                ? "bg-accent/10 font-medium text-accent"
                : "text-ink-dim hover:bg-surface-2 hover:text-ink",
            )}
          >
            {p.label}
          </button>
        ))}
        {recentCustomRanges.length > 0 ? (
          <>
            <div className="mt-1 border-t border-edge pt-1.5 text-xs font-medium text-ink-faint">
              最近使用
            </div>
            {recentCustomRanges.map((r) => (
              <button
                key={`${r.from}-${r.to}`}
                type="button"
                onClick={() => {
                  setCustomRange(r.from, r.to);
                  onClose();
                }}
                className="min-h-8 rounded px-2.5 text-left text-xs text-ink-dim transition-colors hover:bg-surface-2 hover:text-ink"
              >
                {formatAbsoluteRangeLabel(r.from, r.to)}
              </button>
            ))}
          </>
        ) : null}
      </div>

      <div className="flex flex-1 flex-col gap-3 p-3">
        <RangeDragBar draft={draft} onChange={setDraft} />

        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <Field label="起始時間" htmlFor="hd-abs-range-from">
            <Input
              id="hd-abs-range-from"
              type="datetime-local"
              data-autofocus
              value={toLocalInputValue(draft.from)}
              max={toLocalInputValue(draft.to)}
              onChange={(e) => {
                const v = fromLocalInputValue(e.target.value);
                if (v !== null) setDraft((d) => ({ ...d, from: v }));
              }}
            />
          </Field>
          <Field label="結束時間" htmlFor="hd-abs-range-to">
            <Input
              id="hd-abs-range-to"
              type="datetime-local"
              value={toLocalInputValue(draft.to)}
              min={toLocalInputValue(draft.from)}
              onChange={(e) => {
                const v = fromLocalInputValue(e.target.value);
                if (v !== null) setDraft((d) => ({ ...d, to: v }));
              }}
            />
          </Field>
        </div>

        {error ? (
          <p role="alert" className="text-xs font-medium text-danger">
            {error}
          </p>
        ) : null}

        <div className="flex items-center justify-end gap-2">
          <Button type="button" variant="secondary" size="sm" onClick={onClose}>
            取消
          </Button>
          <Button type="button" variant="primary" size="sm" onClick={apply} disabled={!!error}>
            套用範圍
          </Button>
        </div>
      </div>
    </div>
  );
}

// Draggable timeline: track = 3x the draft window, centered. Two handles
// (from/to) plus a middle segment that pans both together. Pointer events
// cover touch; keyboard (Tab to a handle, arrow keys, Home/End) covers the
// rest. Local `draft` only — nothing commits until 套用範圍 / Enter.
function RangeDragBar({
  draft,
  onChange,
}: {
  draft: AbsoluteRange;
  onChange: (next: AbsoluteRange) => void;
}) {
  const trackRef = useRef<HTMLDivElement>(null);
  const dragState = useRef<{
    mode: "from" | "to" | "pan";
    startClientX: number;
    startDraft: AbsoluteRange;
    trackWidthPx: number;
    trackSeconds: number;
  } | null>(null);

  const width = Math.max(MIN_WINDOW_SECONDS, draft.to - draft.from);
  const center = (draft.from + draft.to) / 2;
  const trackSeconds = width * 3;
  const trackFrom = center - trackSeconds / 2;
  const trackTo = center + trackSeconds / 2;

  const pctFor = (epoch: number) =>
    Math.min(100, Math.max(0, ((epoch - trackFrom) / trackSeconds) * 100));

  const beginDrag = (mode: "from" | "to" | "pan") => (e: React.PointerEvent) => {
    const track = trackRef.current;
    if (!track) return;
    (e.target as Element).setPointerCapture(e.pointerId);
    dragState.current = {
      mode,
      startClientX: e.clientX,
      startDraft: draft,
      trackWidthPx: track.getBoundingClientRect().width,
      trackSeconds,
    };
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const s = dragState.current;
    if (!s) return;
    const deltaPx = e.clientX - s.startClientX;
    const deltaSeconds = (deltaPx / s.trackWidthPx) * s.trackSeconds;
    const nowSec = Math.floor(Date.now() / 1000);

    if (s.mode === "from") {
      const next = Math.min(s.startDraft.from + deltaSeconds, s.startDraft.to - MIN_WINDOW_SECONDS);
      onChange({ from: Math.round(next), to: s.startDraft.to });
    } else if (s.mode === "to") {
      const next = Math.max(s.startDraft.to + deltaSeconds, s.startDraft.from + MIN_WINDOW_SECONDS);
      onChange({ from: s.startDraft.from, to: Math.min(Math.round(next), nowSec) });
    } else {
      let from = s.startDraft.from + deltaSeconds;
      let to = s.startDraft.to + deltaSeconds;
      if (to > nowSec) {
        from -= to - nowSec;
        to = nowSec;
      }
      onChange({ from: Math.round(from), to: Math.round(to) });
    }
  };

  const endDrag = () => {
    dragState.current = null;
  };

  const nudge = (mode: "from" | "to", deltaSeconds: number) => {
    const nowSec = Math.floor(Date.now() / 1000);
    if (mode === "from") {
      onChange({ from: Math.min(draft.from + deltaSeconds, draft.to - MIN_WINDOW_SECONDS), to: draft.to });
    } else {
      onChange({
        from: draft.from,
        to: Math.min(Math.max(draft.to + deltaSeconds, draft.from + MIN_WINDOW_SECONDS), nowSec),
      });
    }
  };

  const onHandleKeyDown = (mode: "from" | "to") => (e: React.KeyboardEvent) => {
    const step = Math.max(60, Math.round(width * 0.02)) * (e.shiftKey ? 5 : 1);
    switch (e.key) {
      case "ArrowLeft":
        e.preventDefault();
        nudge(mode, -step);
        break;
      case "ArrowRight":
        e.preventDefault();
        nudge(mode, step);
        break;
      case "Home":
        e.preventDefault();
        if (mode === "from") onChange({ from: Math.round(trackFrom), to: draft.to });
        break;
      case "End":
        e.preventDefault();
        if (mode === "to") {
          const nowSec = Math.floor(Date.now() / 1000);
          onChange({ from: draft.from, to: Math.min(Math.round(trackTo), nowSec) });
        }
        break;
      default:
        return;
    }
  };

  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between text-[11px] tabular-nums text-ink-faint">
        <span>{formatAbsoluteInstant(trackFrom)}</span>
        <span>{formatAbsoluteInstant(trackTo)}</span>
      </div>
      <div
        ref={trackRef}
        className="relative h-8 rounded-md bg-surface-3"
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
      >
        <div
          onPointerDown={beginDrag("pan")}
          className="absolute inset-y-0 cursor-grab rounded bg-accent/25 active:cursor-grabbing"
          style={{
            left: `${pctFor(draft.from)}%`,
            right: `${100 - pctFor(draft.to)}%`,
          }}
        />
        <div
          role="slider"
          tabIndex={0}
          aria-label="起始時間（可拖曳）"
          aria-valuemin={trackFrom}
          aria-valuemax={draft.to}
          aria-valuenow={draft.from}
          aria-valuetext={formatAbsoluteInstant(draft.from)}
          onPointerDown={beginDrag("from")}
          onKeyDown={onHandleKeyDown("from")}
          className="absolute top-0 h-full w-2.5 -translate-x-1/2 cursor-ew-resize rounded-full bg-accent shadow-hover focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
          style={{ left: `${pctFor(draft.from)}%` }}
        />
        <div
          role="slider"
          tabIndex={0}
          aria-label="結束時間（可拖曳）"
          aria-valuemin={draft.from}
          aria-valuemax={trackTo}
          aria-valuenow={draft.to}
          aria-valuetext={formatAbsoluteInstant(draft.to)}
          onPointerDown={beginDrag("to")}
          onKeyDown={onHandleKeyDown("to")}
          className="absolute top-0 h-full w-2.5 -translate-x-1/2 cursor-ew-resize rounded-full bg-accent shadow-hover focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
          style={{ left: `${pctFor(draft.to)}%` }}
        />
      </div>
    </div>
  );
}
