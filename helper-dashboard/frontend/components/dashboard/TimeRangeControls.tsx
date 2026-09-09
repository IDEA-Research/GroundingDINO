"use client";

import React, { useRef, useState } from "react";

import { AbsoluteRangePicker } from "@/components/dashboard/AbsoluteRangePicker";
import { Button } from "@/components/ui/Button";
import { IconButton } from "@/components/ui/IconButton";
import { cx } from "@/components/ui/cx";
import {
  formatDurationLabel,
  useTimeRange,
  type RangeExpr,
} from "@/lib/time-range";

// Two SEPARATE controls (spec 五.3), both pure view-layer (they never touch
// the DashboardSpec):
//   1. 時間範圍 — single-select radiogroup: 依儀表板設定 | 5分 | 15分 | 1小時 |
//      6小時 | 24小時 | 72小時. Arrow keys move the selection (roving
//      tabindex), the checked item is the only tab stop.
//   2. 自動更新 — on/off switch + interval readout + 立即更新.
// The last-updated clock lives in the DataStatusStrip (data arrival), not
// here — a ticking clock next to a range picker made a dead feed look live.

export const TIME_RANGE_CONTROLS_ID = "hd-time-range-controls";

// Used by 「調整時間範圍」 actions: scrolls to and focuses the active range.
export function focusTimeRangeControls(): void {
  if (typeof document === "undefined") return;
  const root = document.getElementById(TIME_RANGE_CONTROLS_ID);
  if (!root) return;
  root.scrollIntoView({ block: "nearest", behavior: "smooth" });
  const checked = root.querySelector<HTMLElement>('[role="radio"][aria-checked="true"]');
  (checked ?? root.querySelector<HTMLElement>('[role="radio"]'))?.focus();
}

export function TimeRangeControls({
  className,
  compact = false,
}: {
  className?: string;
  // Hide the interval readout (tight header rows).
  compact?: boolean;
}) {
  const {
    range,
    setRange,
    panRange,
    zoomOutRange,
    presets,
    autoRefresh,
    refreshNow,
    inFlightCount,
  } = useTimeRange();
  const radioRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const customTriggerRef = useRef<HTMLButtonElement>(null);
  const [customOpen, setCustomOpen] = useState(false);

  // -1 when "custom" is active: no preset pill should read as checked, and
  // Math.max(0, -1) would have wrongly re-highlighted 依儀表板設定.
  const currentIdx = presets.findIndex((p) => p.expr === range.expr);
  // pan/zoom only make sense against a single active window (a preset or a
  // custom range) — "依儀表板設定" leaves each widget on its own window.
  const canPanZoom = range.mode === "override";

  const select = (i: number) => {
    const p = presets[i];
    if (!p) return;
    setRange(p.expr as RangeExpr);
    radioRefs.current[i]?.focus();
  };

  const onRadioKeyDown = (e: React.KeyboardEvent, i: number) => {
    let next: number | null = null;
    switch (e.key) {
      case "ArrowRight":
      case "ArrowDown":
        next = (i + 1) % presets.length;
        break;
      case "ArrowLeft":
      case "ArrowUp":
        next = (i - 1 + presets.length) % presets.length;
        break;
      case "Home":
        next = 0;
        break;
      case "End":
        next = presets.length - 1;
        break;
      case " ":
      case "Enter":
        next = i;
        break;
      default:
        return;
    }
    e.preventDefault();
    select(next);
  };

  return (
    <div
      id={TIME_RANGE_CONTROLS_ID}
      className={cx("flex flex-wrap items-center gap-x-3 gap-y-2", className)}
    >
      <div className="flex items-center gap-2">
        <span id={`${TIME_RANGE_CONTROLS_ID}-label`} className="text-sm text-ink-dim">
          時間範圍
        </span>
        <div
          role="radiogroup"
          aria-labelledby={`${TIME_RANGE_CONTROLS_ID}-label`}
          className="flex flex-wrap items-center gap-0.5 rounded-md border border-edge bg-surface p-0.5"
        >
          {presets.map((p, i) => {
            const checked = i === currentIdx;
            return (
              <button
                key={p.expr}
                ref={(el) => {
                  radioRefs.current[i] = el;
                }}
                type="button"
                role="radio"
                aria-checked={checked}
                tabIndex={checked || (currentIdx === -1 && i === 0) ? 0 : -1}
                onClick={() => select(i)}
                onKeyDown={(e) => onRadioKeyDown(e, i)}
                className={cx(
                  "min-h-8 whitespace-nowrap rounded px-2.5 text-sm transition-colors max-lg:min-h-10",
                  checked
                    ? "bg-accent/10 font-medium text-accent"
                    : "text-ink-dim hover:bg-surface-2 hover:text-ink",
                )}
              >
                {p.label}
              </button>
            );
          })}
        </div>

        <div className="flex items-center gap-0.5">
          <IconButton
            label="往前一個範圍寬度"
            size="sm"
            variant="ghost"
            disabled={!canPanZoom}
            onClick={() => panRange("back")}
          >
            <svg viewBox="0 0 16 16" width="14" height="14" fill="none" aria-hidden="true">
              <path d="M10 3 5 8l5 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </IconButton>
          <IconButton
            label="往後一個範圍寬度"
            size="sm"
            variant="ghost"
            disabled={!canPanZoom}
            onClick={() => panRange("forward")}
          >
            <svg viewBox="0 0 16 16" width="14" height="14" fill="none" aria-hidden="true">
              <path d="M6 3l5 5-5 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </IconButton>
          <IconButton
            label="縮小（範圍加倍）"
            size="sm"
            variant="ghost"
            disabled={!canPanZoom}
            onClick={zoomOutRange}
          >
            <svg viewBox="0 0 16 16" width="14" height="14" fill="none" aria-hidden="true">
              <circle cx="7" cy="7" r="4.5" stroke="currentColor" strokeWidth="1.6" />
              <path d="M5 7h4M11 11l3 3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
            </svg>
          </IconButton>
        </div>

        <div className="relative">
          <Button
            ref={customTriggerRef}
            type="button"
            variant="secondary"
            size="sm"
            onClick={() => setCustomOpen((v) => !v)}
            aria-expanded={customOpen}
          >
            {range.expr === "custom" ? range.label : "自訂範圍…"}
          </Button>
          <AbsoluteRangePicker
            open={customOpen}
            onClose={() => setCustomOpen(false)}
            triggerRef={customTriggerRef}
          />
        </div>
      </div>

      <div className="flex items-center gap-2">
        <button
          type="button"
          role="switch"
          aria-checked={autoRefresh.enabled}
          onClick={() => autoRefresh.setEnabled(!autoRefresh.enabled)}
          className="inline-flex min-h-8 items-center gap-2 rounded-md px-1.5 text-sm text-ink hover:bg-surface-2 max-lg:min-h-10"
        >
          <span
            aria-hidden="true"
            className={cx(
              "relative inline-block h-5 w-9 shrink-0 rounded-full border transition-colors",
              autoRefresh.enabled
                ? "border-accent bg-accent"
                : "border-edge-strong bg-surface-3",
            )}
          >
            <span
              className={cx(
                "absolute top-0.5 h-3.5 w-3.5 rounded-full bg-surface shadow-card transition-transform",
                autoRefresh.enabled ? "left-0.5 translate-x-4" : "left-0.5",
              )}
            />
          </span>
          <span>自動更新</span>
        </button>
        {!compact ? (
          <span className="font-mono text-xs tabular-nums text-ink-faint">
            {autoRefresh.enabled
              ? `每 ${formatDurationLabel(autoRefresh.intervalSeconds)}`
              : "已暫停"}
          </span>
        ) : null}
        <Button
          size="sm"
          variant="secondary"
          onClick={refreshNow}
          loading={inFlightCount > 0}
          aria-label="立即更新"
        >
          立即更新
        </Button>
      </div>
    </div>
  );
}
