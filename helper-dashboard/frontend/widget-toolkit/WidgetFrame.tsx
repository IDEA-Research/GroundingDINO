"use client";

import React from "react";

import { Badge } from "@/components/ui/Badge";
import { Spinner } from "@/components/ui/Button";
import { IconButton } from "@/components/ui/IconButton";
import { Menu } from "@/components/ui/Menu";
import { cx } from "@/components/ui/cx";
import type { QueryStatus, WidgetData } from "@/lib/useQueryData";

// The single card primitive every widget renders inside. Carries the
// data-widget-id hook the backend Playwright evaluator locates widgets by —
// never remove it. The root is a size container (.hd-card) so the stat
// numeral scales with card width.
//
// Header: title (wraps to 2 lines, never uppercase) / description caption,
// then on the right: unit, effective-range badge, 模擬 badge, stale/refresh
// indicators. Status is conveyed with TEXT badges, never color alone.

export interface WidgetFrameProps {
  widget: {
    id: string;
    title: string;
    description?: string | null;
    type?: string;
  };
  children: React.ReactNode;
  // Effective time window label for range widgets (依儀表板設定 / 5分 ...).
  badge?: string | null;
  unit?: string | null;
  status?: QueryStatus | null;
  // Short reason shown as a badge when status is stale (資料過期 / 更新失敗).
  reason?: string | null;
  isRefreshing?: boolean;
  simulated?: boolean;
  // e.g. <NonDiagnosticFooter/> (LD-7) — rendered below the body.
  footer?: React.ReactNode;
  headerExtra?: React.ReactNode;
  // Compact: no header/padding (the gauge folded under a stat card).
  compact?: boolean;
  className?: string;
  bodyClassName?: string;
  // Escape hatch for the rare data-driven style Tailwind classes can't
  // express (e.g. StatCardWidget's optional severity background wash) —
  // not for general use, prefer className for anything static.
  style?: React.CSSProperties;
  // Panel hover menu callbacks (Grafana panel-menu parity). Each is a bare
  // callback — the actual behavior (opening a fullscreen view, an edit
  // drawer, cloning/removing the widget in the dashboard spec) is owned by
  // whoever composes WidgetFrame, not by WidgetFrame itself. The "..."
  // trigger only appears once at least one of these is provided, so every
  // widget that does not pass them (i.e. every widget today, pre-Phase 5)
  // renders byte-for-byte the same as before this prop existed. `onEdit`
  // is expected to stay a placeholder until Phase 5 builds the edit
  // drawer; `onDuplicate`/`onRemove` likewise until they're wired to the
  // dashboard data layer.
  onView?: () => void;
  onEdit?: () => void;
  onDuplicate?: () => void;
  onRemove?: () => void;
}

export function WidgetFrame({
  widget,
  children,
  badge,
  unit,
  status,
  reason,
  isRefreshing = false,
  simulated = false,
  footer,
  headerExtra,
  compact = false,
  className,
  bodyClassName,
  style,
  onView,
  onEdit,
  onDuplicate,
  onRemove,
}: WidgetFrameProps) {
  const stale = status === "stale";
  const staleText = reason && reason.trim() ? reason : "資料過期";
  const hasMenu = Boolean(onView || onEdit || onDuplicate || onRemove);

  if (compact) {
    return (
      <div
        data-widget-id={widget.id}
        data-widget-type={widget.type}
        data-state={stale ? "stale" : undefined}
        className={cx("hd-card flex h-full min-h-0 flex-col", className)}
      >
        <div className={cx("min-h-0 flex-1", bodyClassName)}>{children}</div>
      </div>
    );
  }

  return (
    <div
      data-widget-id={widget.id}
      data-widget-type={widget.type}
      data-state={stale ? "stale" : undefined}
      className={cx(
        "hd-card group/panel flex h-full min-h-0 flex-col rounded-lg border border-edge bg-surface p-3 text-ink shadow-card",
        className,
      )}
      style={style}
    >
      {/* dash-drag-handle: the panel's react-grid-layout draggableHandle
          target (DashboardGrid.tsx). Title/description/badges/menu only —
          the info button and "..." menu are real interactive elements
          that live here too; react-grid-layout only intercepts the
          mousedown that STARTS a drag, so nested buttons keep working
          (LineChartWidget's pinned-value badge already relies on this). */}
      <div className="dash-drag-handle mb-2 flex shrink-0 items-start justify-between gap-2">
        <div className="flex min-w-0 flex-1 items-start gap-1">
          <h3
            className="line-clamp-2 min-w-0 flex-1 text-sm font-medium leading-5 text-ink"
            title={widget.title}
          >
            {widget.title}
          </h3>
          {widget.description ? (
            <span className="group/desc relative mt-0.5 inline-flex shrink-0">
              <button
                type="button"
                aria-label="面板說明"
                className="inline-flex rounded text-ink-faint hover:text-ink focus-visible:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-focus"
              >
                <SeverityIcon severity="info" className="h-3.5 w-3.5" />
              </button>
              {/* Hover/focus only — Grafana's panel-description pattern.
                  A native `title` attribute would ALSO show a tooltip but
                  only after a long OS-controlled delay and cannot be
                  styled, so this uses an explicit group-hover panel
                  instead (kept in the DOM, not remounted, so it never
                  flashes/reflows the header). */}
              <span
                role="tooltip"
                className="pointer-events-none absolute left-0 top-full z-20 mt-1 hidden w-max max-w-[16rem] rounded-md border border-edge-strong bg-surface-2 px-2 py-1 text-xs leading-snug text-ink shadow-pop group-hover/desc:block group-focus-within/desc:block"
              >
                {widget.description}
              </span>
            </span>
          ) : null}
        </div>
        <div className="flex shrink-0 flex-wrap items-center justify-end gap-1">
          {headerExtra}
          {unit ? (
            <span className="font-mono text-xs text-ink-faint">{unit}</span>
          ) : null}
          {badge ? <Badge tone="neutral">{badge}</Badge> : null}
          {simulated ? <Badge tone="warn">模擬</Badge> : null}
          {stale ? <Badge tone="warn">{staleText}</Badge> : null}
          {status === "offline" ? <Badge tone="danger">來源中斷</Badge> : null}
          {isRefreshing ? (
            <span
              className="inline-flex items-center text-ink-faint"
              role="status"
              aria-label="更新中"
              title="更新中"
            >
              <Spinner className="h-3.5 w-3.5" />
            </span>
          ) : null}
          {hasMenu ? (
            <Menu
              align="end"
              label={`${widget.title} 面板選單`}
              trigger={
                <IconButton
                  label="面板選單"
                  size="sm"
                  variant="ghost"
                  className="opacity-0 transition-opacity group-hover/panel:opacity-100 group-focus-within/panel:opacity-100"
                >
                  <MoreIcon className="h-4 w-4" />
                </IconButton>
              }
              items={[
                {
                  label: "全螢幕檢視",
                  onSelect: onView,
                  disabled: !onView,
                },
                { label: "編輯", onSelect: onEdit, disabled: !onEdit },
                {
                  label: "複製",
                  onSelect: onDuplicate,
                  disabled: !onDuplicate,
                },
                {
                  label: "移除",
                  onSelect: onRemove,
                  disabled: !onRemove,
                  tone: "danger",
                },
              ]}
            />
          ) : null}
        </div>
      </div>
      <div className={cx("min-h-0 flex-1 overflow-hidden", bodyClassName)}>
        {children}
      </div>
      {footer}
    </div>
  );
}

// Pull the frame's status props straight from a WidgetData.
export function frameStatusProps(data: WidgetData): {
  status: QueryStatus;
  reason: string | null;
  isRefreshing: boolean;
  simulated: boolean;
} {
  return {
    status: data.status,
    reason: data.reason,
    isRefreshing: data.isRefreshing,
    simulated: data.simulated,
  };
}

// LD-7: non-suppressible decision-support footer on every widget that
// displays a threshold state or alert. Text is >= 4.5:1 (ink-faint).
export function NonDiagnosticFooter({ className }: { className?: string }) {
  return (
    <div
      data-non-diagnostic="true"
      className={cx(
        "mt-1 shrink-0 border-t border-edge pt-1 text-xs text-ink-faint",
        className,
      )}
    >
      決策支援,非診斷
    </div>
  );
}

export function fmtClock(epochSeconds: number): string {
  const d = new Date(epochSeconds * 1000);
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

export function fmtClockMs(epochMs: number): string {
  return fmtClock(epochMs / 1000);
}

// Fixed decimals per widget (options.decimals) or a stable magnitude rule.
export function formatValue(
  v: number | null | undefined,
  decimals?: number | null,
): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  if (typeof decimals === "number" && Number.isFinite(decimals)) {
    return v.toFixed(Math.max(0, Math.min(6, Math.round(decimals))));
  }
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

// Inline status glyph paired with a text label (never color alone).
export function SeverityIcon({
  severity,
  className,
}: {
  severity: "danger" | "warn" | "ok" | "info";
  className?: string;
}) {
  const common = {
    "aria-hidden": true as const,
    viewBox: "0 0 16 16",
    className: cx("h-4 w-4 shrink-0", className),
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.8,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
  };
  switch (severity) {
    case "danger":
      return (
        <svg {...common}>
          <path d="M5.2 1.5h5.6l3.7 3.7v5.6l-3.7 3.7H5.2L1.5 10.8V5.2z" />
          <path d="M8 4.8v3.9M8 11.2h.01" />
        </svg>
      );
    case "warn":
      return (
        <svg {...common}>
          <path d="M8 2 14.5 13.5h-13z" />
          <path d="M8 6.2v3.3M8 11.8h.01" />
        </svg>
      );
    case "ok":
      return (
        <svg {...common}>
          <circle cx="8" cy="8" r="6.3" />
          <path d="m5.2 8.2 1.9 1.9 3.7-4" />
        </svg>
      );
    default:
      return (
        <svg {...common}>
          <circle cx="8" cy="8" r="6.3" />
          <path d="M8 7.2v4M8 4.8h.01" />
        </svg>
      );
  }
}

// Panel hover-menu trigger glyph ("more" / kebab, vertical).
function MoreIcon({ className }: { className?: string }) {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 16 16"
      className={cx("h-4 w-4 shrink-0", className)}
      fill="currentColor"
    >
      <circle cx="8" cy="3.4" r="1.35" />
      <circle cx="8" cy="8" r="1.35" />
      <circle cx="8" cy="12.6" r="1.35" />
    </svg>
  );
}
