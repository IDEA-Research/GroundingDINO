"use client";

import React from "react";

// The single card primitive every widget renders inside. Bedside-monitor
// grammar: the frame (border, title, badges) stays low-luminance so only
// the data inside is bright. Carries the data-widget-id hook the backend
// Playwright evaluator locates widgets by — never remove it.
export function WidgetFrame({
  widget,
  children,
  badge,
  unit,
}: {
  widget: { id: string; title: string; description?: string | null };
  children: React.ReactNode;
  badge?: string | null;
  unit?: string | null;
}) {
  return (
    <div
      data-widget-id={widget.id}
      className="flex h-full flex-col rounded-lg border border-edge bg-surface p-3"
    >
      <div className="mb-2 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate text-xs font-semibold uppercase tracking-wide text-ink-dim">
            {widget.title}
          </div>
          {widget.description ? (
            <div className="truncate text-xs text-ink-faint">
              {widget.description}
            </div>
          ) : null}
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          {unit ? (
            <span className="font-mono text-2xs text-ink-faint">{unit}</span>
          ) : null}
          {badge ? (
            <span className="rounded border border-edge bg-surface-2 px-1.5 py-0.5 text-2xs uppercase tracking-wide text-ink-faint">
              {badge}
            </span>
          ) : null}
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-hidden">{children}</div>
    </div>
  );
}
