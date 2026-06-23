"use client";

import React from "react";

export function WidgetFrame({
  widget,
  children,
  badge,
}: {
  widget: { id: string; title: string; description?: string | null };
  children: React.ReactNode;
  badge?: string | null;
}) {
  return (
    <div
      data-widget-id={widget.id}
      className="flex h-full flex-col rounded-lg border border-slate-800 bg-panel2 p-3 shadow-sm"
    >
      <div className="mb-2 flex items-start justify-between">
        <div>
          <div className="text-sm font-semibold text-slate-100">
            {widget.title}
          </div>
          {widget.description ? (
            <div className="text-xs text-slate-400">{widget.description}</div>
          ) : null}
        </div>
        {badge ? (
          <span className="rounded bg-slate-700 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-slate-300">
            {badge}
          </span>
        ) : null}
      </div>
      <div className="flex-1 overflow-hidden">{children}</div>
    </div>
  );
}
