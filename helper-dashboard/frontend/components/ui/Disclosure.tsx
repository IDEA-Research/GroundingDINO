"use client";

import React, { useId, useState } from "react";

import { cx } from "./cx";

// 「詳細資訊」 / 「技術細節」 expander: a real button with aria-expanded and
// aria-controls; content is removed from the DOM while collapsed.
export interface DisclosureProps {
  summary: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
  className?: string;
  onToggle?: (open: boolean) => void;
}

export function Disclosure({
  summary,
  children,
  defaultOpen = false,
  className,
  onToggle,
}: DisclosureProps) {
  const [open, setOpen] = useState(defaultOpen);
  const id = useId();
  const panelId = `${id}-panel`;
  return (
    <div className={cx("text-sm", className)}>
      <button
        type="button"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => {
          setOpen((o) => {
            onToggle?.(!o);
            return !o;
          });
        }}
        className="inline-flex min-h-8 items-center gap-1.5 rounded-md px-1.5 text-sm text-ink-dim hover:bg-surface-2 hover:text-ink max-lg:min-h-10"
      >
        <svg
          aria-hidden="true"
          viewBox="0 0 16 16"
          className={cx(
            "h-3.5 w-3.5 shrink-0 transition-transform",
            open && "rotate-90",
          )}
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M6 3.5 10.5 8 6 12.5" />
        </svg>
        <span>{summary}</span>
      </button>
      {open ? (
        <div id={panelId} className="mt-1.5 pl-1">
          {children}
        </div>
      ) : null}
    </div>
  );
}
