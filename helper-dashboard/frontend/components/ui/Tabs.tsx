"use client";

import React, { useId, useRef } from "react";

import { Badge } from "./Badge";
import { cx } from "./cx";

export interface TabItem {
  id: string;
  label: React.ReactNode;
  badge?: React.ReactNode;
  disabled?: boolean;
}

export interface TabsProps {
  tabs: TabItem[];
  value: string;
  onChange: (id: string) => void;
  // Renders the active panel.
  children: (id: string) => React.ReactNode;
  className?: string;
  panelClassName?: string;
  "aria-label"?: string;
}

// WAI-ARIA tabs: roving tabindex, Left/Right/Home/End, automatic activation.
export function Tabs({
  tabs,
  value,
  onChange,
  children,
  className,
  panelClassName,
  "aria-label": ariaLabel,
}: TabsProps) {
  const id = useId();
  const refs = useRef<(HTMLButtonElement | null)[]>([]);

  const enabled = tabs.map((t, i) => (t.disabled ? -1 : i)).filter((i) => i >= 0);

  const focusAt = (i: number) => {
    const target = tabs[i];
    if (!target) return;
    refs.current[i]?.focus();
    onChange(target.id);
  };

  const onKeyDown = (e: React.KeyboardEvent, i: number) => {
    if (enabled.length === 0) return;
    const pos = enabled.indexOf(i);
    let next: number | null = null;
    switch (e.key) {
      case "ArrowRight":
        next = enabled[(pos + 1) % enabled.length];
        break;
      case "ArrowLeft":
        next = enabled[(pos - 1 + enabled.length) % enabled.length];
        break;
      case "Home":
        next = enabled[0];
        break;
      case "End":
        next = enabled[enabled.length - 1];
        break;
      default:
        return;
    }
    e.preventDefault();
    if (next !== null) focusAt(next);
  };

  return (
    <div className={cx("flex min-h-0 flex-col", className)}>
      <div
        role="tablist"
        aria-label={ariaLabel}
        className="flex shrink-0 gap-1 overflow-x-auto border-b border-edge"
      >
        {tabs.map((t, i) => {
          const selected = t.id === value;
          return (
            <button
              key={t.id}
              ref={(el) => {
                refs.current[i] = el;
              }}
              type="button"
              role="tab"
              id={`${id}-tab-${t.id}`}
              aria-selected={selected}
              aria-controls={`${id}-panel-${t.id}`}
              aria-disabled={t.disabled || undefined}
              disabled={t.disabled}
              tabIndex={selected ? 0 : -1}
              onClick={() => onChange(t.id)}
              onKeyDown={(e) => onKeyDown(e, i)}
              className={cx(
                "-mb-px inline-flex min-h-9 items-center gap-1.5 whitespace-nowrap border-b-2 px-3 text-sm transition-colors max-lg:min-h-10",
                selected
                  ? "border-accent font-medium text-ink"
                  : "border-transparent text-ink-dim hover:border-edge-strong hover:text-ink",
                t.disabled && "cursor-not-allowed opacity-50",
              )}
            >
              {t.label}
              {t.badge !== undefined && t.badge !== null ? (
                <Badge tone={selected ? "accent" : "neutral"}>{t.badge}</Badge>
              ) : null}
            </button>
          );
        })}
      </div>
      <div
        role="tabpanel"
        id={`${id}-panel-${value}`}
        aria-labelledby={`${id}-tab-${value}`}
        tabIndex={0}
        className={cx("min-h-0 flex-1", panelClassName)}
      >
        {children(value)}
      </div>
    </div>
  );
}
