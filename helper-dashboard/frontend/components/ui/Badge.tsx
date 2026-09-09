"use client";

import React from "react";

import { cx } from "./cx";

export type BadgeTone =
  | "neutral"
  | "ok"
  | "warn"
  | "danger"
  | "info"
  | "accent";

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  tone?: BadgeTone;
  // Leading status dot (color follows the tone).
  dot?: boolean;
  children: React.ReactNode;
}

const TONE: Record<BadgeTone, string> = {
  neutral: "border-edge bg-surface-2 text-ink-dim",
  ok: "border-ok/30 bg-ok/10 text-ok",
  warn: "border-warn/30 bg-warn/10 text-warn",
  danger: "border-danger/30 bg-danger/10 text-danger",
  info: "border-info/30 bg-info/10 text-info",
  accent: "border-accent/30 bg-accent/10 text-accent",
};

const DOT: Record<BadgeTone, string> = {
  neutral: "bg-ink-faint",
  ok: "bg-ok",
  warn: "bg-warn",
  danger: "bg-danger",
  info: "bg-info",
  accent: "bg-accent",
};

// 12px text (CJK-safe caption size). Use `text-2xs` via className ONLY for
// Latin-only content.
export function Badge({
  tone = "neutral",
  dot = false,
  className,
  children,
  ...rest
}: BadgeProps) {
  return (
    <span
      className={cx(
        "inline-flex max-w-full items-center gap-1 whitespace-nowrap rounded border px-1.5 py-0.5 text-xs font-medium leading-4",
        TONE[tone],
        className,
      )}
      {...rest}
    >
      {dot ? (
        <span
          aria-hidden="true"
          className={cx("inline-block h-1.5 w-1.5 shrink-0 rounded-full", DOT[tone])}
        />
      ) : null}
      <span className="truncate">{children}</span>
    </span>
  );
}
