"use client";

import React from "react";

import { cx } from "./cx";

// Shimmering placeholder block (CSS in globals.css; static under
// prefers-reduced-motion). Size it with className (h-4 w-24 ...).
export function Skeleton({
  className,
  style,
}: {
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <div
      aria-hidden="true"
      className={cx("hd-skeleton", className)}
      style={style}
    />
  );
}
