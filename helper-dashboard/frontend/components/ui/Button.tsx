"use client";

import React, { forwardRef } from "react";

import { cx } from "./cx";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
export type ButtonSize = "sm" | "md";

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
  iconLeft?: React.ReactNode;
  iconRight?: React.ReactNode;
}

const VARIANT: Record<ButtonVariant, string> = {
  primary:
    "border border-transparent bg-accent text-accent-ink hover:brightness-110 active:brightness-95 disabled:opacity-50",
  secondary:
    "border border-edge-strong bg-surface text-ink hover:bg-surface-2 active:bg-surface-3 disabled:opacity-50",
  ghost:
    "border border-transparent bg-transparent text-ink-dim hover:bg-surface-2 hover:text-ink active:bg-surface-3 disabled:opacity-50",
  danger:
    "border border-danger/40 bg-danger/10 text-danger hover:bg-danger/15 active:bg-danger/20 disabled:opacity-50",
};

// >= 36px (md) / 32px (sm) on desktop, >= 40px below lg for touch.
const SIZE: Record<ButtonSize, string> = {
  sm: "min-h-8 gap-1.5 px-2.5 text-sm max-lg:min-h-10",
  md: "min-h-9 gap-2 px-3 text-sm max-lg:min-h-10",
};

export function Spinner({ className }: { className?: string }) {
  return (
    <svg
      className={cx("hd-spin h-4 w-4 shrink-0", className)}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
      focusable="false"
    >
      <circle
        cx="12"
        cy="12"
        r="9"
        stroke="currentColor"
        strokeOpacity="0.25"
        strokeWidth="3"
      />
      <path
        d="M21 12a9 9 0 0 0-9-9"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
      />
    </svg>
  );
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  function Button(
    {
      variant = "secondary",
      size = "md",
      loading = false,
      iconLeft,
      iconRight,
      className,
      children,
      disabled,
      type = "button",
      ...rest
    },
    ref,
  ) {
    return (
      <button
        ref={ref}
        type={type}
        className={cx(
          "inline-flex select-none items-center justify-center whitespace-nowrap rounded-md font-medium leading-none transition-colors disabled:cursor-not-allowed",
          VARIANT[variant],
          SIZE[size],
          className,
        )}
        disabled={disabled || loading}
        aria-busy={loading || undefined}
        {...rest}
      >
        {loading ? <Spinner /> : iconLeft}
        {children}
        {iconRight}
      </button>
    );
  },
);
