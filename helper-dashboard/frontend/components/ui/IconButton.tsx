"use client";

import React, { forwardRef } from "react";

import type { ButtonSize, ButtonVariant } from "./Button";
import { Spinner } from "./Button";
import { cx } from "./cx";

export interface IconButtonProps
  extends Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, "aria-label"> {
  // Required: icon-only controls must always have an accessible name.
  label: string;
  size?: ButtonSize;
  variant?: ButtonVariant;
  loading?: boolean;
  children: React.ReactNode;
}

const VARIANT: Record<ButtonVariant, string> = {
  primary:
    "border border-transparent bg-accent text-accent-ink hover:brightness-110 disabled:opacity-50",
  secondary:
    "border border-edge-strong bg-surface text-ink hover:bg-surface-2 disabled:opacity-50",
  ghost:
    "border border-transparent bg-transparent text-ink-dim hover:bg-surface-2 hover:text-ink disabled:opacity-50",
  danger:
    "border border-danger/40 bg-danger/10 text-danger hover:bg-danger/15 disabled:opacity-50",
};

// Square targets: 36px (md) / 32px (sm) on desktop, 40px below lg.
const SIZE: Record<ButtonSize, string> = {
  sm: "h-8 w-8 min-h-8 min-w-8 max-lg:h-10 max-lg:w-10",
  md: "h-9 w-9 min-h-9 min-w-9 max-lg:h-10 max-lg:w-10",
};

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  function IconButton(
    {
      label,
      size = "md",
      variant = "ghost",
      loading = false,
      className,
      children,
      disabled,
      type = "button",
      title,
      ...rest
    },
    ref,
  ) {
    return (
      <button
        ref={ref}
        type={type}
        aria-label={label}
        title={title ?? label}
        className={cx(
          "inline-flex shrink-0 select-none items-center justify-center rounded-md p-0 transition-colors disabled:cursor-not-allowed",
          VARIANT[variant],
          SIZE[size],
          className,
        )}
        disabled={disabled || loading}
        aria-busy={loading || undefined}
        {...rest}
      >
        {loading ? <Spinner /> : children}
      </button>
    );
  },
);
