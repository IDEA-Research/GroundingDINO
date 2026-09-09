"use client";

import React, { forwardRef } from "react";

import { cx } from "./cx";

// Wraps ANY input with a visible label (+ optional hint / error). Pass the
// same `htmlFor` as the input's `id`; use fieldDescribedBy() for the
// input's aria-describedby so hint/error are announced.
export interface FieldProps {
  label: React.ReactNode;
  htmlFor: string;
  hint?: React.ReactNode;
  error?: React.ReactNode;
  required?: boolean;
  className?: string;
  children: React.ReactNode;
}

export function fieldHintId(htmlFor: string): string {
  return `${htmlFor}-hint`;
}

export function fieldErrorId(htmlFor: string): string {
  return `${htmlFor}-error`;
}

export function fieldDescribedBy(
  htmlFor: string,
  opts: { hint?: unknown; error?: unknown },
): string | undefined {
  const ids: string[] = [];
  if (opts.hint) ids.push(fieldHintId(htmlFor));
  if (opts.error) ids.push(fieldErrorId(htmlFor));
  return ids.length ? ids.join(" ") : undefined;
}

export function Field({
  label,
  htmlFor,
  hint,
  error,
  required = false,
  className,
  children,
}: FieldProps) {
  return (
    <div className={cx("flex flex-col gap-1", className)}>
      <label htmlFor={htmlFor} className="text-sm font-medium text-ink">
        {label}
        {required ? (
          <span aria-hidden="true" className="ml-0.5 text-danger">
            *
          </span>
        ) : null}
      </label>
      {children}
      {hint ? (
        <p id={fieldHintId(htmlFor)} className="text-xs text-ink-faint">
          {hint}
        </p>
      ) : null}
      {error ? (
        <p
          id={fieldErrorId(htmlFor)}
          role="alert"
          className="text-xs font-medium text-danger"
        >
          {error}
        </p>
      ) : null}
    </div>
  );
}

// Shared input styling so every text control matches the Button metrics.
export const inputClassName =
  "min-h-9 w-full rounded-md border border-edge-strong bg-surface px-3 text-sm text-ink placeholder:text-ink-faint disabled:cursor-not-allowed disabled:opacity-50 max-lg:min-h-10";

export const Input = forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement> & { invalid?: boolean }
>(function Input({ className, invalid, ...rest }, ref) {
  return (
    <input
      ref={ref}
      aria-invalid={invalid || undefined}
      className={cx(inputClassName, invalid && "border-danger", className)}
      {...rest}
    />
  );
});

export const Textarea = forwardRef<
  HTMLTextAreaElement,
  React.TextareaHTMLAttributes<HTMLTextAreaElement> & { invalid?: boolean }
>(function Textarea({ className, invalid, ...rest }, ref) {
  return (
    <textarea
      ref={ref}
      aria-invalid={invalid || undefined}
      className={cx(
        inputClassName,
        "min-h-[5.5rem] resize-y py-2 leading-relaxed",
        invalid && "border-danger",
        className,
      )}
      {...rest}
    />
  );
});

export const Select = forwardRef<
  HTMLSelectElement,
  React.SelectHTMLAttributes<HTMLSelectElement>
>(function Select({ className, children, ...rest }, ref) {
  return (
    <select ref={ref} className={cx(inputClassName, "pr-8", className)} {...rest}>
      {children}
    </select>
  );
});
