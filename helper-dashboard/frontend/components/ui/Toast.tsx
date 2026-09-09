"use client";

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { Button } from "./Button";
import { IconButton } from "./IconButton";
import { cx } from "./cx";

export type ToastTone = "neutral" | "ok" | "warn" | "danger" | "info";

export interface ToastOptions {
  title: React.ReactNode;
  description?: React.ReactNode;
  tone?: ToastTone;
  action?: { label: string; onClick: () => void };
  // 0 = sticky until dismissed. Default 5000ms, 10000ms when an action is
  // attached (e.g. 復原 after a delete).
  durationMs?: number;
}

export interface ToastRecord extends ToastOptions {
  id: string;
  createdAt: number;
}

interface ToastContextValue {
  toasts: ToastRecord[];
  show: (opts: ToastOptions) => string;
  dismiss: (id: string) => void;
  clear: () => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

const MAX_VISIBLE = 4;

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<ToastRecord[]>([]);
  const timers = useRef(new Map<string, number>());
  const counter = useRef(0);

  const dismiss = useCallback((id: string) => {
    const t = timers.current.get(id);
    if (t) {
      window.clearTimeout(t);
      timers.current.delete(id);
    }
    setToasts((list) => list.filter((x) => x.id !== id));
  }, []);

  const show = useCallback(
    (opts: ToastOptions) => {
      counter.current += 1;
      const id = `toast-${Date.now()}-${counter.current}`;
      const duration =
        opts.durationMs ?? (opts.action ? 10000 : 5000);
      setToasts((list) => {
        const next = [...list, { ...opts, id, createdAt: Date.now() }];
        // Drop the oldest beyond the cap so the stack never floods.
        while (next.length > MAX_VISIBLE) {
          const victim = next.shift();
          if (victim) {
            const t = timers.current.get(victim.id);
            if (t) window.clearTimeout(t);
            timers.current.delete(victim.id);
          }
        }
        return next;
      });
      if (duration > 0) {
        const t = window.setTimeout(() => dismiss(id), duration);
        timers.current.set(id, t);
      }
      return id;
    },
    [dismiss],
  );

  const clear = useCallback(() => {
    timers.current.forEach((t) => window.clearTimeout(t));
    timers.current.clear();
    setToasts([]);
  }, []);

  useEffect(() => {
    const map = timers.current;
    return () => {
      map.forEach((t) => window.clearTimeout(t));
      map.clear();
    };
  }, []);

  const value = useMemo(
    () => ({ toasts, show, dismiss, clear }),
    [toasts, show, dismiss, clear],
  );

  return <ToastContext.Provider value={value}>{children}</ToastContext.Provider>;
}

const NOOP_CONTEXT: ToastContextValue = {
  toasts: [],
  show: () => "",
  dismiss: () => {},
  clear: () => {},
};

// Works without a provider (no-op) so isolated renders never crash.
export function useToast(): Pick<ToastContextValue, "show" | "dismiss" | "clear"> {
  const ctx = useContext(ToastContext) ?? NOOP_CONTEXT;
  return { show: ctx.show, dismiss: ctx.dismiss, clear: ctx.clear };
}

const TONE_BAR: Record<ToastTone, string> = {
  neutral: "bg-ink-faint",
  ok: "bg-ok",
  warn: "bg-warn",
  danger: "bg-danger",
  info: "bg-info",
};

const TONE_TEXT: Record<ToastTone, string> = {
  neutral: "",
  ok: "text-ok",
  warn: "text-warn",
  danger: "text-danger",
  info: "text-info",
};

function ToastCard({
  toast,
  onDismiss,
}: {
  toast: ToastRecord;
  onDismiss: () => void;
}) {
  const tone = toast.tone ?? "neutral";
  return (
    <div
      role={tone === "danger" ? "alert" : "status"}
      className="pointer-events-auto flex w-full max-w-sm items-stretch overflow-hidden rounded-lg border border-edge-strong bg-surface text-sm text-ink shadow-pop"
    >
      <div aria-hidden="true" className={cx("w-1 shrink-0", TONE_BAR[tone])} />
      <div className="flex min-w-0 flex-1 items-start gap-2 px-3 py-2.5">
        <div className="min-w-0 flex-1">
          <div className={cx("font-medium", TONE_TEXT[tone])}>{toast.title}</div>
          {toast.description ? (
            <div className="mt-0.5 text-sm text-ink-dim">{toast.description}</div>
          ) : null}
        </div>
        {toast.action ? (
          <Button
            size="sm"
            variant="secondary"
            onClick={() => {
              toast.action?.onClick();
              onDismiss();
            }}
          >
            {toast.action.label}
          </Button>
        ) : null}
        <IconButton label="關閉通知" size="sm" onClick={onDismiss}>
          <svg
            aria-hidden="true"
            viewBox="0 0 16 16"
            className="h-3.5 w-3.5"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
          >
            <path d="M3.5 3.5l9 9M12.5 3.5l-9 9" />
          </svg>
        </IconButton>
      </div>
    </div>
  );
}

// Mount ONCE (in the page shell). Bottom-right on desktop, bottom-center on
// small screens, above drawers but never covering the top bar.
export function ToastViewport({ className }: { className?: string }) {
  const ctx = useContext(ToastContext);
  if (!ctx) return null;
  return (
    <div
      aria-live="polite"
      aria-atomic="false"
      className={cx(
        "pointer-events-none fixed inset-x-3 bottom-3 z-[60] flex flex-col items-center gap-2 sm:inset-x-auto sm:right-4 sm:bottom-4 sm:items-end",
        className,
      )}
    >
      {ctx.toasts.map((t) => (
        <ToastCard key={t.id} toast={t} onDismiss={() => ctx.dismiss(t.id)} />
      ))}
    </div>
  );
}
