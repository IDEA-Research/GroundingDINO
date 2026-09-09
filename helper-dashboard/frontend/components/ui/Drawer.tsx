"use client";

import React, {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
} from "react";

import { IconButton } from "./IconButton";
import { cx } from "./cx";

export const DRAWER_MIN_WIDTH = 360;
export const DRAWER_DEFAULT_WIDTH = 440;

export interface DrawerProps {
  open: boolean;
  onClose: () => void;
  side?: "right" | "bottom";
  title: React.ReactNode;
  // Desktop width in px (clamped to 360..90vw). Ignored below lg (full
  // width) and when maximized.
  width?: number;
  resizable?: boolean;
  maximized?: boolean;
  onToggleMaximize?: () => void;
  children: React.ReactNode;
  footer?: React.ReactNode;
  // Extra header controls (rendered before maximize/close).
  headerExtra?: React.ReactNode;
  // Keep children mounted (display:none) while closed so in-flight work
  // (e.g. the Helper chat job) survives collapsing the drawer.
  keepMounted?: boolean;
  className?: string;
  bodyClassName?: string;
  id?: string;
}

function useIsNarrow(): boolean {
  const [narrow, setNarrow] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mql = window.matchMedia("(max-width: 1023.98px)");
    const update = () => setNarrow(mql.matches);
    update();
    mql.addEventListener("change", update);
    return () => mql.removeEventListener("change", update);
  }, []);
  return narrow;
}

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

// Right-side (or bottom) panel. Below lg it OVERLAYS: role=dialog +
// aria-modal, backdrop, focus trap, body scroll lock, full width beneath
// the top bar (set --hd-topbar-h on :root / the top bar to offset it).
// At lg+ it is a non-modal side panel: role=complementary, no backdrop,
// the page stays interactive. Escape closes in both modes.
export function Drawer({
  open,
  onClose,
  side = "right",
  title,
  width = DRAWER_DEFAULT_WIDTH,
  resizable = false,
  maximized = false,
  onToggleMaximize,
  children,
  footer,
  headerExtra,
  keepMounted = false,
  className,
  bodyClassName,
  id,
}: DrawerProps) {
  const narrow = useIsNarrow();
  const autoId = useId();
  const drawerId = id ?? `${autoId}-drawer`;
  const titleId = `${drawerId}-title`;
  const panelRef = useRef<HTMLDivElement | null>(null);
  const lastActiveRef = useRef<HTMLElement | null>(null);
  const [userWidth, setUserWidth] = useState<number | null>(null);
  const dragRef = useRef<{ startX: number; startW: number } | null>(null);

  const modal = open && narrow;

  const clampWidth = useCallback((w: number) => {
    const max =
      typeof window !== "undefined"
        ? Math.floor(window.innerWidth * 0.9)
        : 1200;
    return Math.max(DRAWER_MIN_WIDTH, Math.min(max, Math.round(w)));
  }, []);

  // Escape closes; Tab is trapped while modal.
  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Escape") {
      e.stopPropagation();
      onClose();
      return;
    }
    if (e.key === "Tab" && modal && panelRef.current) {
      const nodes = Array.from(
        panelRef.current.querySelectorAll<HTMLElement>(FOCUSABLE),
      ).filter((el) => el.offsetParent !== null || el === document.activeElement);
      if (nodes.length === 0) {
        e.preventDefault();
        return;
      }
      const first = nodes[0];
      const last = nodes[nodes.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
  };

  // Focus management + scroll lock.
  useEffect(() => {
    if (!open) return;
    lastActiveRef.current = document.activeElement as HTMLElement | null;
    const panel = panelRef.current;
    const t = window.setTimeout(() => {
      if (!panel) return;
      const first = panel.querySelector<HTMLElement>(FOCUSABLE);
      (first ?? panel).focus();
    }, 0);
    let prevOverflow: string | null = null;
    if (narrow) {
      prevOverflow = document.body.style.overflow;
      document.body.style.overflow = "hidden";
    }
    return () => {
      window.clearTimeout(t);
      if (prevOverflow !== null) document.body.style.overflow = prevOverflow;
      const back = lastActiveRef.current;
      if (back && typeof back.focus === "function") back.focus();
    };
  }, [open, narrow]);

  // Resize handle (pointer drag + keyboard).
  useEffect(() => {
    if (!resizable) return;
    const onMove = (e: PointerEvent) => {
      const d = dragRef.current;
      if (!d) return;
      const delta = d.startX - e.clientX; // dragging left widens a right drawer
      setUserWidth(clampWidth(d.startW + delta));
    };
    const onUp = () => {
      dragRef.current = null;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [resizable, clampWidth]);

  if (!open && !keepMounted) return null;

  const effectiveWidth = clampWidth(userWidth ?? width);
  const desktopStyle: React.CSSProperties =
    side === "right" && !narrow
      ? { width: maximized ? "90vw" : `${effectiveWidth}px` }
      : {};

  const positionClass =
    side === "bottom"
      ? "inset-x-0 bottom-0 h-[70vh] rounded-t-lg border-t"
      : narrow
        ? "inset-x-0 bottom-0 border-t"
        : "right-0 bottom-0 border-l";

  return (
    <>
      {modal ? (
        <div
          aria-hidden="true"
          onClick={onClose}
          className="fixed inset-0 z-40 bg-canvas/60"
          style={{ top: "var(--hd-topbar-h, 0px)" }}
        />
      ) : null}
      <div
        ref={panelRef}
        id={drawerId}
        role={modal ? "dialog" : "complementary"}
        aria-modal={modal || undefined}
        aria-labelledby={titleId}
        tabIndex={-1}
        onKeyDown={onKeyDown}
        // Display is toggled via classes, not the `hidden` attribute: the
        // `flex` utility outranks preflight's `[hidden]{display:none}`, so a
        // keepMounted drawer would otherwise stay visible when closed.
        className={cx(
          "fixed z-50 flex-col border-edge bg-surface text-ink shadow-pop outline-none",
          open ? "flex" : "hidden",
          positionClass,
          className,
        )}
        style={{
          top: side === "bottom" ? undefined : "var(--hd-topbar-h, 0px)",
          ...desktopStyle,
        }}
      >
        {resizable && side === "right" && !narrow && !maximized ? (
          <div
            role="separator"
            aria-orientation="vertical"
            aria-label="調整面板寬度"
            aria-valuenow={effectiveWidth}
            aria-valuemin={DRAWER_MIN_WIDTH}
            tabIndex={0}
            onPointerDown={(e) => {
              dragRef.current = { startX: e.clientX, startW: effectiveWidth };
              document.body.style.cursor = "col-resize";
              document.body.style.userSelect = "none";
            }}
            onKeyDown={(e) => {
              if (e.key === "ArrowLeft") {
                e.preventDefault();
                setUserWidth(clampWidth(effectiveWidth + 24));
              } else if (e.key === "ArrowRight") {
                e.preventDefault();
                setUserWidth(clampWidth(effectiveWidth - 24));
              }
            }}
            className="absolute inset-y-0 -left-1 w-2 cursor-col-resize hover:bg-accent/30 focus-visible:bg-accent/40"
          />
        ) : null}
        <div className="flex shrink-0 items-center gap-2 border-b border-edge px-3 py-2">
          <h2 id={titleId} className="min-w-0 flex-1 truncate text-base font-semibold">
            {title}
          </h2>
          {headerExtra}
          {onToggleMaximize && !narrow ? (
            <IconButton
              label={maximized ? "還原面板大小" : "放大面板"}
              onClick={onToggleMaximize}
              aria-pressed={maximized}
            >
              <svg
                aria-hidden="true"
                viewBox="0 0 16 16"
                className="h-4 w-4"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                {maximized ? (
                  <path d="M6 2v4H2M10 14v-4h4M2 10h4v4M14 6h-4V2" />
                ) : (
                  <path d="M2 6V2h4M14 10v4h-4M2 10v4h4M14 6V2h-4" />
                )}
              </svg>
            </IconButton>
          ) : null}
          <IconButton label="關閉" onClick={onClose}>
            <svg
              aria-hidden="true"
              viewBox="0 0 16 16"
              className="h-4 w-4"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
            >
              <path d="M3.5 3.5l9 9M12.5 3.5l-9 9" />
            </svg>
          </IconButton>
        </div>
        <div className={cx("min-h-0 flex-1 overflow-auto", bodyClassName)}>
          {children}
        </div>
        {footer ? (
          <div className="shrink-0 border-t border-edge px-3 py-2">{footer}</div>
        ) : null}
      </div>
    </>
  );
}
