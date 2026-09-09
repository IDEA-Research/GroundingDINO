"use client";

import React, { useEffect, useRef, useState, type RefObject } from "react";

// Right-side Helper drawer. Default closed; opened by 「詢問 Helper」.
//
// The child (ChatPanel) stays MOUNTED while closed (display:none via the
// `hidden` utility, never unmounted) so the draft text and any in-flight
// job survive collapsing the drawer. At >= lg the drawer is an inline
// column beside the dashboard (non-modal); below lg it becomes a
// full-height sheet under the top bar (role=dialog, aria-modal, focus
// trap, Escape closes, body scroll locked). It never covers the top bar:
// its top edge is --hd-topbar-h (set by app/page.tsx).

const FOCUSABLE =
  'a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),' +
  'textarea:not([disabled]),[tabindex]:not([tabindex="-1"])';

export function useMinWidth(px: number): boolean {
  const [matches, setMatches] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mql = window.matchMedia(`(min-width: ${px}px)`);
    const update = () => setMatches(mql.matches);
    update();
    mql.addEventListener("change", update);
    return () => mql.removeEventListener("change", update);
  }, [px]);
  return matches;
}

function focusableIn(root: HTMLElement): HTMLElement[] {
  return Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE)).filter(
    (el) => el.getAttribute("aria-hidden") !== "true" && el.offsetParent !== null,
  );
}

// Minimal focus trap shared by the shell's own popovers/sheets (the
// Helper drawer, the 開啟 picker, the save popover). Tab/Shift+Tab cycle
// inside `ref`; Escape calls `onEscape`; focus returns to the previously
// focused element on close. Initial focus goes to `[data-autofocus]`,
// else the first focusable, else the root (which must be tabIndex=-1).
export function useFocusTrap(
  ref: RefObject<HTMLElement>,
  active: boolean,
  onEscape?: () => void,
): void {
  const escRef = useRef(onEscape);
  escRef.current = onEscape;

  useEffect(() => {
    if (!active) return;
    const root = ref.current;
    if (!root) return;
    const prev =
      typeof document !== "undefined"
        ? (document.activeElement as HTMLElement | null)
        : null;

    const raf = requestAnimationFrame(() => {
      const initial =
        root.querySelector<HTMLElement>("[data-autofocus]") ??
        focusableIn(root)[0] ??
        root;
      initial.focus();
    });

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        // Native listener runs before React's delegated handlers, so let
        // nested widgets own their Escape: an open Menu list closes itself,
        // and inline editors opt out with data-trap-ignore-escape.
        const t = e.target as HTMLElement | null;
        if (t?.closest?.('[role="menu"], [data-trap-ignore-escape]')) return;
        e.stopPropagation();
        escRef.current?.();
        return;
      }
      if (e.key !== "Tab") return;
      const list = focusableIn(root);
      if (list.length === 0) {
        e.preventDefault();
        root.focus();
        return;
      }
      const idx = list.indexOf(document.activeElement as HTMLElement);
      if (e.shiftKey) {
        if (idx <= 0) {
          e.preventDefault();
          list[list.length - 1].focus();
        }
      } else if (idx === -1 || idx === list.length - 1) {
        e.preventDefault();
        list[0].focus();
      }
    };
    root.addEventListener("keydown", onKey);
    return () => {
      cancelAnimationFrame(raf);
      root.removeEventListener("keydown", onKey);
      if (prev && document.contains(prev) && typeof prev.focus === "function") {
        prev.focus();
      }
    };
  }, [ref, active]);
}

export function HelperDrawer({
  open,
  onClose,
  children,
}: {
  open: boolean;
  onClose: () => void;
  children: React.ReactNode;
}) {
  const ref = useRef<HTMLElement>(null);
  const isLg = useMinWidth(1024);
  const overlay = open && !isLg;

  useFocusTrap(ref, overlay, onClose);

  // Inline (>= lg) mode: Escape still closes when focus is inside, and the
  // chat input receives focus on open.
  useEffect(() => {
    if (!open || !isLg) return;
    const el = ref.current;
    if (!el) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
      }
    };
    el.addEventListener("keydown", onKey);
    const target = el.querySelector<HTMLElement>("[data-autofocus]");
    target?.focus();
    return () => el.removeEventListener("keydown", onKey);
  }, [open, isLg, onClose]);

  // Body scroll lock while the sheet overlays the page.
  useEffect(() => {
    if (!overlay) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [overlay]);

  return (
    <>
      {overlay ? (
        <div
          className="fixed inset-x-0 bottom-0 top-[var(--hd-topbar-h,3.5rem)] z-30 bg-ink/30"
          onClick={onClose}
          aria-hidden="true"
        />
      ) : null}
      <aside
        ref={ref}
        id="hd-helper-drawer"
        role={overlay ? "dialog" : "complementary"}
        aria-modal={overlay ? true : undefined}
        aria-label="Helper 對話"
        aria-hidden={open ? undefined : true}
        tabIndex={-1}
        className={
          (open ? "flex " : "hidden ") +
          "fixed inset-x-0 bottom-0 top-[var(--hd-topbar-h,3.5rem)] z-40 flex-col border-t border-edge bg-canvas shadow-pop " +
          "lg:static lg:inset-auto lg:z-auto lg:h-full lg:w-[400px] lg:shrink-0 lg:border-l lg:border-t-0 lg:shadow-none"
        }
      >
        {children}
      </aside>
    </>
  );
}
