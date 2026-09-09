"use client";

import React, {
  cloneElement,
  isValidElement,
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
} from "react";

import { cx } from "./cx";

export interface MenuItem {
  label: React.ReactNode;
  onSelect?: () => void;
  tone?: "danger";
  disabled?: boolean;
  description?: React.ReactNode;
  icon?: React.ReactNode;
  // Optional drill-in submenu (e.g. 管理 -> 重新命名 / 複製 / 刪除). The item
  // opens the nested list in place with a 「返回」 entry.
  items?: MenuItem[];
}

export interface MenuProps {
  // Any node; when it is a React element (Button / IconButton) it receives
  // aria-haspopup / aria-expanded / aria-controls automatically.
  trigger: React.ReactNode;
  items: MenuItem[];
  align?: "start" | "end";
  // Accessible name for the menu list.
  label?: string;
  className?: string;
  menuClassName?: string;
}

// Keyboard: Arrow Up/Down move, Home/End jump, Enter/Space select, Escape
// closes (focus returns to the trigger), Tab/Shift+Tab stay inside the menu
// (focus-trapped). Click outside closes.
export function Menu({
  trigger,
  items,
  align = "end",
  label,
  className,
  menuClassName,
}: MenuProps) {
  const [open, setOpen] = useState(false);
  const [path, setPath] = useState<number[]>([]);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const triggerWrapRef = useRef<HTMLDivElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const itemRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const id = useId();
  const menuId = `${id}-menu`;

  // Resolve the currently visible list (root or a drilled-in submenu).
  let visible: MenuItem[] = items;
  for (const idx of path) {
    const next = visible[idx]?.items;
    if (!next) break;
    visible = next;
  }
  const inSubmenu = path.length > 0;

  const focusTrigger = useCallback(() => {
    const wrap = triggerWrapRef.current;
    if (!wrap) return;
    const focusable = wrap.querySelector<HTMLElement>(
      'button, [href], input, [tabindex]:not([tabindex="-1"])',
    );
    (focusable ?? wrap).focus();
  }, []);

  const close = useCallback(
    (returnFocus = true) => {
      setOpen(false);
      setPath([]);
      if (returnFocus) focusTrigger();
    },
    [focusTrigger],
  );

  const enabledIdx = visible
    .map((it, i) => (it.disabled ? -1 : i))
    .filter((i) => i >= 0);
  // With a submenu open the first entry is the synthetic 「返回」 button.
  const offset = inSubmenu ? 1 : 0;

  const focusItem = (pos: number) => {
    itemRefs.current[pos]?.focus();
  };

  useEffect(() => {
    if (!open) return;
    // Focus the first enabled entry once the list is painted.
    const t = window.setTimeout(() => {
      const first = inSubmenu ? 0 : enabledIdx.length ? enabledIdx[0] : 0;
      focusItem(first + (inSubmenu ? 0 : 0));
    }, 0);
    const onDocDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        close(false);
      }
    };
    document.addEventListener("mousedown", onDocDown);
    return () => {
      window.clearTimeout(t);
      document.removeEventListener("mousedown", onDocDown);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, path.join("/")]);

  const allFocusable = (): number[] => {
    const out: number[] = [];
    if (inSubmenu) out.push(0);
    for (const i of enabledIdx) out.push(i + offset);
    return out;
  };

  const onMenuKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const focusables = allFocusable();
    if (focusables.length === 0) {
      if (e.key === "Escape") {
        e.preventDefault();
        close();
      }
      return;
    }
    const current = itemRefs.current.findIndex(
      (el) => el === document.activeElement,
    );
    const pos = Math.max(0, focusables.indexOf(current));
    switch (e.key) {
      case "ArrowDown":
      case "Tab": {
        if (e.key === "Tab" && e.shiftKey) {
          e.preventDefault();
          focusItem(focusables[(pos - 1 + focusables.length) % focusables.length]);
          return;
        }
        e.preventDefault();
        focusItem(focusables[(pos + 1) % focusables.length]);
        return;
      }
      case "ArrowUp":
        e.preventDefault();
        focusItem(focusables[(pos - 1 + focusables.length) % focusables.length]);
        return;
      case "Home":
        e.preventDefault();
        focusItem(focusables[0]);
        return;
      case "End":
        e.preventDefault();
        focusItem(focusables[focusables.length - 1]);
        return;
      case "ArrowLeft":
        if (inSubmenu) {
          e.preventDefault();
          setPath((p) => p.slice(0, -1));
        }
        return;
      case "ArrowRight": {
        const item = visible[current - offset];
        if (item?.items && !item.disabled) {
          e.preventDefault();
          setPath((p) => [...p, current - offset]);
        }
        return;
      }
      case "Escape":
        e.preventDefault();
        close();
        return;
      default:
        return;
    }
  };

  const onTriggerKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      setOpen(true);
    } else if (e.key === "Escape" && open) {
      e.preventDefault();
      close();
    }
  };

  const triggerNode = isValidElement(trigger)
    ? cloneElement(trigger as React.ReactElement<Record<string, unknown>>, {
        "aria-haspopup": "menu",
        "aria-expanded": open,
        "aria-controls": open ? menuId : undefined,
      })
    : (
        <button
          type="button"
          aria-haspopup="menu"
          aria-expanded={open}
          aria-controls={open ? menuId : undefined}
          className="inline-flex min-h-9 items-center gap-1 rounded-md border border-edge-strong bg-surface px-3 text-sm text-ink hover:bg-surface-2 max-lg:min-h-10"
        >
          {trigger}
        </button>
      );

  const select = (item: MenuItem, idx: number) => {
    if (item.disabled) return;
    if (item.items) {
      setPath((p) => [...p, idx]);
      return;
    }
    close();
    item.onSelect?.();
  };

  itemRefs.current = [];

  return (
    <div ref={rootRef} className={cx("relative inline-block", className)}>
      <div
        ref={triggerWrapRef}
        className="inline-flex"
        onClick={() => {
          if (open) close(false);
          else setOpen(true);
        }}
        onKeyDown={onTriggerKeyDown}
      >
        {triggerNode}
      </div>
      {open ? (
        <div
          ref={menuRef}
          id={menuId}
          role="menu"
          aria-label={label}
          onKeyDown={onMenuKeyDown}
          className={cx(
            "absolute z-50 mt-1 max-h-[60vh] min-w-[12rem] max-w-[min(20rem,calc(100vw-1.5rem))] overflow-auto rounded-md border border-edge-strong bg-surface p-1 shadow-pop",
            align === "end" ? "right-0" : "left-0",
            menuClassName,
          )}
        >
          {inSubmenu ? (
            <button
              type="button"
              role="menuitem"
              tabIndex={-1}
              ref={(el) => {
                itemRefs.current[0] = el;
              }}
              onClick={() => setPath((p) => p.slice(0, -1))}
              className="flex min-h-9 w-full items-center gap-2 rounded px-2 text-left text-sm text-ink-dim hover:bg-surface-2 hover:text-ink max-lg:min-h-10"
            >
              <span aria-hidden="true">‹</span>
              返回
            </button>
          ) : null}
          {visible.length === 0 ? (
            <div className="px-2 py-2 text-sm text-ink-faint">沒有項目</div>
          ) : null}
          {visible.map((item, i) => (
            <button
              key={i}
              type="button"
              role="menuitem"
              tabIndex={-1}
              ref={(el) => {
                itemRefs.current[i + offset] = el;
              }}
              aria-disabled={item.disabled || undefined}
              disabled={item.disabled}
              aria-haspopup={item.items ? "menu" : undefined}
              onClick={() => select(item, i)}
              className={cx(
                "flex min-h-9 w-full items-start gap-2 rounded px-2 py-1.5 text-left text-sm max-lg:min-h-10",
                item.tone === "danger"
                  ? "text-danger hover:bg-danger/10"
                  : "text-ink hover:bg-surface-2",
                item.disabled && "cursor-not-allowed opacity-50 hover:bg-transparent",
              )}
            >
              {item.icon ? (
                <span aria-hidden="true" className="mt-0.5 shrink-0">
                  {item.icon}
                </span>
              ) : null}
              <span className="min-w-0 flex-1">
                <span className="block truncate">{item.label}</span>
                {item.description ? (
                  <span className="block text-xs text-ink-faint">
                    {item.description}
                  </span>
                ) : null}
              </span>
              {item.items ? (
                <span aria-hidden="true" className="shrink-0 text-ink-faint">
                  ›
                </span>
              ) : null}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
