"use client";

import React, { useEffect, useRef, useState } from "react";

import { useFocusTrap } from "@/components/chat/HelperDrawer";
import { Button, Field, Menu, Skeleton, fieldDescribedBy } from "@/components/ui";
import type { SavedDashboardSummary } from "@/lib/spec-schema";

// 「開啟」 popover in the top bar: the saved library (each entry with a
// labelled 管理 menu — 開啟 / 重新命名 / 複製 / 刪除 — instead of a bare ✕),
// 從範本建立, and 資料來源狀態. The same list is reused by the home
// empty state. All library mutations are implemented by the shell
// (app/page.tsx) and passed in as `SavedLibraryApi` so the toast/undo
// flow has one owner.

export interface SavedLibraryApi {
  // null while the first list request is in flight.
  entries: SavedDashboardSummary[] | null;
  error: string | null;
  refresh: () => void;
  open: (savedId: string) => Promise<void>;
  rename: (savedId: string, name: string) => Promise<void>;
  duplicate: (savedId: string) => Promise<void>;
  // Soft delete + 10 s 復原 toast (owner: shell).
  remove: (savedId: string) => Promise<void>;
  // saved_id of the entry currently being acted on (disables its row).
  busyId: string | null;
  // saved_id the on-screen spec came from, if any.
  currentSavedId: string | null;
}

function entrySubtitle(e: SavedDashboardSummary): string {
  if (e.description) return e.description;
  const parts: string[] = [];
  if (e.tags.length > 0) parts.push(e.tags.join("、"));
  parts.push(`${e.widget_types.length} 個元件`);
  if (e.used_count > 0) parts.push(`開啟 ${e.used_count} 次`);
  return parts.join(" · ");
}

export function SavedDashboardList({
  library,
  onAfterOpen,
  className,
}: {
  library: SavedLibraryApi;
  onAfterOpen?: () => void;
  className?: string;
}) {
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [renameError, setRenameError] = useState<string | null>(null);

  function startRename(e: SavedDashboardSummary) {
    setRenamingId(e.saved_id);
    setDraft(e.name);
    setRenameError(null);
  }

  async function submitRename(ev: React.FormEvent, id: string) {
    ev.preventDefault();
    const n = draft.trim();
    if (!n) return;
    try {
      await library.rename(id, n);
      setRenamingId(null);
    } catch (err) {
      setRenameError(err instanceof Error ? err.message : String(err));
    }
  }

  if (library.entries === null) {
    return (
      <div className={className} aria-busy="true" aria-label="載入已儲存儀表板中">
        <Skeleton className="mb-2 h-12 w-full rounded-md" />
        <Skeleton className="mb-2 h-12 w-full rounded-md" />
        <Skeleton className="h-12 w-full rounded-md" />
      </div>
    );
  }

  if (library.error && library.entries.length === 0) {
    return (
      <div className={className}>
        <div className="flex flex-wrap items-center gap-2 rounded-md border border-danger/40 bg-danger/10 p-2 text-sm text-ink">
          <span className="min-w-0 flex-1">無法取得已儲存清單：{library.error}</span>
          <Button type="button" variant="secondary" size="sm" onClick={library.refresh}>
            重試
          </Button>
        </div>
      </div>
    );
  }

  if (library.entries.length === 0) {
    return (
      <div className={className}>
        <div className="rounded-md border border-dashed border-edge-strong p-3 text-sm text-ink-dim">
          尚未儲存任何儀表板。
        </div>
      </div>
    );
  }

  return (
    <ul className={"space-y-1.5 " + (className ?? "")} aria-label="已儲存儀表板">
      {library.entries.map((e) => {
        const busy = library.busyId === e.saved_id;
        const isCurrent = library.currentSavedId === e.saved_id;
        if (renamingId === e.saved_id) {
          const inputId = `hd-rename-${e.saved_id}`;
          return (
            <li key={e.saved_id} className="rounded-md border border-accent/40 bg-surface p-2">
              <form onSubmit={(ev) => submitRename(ev, e.saved_id)} className="space-y-2">
                <Field label="新名稱" htmlFor={inputId} error={renameError ?? undefined}>
                  <input
                    id={inputId}
                    autoFocus
                    data-trap-ignore-escape
                    aria-describedby={fieldDescribedBy(inputId, { error: renameError })}
                    value={draft}
                    maxLength={96}
                    onChange={(ev) => setDraft(ev.target.value)}
                    onKeyDown={(ev) => {
                      if (ev.key === "Escape") {
                        ev.preventDefault();
                        ev.stopPropagation();
                        setRenamingId(null);
                      }
                    }}
                    className="min-h-[40px] w-full rounded-md border border-edge bg-canvas px-2 text-sm text-ink focus:outline-none focus:ring-2 focus:ring-focus"
                  />
                </Field>
                <div className="flex justify-end gap-2">
                  <Button type="button" variant="ghost" size="sm" onClick={() => setRenamingId(null)}>
                    取消
                  </Button>
                  <Button type="submit" variant="primary" size="sm" loading={busy} disabled={!draft.trim()}>
                    確定
                  </Button>
                </div>
              </form>
            </li>
          );
        }
        return (
          <li
            key={e.saved_id}
            className={
              "flex items-center gap-2 rounded-md border bg-surface pl-1 pr-1.5 " +
              (isCurrent ? "border-accent/50" : "border-edge")
            }
          >
            <button
              type="button"
              disabled={busy}
              onClick={() => {
                library
                  .open(e.saved_id)
                  .then(() => onAfterOpen?.())
                  .catch(() => {
                    // already surfaced as a toast by the shell
                  });
              }}
              className="min-h-[44px] min-w-0 flex-1 rounded-md px-2 py-1.5 text-left hover:bg-surface-2 focus:outline-none focus:ring-2 focus:ring-focus disabled:opacity-60"
              aria-label={`開啟 ${e.name}`}
            >
              <div className="flex items-center gap-2">
                <span className="truncate text-sm font-medium text-ink">{e.name}</span>
                {isCurrent ? (
                  <span className="shrink-0 text-xs text-accent">目前</span>
                ) : null}
              </div>
              <div className="truncate text-xs text-ink-dim">{entrySubtitle(e)}</div>
            </button>
            <Menu
              align="end"
              label={`管理 ${e.name}`}
              trigger={
                <Button type="button" variant="ghost" size="sm" aria-label={`管理 ${e.name}`}>
                  管理
                </Button>
              }
              items={[
                {
                  label: "開啟",
                  onSelect: () => {
                    library
                  .open(e.saved_id)
                  .then(() => onAfterOpen?.())
                  .catch(() => {
                    // already surfaced as a toast by the shell
                  });
                  },
                  disabled: busy,
                },
                {
                  label: "重新命名",
                  onSelect: () => startRename(e),
                  disabled: busy,
                },
                {
                  label: "複製",
                  description: `建立「${e.name} 副本」`,
                  onSelect: () => {
                    void library.duplicate(e.saved_id);
                  },
                  disabled: busy,
                },
                {
                  label: "刪除",
                  description: "可在 10 秒內復原",
                  tone: "danger",
                  onSelect: () => {
                    void library.remove(e.saved_id);
                  },
                  disabled: busy,
                },
              ]}
            />
          </li>
        );
      })}
    </ul>
  );
}

export function DashboardPickerMenu({
  library,
  onOpenTemplates,
  onOpenDataSource,
}: {
  library: SavedLibraryApi;
  onOpenTemplates: () => void;
  onOpenDataSource: () => void;
}) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);
  const popRef = useRef<HTMLDivElement>(null);

  useFocusTrap(popRef, open, () => setOpen(false));

  useEffect(() => {
    if (!open) return;
    library.refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // Click outside closes — but not clicks inside a (possibly portaled)
  // 管理 menu list.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const t = e.target as HTMLElement | null;
      if (!t) return;
      if (wrapRef.current?.contains(t)) return;
      if (t.closest?.('[role="menu"]')) return;
      setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  const count = library.entries?.length ?? 0;

  return (
    <div ref={wrapRef} className="relative shrink-0">
      <Button
        type="button"
        variant="secondary"
        size="sm"
        aria-haspopup="dialog"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        開啟{count > 0 ? ` (${count})` : ""}
      </Button>
      {open ? (
        <div
          ref={popRef}
          role="dialog"
          aria-label="開啟儀表板"
          tabIndex={-1}
          className="fixed inset-x-3 top-[calc(var(--hd-topbar-h,3.5rem)+0.25rem)] z-50 rounded-lg border border-edge bg-surface p-3 shadow-pop sm:absolute sm:inset-x-auto sm:left-0 sm:top-full sm:mt-1 sm:w-[24rem]"
        >
          <div className="mb-2 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-ink">已儲存儀表板</h2>
            <Button type="button" variant="ghost" size="sm" onClick={() => setOpen(false)} aria-label="關閉清單">
              關閉
            </Button>
          </div>
          <div className="max-h-[50vh] overflow-y-auto pr-0.5">
            <SavedDashboardList library={library} onAfterOpen={() => setOpen(false)} />
          </div>
          <div className="mt-3 grid grid-cols-2 gap-2 border-t border-edge pt-3">
            <Button
              type="button"
              variant="secondary"
              size="md"
              onClick={() => {
                setOpen(false);
                onOpenTemplates();
              }}
            >
              從範本建立
            </Button>
            <Button
              type="button"
              variant="secondary"
              size="md"
              onClick={() => {
                setOpen(false);
                onOpenDataSource();
              }}
            >
              資料來源狀態
            </Button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
