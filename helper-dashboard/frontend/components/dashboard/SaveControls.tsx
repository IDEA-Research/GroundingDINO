"use client";

import React, { useEffect, useRef, useState } from "react";

import { useFocusTrap } from "@/components/chat/HelperDrawer";
import { Badge, Button, Field, fieldDescribedBy, useToast } from "@/components/ui";
import { api, describeApiError } from "@/lib/api";
import type { DashboardSpec, SavedDashboardEntry } from "@/lib/spec-schema";

// Save-state tracking + the 儲存 control.
//
// The shell keeps {savedId, savedSpecHash} for the spec on screen; the
// hash covers exactly {title, description, widgets, layout, variables}
// (never dashboard_id — every load/clone mints a new one). Compared on
// each render:
//   saved   — from the library and unchanged        -> 已儲存
//   dirty   — from the library, spec changed since  -> 有未儲存變更
//   unsaved — never saved (chat-created / template) -> 尚未儲存

export type SaveState = "saved" | "dirty" | "unsaved";

function stableStringify(v: unknown): string {
  if (v === null || typeof v !== "object") return JSON.stringify(v) ?? "null";
  if (Array.isArray(v)) return `[${v.map(stableStringify).join(",")}]`;
  const obj = v as Record<string, unknown>;
  const keys = Object.keys(obj)
    .filter((k) => obj[k] !== undefined)
    .sort();
  return `{${keys
    .map((k) => `${JSON.stringify(k)}:${stableStringify(obj[k])}`)
    .join(",")}}`;
}

// FNV-1a over the stable JSON — cheap, deterministic, good enough to
// detect "did anything change".
export function computeSpecHash(spec: DashboardSpec | null | undefined): string | null {
  if (!spec) return null;
  const s = stableStringify({
    title: spec.title,
    description: spec.description,
    widgets: spec.widgets,
    layout: spec.layout,
    variables: spec.variables,
  });
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i += 1) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}

// Where the on-screen spec came from, when it came from the library.
export interface SavedOrigin {
  savedId: string;
  name: string;
  description: string;
  tags: string[];
  // Hash of the spec as it was when loaded/saved.
  hash: string;
}

export function saveStateFor(
  spec: DashboardSpec | null,
  origin: SavedOrigin | null,
): SaveState | null {
  if (!spec) return null;
  if (!origin) return "unsaved";
  return computeSpecHash(spec) === origin.hash ? "saved" : "dirty";
}

const CHIP: Record<SaveState, { label: string; tone: "ok" | "warn" | "neutral" }> = {
  saved: { label: "已儲存", tone: "ok" },
  dirty: { label: "有未儲存變更", tone: "warn" },
  unsaved: { label: "尚未儲存", tone: "neutral" },
};

export function SaveStateChip({ state }: { state: SaveState | null }) {
  if (!state) return null;
  const c = CHIP[state];
  return (
    <span data-save-state={state} className="shrink-0">
      <Badge tone={c.tone}>{c.label}</Badge>
    </span>
  );
}

// 儲存 button + its popover. From the library: 覆寫「name」 / 另存新檔;
// otherwise a single name form (save as new).
export function SaveButton({
  spec,
  origin,
  onSaved,
}: {
  spec: DashboardSpec | null;
  origin: SavedOrigin | null;
  // Called with the NEW library entry (both overwrite and save-as-new
  // create a fresh saved_id).
  onSaved: (entry: SavedDashboardEntry) => void;
}) {
  const toast = useToast();
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<"choose" | "name">("name");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const popRef = useRef<HTMLDivElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  useFocusTrap(popRef, open, () => setOpen(false));

  // Click outside closes.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  if (!spec) return null;

  function openPopover() {
    setError(null);
    setName(origin ? `${origin.name} 副本` : spec?.title ?? "");
    setMode(origin ? "choose" : "name");
    setOpen(true);
  }

  async function saveAsNew(e: React.FormEvent) {
    e.preventDefault();
    if (!spec) return;
    const n = name.trim();
    if (!n || busy) return;
    setBusy(true);
    setError(null);
    try {
      const r = await api.saveDashboard({ dashboard_id: spec.dashboard_id, name: n });
      toast.show({
        title: "儲存成功",
        description: `已儲存為「${r.saved.name}」`,
        tone: "ok",
        durationMs: 4000,
      });
      setOpen(false);
      onSaved(r.saved);
    } catch (err) {
      const msg = describeApiError(err);
      setError(msg);
      toast.show({ title: "儲存失敗", description: msg, tone: "danger", durationMs: 6000 });
    } finally {
      setBusy(false);
    }
  }

  async function overwrite() {
    if (!spec || !origin || busy) return;
    setBusy(true);
    setError(null);
    try {
      // Save the new version FIRST, then retire the old entry — if the
      // save fails nothing is lost.
      const r = await api.saveDashboard({
        dashboard_id: spec.dashboard_id,
        name: origin.name,
        description: origin.description,
        tags: origin.tags,
      });
      try {
        await api.deleteSaved(origin.savedId);
      } catch (delErr) {
        toast.show({
          title: "舊版本未能移除",
          description: `新版本已儲存，但舊的「${origin.name}」仍在清單中：${describeApiError(delErr)}`,
          tone: "warn",
          durationMs: 8000,
        });
      }
      toast.show({
        title: "儲存成功",
        description: `已覆寫「${origin.name}」`,
        tone: "ok",
        durationMs: 4000,
      });
      setOpen(false);
      onSaved(r.saved);
    } catch (err) {
      const msg = describeApiError(err);
      setError(msg);
      toast.show({ title: "儲存失敗", description: msg, tone: "danger", durationMs: 6000 });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div ref={wrapRef} className="relative shrink-0">
      <Button
        type="button"
        variant="secondary"
        size="sm"
        onClick={() => (open ? setOpen(false) : openPopover())}
        aria-haspopup="dialog"
        aria-expanded={open}
      >
        儲存
      </Button>
      {open ? (
        <div
          ref={popRef}
          role="dialog"
          aria-label="儲存儀表板"
          tabIndex={-1}
          className="fixed inset-x-3 top-[calc(var(--hd-topbar-h,3.5rem)+0.25rem)] z-50 rounded-lg border border-edge bg-surface p-3 shadow-pop sm:absolute sm:inset-x-auto sm:right-0 sm:top-full sm:mt-1 sm:w-[20rem]"
        >
          {mode === "choose" && origin ? (
            <div className="space-y-2">
              <div className="text-sm text-ink-dim">
                此儀表板來自已儲存的「{origin.name}」。
              </div>
              <Button
                type="button"
                variant="primary"
                size="md"
                className="w-full justify-start"
                loading={busy}
                onClick={overwrite}
              >
                覆寫「{origin.name}」
              </Button>
              <Button
                type="button"
                variant="secondary"
                size="md"
                className="w-full justify-start"
                disabled={busy}
                onClick={() => setMode("name")}
              >
                另存新檔
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="w-full"
                disabled={busy}
                onClick={() => setOpen(false)}
              >
                取消
              </Button>
              {error ? <div className="text-xs text-danger">{error}</div> : null}
            </div>
          ) : (
            <form onSubmit={saveAsNew} className="space-y-2">
              <Field
                label="儀表板名稱"
                htmlFor="hd-save-name"
                hint="會以目前畫面的內容建立新的已儲存項目。"
                error={error ?? undefined}
              >
                <input
                  id="hd-save-name"
                  data-autofocus
                  aria-describedby={fieldDescribedBy("hd-save-name", { hint: true, error })}
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  maxLength={96}
                  autoComplete="off"
                  className="min-h-[40px] w-full rounded-md border border-edge bg-canvas px-2 text-sm text-ink focus:outline-none focus:ring-2 focus:ring-focus"
                />
              </Field>
              <div className="flex justify-end gap-2">
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  disabled={busy}
                  onClick={() => (origin ? setMode("choose") : setOpen(false))}
                >
                  {origin ? "返回" : "取消"}
                </Button>
                <Button
                  type="submit"
                  variant="primary"
                  size="sm"
                  loading={busy}
                  disabled={!name.trim()}
                >
                  確認儲存
                </Button>
              </div>
            </form>
          )}
        </div>
      ) : null}
    </div>
  );
}
