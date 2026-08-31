"use client";

import React, { useCallback, useMemo, useRef, useState } from "react";

import { ChatPanel, type ChatMessage } from "@/components/chat/ChatPanel";
import { DashboardPreview } from "@/components/dashboard/DashboardPreview";
import {
  SavedDashboardsBar,
  type SavedDashboardsBarHandle,
} from "@/components/dashboard/SavedDashboardsBar";
import { DevFixMonitor } from "@/components/devfix/DevFixMonitor";
import { JsonInspector } from "@/components/inspector/JsonInspector";
import { api } from "@/lib/api";
import type {
  DashboardSpec,
  PatchSpec,
  ReviewTrailEvent,
} from "@/lib/spec-schema";
import { useBackendStatus } from "@/lib/useBackendStatus";

function newId(): string {
  return Math.random().toString(36).slice(2, 10);
}

// Responsive shell. lg+ keeps the 3-column monitor layout
// (chat 3 / dashboard 6 / developer panel 3). Below lg the dashboard owns
// the screen: chat collapses into a bottom sheet and the developer panel
// becomes an overlay behind the 開發者 toggle. Each panel is a SINGLE
// component instance restyled per breakpoint — no duplicated state.
export default function HomePage() {
  const [sessionId] = useState<string>(() => `s-${newId()}`);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [spec, setSpec] = useState<DashboardSpec | null>(null);
  const [lastPatch, setLastPatch] = useState<PatchSpec | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [reviewTrail, setReviewTrail] = useState<ReviewTrailEvent[]>([]);
  const [busy, setBusy] = useState(false);
  const [busyMessage, setBusyMessage] = useState<string | null>(null);
  // Mobile-only panel states (ignored at lg+ where CSS pins the layout).
  const [chatOpen, setChatOpen] = useState(true);
  const [devOpen, setDevOpen] = useState(false);
  const backend = useBackendStatus();
  const barRef = useRef<SavedDashboardsBarHandle | null>(null);

  const currentDashboardId = useMemo(
    () => spec?.dashboard_id ?? null,
    [spec?.dashboard_id],
  );

  // Loading from the library hands back a persisted CLONE with a fresh
  // dashboard_id (contract C3); rendering it makes that clone the
  // current_dashboard_id every subsequent chat message carries.
  const handleLoadSaved = useCallback((dashboard: DashboardSpec) => {
    setSpec(dashboard);
  }, []);

  const send = useCallback(
    async (text: string) => {
      setMessages((prev) => [
        ...prev,
        { id: newId(), from: "user", text },
      ]);
      setBusy(true);
      // Best-effort hint — the real review step runs on the backend.
      setBusyMessage("正在產生並審查儀表板…");
      try {
        const res = await api.sendMessage(
          {
            session_id: sessionId,
            message: text,
            current_dashboard_id: currentDashboardId,
          },
          {
            onProgress: (p) => {
              const msg = p.message || `處理中…(${p.status})`;
              const pct =
                typeof p.pct === "number" && Number.isFinite(p.pct)
                  ? ` ${Math.max(0, Math.min(100, Math.round(p.pct)))}%`
                  : "";
              setBusyMessage(`${msg}${pct}`);
            },
          },
        );
        setMessages((prev) => [
          ...prev,
          {
            id: newId(),
            from: "helper",
            text: res.user_reply,
            meta: metaLabel(res.intent_type, res.runtime_used, res.fallback_reason),
            questions: res.clarification_questions || undefined,
            savePromptFor: res.save_prompt_for,
          },
        ]);
        if (res.dashboard) setSpec(res.dashboard);
        if (res.patch) setLastPatch(res.patch);
        setWarnings(res.warnings || []);
        setReviewTrail(res.review_trail || []);
        // Cheap GET so saves made through the chat flow appear in the bar.
        barRef.current?.refresh();
      } catch (e) {
        console.error("[chat-ui] sendMessage failed", {
          error: e,
          message: e instanceof Error ? e.message : String(e),
          stack: e instanceof Error ? e.stack : undefined,
          sessionId,
          currentDashboardId,
          text,
        });
        setMessages((prev) => [
          ...prev,
          {
            id: newId(),
            from: "helper",
            text: improvedErrorMessage(e, backend),
            meta: "error",
          },
        ]);
      } finally {
        setBusy(false);
        setBusyMessage(null);
      }
    },
    [sessionId, currentDashboardId, backend],
  );

  return (
    <>
      <div className="h-screen lg:grid lg:grid-cols-12">
        {/* Chat: left column at lg+, collapsible bottom sheet below lg. */}
        <aside
          className={
            "fixed inset-x-0 bottom-0 z-40 flex flex-col border-t border-edge bg-canvas " +
            (chatOpen ? "h-[45vh] " : "h-10 ") +
            "lg:static lg:z-auto lg:col-span-3 lg:h-full lg:border-r lg:border-t-0"
          }
        >
          <button
            type="button"
            onClick={() => setChatOpen((v) => !v)}
            aria-expanded={chatOpen}
            className="flex h-10 shrink-0 items-center justify-between px-3 text-xs text-ink-dim lg:hidden"
          >
            <span className="font-semibold text-ink">對話</span>
            <span>{chatOpen ? "收合" : "展開"}</span>
          </button>
          <div
            className={
              (chatOpen ? "flex " : "hidden ") + "min-h-0 flex-1 flex-col lg:flex"
            }
          >
            <ChatPanel
              messages={messages}
              onSend={send}
              onQuickReply={send}
              busy={busy}
              busyMessage={busyMessage}
              backend={backend}
            />
          </div>
        </aside>

        <main className="flex h-full min-h-0 flex-col pb-10 lg:col-span-6 lg:border-r lg:border-edge lg:pb-0">
          <div className="flex items-start gap-2 border-b border-edge px-2 py-1.5">
            <div className="min-w-0 flex-1">
              <SavedDashboardsBar ref={barRef} onLoad={handleLoadSaved} />
            </div>
            <SaveCurrentControl
              dashboardId={currentDashboardId}
              onSaved={() => barRef.current?.refresh()}
            />
            <button
              type="button"
              onClick={() => setDevOpen(true)}
              className="shrink-0 rounded border border-edge-strong bg-surface-3 px-2 py-0.5 text-xs text-ink-dim hover:bg-surface-2 hover:text-ink lg:hidden"
            >
              開發者
            </button>
          </div>
          <div className="min-h-0 flex-1">
            <DashboardPreview
              spec={spec}
              reviewing={busy}
              reviewTrail={reviewTrail}
            />
          </div>
        </main>

        {/* Developer panel: right column at lg+, slide-over below lg. */}
        {devOpen ? (
          <div
            className="fixed inset-0 z-40 bg-black/60 lg:hidden"
            onClick={() => setDevOpen(false)}
            aria-hidden="true"
          />
        ) : null}
        <aside
          className={
            (devOpen
              ? "fixed inset-y-0 right-0 z-50 w-[90vw] max-w-md border-l border-edge bg-canvas shadow-xl shadow-black/50 "
              : "hidden ") +
            "h-full lg:static lg:z-auto lg:col-span-3 lg:block lg:w-auto lg:max-w-none lg:border-l-0 lg:shadow-none"
          }
        >
          <JsonInspector
            spec={spec}
            lastPatch={lastPatch}
            warnings={warnings}
            reviewTrail={reviewTrail}
            onClose={() => setDevOpen(false)}
          />
        </aside>
      </div>
      {/* Floating, fixed bottom-right: tails devfix runs live without
          taking a slot in the dashboard grid. */}
      <DevFixMonitor />
    </>
  );
}

// Direct save of the CURRENT dashboard into the library — inline name
// input (no window.prompt), then the bar refetches. Bypasses the
// chat-state-fragile _PENDING_SAVE flow entirely.
function SaveCurrentControl({
  dashboardId,
  onSaved,
}: {
  dashboardId: string | null;
  onSaved: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!dashboardId) return null;

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const n = name.trim();
    if (!n || saving || !dashboardId) return;
    setSaving(true);
    setError(null);
    try {
      await api.saveDashboard({ dashboard_id: dashboardId, name: n });
      setName("");
      setOpen(false);
      onSaved();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="flex shrink-0 items-center gap-1">
      {open ? (
        <form onSubmit={submit} className="flex items-center gap-1">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="名稱…"
            autoFocus
            className="w-32 rounded border border-edge bg-surface-2 px-2 py-0.5 text-xs text-ink outline-none placeholder:text-ink-faint focus:ring-1 focus:ring-ring"
          />
          <button
            type="submit"
            disabled={saving || !name.trim()}
            className="rounded border border-ok/40 bg-ok/15 px-2 py-0.5 text-xs font-medium text-ok hover:bg-ok/25 disabled:opacity-40"
          >
            確認
          </button>
          <button
            type="button"
            onClick={() => {
              setOpen(false);
              setError(null);
            }}
            className="rounded border border-edge-strong bg-surface-3 px-2 py-0.5 text-xs text-ink-dim hover:bg-surface-2 hover:text-ink"
          >
            取消
          </button>
        </form>
      ) : (
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="rounded border border-edge-strong bg-surface-3 px-2 py-0.5 text-xs text-ink-dim hover:bg-surface-2 hover:text-ink"
        >
          儲存
        </button>
      )}
      {error ? (
        <span
          className="max-w-[16rem] truncate text-xs text-alarm-high"
          title={error}
        >
          {error}
        </span>
      ) : null}
    </div>
  );
}

function metaLabel(
  intent: string,
  runtime: string | null,
  fallback: string | null,
): string {
  let s = intent;
  if (runtime) s += ` · ${runtime}`;
  if (fallback) s += ` (fallback)`;
  return s;
}

function improvedErrorMessage(
  e: unknown,
  backend: ReturnType<typeof useBackendStatus>,
): string {
  const status = backend.httpCode ? ` (HTTP ${backend.httpCode})` : "";
  if (backend.status === "down") {
    return (
      `後端無法連線${status}。` +
      `Helper Dashboard 後端在 /api 沒有回應,我會在背景持續重試。詳細:` +
      (e instanceof Error ? e.message : String(e))
    );
  }
  return (
    `連線後端時發生問題。${e instanceof Error ? e.message : String(e)}`
  );
}
