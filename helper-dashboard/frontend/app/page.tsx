"use client";

import React, { useCallback, useMemo, useState } from "react";

import { ChatPanel, type ChatMessage } from "@/components/chat/ChatPanel";
import { DashboardPreview } from "@/components/dashboard/DashboardPreview";
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

export default function HomePage() {
  const [sessionId] = useState<string>(() => `s-${newId()}`);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [spec, setSpec] = useState<DashboardSpec | null>(null);
  const [lastPatch, setLastPatch] = useState<PatchSpec | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [reviewTrail, setReviewTrail] = useState<ReviewTrailEvent[]>([]);
  const [busy, setBusy] = useState(false);
  const [busyMessage, setBusyMessage] = useState<string | null>(null);
  const backend = useBackendStatus();

  const currentDashboardId = useMemo(
    () => spec?.dashboard_id ?? null,
    [spec?.dashboard_id],
  );

  const send = useCallback(
    async (text: string) => {
      setMessages((prev) => [
        ...prev,
        { id: newId(), from: "user", text },
      ]);
      setBusy(true);
      // Best-effort hint — the real review step runs on the backend.
      setBusyMessage("Reviewing rendered dashboard…");
      try {
        const res = await api.sendMessage(
          {
            session_id: sessionId,
            message: text,
            current_dashboard_id: currentDashboardId,
          },
          {
            onProgress: (p) => {
              const msg = p.message || `Working… (${p.status})`;
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
    <div className="grid h-screen grid-cols-12 gap-0">
      <aside className="col-span-3 h-full border-r border-slate-800">
        <ChatPanel
          messages={messages}
          onSend={send}
          onQuickReply={send}
          busy={busy}
          busyMessage={busyMessage}
          backend={backend}
        />
      </aside>
      <main className="col-span-6 h-full border-r border-slate-800">
        <DashboardPreview spec={spec} reviewing={busy} />
      </main>
      <aside className="col-span-3 h-full">
        <JsonInspector
          spec={spec}
          lastPatch={lastPatch}
          warnings={warnings}
          reviewTrail={reviewTrail}
        />
      </aside>
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
      `Backend is unreachable${status}. ` +
      `The Helper Dashboard backend is not responding at /api. ` +
      `I'll keep retrying in the background. Detail: ` +
      (e instanceof Error ? e.message : String(e))
    );
  }
  return (
    `Something went wrong reaching the backend. ${
      e instanceof Error ? e.message : String(e)
    }`
  );
}
