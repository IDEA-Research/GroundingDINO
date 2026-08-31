"use client";

import React, { useEffect, useRef, useState } from "react";

import type { BackendStatusState } from "@/lib/useBackendStatus";

export interface ChatMessage {
  id: string;
  from: "user" | "helper";
  text: string;
  meta?: string;
  // For ClarificationRequest messages — clickable question suggestions.
  questions?: string[];
  // For save-prompt messages — offer Yes/No chips.
  savePromptFor?: string | null;
}

export function ChatPanel({
  messages,
  onSend,
  onQuickReply,
  busy,
  busyMessage,
  backend,
}: {
  messages: ChatMessage[];
  onSend: (text: string) => void;
  onQuickReply: (text: string) => void;
  busy: boolean;
  busyMessage?: string | null;
  backend: BackendStatusState;
}) {
  const [text, setText] = useState("");
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages.length, busy]);

  function submit(e: React.FormEvent) {
    e.preventDefault();
    const t = text.trim();
    if (!t || busy) return;
    onSend(t);
    setText("");
  }

  const dotColor =
    backend.status === "up"
      ? "bg-ok"
      : backend.status === "down"
        ? "bg-alarm-high"
        : "bg-alarm-med";
  const dotLabel =
    backend.status === "up"
      ? "後端連線正常"
      : backend.status === "down"
        ? `後端無法連線${backend.httpCode ? ` (HTTP ${backend.httpCode})` : ""}`
        : "檢查後端連線中…";

  return (
    <div className="flex h-full flex-col bg-canvas">
      <div className="flex items-center justify-between border-b border-edge px-3 py-2">
        <div className="text-sm font-semibold text-ink">Helper</div>
        <div
          title={dotLabel}
          className="flex items-center gap-1.5 text-xs text-ink-faint"
        >
          <span
            data-backend-status={backend.status}
            className={`inline-block h-2 w-2 rounded-full ${dotColor}`}
          />
          <span className="hidden md:inline">{dotLabel}</span>
        </div>
      </div>
      <div
        ref={scrollRef}
        className="flex-1 space-y-2 overflow-y-auto p-3"
      >
        {messages.length === 0 ? (
          <div className="rounded-lg border border-edge bg-surface-2/60 p-3 text-sm text-ink-dim">
            嗨 — 告訴我你想要的儀表板。例如:「幫我建立新生兒 HR 與 SpO2
            監控儀表板」。
          </div>
        ) : null}
        {messages.map((m) => (
          <div
            key={m.id}
            className={
              "max-w-[85%] rounded-lg border px-3 py-2 text-sm " +
              (m.from === "user"
                ? "ml-auto border-chart-4/30 bg-chart-4/15 text-ink"
                : "border-edge bg-surface-2 text-ink")
            }
          >
            <div className="whitespace-pre-wrap">{m.text}</div>

            {m.questions && m.questions.length > 0 ? (
              <div className="mt-2 space-y-1">
                {m.questions.map((q, i) => (
                  <button
                    key={i}
                    type="button"
                    onClick={() => onQuickReply(q)}
                    className="block w-full rounded border border-edge-strong bg-surface px-2 py-1 text-left text-xs text-ink-dim hover:bg-surface-3 hover:text-ink"
                  >
                    {q}
                  </button>
                ))}
              </div>
            ) : null}

            {m.savePromptFor ? (
              <div className="mt-2 flex flex-wrap items-center gap-1.5">
                <span className="text-2xs uppercase tracking-wide text-ink-faint">
                  另存為
                </span>
                <SaveQuick onPick={(name) => onQuickReply(name)} />
                <button
                  type="button"
                  onClick={() => onQuickReply("no")}
                  className="rounded border border-edge-strong bg-surface-3 px-2 py-0.5 text-xs text-ink-dim hover:bg-surface-2 hover:text-ink"
                >
                  略過
                </button>
              </div>
            ) : null}

            {m.meta ? (
              <div className="mt-1 font-mono text-2xs uppercase tracking-wide text-ink-faint">
                {m.meta}
              </div>
            ) : null}
          </div>
        ))}
        {busy ? (
          <div className="max-w-[85%] rounded-lg border border-edge bg-surface-2 px-3 py-2 text-xs italic text-ink-faint">
            {busyMessage || "Helper 思考中…"}
          </div>
        ) : null}
      </div>
      <form
        onSubmit={submit}
        className="flex gap-2 border-t border-edge p-2"
      >
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={busy}
          placeholder={
            backend.status === "down"
              ? "後端無法連線 — 等待重新連線…"
              : "輸入訊息…"
          }
          className="flex-1 rounded-md border border-edge bg-surface-2 px-3 py-2 text-sm text-ink outline-none placeholder:text-ink-faint focus:ring-1 focus:ring-ring"
        />
        <button
          type="submit"
          disabled={busy || !text.trim() || backend.status === "down"}
          className="rounded-md border border-vital-spo2/40 bg-vital-spo2/15 px-3 py-2 text-sm font-medium text-vital-spo2 hover:bg-vital-spo2/25 disabled:opacity-40"
        >
          傳送
        </button>
      </form>
    </div>
  );
}

function SaveQuick({ onPick }: { onPick: (name: string) => void }) {
  const [v, setV] = useState("");
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (v.trim()) onPick(v.trim());
      }}
      className="flex gap-1"
    >
      <input
        value={v}
        onChange={(e) => setV(e.target.value)}
        placeholder="名稱…"
        className="rounded border border-edge bg-surface px-2 py-0.5 text-xs text-ink outline-none placeholder:text-ink-faint focus:ring-1 focus:ring-ring"
      />
      <button
        type="submit"
        className="rounded border border-ok/40 bg-ok/15 px-2 py-0.5 text-xs font-medium text-ok hover:bg-ok/25"
      >
        儲存
      </button>
    </form>
  );
}
