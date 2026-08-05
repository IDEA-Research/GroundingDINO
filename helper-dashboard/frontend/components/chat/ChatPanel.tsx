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
      ? "bg-emerald-400"
      : backend.status === "down"
        ? "bg-rose-500"
        : "bg-amber-400";
  const dotLabel =
    backend.status === "up"
      ? "backend up"
      : backend.status === "down"
        ? `backend unreachable${backend.httpCode ? ` (HTTP ${backend.httpCode})` : ""}`
        : "checking backend…";

  return (
    <div className="flex h-full flex-col bg-panel">
      <div className="flex items-center justify-between border-b border-slate-800 px-3 py-2">
        <div className="text-sm font-semibold">Helper</div>
        <div
          title={dotLabel}
          className="flex items-center gap-1.5 text-[10px] text-slate-400"
        >
          <span className={`inline-block h-2 w-2 rounded-full ${dotColor}`} />
          <span className="hidden md:inline">{dotLabel}</span>
        </div>
      </div>
      <div
        ref={scrollRef}
        className="flex-1 space-y-2 overflow-y-auto p-3"
      >
        {messages.length === 0 ? (
          <div className="rounded bg-slate-900/60 p-3 text-sm text-slate-300">
            Hi — tell me what dashboard you'd like. Example: "Show me a CPU
            and memory dashboard for my nodes."
          </div>
        ) : null}
        {messages.map((m) => (
          <div
            key={m.id}
            className={
              "max-w-[85%] rounded-lg px-3 py-2 text-sm " +
              (m.from === "user"
                ? "ml-auto bg-sky-700 text-white"
                : "bg-slate-800 text-slate-100")
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
                    className="block w-full rounded border border-slate-600 bg-slate-900 px-2 py-1 text-left text-xs text-slate-200 hover:bg-slate-700"
                  >
                    {q}
                  </button>
                ))}
              </div>
            ) : null}

            {m.savePromptFor ? (
              <div className="mt-2 flex flex-wrap gap-1.5">
                <span className="text-[10px] uppercase tracking-wide text-slate-400">
                  save as
                </span>
                <SaveQuick onPick={(name) => onQuickReply(name)} />
                <button
                  type="button"
                  onClick={() => onQuickReply("no")}
                  className="rounded bg-slate-700 px-2 py-0.5 text-[11px] text-slate-100 hover:bg-slate-600"
                >
                  skip
                </button>
              </div>
            ) : null}

            {m.meta ? (
              <div className="mt-1 text-[10px] uppercase tracking-wide text-slate-300/70">
                {m.meta}
              </div>
            ) : null}
          </div>
        ))}
        {busy ? (
          <div className="max-w-[85%] rounded-lg bg-slate-800 px-3 py-2 text-xs italic text-slate-400">
            {busyMessage || "Helper is thinking…"}
          </div>
        ) : null}
      </div>
      <form
        onSubmit={submit}
        className="flex gap-2 border-t border-slate-800 p-2"
      >
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={busy}
          placeholder={
            backend.status === "down"
              ? "Backend unreachable — waiting for reconnect…"
              : "Ask Helper for a dashboard or a change…"
          }
          className="flex-1 rounded bg-slate-900 px-3 py-2 text-sm text-slate-100 outline-none placeholder:text-slate-500 focus:ring-1 focus:ring-sky-500"
        />
        <button
          type="submit"
          disabled={busy || !text.trim() || backend.status === "down"}
          className="rounded bg-sky-600 px-3 py-2 text-sm font-medium text-white disabled:opacity-40"
        >
          Send
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
        placeholder="name…"
        className="rounded bg-slate-900 px-2 py-0.5 text-[11px] text-slate-100 outline-none placeholder:text-slate-500"
      />
      <button
        type="submit"
        className="rounded bg-emerald-600 px-2 py-0.5 text-[11px] font-medium text-white hover:bg-emerald-500"
      >
        save
      </button>
    </form>
  );
}
