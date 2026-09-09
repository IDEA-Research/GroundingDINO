"use client";

import React, { useContext, useEffect, useRef, useState } from "react";

import { Button, Disclosure, Field, fieldDescribedBy } from "@/components/ui";
import { DevModeContext } from "@/lib/dev-mode";
import type { BackendStatusState } from "@/lib/useBackendStatus";

export interface ChatMessage {
  id: string;
  from: "user" | "helper";
  // 繁中, user-facing.
  text: string;
  // Developer provenance (intent · runtime · fallback). NEVER rendered in
  // the user UI — only inside 「技術細節」 when the developer workspace is
  // unlocked.
  meta?: string;
  // Raw error/diagnostic detail; same rule as `meta`.
  detail?: string;
  // For ClarificationRequest messages — clickable question suggestions.
  questions?: string[];
  // For save-prompt messages — offer a name input / skip.
  savePromptFor?: string | null;
}

export function ChatPanel({
  messages,
  onSend,
  onQuickReply,
  busy,
  busyMessage,
  backend,
  onClose,
}: {
  messages: ChatMessage[];
  onSend: (text: string) => void;
  onQuickReply: (text: string) => void;
  busy: boolean;
  busyMessage?: string | null;
  backend: BackendStatusState;
  onClose?: () => void;
}) {
  const { unlocked } = useContext(DevModeContext);
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

  const down = backend.status === "down";
  const dotColor =
    backend.status === "up"
      ? "bg-ok"
      : down
        ? "bg-danger"
        : "bg-warn";
  const dotLabel =
    backend.status === "up"
      ? "伺服器連線正常"
      : down
        ? "伺服器無法連線"
        : "檢查連線中…";

  return (
    <div className="flex h-full min-h-0 flex-col bg-canvas">
      <div className="flex items-center gap-2 border-b border-edge px-3 py-2">
        <div className="min-w-0 flex-1">
          <div className="text-sm font-semibold text-ink">Helper</div>
          <div className="text-xs text-ink-dim">儀表板助理</div>
        </div>
        <div
          className="flex items-center gap-1.5 text-xs text-ink-dim"
          title={dotLabel}
        >
          <span
            data-backend-status={backend.status}
            aria-hidden="true"
            className={`inline-block h-2.5 w-2.5 rounded-full ${dotColor}`}
          />
          <span>{dotLabel}</span>
        </div>
        {onClose ? (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={onClose}
            aria-label="關閉 Helper 對話"
          >
            關閉
          </Button>
        ) : null}
      </div>

      <div
        ref={scrollRef}
        role="log"
        aria-live="polite"
        aria-label="對話內容"
        className="min-h-0 flex-1 space-y-2 overflow-y-auto p-3"
      >
        {messages.length === 0 ? (
          <div className="rounded-lg border border-edge bg-surface p-3 text-sm leading-relaxed text-ink-dim">
            你好，我是 Helper。告訴我你想看的監控畫面，例如「幫我建立新生兒 HR 與
            SpO2 監控儀表板」，或直接說要調整目前儀表板的哪個部分。
          </div>
        ) : null}

        {messages.map((m) => (
          <div
            key={m.id}
            className={
              "max-w-[88%] rounded-lg border px-3 py-2 text-sm leading-relaxed " +
              (m.from === "user"
                ? "ml-auto border-accent/30 bg-accent/10 text-ink"
                : "border-edge bg-surface text-ink")
            }
          >
            <div className="whitespace-pre-wrap break-words">{m.text}</div>

            {m.questions && m.questions.length > 0 ? (
              <div className="mt-2 space-y-1.5">
                {m.questions.map((q, i) => (
                  <Button
                    key={i}
                    type="button"
                    variant="secondary"
                    size="sm"
                    className="w-full justify-start text-left"
                    onClick={() => onQuickReply(q)}
                  >
                    {q}
                  </Button>
                ))}
              </div>
            ) : null}

            {m.savePromptFor ? (
              <SaveQuick
                id={`hd-save-name-${m.id}`}
                onPick={(name) => onQuickReply(name)}
                onSkip={() => onQuickReply("略過")}
              />
            ) : null}

            {unlocked && (m.meta || m.detail) ? (
              <div className="mt-2">
                <Disclosure summary="技術細節">
                  <pre className="whitespace-pre-wrap break-all font-mono text-xs text-ink-dim">
                    {[m.meta, m.detail].filter(Boolean).join("\n")}
                  </pre>
                </Disclosure>
              </div>
            ) : null}
          </div>
        ))}

        {busy ? (
          <div
            role="status"
            className="flex max-w-[88%] items-center gap-2 rounded-lg border border-edge bg-surface px-3 py-2 text-sm text-ink-dim"
          >
            <span
              aria-hidden="true"
              className="inline-block h-2 w-2 shrink-0 animate-pulse rounded-full bg-accent motion-reduce:animate-none"
            />
            <span>{busyMessage || "Helper 思考中…"}</span>
          </div>
        ) : null}
      </div>

      <form
        onSubmit={submit}
        className="flex items-end gap-2 border-t border-edge p-3"
      >
        <div className="min-w-0 flex-1">
          <Field
            label="訊息"
            htmlFor="hd-chat-input"
            hint={down ? "伺服器無法連線，暫時無法傳送。" : undefined}
          >
            <input
              id="hd-chat-input"
              data-autofocus
              aria-describedby={fieldDescribedBy("hd-chat-input", { hint: down })}
              value={text}
              onChange={(e) => setText(e.target.value)}
              disabled={busy}
              autoComplete="off"
              placeholder={down ? "等待伺服器重新連線…" : "描述你想看的畫面或要調整的地方"}
              className="min-h-[40px] w-full rounded-md border border-edge bg-surface px-3 text-sm text-ink placeholder:text-ink-faint focus:outline-none focus:ring-2 focus:ring-focus disabled:opacity-60"
            />
          </Field>
        </div>
        <Button
          type="submit"
          variant="primary"
          size="md"
          disabled={busy || !text.trim() || down}
          className="min-h-[40px]"
        >
          傳送
        </Button>
      </form>
    </div>
  );
}

function SaveQuick({
  id,
  onPick,
  onSkip,
}: {
  id: string;
  onPick: (name: string) => void;
  onSkip: () => void;
}) {
  const [v, setV] = useState("");
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (v.trim()) onPick(v.trim());
      }}
      className="mt-2 flex flex-wrap items-end gap-2"
    >
      <div className="min-w-0 flex-1">
        <Field label="儲存名稱" htmlFor={id}>
          <input
            id={id}
            value={v}
            onChange={(e) => setV(e.target.value)}
            autoComplete="off"
            placeholder="例如：新生兒 HR 與 SpO2"
            className="min-h-[36px] w-full rounded-md border border-edge bg-canvas px-2 text-sm text-ink placeholder:text-ink-faint focus:outline-none focus:ring-2 focus:ring-focus"
          />
        </Field>
      </div>
      <Button type="submit" variant="primary" size="sm" disabled={!v.trim()}>
        儲存
      </Button>
      <Button type="button" variant="ghost" size="sm" onClick={onSkip}>
        略過
      </Button>
    </form>
  );
}
