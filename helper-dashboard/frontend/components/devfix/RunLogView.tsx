"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";

import { Badge, Button, Disclosure, Field } from "@/components/ui";
import { api, describeApiError, isApiError } from "@/lib/api";
import type { DevRunEvent } from "@/lib/spec-schema";
import {
  EVENTS_POLL_MS,
  isActiveStatus,
  type RunStatusOrIdle,
} from "@/lib/useDevfixRuns";

// Big Guy run log. Summary-first (notes + errors + finish + the final
// reply), with 全部/工具呼叫/編輯/指令/錯誤/思考 filters, a search box and
// per-entry 顯示全文. Streams GET /runs/{id}/events?after=N every 2 s while
// the run is active; if that route is unavailable (older backend) it
// falls back to the rendered log lines of GET /runs/{id}?after=N.

export type LogKind =
  | "text"
  | "thinking"
  | "tool"
  | "edit"
  | "bash"
  | "read"
  | "error"
  | "note"
  | "finish"
  | "step"
  | "other";

export interface LogEntry {
  i: number;
  kind: LogKind;
  title: string;
  body: string;
  ts: number | null;
  status: string | null;
}

type Filter = "all" | "tool" | "edit" | "bash" | "error" | "thinking";

const FILTERS: { id: Filter; label: string }[] = [
  { id: "all", label: "全部" },
  { id: "tool", label: "工具呼叫" },
  { id: "edit", label: "編輯" },
  { id: "bash", label: "指令" },
  { id: "error", label: "錯誤" },
  { id: "thinking", label: "思考" },
];

const KIND_LABEL: Record<LogKind, string> = {
  text: "回覆",
  thinking: "思考",
  tool: "工具",
  edit: "編輯",
  bash: "指令",
  read: "讀取",
  error: "錯誤",
  note: "備註",
  finish: "結束",
  step: "步驟",
  other: "事件",
};

type Tone = "neutral" | "ok" | "warn" | "danger" | "info" | "accent";

function kindTone(k: LogKind, status: string | null): Tone {
  if (k === "error" || status === "error") return "danger";
  if (k === "edit") return "accent";
  if (k === "bash") return "info";
  if (k === "finish") return "ok";
  if (k === "note") return "warn";
  if (k === "thinking") return "neutral";
  return "neutral";
}

// Row background tone — same danger-for-errors as the Badge, but 備註/結束
// (note/finish) are deliberately flattened to neutral so only the things
// that actually need attention (errors, edits, commands) tint the row.
function rowTone(k: LogKind, status: string | null): Tone {
  if (k === "error" || status === "error") return "danger";
  if (k === "note" || k === "finish") return "neutral";
  return kindTone(k, status);
}

// Full-row border + low-opacity fill per tone, matching the `bg-x/10`
// convention already used for Badge chips and the diff view's +/- lines
// (RunDiffView.tsx) — extended here to the whole log row.
const ROW_TONE: Record<Tone, string> = {
  neutral: "border-edge bg-surface",
  ok: "border-ok/30 bg-ok/10",
  warn: "border-warn/30 bg-warn/10",
  danger: "border-danger/30 bg-danger/10",
  info: "border-info/30 bg-info/10",
  accent: "border-accent/30 bg-accent/10",
};

// Solid fill for the log-level distribution bar's segments (same palette).
const BAR_TONE: Record<Tone, string> = {
  neutral: "bg-ink-faint",
  ok: "bg-ok",
  warn: "bg-warn",
  danger: "bg-danger",
  info: "bg-info",
  accent: "bg-accent",
};

function obj(v: unknown): Record<string, unknown> | null {
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

function str(...vals: unknown[]): string | null {
  for (const v of vals) if (typeof v === "string" && v.length > 0) return v;
  return null;
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function pretty(v: unknown): string {
  if (v === undefined || v === null) return "";
  if (typeof v === "string") return v;
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}

const EDIT_TOOLS = new Set(["edit", "write", "patch", "multiedit", "apply_patch"]);

// Tolerant normalisation of an OpenCode `--format json` event (or a
// runtime note) into one log entry. Unknown shapes still render (kind
// "other" with the JSON body) so nothing is silently dropped.
export function normalizeEvent(raw: DevRunEvent, i: number): LogEntry {
  const type = (str(raw.type, raw.kind) ?? "other").toLowerCase();
  const part = obj(raw.part) ?? raw;
  const isoTs = (v: unknown): number | null =>
    typeof v === "string" && Number.isFinite(Date.parse(v)) ? Date.parse(v) : null;
  const ts = num(raw.timestamp) ?? num(part.timestamp) ?? isoTs(raw.ts) ?? isoTs(part.ts);
  const text = str(part.text, part.message, part.content, raw.text, raw.message);

  if (type === "text") {
    return { i, kind: "text", title: "Big Guy", body: text ?? "", ts, status: null };
  }
  if (type === "reasoning" || type === "thinking") {
    return { i, kind: "thinking", title: "思考", body: text ?? "", ts, status: null };
  }
  if (type === "tool_use" || type === "tool" || type === "tool_call" || type === "tool_result") {
    const tool = (str(part.tool, part.name, raw.tool) ?? "tool").toLowerCase();
    const state = obj(part.state) ?? {};
    const status = str(state.status, part.status, raw.status);
    const input = state.input ?? part.input ?? raw.input;
    const inp = obj(input) ?? {};
    const output = str(state.output, part.output, raw.output);
    const err = str(state.error, part.error, raw.error);
    const title =
      str(state.title, part.title, raw.title) ??
      str(inp.filePath, inp.path, inp.file, raw.path) ??
      str(inp.command, raw.command) ??
      tool;
    const kind: LogKind = EDIT_TOOLS.has(tool)
      ? "edit"
      : tool === "bash"
        ? "bash"
        : tool === "read" || tool === "glob" || tool === "grep" || tool === "list"
          ? "read"
          : "tool";
    const parts: string[] = [];
    const cmd = str(inp.command);
    if (cmd) parts.push(`$ ${cmd}`);
    else if (Object.keys(inp).length > 0) parts.push(pretty(inp));
    if (output) parts.push(output);
    if (err) parts.push(`error: ${err}`);
    return {
      i,
      kind: err || status === "error" ? "error" : kind,
      title: `${tool} · ${title}`,
      body: parts.join("\n"),
      ts,
      status: status ?? null,
    };
  }
  if (type === "error") {
    const e = obj(raw.error) ?? obj(part.error);
    return {
      i,
      kind: "error",
      title: str(e?.name, "錯誤") ?? "錯誤",
      body: text ?? str(e?.message) ?? pretty(raw.error ?? part),
      ts,
      status: "error",
    };
  }
  if (type === "note" || type === "info") {
    return { i, kind: "note", title: "備註", body: text ?? pretty(part), ts, status: null };
  }
  if (type === "finish" || type === "run_finished" || type === "done") {
    return {
      i,
      kind: "finish",
      title: `結束${str(raw.status, part.status) ? ` · ${str(raw.status, part.status)}` : ""}`,
      body: text ?? str(raw.error_kind, part.error_kind) ?? "",
      ts,
      status: str(raw.status, part.status),
    };
  }
  if (type === "step_start" || type === "step_finish" || type === "step") {
    return {
      i,
      kind: "step",
      title: type === "step_start" ? "步驟開始" : "步驟結束",
      body: type === "step_finish" ? pretty(part) : "",
      ts,
      status: null,
    };
  }
  return { i, kind: "other", title: type, body: text ?? pretty(raw), ts, status: null };
}

// Rendered log lines (fallback) are classified by their emoji prefix.
export function lineToEntry(line: string, i: number): LogEntry {
  const t = line.trim();
  const pick = (kind: LogKind, title: string): LogEntry => ({
    i,
    kind,
    title,
    body: t,
    ts: null,
    status: kind === "error" ? "error" : null,
  });
  if (t.startsWith("✏️")) return pick("edit", "編輯");
  if (t.startsWith("▶️")) return pick("bash", "指令");
  if (t.startsWith("📖")) return pick("read", "讀取");
  if (t.startsWith("🔧")) return pick("tool", "工具");
  if (t.startsWith("💭")) return pick("thinking", "思考");
  if (t.startsWith("💬")) return pick("text", "Big Guy");
  if (t.startsWith("❌") || t.startsWith("⛔")) return pick("error", "錯誤");
  if (t.startsWith("ℹ️")) return pick("note", "備註");
  if (t.startsWith("🏁")) return pick("finish", "結束");
  return pick("other", "紀錄");
}

function matches(e: LogEntry, f: Filter): boolean {
  switch (f) {
    case "all":
      return true;
    case "tool":
      return e.kind === "tool" || e.kind === "edit" || e.kind === "bash" || e.kind === "read";
    case "edit":
      return e.kind === "edit";
    case "bash":
      return e.kind === "bash";
    case "error":
      return e.kind === "error" || e.status === "error";
    case "thinking":
      return e.kind === "thinking";
    default:
      return true;
  }
}

function summaryOf(entries: LogEntry[]): LogEntry[] {
  const out = entries.filter(
    (e) => e.kind === "note" || e.kind === "error" || e.kind === "finish",
  );
  const lastText = [...entries].reverse().find((e) => e.kind === "text" && e.body.trim());
  if (lastText && !out.includes(lastText)) out.push(lastText);
  return out.sort((a, b) => a.i - b.i);
}

const PREVIEW_CHARS = 280;

function fmtTs(ts: number | null): string {
  if (ts === null) return "";
  const ms = ts > 1e12 ? ts : ts * 1000;
  const d = new Date(ms);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function EntryRow({ e }: { e: LogEntry }) {
  const long = e.body.length > PREVIEW_CHARS;
  return (
    <li className={`rounded-md border px-2 py-1.5 ${ROW_TONE[rowTone(e.kind, e.status)]}`}>
      <div className="flex flex-wrap items-center gap-1.5 text-xs">
        <span className="w-16 shrink-0 font-mono tabular-nums text-ink-faint">
          {e.ts !== null ? fmtTs(e.ts) : ""}
        </span>
        <Badge tone={kindTone(e.kind, e.status)}>{KIND_LABEL[e.kind]}</Badge>
        <span className="min-w-0 flex-1 truncate font-medium text-ink" title={e.title}>
          {e.title}
        </span>
        {e.status && e.status !== "completed" ? (
          <span className="font-mono text-ink-dim">{e.status}</span>
        ) : null}
      </div>
      {e.body ? (
        long ? (
          <>
            <pre className="mt-1 whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-ink-dim">
              {e.body.slice(0, PREVIEW_CHARS)}…
            </pre>
            <Disclosure summary="顯示全文">
              <pre className="mt-1 max-h-80 overflow-auto whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-ink-dim">
                {e.body}
              </pre>
            </Disclosure>
          </>
        ) : (
          <pre className="mt-1 whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-ink-dim">
            {e.body}
          </pre>
        )
      ) : null}
    </li>
  );
}

// Grafana-style log-level distribution: one horizontal stacked bar showing
// each event kind's share of the currently loaded batch. Computed from the
// full loaded stream (not the active filter/search) so it stays a stable
// at-a-glance overview — "was this run mostly tool calls, or errors?" —
// rather than shrinking to whatever the user is currently filtering on.
const KIND_ORDER: LogKind[] = [
  "error",
  "edit",
  "bash",
  "tool",
  "read",
  "thinking",
  "text",
  "note",
  "finish",
  "step",
  "other",
];

function LogLevelBar({ entries }: { entries: LogEntry[] }) {
  const total = entries.length;
  if (total === 0) return null;
  const counts = new Map<LogKind, number>();
  for (const e of entries) counts.set(e.kind, (counts.get(e.kind) ?? 0) + 1);
  const segments = KIND_ORDER.filter((k) => (counts.get(k) ?? 0) > 0).map((k) => ({
    kind: k,
    count: counts.get(k) as number,
  }));

  return (
    <div className="flex min-w-[10rem] flex-1 items-center gap-2">
      <span className="shrink-0 text-xs text-ink-dim">種類分佈</span>
      <div
        role="img"
        aria-label={`共 ${total} 筆：${segments.map((s) => `${KIND_LABEL[s.kind]} ${s.count}`).join("、")}`}
        className="flex h-2 flex-1 overflow-hidden rounded-full bg-surface-2"
      >
        {segments.map((s) => (
          <div
            key={s.kind}
            title={`${KIND_LABEL[s.kind]} · ${s.count}`}
            className={BAR_TONE[rowTone(s.kind, null)]}
            style={{ width: `${(s.count / total) * 100}%` }}
          />
        ))}
      </div>
      <span className="shrink-0 font-mono tabular-nums text-xs text-ink-faint">{total}</span>
    </div>
  );
}

export function RunLogView({
  runId,
  status,
  onLatest,
}: {
  runId: string;
  status: RunStatusOrIdle;
  // Latest entry title — the 目前任務 card shows it as progress.
  onLatest?: (summary: string | null) => void;
}) {
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState(false);
  const [mode, setMode] = useState<"events" | "lines">("events");
  const cursor = useRef(0);
  const count = useRef(0);
  const listRef = useRef<HTMLOListElement>(null);
  const active = isActiveStatus(status);

  // New run -> reset the stream.
  useEffect(() => {
    cursor.current = 0;
    count.current = 0;
    setEntries([]);
    setError(null);
    setMode("events");
  }, [runId]);

  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    async function tick() {
      try {
        if (mode === "events") {
          const r = await api.devfixEvents(runId, cursor.current);
          if (stopped) return;
          cursor.current = r.next_cursor;
          if (r.events.length > 0) {
            const fresh = r.events.map((ev) => normalizeEvent(ev, count.current++));
            setEntries((prev) => [...prev, ...fresh]);
          }
        } else {
          const r = await api.devfixRun(runId, cursor.current);
          if (stopped) return;
          cursor.current = r.next_cursor;
          if (r.lines.length > 0) {
            const fresh = r.lines.map((l) => lineToEntry(l, count.current++));
            setEntries((prev) => [...prev, ...fresh]);
          }
        }
        setError(null);
      } catch (e) {
        if (stopped) return;
        if (mode === "events" && isApiError(e) && (e.status === 404 || e.status === 405)) {
          // Older backend without /events — switch to rendered lines.
          cursor.current = 0;
          setMode("lines");
          return;
        }
        setError(e instanceof Error ? e.message : describeApiError(e));
      }
      if (!stopped && active) timer = setTimeout(tick, EVENTS_POLL_MS);
    }

    tick();
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [runId, active, mode]);

  useEffect(() => {
    onLatest?.(entries.length > 0 ? entries[entries.length - 1].title : null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [entries.length]);

  // Follow the newest entry while the full log is open.
  useEffect(() => {
    if (!expanded) return;
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [entries.length, expanded]);

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    const base = expanded ? entries : summaryOf(entries);
    return base.filter(
      (e) =>
        matches(e, filter) &&
        (!q || e.title.toLowerCase().includes(q) || e.body.toLowerCase().includes(q)),
    );
  }, [entries, expanded, filter, query]);

  const errorCount = entries.filter((e) => e.kind === "error" || e.status === "error").length;

  return (
    <div className="flex min-h-0 flex-col gap-2">
      <LogLevelBar entries={entries} />

      <div className="flex flex-wrap items-end gap-2">
        <div role="group" aria-label="紀錄篩選" className="flex flex-wrap gap-1">
          {FILTERS.map((f) => (
            <Button
              key={f.id}
              type="button"
              size="sm"
              variant={filter === f.id ? "primary" : "ghost"}
              aria-pressed={filter === f.id}
              onClick={() => setFilter(f.id)}
            >
              {f.label}
              {f.id === "error" && errorCount > 0 ? ` (${errorCount})` : ""}
            </Button>
          ))}
        </div>
        <div className="min-w-[10rem] flex-1">
          <Field label="搜尋紀錄" htmlFor={`hd-log-search-${runId}`}>
            <input
              id={`hd-log-search-${runId}`}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="檔名、指令、關鍵字…"
              autoComplete="off"
              className="min-h-[36px] w-full rounded-md border border-edge bg-canvas px-2 text-sm text-ink placeholder:text-ink-faint focus:outline-none focus:ring-2 focus:ring-focus"
            />
          </Field>
        </div>
        <Button
          type="button"
          size="sm"
          variant="secondary"
          aria-pressed={expanded}
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded ? "只看摘要" : `展開完整紀錄 (${entries.length})`}
        </Button>
      </div>

      <div className="flex items-center gap-2 text-xs text-ink-dim">
        <span>{expanded ? "完整紀錄" : "摘要（備註、錯誤、結束與最後回覆）"}</span>
        <span>· {visible.length} 筆</span>
        {mode === "lines" ? <span>· 純文字紀錄</span> : null}
        {active ? (
          <span className="flex items-center gap-1">
            <span
              aria-hidden="true"
              className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-warn motion-reduce:animate-none"
            />
            即時更新中
          </span>
        ) : null}
      </div>

      {error ? (
        <div className="rounded-md border border-danger/40 bg-danger/10 p-2 font-mono text-xs text-danger">
          {error}
        </div>
      ) : null}

      <ol
        ref={listRef}
        aria-live={active ? "polite" : undefined}
        className="max-h-[50vh] min-h-[6rem] space-y-1.5 overflow-y-auto pr-0.5"
      >
        {visible.length === 0 ? (
          <li className="rounded-md border border-dashed border-edge-strong p-3 text-sm text-ink-dim">
            {entries.length === 0
              ? active
                ? "等待紀錄輸出…"
                : "此任務沒有紀錄。"
              : expanded
                ? "沒有符合篩選條件的紀錄。"
                : "摘要中沒有符合的項目；可展開完整紀錄。"}
          </li>
        ) : (
          visible.map((e) => <EntryRow key={e.i} e={e} />)
        )}
      </ol>
    </div>
  );
}
