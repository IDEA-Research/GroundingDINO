"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";

import { Badge, Button, Disclosure, Skeleton } from "@/components/ui";
import { api, describeApiError } from "@/lib/api";
import type { DevRunDiffFile, DevRunDiffResponse } from "@/lib/spec-schema";

// Unified diff from GET /runs/{id}/diff, split per file (collapsible),
// +/- colouring, monospace. Untracked files arrive as new-file diffs.

export interface FileDiff {
  path: string;
  lines: string[];
}

export function splitPatch(patch: string): FileDiff[] {
  const out: FileDiff[] = [];
  let cur: FileDiff | null = null;
  for (const line of patch.split("\n")) {
    if (line.startsWith("diff --git ")) {
      const m = /^diff --git a\/(.+?) b\/(.+)$/.exec(line);
      cur = { path: m ? m[2] : line.slice("diff --git ".length), lines: [] };
      out.push(cur);
      continue;
    }
    if (!cur) {
      if (!line.trim()) continue;
      cur = { path: "(patch)", lines: [] };
      out.push(cur);
    }
    cur.lines.push(line);
  }
  return out;
}

function lineClass(l: string): string {
  if (l.startsWith("+++") || l.startsWith("---")) return "text-ink-dim";
  if (l.startsWith("+")) return "bg-ok/10 text-ok";
  if (l.startsWith("-")) return "bg-danger/10 text-danger";
  if (l.startsWith("@@")) return "text-info";
  if (l.startsWith("index ") || l.startsWith("new file") || l.startsWith("deleted file")) {
    return "text-ink-faint";
  }
  return "text-ink-dim";
}

function statusTone(s: string): "ok" | "danger" | "info" | "neutral" {
  const v = s.toLowerCase();
  if (v.startsWith("a") || v === "new" || v === "added" || v === "untracked") return "ok";
  if (v.startsWith("d") || v === "deleted") return "danger";
  if (v.startsWith("m") || v === "modified") return "info";
  return "neutral";
}

// % floor so a small diff still shows a visible sliver against the widest
// file in the run, rather than collapsing to nothing.
const MIN_BAR_SHARE = 6;

// Compact stacked bar for one file's additions/deletions. Bar *length* is
// relative to the largest single-file change in this diff (maxChange), so
// files can be compared to each other at a glance; the fill is then split
// green/red by each file's own add:delete ratio. additions-only ("added")
// and deletions-only ("deleted") files degrade to a single solid segment
// rather than an empty or half-drawn bar.
function DiffChangeBar({
  additions,
  deletions,
  maxChange,
}: {
  additions: number;
  deletions: number;
  maxChange: number;
}) {
  const add = additions || 0;
  const del = deletions || 0;
  const total = add + del;
  if (total <= 0) return null;

  const widthPct = Math.max(MIN_BAR_SHARE, Math.round((total / maxChange) * 100));
  const addSharePct = (add / total) * 100;
  const delSharePct = 100 - addSharePct;

  return (
    <span
      className="inline-flex h-1.5 w-20 shrink-0 overflow-hidden rounded-full bg-surface-3"
      role="img"
      aria-label={`+${add} −${del}`}
    >
      <span className="flex h-full" style={{ width: `${widthPct}%` }}>
        {add > 0 ? (
          <span className="h-full bg-ok" style={{ width: `${addSharePct}%` }} />
        ) : null}
        {del > 0 ? (
          <span className="h-full bg-danger" style={{ width: `${delSharePct}%` }} />
        ) : null}
      </span>
    </span>
  );
}

export function RunDiffView({ runId, active }: { runId: string; active: boolean }) {
  const [data, setData] = useState<DevRunDiffResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const r = await api.devfixDiff(runId);
      setData(r);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : describeApiError(e));
    } finally {
      setLoading(false);
    }
  }, [runId]);

  // Load on mount and again when the run stops (final diff).
  useEffect(() => {
    load();
  }, [load, active]);

  const files = useMemo(() => (data ? splitPatch(data.patch) : []), [data]);
  const byPath = useMemo(() => {
    const m = new Map<string, DevRunDiffFile>();
    data?.files.forEach((f) => m.set(f.path, f));
    return m;
  }, [data]);

  const totalAdd = data?.files.reduce((n, f) => n + (f.additions || 0), 0) ?? 0;
  const totalDel = data?.files.reduce((n, f) => n + (f.deletions || 0), 0) ?? 0;
  // Widest single-file change in this diff; the per-file bar's length is
  // relative to this so files can be compared to each other at a glance.
  const maxChange = useMemo(() => {
    const max =
      data?.files.reduce((m, f) => Math.max(m, (f.additions || 0) + (f.deletions || 0)), 0) ?? 0;
    return max > 0 ? max : 1;
  }, [data]);

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2 text-xs text-ink-dim">
        {data ? (
          <>
            <span>{data.files.length} 個檔案</span>
            <span className="font-mono text-ok">+{totalAdd}</span>
            <span className="font-mono text-danger">−{totalDel}</span>
          </>
        ) : null}
        {active ? <span>· 任務仍在進行，差異可能持續變動</span> : null}
        <Button type="button" size="sm" variant="ghost" loading={loading} onClick={load}>
          重新載入
        </Button>
      </div>

      {error ? (
        <div className="rounded-md border border-danger/40 bg-danger/10 p-2 font-mono text-xs text-danger">
          {error}
        </div>
      ) : null}

      {!data && loading ? (
        <div className="space-y-2">
          <Skeleton className="h-6 w-2/3 rounded" />
          <Skeleton className="h-24 w-full rounded" />
        </div>
      ) : null}

      {data && files.length === 0 ? (
        <div className="rounded-md border border-dashed border-edge-strong p-3 text-sm text-ink-dim">
          沒有程式差異。
        </div>
      ) : null}

      <div className="space-y-2">
        {files.map((f, idx) => {
          const info = byPath.get(f.path);
          return (
            <div key={`${f.path}-${idx}`} className="rounded-md border border-edge bg-surface">
              <Disclosure
                defaultOpen={files.length <= 3}
                summary={
                  <span className="flex min-w-0 flex-wrap items-center gap-2 font-mono text-xs">
                    <span className="min-w-0 break-all text-ink">{f.path}</span>
                    {info ? (
                      <>
                        <Badge tone={statusTone(info.status)}>{info.status}</Badge>
                        <span className="text-ok">+{info.additions}</span>
                        <span className="text-danger">−{info.deletions}</span>
                        <DiffChangeBar
                          additions={info.additions}
                          deletions={info.deletions}
                          maxChange={maxChange}
                        />
                      </>
                    ) : null}
                  </span>
                }
              >
                <pre className="max-h-96 overflow-auto border-t border-edge bg-canvas p-2 font-mono text-xs leading-relaxed">
                  {f.lines.map((l, i) => (
                    <span key={i} className={`block whitespace-pre ${lineClass(l)}`}>
                      {l || " "}
                    </span>
                  ))}
                </pre>
              </Disclosure>
            </div>
          );
        })}
      </div>
    </div>
  );
}
