"use client";

import React from "react";

import { Badge } from "@/components/ui";
import type { DevRunMeta } from "@/lib/spec-schema";

// 已讀取 / 已修改 / 已套用 — three separate lists, because "changed in the
// worktree" is not "copied back to the live tree", and neither means the
// change is live (backend restart / frontend rebuild still needed).

const APPLIED_LABEL: Record<string, { label: string; tone: "ok" | "neutral" | "danger" }> = {
  applied: { label: "已套用到專案", tone: "ok" },
  not_applied: { label: "未套用", tone: "neutral" },
  violation: { label: "違反允許清單，未套用", tone: "danger" },
  "n/a": { label: "不適用", tone: "neutral" },
};

function FileList({ title, files, empty }: { title: string; files: string[]; empty: string }) {
  return (
    <section className="rounded-md border border-edge bg-surface p-2">
      <h4 className="mb-1 flex items-center gap-2 text-sm font-medium text-ink">
        {title}
        <span className="text-xs text-ink-dim">{files.length}</span>
      </h4>
      {files.length === 0 ? (
        <div className="text-xs text-ink-dim">{empty}</div>
      ) : (
        <ul className="max-h-48 space-y-0.5 overflow-y-auto font-mono text-xs text-ink-dim">
          {files.map((f) => (
            <li key={f} className="break-all">
              {f}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export function RunFilesView({ meta }: { meta: DevRunMeta }) {
  const read = meta.files_read ?? [];
  const changed = meta.files_changed ?? [];
  const applied = meta.applied?.files ?? [];
  const appliedStatus = meta.applied?.status ?? "n/a";
  const a = APPLIED_LABEL[appliedStatus] ?? { label: appliedStatus, tone: "neutral" as const };
  // Auto-fix rolled the live tree back (meta.rollback).
  const rollback = meta.rollback;
  const rolledBack = (rollback?.restored?.length ?? 0) + (rollback?.deleted?.length ?? 0);

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2 text-xs text-ink-dim">
        <Badge tone={a.tone}>{a.label}</Badge>
        {rolledBack > 0 ? (
          <Badge tone="danger">已回復 {rolledBack} 個檔案</Badge>
        ) : null}
        <span>已套用 ≠ 已上線：後端需重啟、前端需重新建置後才會生效。</span>
      </div>
      <div className="grid gap-2 md:grid-cols-3">
        <FileList title="已讀取" files={read} empty="沒有記錄到讀取的檔案。" />
        <FileList title="已修改" files={changed} empty="工作目錄中沒有變更。" />
        <FileList title="已套用" files={applied} empty="沒有檔案被複製回專案。" />
      </div>
    </div>
  );
}
