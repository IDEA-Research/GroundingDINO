"use client";

import React from "react";

import { Badge, cx } from "@/components/ui";
import type { DevRunMeta } from "@/lib/spec-schema";

// 已讀取 → 已修改 → 已套用 — a three-stage pipeline, because "changed in the
// worktree" is not "copied back to the live tree", and neither means the
// change is live (backend restart / frontend rebuild still needed). The
// "套用" (applied) stage is the one that matters most, so its status drives
// the accent color of that column.

type AppliedTone = "ok" | "neutral" | "danger";

const APPLIED_LABEL: Record<string, { label: string; tone: AppliedTone }> = {
  applied: { label: "已套用到專案", tone: "ok" },
  not_applied: { label: "未套用", tone: "neutral" },
  violation: { label: "違反允許清單，未套用", tone: "danger" },
  "n/a": { label: "不適用", tone: "neutral" },
};

// Column accent follows the applied status — this is the one signal in the
// pipeline worth making loud ("改了不代表上線").
const STAGE_ACCENT: Record<AppliedTone, string> = {
  ok: "border-ok/40 bg-ok/5",
  neutral: "border-edge bg-surface",
  danger: "border-danger/40 bg-danger/5",
};

function FileList({
  title,
  files,
  empty,
  echo,
  accentClass,
}: {
  title: string;
  files: string[];
  empty: string;
  echo?: Set<string>;
  accentClass?: string;
}) {
  return (
    <section
      className={cx("rounded-md border p-2", accentClass ?? "border-edge bg-surface")}
    >
      <h4 className="mb-1 flex items-center gap-2 text-sm font-medium text-ink">
        {title}
        <span className="text-xs text-ink-dim">{files.length}</span>
      </h4>
      {files.length === 0 ? (
        <div className="text-xs text-ink-dim">{empty}</div>
      ) : (
        <ul className="max-h-48 space-y-0.5 overflow-y-auto font-mono text-xs text-ink-dim">
          {files.map((f) => (
            <li
              key={f}
              className={cx("break-all", echo?.has(f) ? "font-medium text-ink" : undefined)}
            >
              {f}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

// Direction cue between pipeline stages: "→" side-by-side on desktop,
// "↓" once columns stack on narrow screens.
function FlowArrow() {
  return (
    <div
      aria-hidden="true"
      className="flex items-center justify-center px-1 text-sm text-ink-faint md:px-0"
    >
      <span className="md:hidden">↓</span>
      <span className="hidden md:inline">→</span>
    </div>
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

  // Files that made it into both "修改" and "套用" get a visual echo so the
  // pipeline reads as one flow instead of three unrelated lists.
  const changedSet = new Set(changed);
  const appliedSet = new Set(applied);

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2 text-xs text-ink-dim">
        <Badge tone={a.tone}>{a.label}</Badge>
        {rolledBack > 0 ? (
          <Badge tone="danger">已回復 {rolledBack} 個檔案</Badge>
        ) : null}
        <span>已套用 ≠ 已上線：後端需重啟、前端需重新建置後才會生效。</span>
      </div>
      <div className="grid grid-cols-1 items-start gap-1 md:grid-cols-[1fr_auto_1fr_auto_1fr] md:gap-2">
        <FileList title="讀取" files={read} empty="沒有記錄到讀取的檔案。" />
        <FlowArrow />
        <FileList
          title="修改"
          files={changed}
          empty="工作目錄中沒有變更。"
          echo={appliedSet}
        />
        <FlowArrow />
        <FileList
          title="套用"
          files={applied}
          empty="沒有檔案被複製回專案。"
          echo={changedSet}
          accentClass={STAGE_ACCENT[a.tone]}
        />
      </div>
    </div>
  );
}
