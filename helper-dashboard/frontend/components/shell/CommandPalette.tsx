"use client";

import React from "react";
import {
  KBarAnimator,
  KBarPortal,
  KBarPositioner,
  KBarProvider,
  KBarResults,
  KBarSearch,
  useMatches,
  useRegisterActions,
  type Action,
} from "kbar";

import { focusTimeRangeControls } from "@/components/dashboard/TimeRangeControls";
import type { SavedLibraryApi } from "@/components/dashboard/DashboardPickerMenu";
import { THEME_LABELS, useTheme } from "@/lib/theme";

// Ctrl/Cmd+K command palette (kbar — the same library Grafana itself uses).
// kbar ships unstyled by design: every visual piece below is our own token
// classes, nothing overridden from a library default.
//
// Mounted inside app/page.tsx's HomeShell (NOT app/layout.tsx): the saved-
// dashboard library state (`library` below) already lives there as a plain
// prop, so registering actions here reuses that exact same fetch instead of
// the palette firing its own duplicate GET /api/saved_dashboards. The
// headless /dashboard/[id] evaluator route has no keyboard user, so scoping
// the palette to the "/" page tree loses nothing.

export function CommandPaletteProvider({
  library,
  hasSpec,
  onOpenTemplates,
  onOpenDataSource,
  onOpenHelper,
  dev,
  children,
}: {
  library: SavedLibraryApi;
  hasSpec: boolean;
  onOpenTemplates: () => void;
  onOpenDataSource: () => void;
  onOpenHelper: () => void;
  dev: { visible: boolean; onOpen: () => void };
  children: React.ReactNode;
}) {
  return (
    <KBarProvider actions={[]}>
      <CommandPaletteActions
        library={library}
        hasSpec={hasSpec}
        onOpenTemplates={onOpenTemplates}
        onOpenDataSource={onOpenDataSource}
        onOpenHelper={onOpenHelper}
        dev={dev}
      />
      <CommandPaletteRenderer />
      {children}
    </KBarProvider>
  );
}

function CommandPaletteActions({
  library,
  hasSpec,
  onOpenTemplates,
  onOpenDataSource,
  onOpenHelper,
  dev,
}: {
  library: SavedLibraryApi;
  hasSpec: boolean;
  onOpenTemplates: () => void;
  onOpenDataSource: () => void;
  onOpenHelper: () => void;
  dev: { visible: boolean; onOpen: () => void };
}) {
  const theme = useTheme();

  const staticActions: Action[] = [
    {
      id: "hd-theme-light",
      name: `切換主題：${THEME_LABELS.light}`,
      section: "外觀",
      keywords: "theme light 淺色",
      perform: () => theme.setTheme("light"),
    },
    {
      id: "hd-theme-dark",
      name: `切換主題：${THEME_LABELS.dark}`,
      section: "外觀",
      keywords: "theme dark 深色",
      perform: () => theme.setTheme("dark"),
    },
    {
      id: "hd-theme-system",
      name: `切換主題：${THEME_LABELS.system}`,
      section: "外觀",
      keywords: "theme system 跟隨系統",
      perform: () => theme.setTheme("system"),
    },
    {
      id: "hd-open-templates",
      name: "從範本建立儀表板",
      section: "儀表板",
      keywords: "template 範本",
      perform: onOpenTemplates,
    },
    {
      id: "hd-open-datasource",
      name: "檢查資料來源狀態",
      section: "儀表板",
      keywords: "datasource prometheus 資料來源",
      perform: onOpenDataSource,
    },
    {
      id: "hd-open-helper",
      name: "詢問 Helper",
      section: "儀表板",
      keywords: "chat helper 對話",
      perform: onOpenHelper,
    },
  ];

  if (hasSpec) {
    staticActions.push({
      id: "hd-focus-timerange",
      name: "聚焦時間範圍控制",
      section: "儀表板",
      keywords: "time range 時間範圍",
      perform: () => focusTimeRangeControls(),
    });
  }

  // Server-side dev gate is the only real boundary (X-Developer-Token,
  // enforced per-request on every /api/devfix|developer|evaluate route) —
  // this visibility check is pure UI convenience, same rule TopBar already
  // follows for its own 開發者 button.
  if (dev.visible) {
    staticActions.push({
      id: "hd-open-dev",
      name: "開啟開發者面板（Big Guy）",
      section: "開發者",
      keywords: "developer dev big guy 開發者",
      perform: dev.onOpen,
    });
  }

  useRegisterActions(staticActions, [
    hasSpec,
    dev.visible,
    onOpenTemplates,
    onOpenDataSource,
    onOpenHelper,
    dev.onOpen,
  ]);

  const recentActions: Action[] = (library.entries ?? []).slice(0, 8).map((e) => ({
    id: `hd-open-saved-${e.saved_id}`,
    name: `開啟已儲存：${e.name}`,
    section: "已儲存的儀表板",
    keywords: `saved ${e.name} ${(e.tags ?? []).join(" ")}`,
    perform: () => {
      void library.open(e.saved_id);
    },
  }));

  useRegisterActions(recentActions, [library.entries, library.open]);

  return null;
}

function CommandPaletteRenderer() {
  return (
    <KBarPortal>
      <KBarPositioner className="z-50 bg-canvas/60 backdrop-blur-sm" style={{ paddingTop: "15vh" }}>
        <KBarAnimator className="w-full max-w-lg overflow-hidden rounded-lg border border-edge bg-surface shadow-pop">
          <KBarSearch
            defaultPlaceholder="輸入指令或搜尋…（Esc 關閉）"
            className="w-full border-b border-edge bg-transparent px-4 py-3 text-sm text-ink outline-none placeholder:text-ink-faint"
          />
          <RenderResults />
        </KBarAnimator>
      </KBarPositioner>
    </KBarPortal>
  );
}

function RenderResults() {
  const { results } = useMatches();
  if (results.length === 0) {
    return <div className="px-4 py-6 text-center text-sm text-ink-faint">沒有符合的指令</div>;
  }
  return (
    <KBarResults
      items={results}
      maxHeight={360}
      onRender={({ item, active }) =>
        typeof item === "string" ? (
          <div className="px-4 pb-1 pt-3 text-xs font-medium text-ink-faint">{item}</div>
        ) : (
          <div
            className={
              "mx-1.5 flex min-h-9 cursor-pointer items-center justify-between rounded-md px-2.5 text-sm " +
              (active ? "bg-accent/10 text-accent" : "text-ink")
            }
          >
            <span>{item.name}</span>
            {item.shortcut?.length ? (
              <span className="font-mono text-xs text-ink-faint">{item.shortcut.join(" ")}</span>
            ) : null}
          </div>
        )
      }
    />
  );
}
