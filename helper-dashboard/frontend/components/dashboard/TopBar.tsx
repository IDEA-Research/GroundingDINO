"use client";

import React, { useEffect, useRef } from "react";

import { Button, Menu } from "@/components/ui";
import type { DashboardSpec, SavedDashboardEntry } from "@/lib/spec-schema";
import { THEME_LABELS, useTheme, type ThemePreference } from "@/lib/theme";
import {
  runStatusLabel,
  runStatusTone,
  type RunStatusOrIdle,
} from "@/lib/useDevfixRuns";

import { DashboardPickerMenu, type SavedLibraryApi } from "./DashboardPickerMenu";
import { SaveButton, SaveStateChip, type SaveState, type SavedOrigin } from "./SaveControls";

// Product top bar: 「Helper Dashboard」, current dashboard + save-state chip,
// 開啟 (library / 從範本建立 / 資料來源狀態), 儲存, 詢問 Helper, theme toggle,
// and — only when the server says a developer token is configured — the
// status-bearing 「開發者」 button (dot + CURRENT run status; history never
// colors it).
//
// The bar may wrap onto two rows on narrow screens; its rendered height is
// published as --hd-topbar-h on <html> so every drawer/sheet/popover sits
// below it (never covers it). No overflow clipping here: the 開啟/儲存/主題
// popovers are positioned children of this bar.

export const TOPBAR_HEIGHT_PX = 56;
export const TOPBAR_HEIGHT_VAR = "--hd-topbar-h";

export interface DevButtonState {
  // capabilities.dev_token_configured
  visible: boolean;
  unlocked: boolean;
  status: RunStatusOrIdle;
  open: boolean;
  onOpen: () => void;
}

const DOT: Record<ReturnType<typeof runStatusTone>, string> = {
  ok: "bg-ok",
  warn: "bg-warn",
  danger: "bg-danger",
  info: "bg-info",
  neutral: "bg-ink-faint",
};

function ThemeToggle() {
  const theme = useTheme();
  const prefs: ThemePreference[] = ["light", "dark", "system"];
  return (
    <Menu
      align="end"
      label="主題"
      trigger={
        <Button type="button" variant="ghost" size="sm" className="whitespace-nowrap">
          主題：{theme.label}
        </Button>
      }
      items={prefs.map((p) => ({
        label: THEME_LABELS[p],
        description: p === theme.theme ? "目前使用" : undefined,
        onSelect: () => theme.setTheme(p),
      }))}
    />
  );
}

export function TopBar({
  spec,
  currentName,
  saveState,
  saveOrigin,
  onSaved,
  library,
  onOpenTemplates,
  onOpenDataSource,
  helperOpen,
  onToggleHelper,
  dev,
}: {
  spec: DashboardSpec | null;
  currentName: string | null;
  saveState: SaveState | null;
  saveOrigin: SavedOrigin | null;
  onSaved: (entry: SavedDashboardEntry) => void;
  library: SavedLibraryApi;
  onOpenTemplates: () => void;
  onOpenDataSource: () => void;
  helperOpen: boolean;
  onToggleHelper: () => void;
  dev: DevButtonState;
}) {
  const ref = useRef<HTMLElement>(null);
  const devTone = runStatusTone(dev.status);
  const devLabel = dev.unlocked ? `開發者 · ${runStatusLabel(dev.status)}` : "開發者";

  // Publish the real height (the bar can wrap on narrow screens).
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const apply = () => {
      const h = Math.ceil(el.getBoundingClientRect().height) || TOPBAR_HEIGHT_PX;
      document.documentElement.style.setProperty(TOPBAR_HEIGHT_VAR, `${h}px`);
    };
    apply();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(apply);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  return (
    <header
      ref={ref}
      role="banner"
      className="relative z-50 flex shrink-0 flex-wrap items-center gap-x-2 gap-y-1 border-b border-edge bg-surface px-3 py-1.5"
      style={{ minHeight: TOPBAR_HEIGHT_PX }}
    >
      <div className="flex min-w-0 items-center gap-2">
        <span className="whitespace-nowrap text-base font-semibold text-ink">
          Helper Dashboard
        </span>
        <span aria-hidden="true" className="hidden text-ink-faint sm:inline">
          /
        </span>
        <span
          className="hidden min-w-0 max-w-[18rem] truncate text-sm text-ink-dim sm:inline"
          title={currentName ?? undefined}
        >
          {currentName ?? "尚未開啟儀表板"}
        </span>
        <SaveStateChip state={saveState} />
      </div>

      <div className="ml-auto flex shrink-0 flex-wrap items-center gap-1.5">
        <DashboardPickerMenu
          library={library}
          onOpenTemplates={onOpenTemplates}
          onOpenDataSource={onOpenDataSource}
        />
        <SaveButton spec={spec} origin={saveOrigin} onSaved={onSaved} />
        <Button
          type="button"
          variant={helperOpen ? "secondary" : "primary"}
          size="sm"
          onClick={onToggleHelper}
          aria-expanded={helperOpen}
          aria-controls="hd-helper-drawer"
        >
          {helperOpen ? "收合 Helper" : "詢問 Helper"}
        </Button>
        <ThemeToggle />
        {dev.visible ? (
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={dev.onOpen}
            aria-expanded={dev.open}
            aria-controls="hd-dev-drawer"
            title={dev.unlocked ? `Big Guy 目前狀態：${runStatusLabel(dev.status)}` : "開啟開發工作區"}
            data-dev-status={dev.unlocked ? dev.status : "locked"}
          >
            <span
              aria-hidden="true"
              className={
                `mr-1.5 inline-block h-2 w-2 rounded-full ${DOT[devTone]}` +
                (dev.status === "running" ? " animate-pulse motion-reduce:animate-none" : "")
              }
            />
            {devLabel}
          </Button>
        ) : null}
      </div>
    </header>
  );
}
