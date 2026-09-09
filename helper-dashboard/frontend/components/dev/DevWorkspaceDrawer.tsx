"use client";

import React, { useCallback, useEffect, useState } from "react";

import { BigGuyPanel } from "@/components/devfix/BigGuyPanel";
import { JsonInspector, type DevTab } from "@/components/inspector/JsonInspector";
import { Button, Drawer, Field, Input, Tabs, fieldDescribedBy, useToast } from "@/components/ui";
import { api, describeApiError, isApiError } from "@/lib/api";
import type {
  Capabilities,
  DashboardSpec,
  DevfixConfig,
  PatchSpec,
  ReviewTrailEvent,
} from "@/lib/spec-schema";
import {
  runStatus,
  runStatusLabel,
  runStatusTone,
  type DevfixRunsState,
} from "@/lib/useDevfixRuns";

// Developer workspace: ONE right-side drawer hosting Big Guy and the old
// inspector tabs (規格 / 補丁 / 審查 / 評估). Rendered by the shell only when
// GET /api/capabilities says a developer token is configured; the content
// is locked behind the token until GET /api/devfix/config accepts it. The
// server re-checks the header on every developer route — this UI is
// convenience, not the boundary.

export type WorkspaceTab = "bigguy" | DevTab;

const TAB_IDS: WorkspaceTab[] = ["bigguy", "spec", "patch", "review", "evaluation"];

function isWorkspaceTab(id: string): id is WorkspaceTab {
  return (TAB_IDS as string[]).includes(id);
}

function TokenGate({
  onUnlocked,
}: {
  onUnlocked: (token: string, config: DevfixConfig) => void;
}) {
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const t = token.trim();
    if (!t || busy) return;
    setBusy(true);
    setError(null);
    try {
      const config = await api.devfixConfig(t);
      onUnlocked(t, config);
      setToken("");
    } catch (err) {
      if (isApiError(err) && err.status === 403) {
        setError("權杖不正確，或伺服器未設定 HELPER_DASHBOARD_DEV_TOKEN。");
      } else {
        setError(describeApiError(err));
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="space-y-3 p-4 text-sm">
      <p className="text-ink-dim">
        開發工作區包含 Big Guy 本機執行、規格 JSON、補丁、審查軌跡與瀏覽器評估。需要伺服器設定的開發者權杖才能使用；權杖只保存在此分頁的工作階段。
      </p>
      <Field
        label="開發者權杖"
        htmlFor="hd-dev-token"
        hint="對應伺服器環境變數 HELPER_DASHBOARD_DEV_TOKEN。"
        error={error ?? undefined}
        required
      >
        <Input
          id="hd-dev-token"
          type="password"
          autoComplete="off"
          aria-describedby={fieldDescribedBy("hd-dev-token", { hint: true, error })}
          value={token}
          onChange={(e) => setToken(e.target.value)}
          invalid={Boolean(error)}
        />
      </Field>
      <Button type="submit" variant="primary" size="md" loading={busy} disabled={!token.trim()}>
        解鎖
      </Button>
    </form>
  );
}

export function DevWorkspaceDrawer({
  open,
  onClose,
  capabilities,
  unlocked,
  onUnlocked,
  onLock,
  spec,
  lastPatch,
  warnings,
  reviewTrail,
  runs,
}: {
  open: boolean;
  onClose: () => void;
  capabilities: Capabilities | null;
  unlocked: boolean;
  // Called after GET /api/devfix/config accepted the token; the shell
  // stores it (sessionStorage) and flips DevModeContext.
  onUnlocked: (token: string) => void;
  onLock: () => void;
  spec: DashboardSpec | null;
  lastPatch: PatchSpec | null;
  warnings: string[];
  reviewTrail: ReviewTrailEvent[];
  runs: DevfixRunsState;
}) {
  const toast = useToast();
  const [tab, setTab] = useState<WorkspaceTab>("bigguy");
  const [maximized, setMaximized] = useState(false);
  const [config, setConfig] = useState<DevfixConfig | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const [configLoading, setConfigLoading] = useState(false);

  const loadConfig = useCallback(async () => {
    setConfigLoading(true);
    try {
      const c = await api.devfixConfig();
      setConfig(c);
      setConfigError(null);
    } catch (e) {
      if (isApiError(e) && e.status === 403) {
        // Token no longer accepted (server restarted with a new token, or
        // the header was rejected) — fall back to the gate.
        setConfig(null);
        setConfigError(null);
        toast.show({
          title: "開發者權杖已失效",
          description: "請重新輸入權杖。",
          tone: "warn",
          durationMs: 6000,
        });
        onLock();
        return;
      }
      setConfigError(e instanceof Error ? e.message : describeApiError(e));
    } finally {
      setConfigLoading(false);
    }
  }, [onLock, toast]);

  useEffect(() => {
    if (open && unlocked && !config) loadConfig();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, unlocked]);

  useEffect(() => {
    if (!unlocked) setConfig(null);
  }, [unlocked]);

  const st = runStatus(runs.current);

  const headerExtra = unlocked ? (
    <div className="flex items-center gap-1.5">
      <Button
        type="button"
        variant={tab === "bigguy" ? "primary" : "secondary"}
        size="sm"
        aria-pressed={tab === "bigguy"}
        onClick={() => setTab("bigguy")}
        iconLeft={
          <span
            aria-hidden="true"
            className={
              "inline-block h-2 w-2 rounded-full " +
              (runStatusTone(st) === "ok"
                ? "bg-ok"
                : runStatusTone(st) === "warn"
                  ? "bg-warn"
                  : runStatusTone(st) === "danger"
                    ? "bg-danger"
                    : runStatusTone(st) === "info"
                      ? "bg-info"
                      : "bg-ink-faint") +
              (st === "running" ? " animate-pulse motion-reduce:animate-none" : "")
            }
          />
        }
      >
        Big Guy · {runStatusLabel(st)}
      </Button>
      <Button type="button" variant="ghost" size="sm" onClick={onLock} title="清除此分頁的開發者權杖">
        鎖定
      </Button>
    </div>
  ) : null;

  return (
    <Drawer
      id="hd-dev-drawer"
      open={open}
      onClose={onClose}
      side="right"
      title="開發工作區"
      width={560}
      resizable
      maximized={maximized}
      onToggleMaximize={() => setMaximized((v) => !v)}
      headerExtra={headerExtra}
      bodyClassName="flex flex-col"
    >
      {!unlocked ? (
        <TokenGate
          onUnlocked={(token, c) => {
            setConfig(c);
            setConfigError(null);
            onUnlocked(token);
          }}
        />
      ) : (
        <Tabs
          className="flex min-h-0 flex-1 flex-col"
          panelClassName="min-h-0 flex-1 overflow-auto"
          tabs={[
            {
              id: "bigguy",
              label: "Big Guy",
              // Tabs wraps `badge` in its own Badge — pass text only.
              badge: st !== "idle" ? runStatusLabel(st) : undefined,
            },
            { id: "spec", label: "規格", badge: spec ? spec.widgets.length : undefined },
            { id: "patch", label: "補丁", badge: lastPatch ? lastPatch.operations.length : undefined },
            { id: "review", label: "審查", badge: warnings.length > 0 ? warnings.length : undefined },
            { id: "evaluation", label: "評估", disabled: capabilities ? !capabilities.evaluate_enabled : false },
          ]}
          value={tab}
          onChange={(id) => {
            if (isWorkspaceTab(id)) setTab(id);
          }}
        >
          {(id) =>
            id === "bigguy" ? (
              <BigGuyPanel
                capabilities={capabilities}
                runs={runs}
                config={config}
                configError={configError}
                configLoading={configLoading}
                onRefreshConfig={loadConfig}
              />
            ) : (
              <JsonInspector
                tab={id as DevTab}
                spec={spec}
                lastPatch={lastPatch}
                warnings={warnings}
                reviewTrail={reviewTrail}
              />
            )
          }
        </Tabs>
      )}
    </Drawer>
  );
}
