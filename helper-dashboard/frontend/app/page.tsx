"use client";

import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { ChatPanel, type ChatMessage } from "@/components/chat/ChatPanel";
import { HelperDrawer } from "@/components/chat/HelperDrawer";
import { DashboardHeader } from "@/components/dashboard/DashboardHeader";
import {
  SavedDashboardList,
  type SavedLibraryApi,
} from "@/components/dashboard/DashboardPickerMenu";
import { DashboardPreview } from "@/components/dashboard/DashboardPreview";
import { DataSourcePanel } from "@/components/dashboard/DataSourcePanel";
import { useDatasourceStatus } from "@/components/dashboard/DataStatusStrip";
import { usePatientFilter } from "@/components/dashboard/PatientSelector";
import {
  computeSpecHash,
  saveStateFor,
  type SavedOrigin,
} from "@/components/dashboard/SaveControls";
import { SimulationBanner } from "@/components/dashboard/SimulationBanner";
import { TemplatePicker } from "@/components/dashboard/TemplatePicker";
import { TOPBAR_HEIGHT_PX, TopBar } from "@/components/dashboard/TopBar";
import { DevWorkspaceDrawer } from "@/components/dev/DevWorkspaceDrawer";
import {
  Button,
  Skeleton,
  ToastProvider,
  ToastViewport,
  useToast,
} from "@/components/ui";
import { api, describeApiError, isApiError } from "@/lib/api";
import { useDashboardDataStatus } from "@/lib/data-status";
import { DevModeContext } from "@/lib/dev-mode";
import { clearDevToken, getDevToken, setDevToken } from "@/lib/dev-session";
import type {
  Capabilities,
  ChatResponse,
  DashboardSpec,
  PatchSpec,
  ReviewTrailEvent,
  SavedDashboardEntry,
  SavedDashboardSummary,
} from "@/lib/spec-schema";
import { TimeRangeProvider } from "@/lib/time-range";
import { useBackendStatus, type BackendStatusState } from "@/lib/useBackendStatus";
import { runStatus, useDevfixRuns } from "@/lib/useDevfixRuns";

// 一般使用者介面 (`/`): dashboard-first shell.
//
//   TopBar
//   main: SimulationBanner · DashboardHeader · DashboardPreview   | HelperDrawer
//   DevWorkspaceDrawer (gated) · DataSourcePanel · TemplatePicker · toasts
//
// Nothing developer-only (JSON, PromQL, patches, run logs, paths) renders
// here unless the workspace is unlocked with the server-side token — and
// even then only inside the drawer / 「技術細節」 disclosures.

function newId(): string {
  return Math.random().toString(36).slice(2, 10);
}

// ---------------------------------------------------------------------
// 恢復上次開啟 (localStorage hd.lastOpened)
// ---------------------------------------------------------------------

const LAST_OPENED_KEY = "hd.lastOpened";

interface LastOpened {
  savedId?: string | null;
  dashboardId: string;
  name: string;
  ts: number;
  specSnapshot: DashboardSpec;
}

function readLastOpened(): LastOpened | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(LAST_OPENED_KEY);
    if (!raw) return null;
    const v = JSON.parse(raw) as Partial<LastOpened>;
    if (
      !v ||
      typeof v.dashboardId !== "string" ||
      typeof v.name !== "string" ||
      !v.specSnapshot ||
      typeof v.specSnapshot !== "object" ||
      !Array.isArray((v.specSnapshot as DashboardSpec).widgets)
    ) {
      return null;
    }
    return {
      savedId: typeof v.savedId === "string" ? v.savedId : null,
      dashboardId: v.dashboardId,
      name: v.name,
      ts: typeof v.ts === "number" ? v.ts : 0,
      specSnapshot: v.specSnapshot as DashboardSpec,
    };
  } catch {
    return null;
  }
}

function writeLastOpened(v: LastOpened): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(LAST_OPENED_KEY, JSON.stringify(v));
  } catch {
    // storage full/unavailable — restore simply won't be offered
  }
}

// ---------------------------------------------------------------------
// Chat helpers
// ---------------------------------------------------------------------

function devMeta(res: ChatResponse): string {
  let s = res.intent_type;
  if (res.runtime_used) s += ` · ${res.runtime_used}`;
  if (res.fallback_reason) s += ` (fallback: ${res.fallback_reason})`;
  return s;
}

function friendlyChatError(backend: BackendStatusState): string {
  if (backend.status === "down") {
    return "目前無法連線到伺服器，這則訊息尚未送出。系統會持續嘗試重新連線，請稍後再試。";
  }
  return "Helper 處理這則訊息時發生問題，請再試一次；若持續發生，請檢查資料來源狀態。";
}

// ---------------------------------------------------------------------
// Empty states
// ---------------------------------------------------------------------

function EmptyState({
  library,
  lastOpened,
  restoring,
  onRestore,
  onTemplates,
  onAskHelper,
  onDataSource,
}: {
  library: SavedLibraryApi;
  lastOpened: LastOpened | null;
  restoring: boolean;
  onRestore: () => void;
  onTemplates: () => void;
  onAskHelper: () => void;
  onDataSource: () => void;
}) {
  const entries = library.entries;

  if (entries === null && library.error) {
    // Unknown library state (first list request failed) — never claim
    // 「尚未儲存任何儀表板」; offer retry + whatever still works.
    return (
      <section
        aria-labelledby="hd-unknown-title"
        className="mx-auto w-full max-w-2xl space-y-4 p-4 sm:p-6"
      >
        <div>
          <h2 id="hd-unknown-title" className="text-lg font-semibold text-ink">
            無法讀取已儲存的儀表板
          </h2>
          <p className="text-sm text-ink-dim">{library.error}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button type="button" variant="primary" size="md" onClick={library.refresh}>
            重試
          </Button>
          {lastOpened ? (
            <Button type="button" variant="secondary" size="md" loading={restoring} onClick={onRestore}>
              恢復上次開啟：{lastOpened.name}
            </Button>
          ) : null}
          <Button type="button" variant="secondary" size="md" onClick={onDataSource}>
            資料來源狀態
          </Button>
          <Button type="button" variant="secondary" size="md" onClick={onAskHelper}>
            詢問 Helper
          </Button>
        </div>
      </section>
    );
  }

  if (entries === null) {
    return (
      <div className="mx-auto w-full max-w-2xl p-6" aria-busy="true" aria-label="載入中">
        <Skeleton className="mb-3 h-7 w-1/2 rounded" />
        <Skeleton className="mb-2 h-12 w-full rounded-md" />
        <Skeleton className="mb-2 h-12 w-full rounded-md" />
        <Skeleton className="h-12 w-full rounded-md" />
      </div>
    );
  }

  if (entries.length > 0) {
    return (
      <section
        aria-labelledby="hd-empty-title"
        className="mx-auto w-full max-w-2xl space-y-4 p-4 sm:p-6"
      >
        <div>
          <h2 id="hd-empty-title" className="text-lg font-semibold text-ink">
            選擇一個儀表板
          </h2>
          <p className="text-sm text-ink-dim">
            開啟已儲存的儀表板、恢復上次開啟的畫面，或請 Helper 建立新的。
          </p>
        </div>
        {lastOpened ? (
          <Button
            type="button"
            variant="primary"
            size="md"
            loading={restoring}
            onClick={onRestore}
            className="w-full justify-start sm:w-auto"
          >
            恢復上次開啟：{lastOpened.name}
          </Button>
        ) : null}
        <div className="rounded-lg border border-edge bg-surface p-3">
          <SavedDashboardList library={library} />
        </div>
        <div className="flex flex-wrap gap-2">
          <Button type="button" variant="secondary" size="md" onClick={onTemplates}>
            從範本建立
          </Button>
          <Button type="button" variant="secondary" size="md" onClick={onAskHelper}>
            詢問 Helper
          </Button>
        </div>
      </section>
    );
  }

  // 真正沒有任何儀表板 -> 首次建立引導
  return (
    <section
      aria-labelledby="hd-first-title"
      className="mx-auto w-full max-w-2xl space-y-4 p-4 sm:p-6"
    >
      <div>
        <h2 id="hd-first-title" className="text-lg font-semibold text-ink">
          建立第一個儀表板
        </h2>
        <p className="text-sm text-ink-dim">
          尚未儲存任何儀表板。可以從範本開始，或用一句話告訴 Helper 你想監測什麼。
        </p>
      </div>
      {library.error ? (
        <div className="flex flex-wrap items-center gap-2 rounded-md border border-warn/40 bg-warn/10 p-2 text-sm text-ink">
          <span className="min-w-0 flex-1">無法讀取已儲存清單：{library.error}</span>
          <Button type="button" variant="secondary" size="sm" onClick={library.refresh}>
            重試
          </Button>
        </div>
      ) : null}
      <div className="grid gap-3 sm:grid-cols-3">
        <button
          type="button"
          onClick={onTemplates}
          className="flex min-h-[7rem] flex-col items-start gap-1 rounded-lg border border-edge bg-surface p-4 text-left hover:border-accent/60 hover:bg-surface-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
        >
          <span className="text-sm font-semibold text-ink">從範本建立</span>
          <span className="text-sm text-ink-dim">心率與血氧、腦部血氧 rSO2、生命徵象總覽。</span>
        </button>
        <button
          type="button"
          onClick={onAskHelper}
          className="flex min-h-[7rem] flex-col items-start gap-1 rounded-lg border border-edge bg-surface p-4 text-left hover:border-accent/60 hover:bg-surface-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
        >
          <span className="text-sm font-semibold text-ink">詢問 Helper</span>
          <span className="text-sm text-ink-dim">例如「幫我建立新生兒 HR 與 SpO2 監控儀表板」。</span>
        </button>
        <button
          type="button"
          onClick={onDataSource}
          className="flex min-h-[7rem] flex-col items-start gap-1 rounded-lg border border-edge bg-surface p-4 text-left hover:border-accent/60 hover:bg-surface-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
        >
          <span className="text-sm font-semibold text-ink">資料來源狀態</span>
          <span className="text-sm text-ink-dim">確認伺服器、Prometheus 與量測輸出端是否連線。</span>
        </button>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------
// Shell
// ---------------------------------------------------------------------

function HomeShell() {
  const toast = useToast();

  // Chat / spec state
  const [sessionId] = useState<string>(() => `s-${newId()}`);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [spec, setSpec] = useState<DashboardSpec | null>(null);
  const specRef = useRef<DashboardSpec | null>(null);
  specRef.current = spec;
  const [origin, setOrigin] = useState<SavedOrigin | null>(null);
  const originRef = useRef<SavedOrigin | null>(null);
  originRef.current = origin;
  const [lastPatch, setLastPatch] = useState<PatchSpec | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [reviewTrail, setReviewTrail] = useState<ReviewTrailEvent[]>([]);
  const [busy, setBusy] = useState(false);
  const [busyMessage, setBusyMessage] = useState<string | null>(null);

  // Surfaces
  const [helperOpen, setHelperOpen] = useState(false);
  const [devOpen, setDevOpen] = useState(false);
  const [templatesOpen, setTemplatesOpen] = useState(false);
  const [dataSourceOpen, setDataSourceOpen] = useState(false);
  const [showAllWidgets, setShowAllWidgets] = useState(false);

  // Capabilities + developer session
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [devUnlocked, setDevUnlocked] = useState(false);

  // Library
  const [entries, setEntries] = useState<SavedDashboardSummary[] | null>(null);
  const [libraryError, setLibraryError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [lastOpened, setLastOpened] = useState<LastOpened | null>(null);
  const [restoring, setRestoring] = useState(false);

  const backend = useBackendStatus();
  const datasource = useDatasourceStatus(30000);
  const dataStatus = useDashboardDataStatus();
  const patient = usePatientFilter(spec?.dashboard_id ?? null);
  const devRuns = useDevfixRuns(devUnlocked);
  const timeRangeRef = useRef<HTMLDivElement>(null);

  const currentDashboardId = spec?.dashboard_id ?? null;

  // ---- mount: CSS var for drawers, last-opened, capabilities, dev token
  useEffect(() => {
    document.documentElement.style.setProperty("--hd-topbar-h", `${TOPBAR_HEIGHT_PX}px`);
    setLastOpened(readLastOpened());
    let cancelled = false;
    api
      .capabilities()
      .then((c) => {
        if (!cancelled) setCapabilities(c);
      })
      .catch(() => {
        if (!cancelled) {
          setCapabilities({
            dev_token_configured: false,
            devfix_local_runs: false,
            auto_fix_enabled: false,
            simulation_available: false,
            evaluate_enabled: false,
          });
        }
      });
    const token = getDevToken();
    if (token) {
      api
        .devfixConfig(token)
        .then(() => {
          if (!cancelled) setDevUnlocked(true);
        })
        .catch((e: unknown) => {
          // Only a definite 403 invalidates the stored token; a network
          // blip at load keeps it for the next unlock attempt.
          if (!cancelled && isApiError(e) && e.status === 403) clearDevToken();
        });
    }
    return () => {
      cancelled = true;
    };
  }, []);

  // ---- remember what is on screen
  useEffect(() => {
    if (!spec) return;
    const v: LastOpened = {
      savedId: origin?.savedId ?? null,
      dashboardId: spec.dashboard_id,
      name: origin?.name ?? spec.title,
      ts: Date.now(),
      specSnapshot: spec,
    };
    writeLastOpened(v);
    setLastOpened(v);
  }, [spec, origin?.savedId, origin?.name]);

  // ---- library
  const refreshLibrary = useCallback(() => {
    api
      .listSaved()
      .then((r) => {
        setEntries(r.saved);
        setLibraryError(null);
      })
      .catch((e: unknown) => {
        // Keep `entries` as-is (null = unknown) so a failed first load is
        // never presented as 「尚未儲存任何儀表板」.
        setLibraryError(describeApiError(e));
      });
  }, []);

  useEffect(() => {
    refreshLibrary();
  }, [refreshLibrary]);

  const adoptSpec = useCallback((dashboard: DashboardSpec, from: SavedOrigin | null) => {
    setSpec(dashboard);
    setOrigin(from);
    setShowAllWidgets(false);
  }, []);

  const openSaved = useCallback(
    async (savedId: string) => {
      setBusyId(savedId);
      try {
        const res = await api.loadSaved(savedId);
        const summary = entries?.find((e) => e.saved_id === savedId);
        adoptSpec(res.dashboard, {
          savedId,
          name: res.name,
          description: summary?.description ?? "",
          tags: summary?.tags ?? [],
          hash: computeSpecHash(res.dashboard) ?? "",
        });
        refreshLibrary(); // used_count bumped server-side
      } catch (e) {
        toast.show({
          title: "無法開啟儀表板",
          description: describeApiError(e),
          tone: "danger",
          durationMs: 6000,
        });
        throw e;
      } finally {
        setBusyId(null);
      }
    },
    [adoptSpec, entries, refreshLibrary, toast],
  );

  const renameSaved = useCallback(
    async (savedId: string, name: string) => {
      setBusyId(savedId);
      try {
        const r = await api.patchSaved(savedId, { name });
        setOrigin((o) => (o && o.savedId === savedId ? { ...o, name: r.saved.name } : o));
        refreshLibrary();
        toast.show({ title: "已重新命名", description: r.saved.name, tone: "ok", durationMs: 3000 });
      } catch (e) {
        toast.show({
          title: "重新命名失敗",
          description: describeApiError(e),
          tone: "danger",
          durationMs: 6000,
        });
        throw e;
      } finally {
        setBusyId(null);
      }
    },
    [refreshLibrary, toast],
  );

  const duplicateSaved = useCallback(
    async (savedId: string) => {
      setBusyId(savedId);
      try {
        const r = await api.duplicateSaved(savedId);
        refreshLibrary();
        toast.show({
          title: "已建立副本",
          description: `「${r.saved.name}」已加入清單`,
          tone: "ok",
          durationMs: 4000,
        });
      } catch (e) {
        toast.show({
          title: "複製失敗",
          description: describeApiError(e),
          tone: "danger",
          durationMs: 6000,
        });
      } finally {
        setBusyId(null);
      }
    },
    [refreshLibrary, toast],
  );

  const removeSaved = useCallback(
    async (savedId: string) => {
      const name = entries?.find((e) => e.saved_id === savedId)?.name ?? savedId;
      setBusyId(savedId);
      try {
        await api.deleteSaved(savedId);
        // Optimistic: drop the row now, refetch to confirm. If the deleted
        // entry is the one on screen, the chip drops to 尚未儲存 — and
        // 復原 puts that origin back (unless another dashboard was opened
        // in between).
        const prevOrigin =
          originRef.current && originRef.current.savedId === savedId ? originRef.current : null;
        const prevDashboardId = specRef.current?.dashboard_id ?? null;
        setEntries((prev) => (prev ? prev.filter((e) => e.saved_id !== savedId) : prev));
        setOrigin((o) => (o && o.savedId === savedId ? null : o));
        refreshLibrary();
        toast.show({
          title: `已刪除「${name}」`,
          description: "10 秒內可復原。",
          tone: "neutral",
          durationMs: 10000,
          action: {
            label: "復原",
            onClick: () => {
              api
                .restoreSaved(savedId)
                .then(() => {
                  refreshLibrary();
                  if (prevOrigin && specRef.current?.dashboard_id === prevDashboardId) {
                    // Same dashboard still on screen -> it is that entry again
                    // (the hash comparison decides 已儲存 vs 有未儲存變更).
                    setOrigin((o) => (o === null ? prevOrigin : o));
                  }
                  toast.show({ title: `已復原「${name}」`, tone: "ok", durationMs: 3000 });
                })
                .catch((e: unknown) => {
                  toast.show({
                    title: "復原失敗",
                    description: describeApiError(e),
                    tone: "danger",
                    durationMs: 6000,
                  });
                });
            },
          },
        });
      } catch (e) {
        toast.show({
          title: "刪除失敗",
          description: describeApiError(e),
          tone: "danger",
          durationMs: 6000,
        });
      } finally {
        setBusyId(null);
      }
    },
    [entries, refreshLibrary, toast],
  );

  const library = useMemo<SavedLibraryApi>(
    () => ({
      entries,
      error: libraryError,
      refresh: refreshLibrary,
      open: openSaved,
      rename: renameSaved,
      duplicate: duplicateSaved,
      remove: removeSaved,
      busyId,
      currentSavedId: origin?.savedId ?? null,
    }),
    [entries, libraryError, refreshLibrary, openSaved, renameSaved, duplicateSaved, removeSaved, busyId, origin?.savedId],
  );

  const handleSaved = useCallback(
    (entry: SavedDashboardEntry) => {
      setOrigin({
        savedId: entry.saved_id,
        name: entry.name,
        description: entry.description,
        tags: entry.tags,
        hash: computeSpecHash(specRef.current) ?? computeSpecHash(entry.spec) ?? "",
      });
      refreshLibrary();
    },
    [refreshLibrary],
  );

  const restoreLast = useCallback(async () => {
    if (!lastOpened) return;
    setRestoring(true);
    try {
      if (lastOpened.savedId) {
        try {
          // Direct call (not openSaved) so a missing entry does not fire
          // a 「無法開啟」 toast before we fall back to the snapshot.
          const res = await api.loadSaved(lastOpened.savedId);
          const summary = entries?.find((e) => e.saved_id === lastOpened.savedId);
          adoptSpec(res.dashboard, {
            savedId: lastOpened.savedId,
            name: res.name,
            description: summary?.description ?? "",
            tags: summary?.tags ?? [],
            hash: computeSpecHash(res.dashboard) ?? "",
          });
          refreshLibrary();
          return;
        } catch {
          toast.show({
            title: "已儲存的項目已不存在",
            description: "改以本機保存的畫面快照恢復；可再次「儲存」保留它。",
            tone: "info",
            durationMs: 6000,
          });
        }
      }
      adoptSpec(lastOpened.specSnapshot, null);
    } finally {
      setRestoring(false);
    }
  }, [adoptSpec, entries, lastOpened, refreshLibrary, toast]);

  // Chat-driven 儲存 (「已儲存為…」) or 開啟 (LoadSavedIntent): the
  // response carries `saved_id`; fetch the entry so the chip shows its
  // name and compares against the dashboard on screen.
  const linkSavedOrigin = useCallback(
    async (savedId: string, shown: DashboardSpec | null, mustMatchEntry: boolean) => {
      if (!shown) return;
      try {
        const { saved } = await api.getSaved(savedId);
        // The user may have opened something else while we fetched.
        if (specRef.current?.dashboard_id !== shown.dashboard_id) return;
        // Chat save: the backend saved the dashboard it last produced for
        // this session, which is only this one if the ids agree. (Loads
        // clone under a fresh id, so they skip this check.)
        if (mustMatchEntry && saved.spec?.dashboard_id !== shown.dashboard_id) return;
        setOrigin({
          savedId,
          name: saved.name,
          description: saved.description ?? "",
          tags: saved.tags ?? [],
          hash: computeSpecHash(shown) ?? "",
        });
      } catch {
        // The chip simply stays 尚未儲存 / 有未儲存變更.
      }
    },
    [],
  );

  // ---- chat
  const send = useCallback(
    async (text: string) => {
      setMessages((prev) => [...prev, { id: newId(), from: "user", text }]);
      setBusy(true);
      setBusyMessage("正在產生並審查儀表板…");
      try {
        const res = await api.sendMessage(
          {
            session_id: sessionId,
            message: text,
            current_dashboard_id: specRef.current?.dashboard_id ?? null,
          },
          {
            onProgress: (p) => {
              const msg = p.message || `處理中…（${p.status}）`;
              const pct =
                typeof p.pct === "number" && Number.isFinite(p.pct)
                  ? ` ${Math.max(0, Math.min(100, Math.round(p.pct)))}%`
                  : "";
              setBusyMessage(`${msg}${pct}`);
            },
          },
        );
        setMessages((prev) => [
          ...prev,
          {
            id: newId(),
            from: "helper",
            text: res.user_reply,
            meta: devMeta(res),
            questions: res.clarification_questions || undefined,
            savePromptFor: res.save_prompt_for,
          },
        ]);
        if (res.dashboard) {
          const prev = specRef.current;
          const same = Boolean(prev && prev.dashboard_id === res.dashboard.dashboard_id);
          setSpec(res.dashboard);
          // A brand-new dashboard is not the library entry we came from;
          // a patch to the same dashboard keeps the origin (chip -> 有未儲存變更).
          if (!same) {
            setOrigin(null);
            setShowAllWidgets(false);
          }
        }
        if (res.patch) setLastPatch(res.patch);
        setWarnings(res.warnings || []);
        setReviewTrail(res.review_trail || []);
        if (res.saved_id) {
          void linkSavedOrigin(
            res.saved_id,
            res.dashboard ?? specRef.current,
            res.dashboard === null,
          );
        }
        refreshLibrary(); // saves made through the chat flow appear in 開啟
      } catch (e) {
        console.error("[chat-ui] sendMessage failed", {
          error: e,
          message: e instanceof Error ? e.message : String(e),
          sessionId,
          currentDashboardId: specRef.current?.dashboard_id ?? null,
        });
        setMessages((prev) => [
          ...prev,
          {
            id: newId(),
            from: "helper",
            text: friendlyChatError(backend),
            meta: "error",
            detail: `${e instanceof Error ? e.message : String(e)}${
              backend.httpCode ? ` (health HTTP ${backend.httpCode})` : ""
            }`,
          },
        ]);
      } finally {
        setBusy(false);
        setBusyMessage(null);
      }
    },
    [sessionId, backend, refreshLibrary, linkSavedOrigin],
  );

  // ---- derived
  const saveState = saveStateFor(spec, origin);
  const currentName = spec ? origin?.name ?? spec.title : null;
  const simulationActive =
    Boolean(datasource.status?.simulation?.active) || dataStatus.anySimulated;
  const simulationScenario =
    datasource.status?.simulation?.scenario ?? dataStatus.simScenario ?? null;
  const devVisible = capabilities?.dev_token_configured === true;
  const devMode = useMemo(() => ({ unlocked: devUnlocked }), [devUnlocked]);

  const openHelper = useCallback(() => setHelperOpen(true), []);
  const closeHelper = useCallback(() => setHelperOpen(false), []);
  const lockDev = useCallback(() => {
    clearDevToken();
    setDevUnlocked(false);
  }, []);

  return (
    <DevModeContext.Provider value={devMode}>
      <div className="flex h-screen flex-col bg-canvas text-ink">
        <TopBar
          spec={spec}
          currentName={currentName}
          saveState={saveState}
          saveOrigin={origin}
          onSaved={handleSaved}
          library={library}
          onOpenTemplates={() => setTemplatesOpen(true)}
          onOpenDataSource={() => setDataSourceOpen(true)}
          helperOpen={helperOpen}
          onToggleHelper={() => setHelperOpen((v) => !v)}
          dev={{
            visible: devVisible,
            unlocked: devUnlocked,
            status: devUnlocked ? runStatus(devRuns.current) : "idle",
            open: devOpen,
            onOpen: () => setDevOpen(true),
          }}
        />

        <div className="flex min-h-0 flex-1">
          {spec ? (
            <TimeRangeProvider refreshInterval={spec.refresh_interval}>
              <main className="flex min-h-0 min-w-0 flex-1 flex-col" aria-label="儀表板">
                <SimulationBanner active={simulationActive} scenario={simulationScenario} />
                <DashboardHeader
                  spec={spec}
                  savedName={origin?.name ?? null}
                  patient={patient}
                  showAllWidgets={showAllWidgets}
                  onToggleShowAll={setShowAllWidgets}
                  backend={backend}
                  datasource={datasource}
                  capabilities={capabilities}
                  onOpenDataSource={() => setDataSourceOpen(true)}
                  timeRangeRef={timeRangeRef}
                />
                <div className="min-h-0 min-w-0 flex-1 overflow-auto">
                  <DashboardPreview
                    spec={spec}
                    reviewing={busy}
                    reviewTrail={reviewTrail}
                    zoneOrder
                    pairDuplicates={!showAllWidgets}
                    filters={patient.filters}
                  />
                </div>
              </main>
            </TimeRangeProvider>
          ) : (
            <main className="flex min-h-0 min-w-0 flex-1 flex-col overflow-y-auto" aria-label="儀表板">
              <SimulationBanner active={simulationActive} scenario={simulationScenario} />
              <EmptyState
                library={library}
                lastOpened={lastOpened}
                restoring={restoring}
                onRestore={restoreLast}
                onTemplates={() => setTemplatesOpen(true)}
                onAskHelper={openHelper}
                onDataSource={() => setDataSourceOpen(true)}
              />
            </main>
          )}

          <HelperDrawer open={helperOpen} onClose={closeHelper}>
            <ChatPanel
              messages={messages}
              onSend={send}
              onQuickReply={send}
              busy={busy}
              busyMessage={busyMessage}
              backend={backend}
              onClose={closeHelper}
            />
          </HelperDrawer>
        </div>

        {devVisible ? (
          <DevWorkspaceDrawer
            open={devOpen}
            onClose={() => setDevOpen(false)}
            capabilities={capabilities}
            unlocked={devUnlocked}
            onUnlocked={(token) => {
              setDevToken(token);
              setDevUnlocked(true);
            }}
            onLock={lockDev}
            spec={spec}
            lastPatch={lastPatch}
            warnings={warnings}
            reviewTrail={reviewTrail}
            runs={devRuns}
          />
        ) : null}

        <DataSourcePanel
          open={dataSourceOpen}
          onClose={() => setDataSourceOpen(false)}
          backend={backend}
          datasource={datasource}
          capabilities={capabilities}
        />

        <TemplatePicker
          open={templatesOpen}
          onClose={() => setTemplatesOpen(false)}
          onCreated={(dashboard) => adoptSpec(dashboard, null)}
        />

        <ToastViewport />
      </div>
    </DevModeContext.Provider>
  );
}

export default function HomePage() {
  return (
    <ToastProvider>
      <HomeShell />
    </ToastProvider>
  );
}
