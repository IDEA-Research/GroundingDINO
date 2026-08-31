"use client";

import React, {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useState,
} from "react";

import { api } from "@/lib/api";
import type { DashboardSpec, SavedDashboardSummary } from "@/lib/spec-schema";

// Saved-dashboards library strip. Clicking a name goes through
// POST /api/saved_dashboards/{id}/load (contract C3): the backend hands
// back a persisted CLONE with a fresh dashboard_id, which is what gets
// rendered — chat patches then target the clone, never the snapshot.
// Delete is a two-step inline confirm. The parent can force a refetch
// via the imperative handle (e.g. after every chat response, so saves
// made through the chat flow appear).

export interface SavedDashboardsBarHandle {
  refresh: () => void;
}

export const SavedDashboardsBar = forwardRef<
  SavedDashboardsBarHandle,
  { onLoad: (dashboard: DashboardSpec) => void }
>(function SavedDashboardsBar({ onLoad }, ref) {
  const [entries, setEntries] = useState<SavedDashboardSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirmingId, setConfirmingId] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  const refresh = useCallback(() => {
    api
      .listSaved()
      .then((r) => {
        setEntries(r.saved);
        setError(null);
      })
      .catch((e: unknown) => {
        setError(e instanceof Error ? e.message : String(e));
      });
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useImperativeHandle(ref, () => ({ refresh }), [refresh]);

  const load = useCallback(
    async (savedId: string) => {
      setBusyId(savedId);
      setError(null);
      try {
        const res = await api.loadSaved(savedId);
        onLoad(res.dashboard);
        refresh(); // used_count was bumped server-side
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusyId(null);
      }
    },
    [onLoad, refresh],
  );

  const remove = useCallback(
    async (savedId: string) => {
      setBusyId(savedId);
      setError(null);
      try {
        await api.deleteSaved(savedId);
        setConfirmingId(null);
        refresh();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusyId(null);
      }
    },
    [refresh],
  );

  return (
    <div className="flex flex-wrap items-center gap-1.5 text-xs">
      <span className="text-2xs uppercase tracking-wide text-ink-faint">
        已存儀表板
      </span>

      {error ? (
        <span className="flex items-center gap-1 text-xs text-alarm-high">
          <span className="max-w-[16rem] truncate" title={error}>
            {error}
          </span>
          <button
            type="button"
            onClick={refresh}
            className="rounded border border-edge-strong bg-surface-3 px-1.5 py-0.5 text-xs text-ink hover:bg-surface-2"
          >
            重試
          </button>
        </span>
      ) : entries === null ? (
        <span className="text-xs text-ink-faint">載入中…</span>
      ) : entries.length === 0 ? (
        <span className="text-xs text-ink-faint">
          尚未儲存任何儀表板
        </span>
      ) : (
        entries.map((s) => (
          <span
            key={s.saved_id}
            className="flex items-center gap-1 rounded border border-edge bg-surface-2 px-1.5 py-0.5"
          >
            <button
              type="button"
              disabled={busyId === s.saved_id}
              onClick={() => load(s.saved_id)}
              title={s.description || s.name}
              className="font-medium text-ink hover:text-vital-spo2 disabled:opacity-50"
            >
              {s.name}
            </button>
            {s.tags.map((t) => (
              <span
                key={t}
                className="rounded bg-surface-3 px-1 py-0.5 text-2xs text-ink-dim"
              >
                {t}
              </span>
            ))}
            {confirmingId === s.saved_id ? (
              <span className="flex items-center gap-1">
                <button
                  type="button"
                  disabled={busyId === s.saved_id}
                  onClick={() => remove(s.saved_id)}
                  className="rounded bg-alarm-high/20 px-1 py-0.5 text-2xs font-semibold text-alarm-high hover:bg-alarm-high/30 disabled:opacity-50"
                >
                  確定刪除?
                </button>
                <button
                  type="button"
                  onClick={() => setConfirmingId(null)}
                  className="rounded bg-surface-3 px-1 py-0.5 text-2xs text-ink-dim hover:bg-surface-2 hover:text-ink"
                >
                  取消
                </button>
              </span>
            ) : (
              <button
                type="button"
                aria-label={`刪除 ${s.name}`}
                onClick={() => setConfirmingId(s.saved_id)}
                className="px-0.5 text-2xs text-ink-faint hover:text-alarm-high"
              >
                ✕
              </button>
            )}
          </span>
        ))
      )}
    </div>
  );
});
