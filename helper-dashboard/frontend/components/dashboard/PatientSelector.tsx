"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Field, fieldDescribedBy } from "@/components/ui";
import { api, describeApiError } from "@/lib/api";
import type { QueryFilters } from "@/lib/spec-schema";

// 「我正在查看誰的資料?」 — patient filter for the whole dashboard.
// Values come from GET /api/dashboard/label-values?label=patient_id
// (Prometheus label values seen in the last 24h) plus 「全部病患」. The
// choice is remembered PER DASHBOARD in localStorage and handed down as
// `filters` to DashboardPreview -> every widget query. With 全部 selected
// and more than one patient present, the header notes that stat values
// are averages across patients.

export const ALL_PATIENTS = "__all__";
const KEY_PREFIX = "hd.patient.";

export function readPatientChoice(dashboardId: string | null): string | null {
  if (!dashboardId || typeof window === "undefined") return null;
  try {
    const v = window.localStorage.getItem(KEY_PREFIX + dashboardId);
    return v && v !== ALL_PATIENTS ? v : null;
  } catch {
    return null;
  }
}

export function writePatientChoice(dashboardId: string | null, value: string | null): void {
  if (!dashboardId || typeof window === "undefined") return;
  try {
    if (value) window.localStorage.setItem(KEY_PREFIX + dashboardId, value);
    else window.localStorage.removeItem(KEY_PREFIX + dashboardId);
  } catch {
    // storage unavailable — the choice still applies for this page load
  }
}

export interface PatientFilterState {
  values: string[];
  loading: boolean;
  error: string | null;
  // null = 全部病患
  selected: string | null;
  setSelected: (v: string | null) => void;
  filters: QueryFilters | undefined;
  // 全部 selected while > 1 patient is present -> 「顯示多位病患平均」.
  multiPatientNote: boolean;
  refresh: () => void;
}

export function usePatientFilter(
  dashboardId: string | null,
  refreshMs = 60000,
): PatientFilterState {
  const [values, setValues] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelectedState] = useState<string | null>(null);
  const cancelled = useRef(false);

  // Restore the per-dashboard choice whenever the dashboard changes.
  useEffect(() => {
    setSelectedState(readPatientChoice(dashboardId));
  }, [dashboardId]);

  const setSelected = useCallback(
    (v: string | null) => {
      setSelectedState(v);
      writePatientChoice(dashboardId, v);
    },
    [dashboardId],
  );

  const refresh = useCallback(async () => {
    if (typeof document !== "undefined" && document.visibilityState === "hidden") return;
    try {
      const r = await api.labelValues("patient_id", 24);
      if (cancelled.current) return;
      const vals = Array.isArray(r.values)
        ? r.values.filter((v): v is string => typeof v === "string" && v.length > 0)
        : [];
      setValues([...vals].sort());
      setError(null);
    } catch (e) {
      if (cancelled.current) return;
      setError(describeApiError(e));
    } finally {
      if (!cancelled.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    cancelled.current = false;
    if (!dashboardId) return;
    refresh();
    const t = setInterval(refresh, refreshMs);
    return () => {
      cancelled.current = true;
      clearInterval(t);
    };
  }, [dashboardId, refresh, refreshMs]);

  const filters = useMemo<QueryFilters | undefined>(
    () => (selected ? { patient_id: selected } : undefined),
    [selected],
  );

  return {
    values,
    loading,
    error,
    selected,
    setSelected,
    filters,
    multiPatientNote: !selected && values.length > 1,
    refresh,
  };
}

export function PatientSelector({
  state,
  id = "hd-patient",
}: {
  state: PatientFilterState;
  id?: string;
}) {
  // Keep a remembered patient selectable even if it dropped out of the
  // last-24h list, so the <select> never shows a value it has no option for.
  const options = useMemo(() => {
    const set = new Set(state.values);
    if (state.selected) set.add(state.selected);
    return Array.from(set).sort();
  }, [state.values, state.selected]);

  // 「顯示多位病患平均」 is the header badge's job (DashboardHeader); the
  // hint only reports a fetch problem.
  const hint = state.error ? "無法取得病患清單，仍可顯示全部病患。" : undefined;

  return (
    <div className="w-full min-w-[10rem] sm:w-auto">
      <Field label="病患" htmlFor={id} hint={hint}>
        <select
          id={id}
          aria-describedby={fieldDescribedBy(id, { hint })}
          value={state.selected ?? ALL_PATIENTS}
          onChange={(e) =>
            state.setSelected(e.target.value === ALL_PATIENTS ? null : e.target.value)
          }
          className="min-h-[40px] w-full rounded-md border border-edge bg-surface px-2 text-sm text-ink focus:outline-none focus:ring-2 focus:ring-focus lg:min-h-[36px]"
        >
          <option value={ALL_PATIENTS}>
            全部病患{state.values.length > 0 ? `（${state.values.length}）` : ""}
          </option>
          {options.map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
      </Field>
    </div>
  );
}
