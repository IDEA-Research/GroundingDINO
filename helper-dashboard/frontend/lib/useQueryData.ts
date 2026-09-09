"use client";

import React, {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";

import { api } from "./api";
import {
  isJoinedTableRow,
  type Freshness,
  type JoinedTableRow,
  type QueryFilters,
  type QuerySpec,
  type SeriesLabelSet,
  type TableColumn,
  type WidgetQueryRequest,
  type WidgetQueryResponse,
} from "./spec-schema";
import { useTimeRange } from "./time-range";

export interface SeriesPoint {
  t: number; // epoch seconds
  // null = gap the backend marked (samples farther apart than 2 x step).
  v: number | null;
}

// Explicit data state machine — there is NO mock fallback anywhere:
//  - "loading": initial fetch in flight (refresh ticks NEVER re-enter
//               loading — old data stays visible, isRefreshing flips on)
//  - "ok":      query succeeded and returned fresh data
//  - "stale":   data on screen but no longer trustworthy as current —
//               either the backend flagged freshness.stale, or a refresh
//               tick FAILED and the previous good data is kept (reason
//               「更新失敗」). Never fabricate; keep showing what we have.
//  - "empty":   query succeeded but returned ZERO series/points — the
//               renderer must show an unmistakable NO-DATA state (silent
//               emptiness once hid an 11-day stale outage)
//  - "offline": Prometheus unreachable (HTTP 502 / "Prometheus unavailable")
//  - "error":   any other failure (HTTP 4xx/5xx, network, bad source)
export type QueryStatus =
  | "loading"
  | "ok"
  | "stale"
  | "empty"
  | "offline"
  | "error";

// Short 繁中 reasons shown next to 「—」 in cards and in the status strip.
export const REASON_TEXT = {
  loading: "載入中",
  empty: "尚無資料",
  offline: "來源中斷",
  stale: "資料過期",
  error: "查詢失敗",
  refreshFailed: "更新失敗",
  filtered: "此篩選無資料",
} as const;

export interface TableRow {
  metric: Record<string, string>;
  // null = missing (rendered 「—」, never 0).
  value: number | null;
  // Sample timestamp (epoch seconds) when the backend provided one.
  ts?: number | null;
}

export interface WidgetData {
  status: QueryStatus;
  points: SeriesPoint[]; // for line charts
  latest: number | null; // for stat / gauge
  table: TableRow[]; // for table/rows (legacy single-query shape)
  // `queries` tables: one row per join-label value with per-column values.
  joined: JoinedTableRow[];
  columns: TableColumn[];
  // Number of series the backend collapsed (by MEAN) into this data.
  // >1 → the renderer shows an "n series (averaged)" annotation.
  nSeries: number;
  errorDetail: string | null; // set when the last fetch failed
  errorStatus: number | null; // HTTP status of the last failure, if known
  // 繁中 short reason for any non-ok status (see REASON_TEXT).
  reason: string | null;
  // Time window the backend actually used (contract C1): step-aligned
  // end (epoch seconds) and resolved range length. Charts pin their
  // x-domain from THESE, never from a client clock.
  end: number | null;
  rangeSeconds: number | null;
  stepSeconds: number | null;
  // Client-side arrival time (epoch ms) of the data currently shown.
  lastUpdated: number | null;
  // Client-side arrival time (epoch ms) of the last SUCCESSFUL response.
  lastSuccessAt: number | null;
  // A refresh tick is in flight while old data stays on screen.
  isRefreshing: boolean;
  freshness: Freshness | null;
  simulated: boolean;
  simScenario: string | null;
  seriesLabels: SeriesLabelSet[];
  filtersApplied: QueryFilters | null;
}

const LOADING: WidgetData = {
  status: "loading",
  points: [],
  latest: null,
  table: [],
  joined: [],
  columns: [],
  nSeries: 0,
  errorDetail: null,
  errorStatus: null,
  reason: REASON_TEXT.loading,
  end: null,
  rangeSeconds: null,
  stepSeconds: null,
  lastUpdated: null,
  lastSuccessAt: null,
  isRefreshing: false,
  freshness: null,
  simulated: false,
  simScenario: null,
  seriesLabels: [],
  filtersApplied: null,
};

export const EMPTY_WIDGET_DATA: WidgetData = LOADING;

// --- Filters context ---------------------------------------------------------
// DashboardPreview provides the active patient/device filters; QueryGate and
// widgets read them. Explicit `opts.filters` on useQueryData wins.

const QueryFiltersContext = createContext<QueryFilters | null>(null);

export function QueryFiltersProvider({
  filters,
  children,
}: {
  filters: QueryFilters | null | undefined;
  children: React.ReactNode;
}) {
  return React.createElement(
    QueryFiltersContext.Provider,
    { value: normalizeFilters(filters) },
    children,
  );
}

export function useQueryFilters(): QueryFilters | null {
  return useContext(QueryFiltersContext);
}

// Drops empty values; null when nothing is set.
export function normalizeFilters(
  f: QueryFilters | null | undefined,
): QueryFilters | null {
  if (!f) return null;
  const out: QueryFilters = {};
  if (typeof f.patient_id === "string" && f.patient_id.trim()) {
    out.patient_id = f.patient_id.trim();
  }
  if (typeof f.device_model === "string" && f.device_model.trim()) {
    out.device_model = f.device_model.trim();
  }
  return Object.keys(out).length > 0 ? out : null;
}

export function hasActiveFilters(f: QueryFilters | null | undefined): boolean {
  return normalizeFilters(f) !== null;
}

// --- Helpers -----------------------------------------------------------------

export function hasContent(d: WidgetData): boolean {
  return (
    d.latest !== null ||
    d.points.some((p) => p.v !== null) ||
    d.table.length > 0 ||
    d.joined.length > 0
  );
}

function finiteOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function statusCodeOf(err: unknown): number | null {
  const anyErr = err as { status?: unknown; message?: unknown } | null;
  if (anyErr && typeof anyErr.status === "number") return anyErr.status;
  const msg = err instanceof Error ? err.message : String(err ?? "");
  const m = /failed:\s*(\d{3})/.exec(msg);
  return m ? parseInt(m[1], 10) : null;
}

function classifyError(err: unknown): {
  status: "offline" | "error";
  errorDetail: string;
  errorStatus: number | null;
  reason: string;
} {
  const errorDetail = err instanceof Error ? err.message : String(err);
  const errorStatus = statusCodeOf(err);
  const offline =
    errorStatus === 502 || /prometheus unavailable/i.test(errorDetail);
  return {
    status: offline ? "offline" : "error",
    errorDetail,
    errorStatus,
    reason: offline ? REASON_TEXT.offline : REASON_TEXT.error,
  };
}

function normalizeResponse(
  resp: WidgetQueryResponse,
  filters: QueryFilters | null,
  prevLastSuccessAt: number | null,
): WidgetData {
  const now = Date.now();
  const end = finiteOrNull(resp.end);
  const rangeSeconds = finiteOrNull(resp.range_seconds);
  const stepSeconds = finiteOrNull(resp.step_seconds);
  const freshness = resp.freshness ?? null;
  const simulated = Boolean(resp.simulated);
  const simScenario = simulated ? resp.sim_scenario ?? null : null;
  const seriesLabels = Array.isArray(resp.series_labels)
    ? resp.series_labels
    : [];
  const filtersApplied = resp.filters_applied ?? filters ?? null;
  const columns = Array.isArray(resp.columns) ? resp.columns : [];

  const base = {
    end,
    rangeSeconds,
    stepSeconds,
    lastUpdated: now,
    lastSuccessAt: now,
    isRefreshing: false,
    freshness,
    simulated,
    simScenario,
    seriesLabels,
    filtersApplied,
    errorDetail: null,
    errorStatus: null,
    columns,
  };

  if (resp.data_status === "empty") {
    return {
      ...base,
      status: "empty",
      points: [],
      latest: null,
      table: [],
      joined: [],
      nSeries: 0,
      reason: filters ? REASON_TEXT.filtered : REASON_TEXT.empty,
    };
  }

  const points: SeriesPoint[] = (resp.points ?? []).map((p) => ({
    t: p.t,
    v: finiteOrNull(p.v),
  }));

  const joined: JoinedTableRow[] = [];
  const table: TableRow[] = [];
  for (const row of resp.table ?? []) {
    if (isJoinedTableRow(row)) {
      joined.push(row);
      const firstKey = columns[0]?.key ?? Object.keys(row.values)[0];
      table.push({
        metric: row.labels ?? {},
        value: firstKey ? finiteOrNull(row.values[firstKey]) : null,
        ts: firstKey ? finiteOrNull(row.ts?.[firstKey]) : null,
      });
    } else {
      table.push({ metric: row.metric ?? {}, value: finiteOrNull(row.value) });
    }
  }

  const stale = Boolean(freshness?.stale);
  return {
    ...base,
    status: stale ? "stale" : "ok",
    points,
    latest: finiteOrNull(resp.latest),
    table,
    joined,
    nSeries: typeof resp.n_series === "number" ? resp.n_series : 0,
    reason: stale ? REASON_TEXT.stale : null,
    lastSuccessAt: now ?? prevLastSuccessAt,
  };
}

export interface UseQueryDataOptions {
  // Table widgets: one instant query per column, joined by `joinLabel`.
  queries?: QuerySpec[] | null;
  // Explicit filters win over the QueryFiltersProvider value.
  filters?: QueryFilters | null;
  joinLabel?: string | null;
}

// Fetch a widget's series from the backend, which runs the PromQL against
// live Prometheus. The browser never talks to Prometheus directly — the
// backend owns that. On any failure the state is an explicit "error" /
// "offline" with detail; data is never fabricated client-side.
//
// Refresh model: the TimeRangeProvider bumps {tick, endEpoch} on a single
// shared beat; every widget refetches on that beat with the same "end".
// Single in-flight guard: if a fetch is still running when the next tick
// lands, that tick is skipped. A failed refresh keeps the last good data
// on screen and marks it "stale" (reason 更新失敗). The global range
// override is a view-layer request rewrite only — the spec (and this
// hook's `query` argument) is never mutated.
export function useQueryData(
  query: QuerySpec | null | undefined,
  opts?: UseQueryDataOptions,
): WidgetData {
  const {
    tick,
    endEpoch,
    range,
    overrideRangeSeconds,
    overrideStepSeconds,
    trackFetch,
  } = useTimeRange();
  const ctxFilters = useQueryFilters();
  const filters = normalizeFilters(opts?.filters ?? ctxFilters);
  const queries = opts?.queries && opts.queries.length > 0 ? opts.queries : null;
  const joinLabel = opts?.joinLabel ?? "patient_id";

  const [data, setData] = useState<WidgetData>(LOADING);

  // Identity of the last query+window+filters this hook fetched for. A
  // change means "new query" (loud loading); same key on a new tick means
  // "silent refresh".
  const lastKeyRef = useRef<string | null>(null);
  // Monotonic id of the newest launched fetch; a response only applies
  // if it is still the newest (stale responses for older queries drop).
  const seqRef = useRef(0);
  const inFlightRef = useRef(false);
  const unmountedRef = useRef(false);

  useEffect(() => {
    unmountedRef.current = false;
    return () => {
      unmountedRef.current = true;
    };
  }, []);

  const overrideKey = range.mode === "override" ? range.expr : "spec";
  const filtersKey = JSON.stringify(filters ?? {});
  const queriesKey = JSON.stringify(
    queries?.map((q) => [q.key ?? "", q.promql, q.label ?? "", q.unit ?? ""]) ??
      [],
  );
  const source = query?.source ?? queries?.[0]?.source ?? "prometheus";
  const promql = query?.promql ?? "";
  const queryType = query?.query_type ?? "instant";
  const specRange = query?.range ?? "";
  const specStep = query?.step ?? "";

  useEffect(() => {
    const key = JSON.stringify([
      source,
      promql,
      queryType,
      specRange,
      specStep,
      overrideKey,
      filtersKey,
      queriesKey,
      joinLabel,
    ]);
    const isNewQuery = key !== lastKeyRef.current;
    lastKeyRef.current = key;
    if (isNewQuery) {
      // Invalidate any in-flight fetch for the previous query/window.
      seqRef.current += 1;
      inFlightRef.current = false;
    }

    if (!query && !queries) {
      if (isNewQuery) {
        setData({
          ...LOADING,
          status: "error",
          errorDetail: "widget has no query",
          reason: REASON_TEXT.error,
          lastUpdated: Date.now(),
        });
      }
      return;
    }

    if (source !== "prometheus") {
      // Fail closed without a round trip — the backend answers the same
      // request with HTTP 400 "mock data source is not allowed".
      if (isNewQuery) {
        setData({
          ...LOADING,
          status: "error",
          errorDetail: "mock data source is not allowed",
          errorStatus: 400,
          reason: REASON_TEXT.error,
          lastUpdated: Date.now(),
        });
      }
      return;
    }

    if (isNewQuery) {
      // Only an initial load (or a changed query/override/filter) shows
      // the loud loading state.
      setData(LOADING);
    } else if (inFlightRef.current) {
      // Refresh tick while the previous fetch is still in flight — skip
      // this tick; the outstanding response will still apply when it
      // lands, and the next tick will refresh again.
      return;
    } else {
      setData((prev) =>
        prev.isRefreshing ? prev : { ...prev, isRefreshing: true },
      );
    }

    // Build the wire request (contract C1). The spec is never mutated:
    // an active override only rewrites what THIS request sends.
    const body: WidgetQueryRequest = {
      source: "prometheus",
      promql: promql || null,
      query_type: queryType,
      range: specRange || null,
      step: specStep || null,
    };
    if (endEpoch !== null) {
      body.end = endEpoch;
    }
    if (filters) {
      body.filters = filters;
    }
    if (queries) {
      body.queries = queries.map((q, i) => ({
        key: q.key && q.key.trim() ? q.key.trim() : `q${i + 1}`,
        promql: q.promql,
        label: q.label ?? null,
        unit: q.unit ?? null,
      }));
      body.join_label = joinLabel;
      body.query_type = "instant";
    }
    if (
      range.mode === "override" &&
      queryType === "range" &&
      !queries &&
      overrideRangeSeconds !== null
    ) {
      body.range = `${overrideRangeSeconds}s`;
      body.step =
        overrideStepSeconds !== null
          ? `${overrideStepSeconds}s`
          : specStep || "30s";
    }

    const seq = ++seqRef.current;
    inFlightRef.current = true;
    trackFetch.start();

    api
      .queryWidget(body as unknown as Parameters<typeof api.queryWidget>[0])
      .then((raw) => {
        const resp = raw as unknown as WidgetQueryResponse;
        trackFetch.finish(true);
        if (seq !== seqRef.current) return; // superseded by a newer query
        inFlightRef.current = false;
        if (unmountedRef.current) return;
        setData((prev) => normalizeResponse(resp, filters, prev.lastSuccessAt));
      })
      .catch((err: unknown) => {
        trackFetch.finish(false);
        if (seq !== seqRef.current) return; // superseded by a newer query
        inFlightRef.current = false;
        if (unmountedRef.current) return;
        const failure = classifyError(err);
        setData((prev) => {
          const keep =
            !isNewQuery &&
            (prev.status === "ok" || prev.status === "stale") &&
            hasContent(prev);
          if (keep) {
            // Failed refresh: keep the last good data visible, mark stale.
            return {
              ...prev,
              status: "stale",
              reason: REASON_TEXT.refreshFailed,
              errorDetail: failure.errorDetail,
              errorStatus: failure.errorStatus,
              isRefreshing: false,
              lastUpdated: Date.now(),
            };
          }
          return {
            ...LOADING,
            status: failure.status,
            errorDetail: failure.errorDetail,
            errorStatus: failure.errorStatus,
            reason: failure.reason,
            lastUpdated: Date.now(),
            lastSuccessAt: prev.lastSuccessAt,
            filtersApplied: filters,
          };
        });
      });
  }, [
    query,
    queries,
    source,
    promql,
    queryType,
    specRange,
    specStep,
    overrideKey,
    filtersKey,
    queriesKey,
    joinLabel,
    tick,
    endEpoch,
    range.mode,
    overrideRangeSeconds,
    overrideStepSeconds,
    trackFetch,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    filters ? JSON.stringify(filters) : "",
  ]);

  return data;
}
