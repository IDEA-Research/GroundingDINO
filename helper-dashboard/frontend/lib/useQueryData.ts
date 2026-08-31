"use client";

import { useEffect, useRef, useState } from "react";
import { api, type WidgetQueryRequest } from "./api";
import type { QuerySpec } from "./spec-schema";
import { useTimeRange } from "./time-range";

export interface SeriesPoint {
  t: number; // epoch seconds
  v: number;
}

// Explicit data state machine — there is NO mock fallback anywhere:
//  - "loading": initial fetch in flight (refresh ticks NEVER re-enter
//               loading — old data stays visible and updates silently)
//  - "ok":      query succeeded and returned data
//  - "empty":   query succeeded but returned ZERO series/points — the
//               renderer must show an unmistakable NO-DATA state (silent
//               emptiness once hid an 11-day stale outage)
//  - "error":   fetch rejected, HTTP error, or a non-prometheus source —
//               the renderer must show a loud error box, never fabricate
export type QueryStatus = "loading" | "ok" | "empty" | "error";

export interface WidgetData {
  status: QueryStatus;
  points: SeriesPoint[]; // for line charts
  latest: number | null; // for stat / gauge
  table: { metric: Record<string, string>; value: number }[]; // for table/rows
  // Number of series the backend collapsed (by MEAN) into this data.
  // >1 → the renderer shows an "n series (averaged)" annotation.
  nSeries: number;
  errorDetail: string | null; // set when status === "error"
  // Time window the backend actually used (contract C1): step-aligned
  // end (epoch seconds) and resolved range length. null until the first
  // response lands, or when the backend predates the contract. Charts
  // pin their x-domain from THESE, never from a client clock.
  end: number | null;
  rangeSeconds: number | null;
  // Client-side arrival time (epoch ms) of the data currently shown.
  lastUpdated: number | null;
}

const LOADING: WidgetData = {
  status: "loading",
  points: [],
  latest: null,
  table: [],
  nSeries: 0,
  errorDetail: null,
  end: null,
  rangeSeconds: null,
  lastUpdated: null,
};

// Fetch a widget's series from the backend, which runs the PromQL against
// live Prometheus. The browser never talks to Prometheus directly — the
// backend owns that. On any failure the state is an explicit "error" with
// detail; data is never fabricated client-side.
//
// Refresh model: the TimeRangeProvider bumps {tick, endEpoch} on a single
// shared beat; every widget refetches on that beat with the same "end".
// Single in-flight guard: if a fetch is still running when the next tick
// lands, that tick is skipped (no overlapping fetches). The global range
// override is a view-layer request rewrite only — the spec (and this
// hook's `query` argument) is never mutated.
export function useQueryData(query: QuerySpec): WidgetData {
  const {
    tick,
    endEpoch,
    override,
    overrideRangeSeconds,
    overrideStepSeconds,
  } = useTimeRange();
  const [data, setData] = useState<WidgetData>(LOADING);

  // Identity of the last query+override this hook fetched for. A change
  // means "new query" (loud loading); same key on a new tick means
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

  const overrideKey = override.mode === "override" ? override.expr : "auto";

  useEffect(() => {
    const key = JSON.stringify([
      query.source,
      query.promql,
      query.query_type,
      query.range ?? "",
      query.step ?? "",
      overrideKey,
    ]);
    const isNewQuery = key !== lastKeyRef.current;
    lastKeyRef.current = key;
    if (isNewQuery) {
      // Invalidate any in-flight fetch for the previous query/window.
      seqRef.current += 1;
      inFlightRef.current = false;
    }

    if (query.source !== "prometheus") {
      // Fail closed without a round trip — the backend answers the same
      // request with HTTP 400 "mock data source is not allowed".
      if (isNewQuery) {
        setData({
          status: "error",
          points: [],
          latest: null,
          table: [],
          nSeries: 0,
          errorDetail: "mock data source is not allowed",
          end: null,
          rangeSeconds: null,
          lastUpdated: null,
        });
      }
      return;
    }

    if (isNewQuery) {
      // Only an initial load (or a changed query/override) shows the
      // loud loading state.
      setData(LOADING);
    } else if (inFlightRef.current) {
      // Refresh tick while the previous fetch is still in flight — skip
      // this tick; the outstanding response will still apply when it
      // lands, and the next tick will refresh again.
      return;
    }

    // Build the wire request (contract C1). The spec is never mutated:
    // an active override only rewrites what THIS request sends.
    const body: WidgetQueryRequest = {
      source: query.source,
      promql: query.promql,
      query_type: query.query_type,
      range: query.range ?? null,
      step: query.step ?? null,
    };
    if (endEpoch !== null) {
      body.end = endEpoch;
    }
    if (
      override.mode === "override" &&
      query.query_type === "range" &&
      overrideRangeSeconds !== null
    ) {
      body.range = `${overrideRangeSeconds}s`;
      body.step =
        overrideStepSeconds !== null
          ? `${overrideStepSeconds}s`
          : query.step ?? "30s";
    }

    const seq = ++seqRef.current;
    inFlightRef.current = true;

    api
      .queryWidget(body)
      .then((resp) => {
        if (seq !== seqRef.current) return; // superseded by a newer query
        inFlightRef.current = false;
        if (unmountedRef.current) return;
        const end = typeof resp.end === "number" ? resp.end : null;
        const rangeSeconds =
          typeof resp.range_seconds === "number" ? resp.range_seconds : null;
        const lastUpdated = Date.now();
        if (resp.data_status === "empty") {
          setData({
            status: "empty",
            points: [],
            latest: null,
            table: [],
            nSeries: 0,
            errorDetail: null,
            end,
            rangeSeconds,
            lastUpdated,
          });
          return;
        }
        setData({
          status: "ok",
          points: resp.points ?? [],
          latest: resp.latest ?? null,
          table: resp.table ?? [],
          nSeries: resp.n_series ?? 0,
          errorDetail: null,
          end,
          rangeSeconds,
          lastUpdated,
        });
      })
      .catch((err: unknown) => {
        if (seq !== seqRef.current) return; // superseded by a newer query
        inFlightRef.current = false;
        if (unmountedRef.current) return;
        setData({
          status: "error",
          points: [],
          latest: null,
          table: [],
          nSeries: 0,
          errorDetail: err instanceof Error ? err.message : String(err),
          end: null,
          rangeSeconds: null,
          lastUpdated: Date.now(),
        });
      });
  }, [
    query.source,
    query.promql,
    query.query_type,
    query.range,
    query.step,
    overrideKey,
    tick,
    endEpoch,
    overrideRangeSeconds,
    overrideStepSeconds,
  ]);

  return data;
}
