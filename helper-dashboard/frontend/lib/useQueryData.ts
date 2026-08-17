"use client";

import { useEffect, useState } from "react";
import { api } from "./api";
import type { QuerySpec } from "./spec-schema";

export interface SeriesPoint {
  t: number; // epoch seconds
  v: number;
}

// Explicit data state machine — there is NO mock fallback anywhere:
//  - "loading": fetch in flight
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
}

const LOADING: WidgetData = {
  status: "loading",
  points: [],
  latest: null,
  table: [],
  nSeries: 0,
  errorDetail: null,
};

// Fetch a widget's series from the backend, which runs the PromQL against
// live Prometheus. The browser never talks to Prometheus directly — the
// backend owns that. On any failure the state is an explicit "error" with
// detail; data is never fabricated client-side.
export function useQueryData(query: QuerySpec): WidgetData {
  const [data, setData] = useState<WidgetData>(LOADING);

  useEffect(() => {
    let cancelled = false;

    if (query.source !== "prometheus") {
      // Fail closed without a round trip — the backend answers the same
      // request with HTTP 400 "mock data source is not allowed".
      setData({
        status: "error",
        points: [],
        latest: null,
        table: [],
        nSeries: 0,
        errorDetail: "mock data source is not allowed",
      });
      return;
    }

    setData(LOADING);
    api
      .queryWidget(query)
      .then((resp) => {
        if (cancelled) return;
        if (resp.data_status === "empty") {
          setData({
            status: "empty",
            points: [],
            latest: null,
            table: [],
            nSeries: 0,
            errorDetail: null,
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
        });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setData({
          status: "error",
          points: [],
          latest: null,
          table: [],
          nSeries: 0,
          errorDetail: err instanceof Error ? err.message : String(err),
        });
      });

    return () => {
      cancelled = true;
    };
  }, [query.source, query.promql, query.query_type, query.range, query.step]);

  return data;
}
