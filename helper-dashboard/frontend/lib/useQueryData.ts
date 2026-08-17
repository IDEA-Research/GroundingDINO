"use client";

import { useEffect, useState } from "react";
import { api } from "./api";
import type { QuerySpec } from "./spec-schema";

export interface SeriesPoint {
  t: number; // epoch seconds
  v: number;
}

export interface WidgetData {
  loading: boolean;
  error: string | null;
  source: "prometheus" | "mock" | "unknown";
  points: SeriesPoint[]; // for line charts
  latest: number | null; // for stat / gauge
  table: { metric: Record<string, string>; value: number }[]; // for table
}

const EMPTY: WidgetData = {
  loading: true,
  error: null,
  source: "unknown",
  points: [],
  latest: null,
  table: [],
};

// Fetch a widget's series from the backend, which runs the PromQL against
// live Prometheus and returns a `source` flag ("prometheus" | "mock"). The
// browser never talks to Prometheus directly — the backend owns that.
//
// A widget whose spec pins `source: "mock"` stays on the deterministic
// client-side mock and never hits the backend. If the backend itself is
// unreachable we fall back to that same mock, honestly labeled `source: mock`
// so the UI is non-empty and the mock badge is shown — never presented as real.
export function useQueryData(query: QuerySpec): WidgetData {
  const [data, setData] = useState<WidgetData>(EMPTY);

  useEffect(() => {
    let cancelled = false;

    if (query.source === "mock") {
      setData(mockFor(query));
      return;
    }

    setData((prev) => ({ ...prev, loading: true, error: null }));
    api
      .queryWidget(query)
      .then((resp) => {
        if (cancelled) return;
        setData({
          loading: false,
          error: null,
          source: resp.source,
          points: resp.points ?? [],
          latest: resp.latest ?? null,
          table: resp.table ?? [],
        });
      })
      .catch(() => {
        if (cancelled) return;
        setData(mockFor(query));
      });

    return () => {
      cancelled = true;
    };
  }, [query.source, query.promql, query.query_type, query.range, query.step]);

  return data;
}

function mockFor(query: QuerySpec): WidgetData {
  const seed = sumChars(query.promql) % 97;
  const now = Math.floor(Date.now() / 1000);

  if (query.query_type === "instant") {
    const v = 40 + (seed % 50);
    return {
      loading: false,
      error: null,
      source: "mock",
      points: [],
      latest: v,
      table: buildTableRows(query.promql, seed),
    };
  }

  const rangeS = parseDuration(query.range || "1h") || 3600;
  const stepS = parseDuration(query.step || "30s") || 30;
  const n = Math.max(1, Math.floor(rangeS / Math.max(stepS, 1)));
  const points: SeriesPoint[] = [];
  for (let i = 0; i < n; i++) {
    const t = now - rangeS + i * stepS;
    const v = 50 + 20 * Math.sin((i + seed) / 6);
    points.push({ t, v });
  }
  return {
    loading: false,
    error: null,
    source: "mock",
    points,
    latest: points.length ? points[points.length - 1].v : null,
    table: [],
  };
}

function buildTableRows(promql: string, seed: number) {
  const rows: { metric: Record<string, string>; value: number }[] = [];
  for (let i = 0; i < 6; i++) {
    rows.push({
      metric: { instance: `node-${(seed + i) % 10}`, job: "mock" },
      value: 10 + ((seed + i) % 90),
    });
  }
  return rows;
}

function sumChars(s: string): number {
  let n = 0;
  for (let i = 0; i < s.length; i++) n += s.charCodeAt(i);
  return n;
}

function parseDuration(s: string): number | null {
  const m = /^(\d+)(ms|s|m|h)$/.exec(s.trim());
  if (!m) return null;
  const v = parseInt(m[1], 10);
  switch (m[2]) {
    case "ms":
      return Math.max(1, Math.round(v / 1000));
    case "s":
      return v;
    case "m":
      return v * 60;
    case "h":
      return v * 3600;
  }
  return null;
}
