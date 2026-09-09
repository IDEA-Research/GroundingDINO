// Threshold direction + activation — the SINGLE rule set every widget
// (stat card, gauge, line chart) uses, mirrored by the backend helper.
// Never invents clinical values: it only decides which side of a spec
// threshold is the alarmed side and which band a reading falls in.
//
// Effective direction (identical to backend/app/specs/threshold_semantics
// .resolve_direction — keep the two in lockstep):
//   1. an explicit `direction` on any threshold wins (first non-null);
//   2. else, with >= 2 thresholds, compare the first MOST severe band
//      (red 3 > amber/orange 2 > yellow 1 > other 0, by colour name or hue)
//      with the first LEAST severe one: equal ranks fall through; the severe
//      band at the LOWER value -> "below", at the HIGHER value -> "above";
//   3. else, if any label contains "<", "≤" or "低於" -> "below";
//   4. else "above".
//
// Activation:
//   above: the highest threshold whose value <= reading is active;
//   below: the lowest threshold whose value >= reading is active;
//   null/NaN readings never activate anything.

import {
  severityForSpec,
  severityScore,
  type Severity,
} from "./chart-theme";
import type { ThresholdDirection, WidgetThreshold } from "./spec-schema";

export const DIRECTION_LABELS: Record<ThresholdDirection, string> = {
  above: "高於上限",
  below: "低於下限",
};

export function resolveThresholdDirection(
  thresholds: WidgetThreshold[] | null | undefined,
): ThresholdDirection {
  const items = (thresholds ?? []).filter((t) => t !== null && t !== undefined);
  if (items.length === 0) return "above";

  // 1. explicit
  for (const t of items) {
    if (t.direction === "above" || t.direction === "below") return t.direction;
  }

  // 2. severity ordering (first max vs first min, like Python max()/min())
  if (items.length >= 2) {
    const ranked: { rank: number; value: number }[] = [];
    for (const t of items) {
      const value = Number(t.value);
      if (!Number.isFinite(value)) continue;
      ranked.push({ rank: severityScore(t.color), value });
    }
    if (ranked.length >= 2) {
      let most = ranked[0];
      let least = ranked[0];
      for (const rv of ranked) {
        if (rv.rank > most.rank) most = rv;
        if (rv.rank < least.rank) least = rv;
      }
      if (most.rank > least.rank) {
        if (most.value < least.value) return "below";
        if (most.value > least.value) return "above";
      }
    }
  }

  // 3. label hint
  for (const t of items) {
    const label = String(t.label ?? "");
    if (label.includes("<") || label.includes("≤") || label.includes("低於")) {
      return "below";
    }
  }

  // 4. default
  return "above";
}

// Ascending for "above", descending for "below" — the order in which
// thresholds are crossed as a reading gets worse.
export function sortedForDirection(
  thresholds: WidgetThreshold[] | null | undefined,
  direction: ThresholdDirection,
): WidgetThreshold[] {
  const list = (thresholds ?? []).filter(
    (t) => t && typeof t.value === "number" && Number.isFinite(t.value),
  );
  return [...list].sort((a, b) =>
    direction === "below" ? b.value - a.value : a.value - b.value,
  );
}

export interface ActiveThreshold {
  threshold: WidgetThreshold;
  severity: Severity;
  direction: ThresholdDirection;
  // Text shown beside the numeral — never color alone.
  label: string;
}

export function thresholdLabel(
  t: WidgetThreshold,
  direction: ThresholdDirection,
): string {
  const l = (t.label ?? "").trim();
  return l || DIRECTION_LABELS[direction];
}

export function activeThreshold(
  value: number | null | undefined,
  thresholds: WidgetThreshold[] | null | undefined,
  direction?: ThresholdDirection,
): ActiveThreshold | null {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return null;
  }
  const dir = direction ?? resolveThresholdDirection(thresholds);
  const sorted = sortedForDirection(thresholds, dir);
  let hit: WidgetThreshold | null = null;
  for (const t of sorted) {
    if (dir === "below" ? value <= t.value : value >= t.value) hit = t;
  }
  if (!hit) return null;
  return {
    threshold: hit,
    severity: severityForSpec(hit.color) ?? "info",
    direction: dir,
    label: thresholdLabel(hit, dir),
  };
}

export interface ThresholdBand {
  // 0..1 fractions of the gauge sweep.
  from: number;
  to: number;
  // null = neutral (no threshold governs this band).
  severity: Severity | null;
  threshold: WidgetThreshold | null;
}

// Gauge rim bands honoring direction. `frac` maps a value onto 0..1.
export function thresholdBands(
  thresholds: WidgetThreshold[] | null | undefined,
  direction: ThresholdDirection,
  frac: (v: number) => number,
): ThresholdBand[] {
  const sorted = sortedForDirection(thresholds, direction);
  const bands: ThresholdBand[] = [];
  if (direction === "below") {
    let cursor = 1;
    let sev: Severity | null = null;
    let cur: WidgetThreshold | null = null;
    for (const t of sorted) {
      const f = frac(t.value);
      if (f < cursor) bands.push({ from: f, to: cursor, severity: sev, threshold: cur });
      cursor = Math.min(cursor, f);
      sev = severityForSpec(t.color) ?? "info";
      cur = t;
    }
    if (cursor > 0) bands.push({ from: 0, to: cursor, severity: sev, threshold: cur });
    return bands;
  }
  let cursor = 0;
  let sev: Severity | null = null;
  let cur: WidgetThreshold | null = null;
  for (const t of sorted) {
    const f = frac(t.value);
    if (f > cursor) bands.push({ from: cursor, to: f, severity: sev, threshold: cur });
    cursor = Math.max(cursor, f);
    sev = severityForSpec(t.color) ?? "info";
    cur = t;
  }
  if (cursor < 1) bands.push({ from: cursor, to: 1, severity: sev, threshold: cur });
  return bands;
}

// "≥ 160" / "≤ 90" style caption for a threshold line or band.
export function thresholdCaption(
  t: WidgetThreshold,
  direction: ThresholdDirection,
  unit?: string | null,
): string {
  const sign = direction === "below" ? "≤" : "≥";
  const u = unit ? ` ${unit}` : "";
  const l = (t.label ?? "").trim();
  return l ? `${l} (${sign} ${t.value}${u})` : `${sign} ${t.value}${u}`;
}
