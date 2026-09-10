// WidgetThreshold[] -> ThresholdEngine adapter.
//
// StatCardWidget / GaugeWidget work off lib/threshold.ts's
// activeThreshold(value, thresholds, direction) — a hand-rolled walk over
// a direction-aware WidgetThreshold[] ladder (no explicit base step, no
// percentage mode). lib/threshold-engine.ts's resolveThreshold() wants a
// Grafana-style ThresholdStep[]: ascending by value, steps[0].value ===
// null (the base/"normal" step). This module is the one place that
// bridges the two — see the mapping note at the bottom of
// threshold-engine.ts — so the "which threshold is active" DECISION
// (walking the ladder and picking a winner) is made in one shared place
// instead of twice.
//
// Only the SELECTION algorithm moves here. Turning the winning threshold
// into a Severity/color stays lib/chart-theme.ts's job (severityForSpec /
// colorForSeverity) exactly as before — that is spec-color-string
// interpretation, a different concern from threshold resolution, and
// resolveThreshold's own `.color` output is not used for that reason (see
// toThresholdSteps below: base color is a placeholder, real severity
// comes from the ORIGINAL WidgetThreshold row via `sorted[level - 1]`).
//
// "above" ladders (e.g. HR [100 amber, 120 red]) already sort ascending —
// prepend the base step and hand the reading straight to resolveThreshold.
//
// "below" ladders (a low-is-bad ladder, e.g. SpO2 [94 amber, 90 red]) are
// the one wrinkle: resolveThreshold only ever walks ascending and compares
// `value >= step.value`. Negating BOTH the reading and every step's value
// flips the comparison direction while preserving which step wins
// (`value <= t.value` <=> `-value >= -t.value`, exact for finite floats —
// negation introduces no rounding), and — because sortedForDirection
// already returns "below" ladders sorted DESCENDING — negating each
// element in place lands them back in ascending order with no extra
// reverse needed: v1 >= v2 >= ... descending implies -v1 <= -v2 <= ...
// ascending.
//
// Equivalence against lib/threshold.ts's activeThreshold() is verified in
// lib/threshold-adapter.test.ts against the real LIVE_SPO2 / HR fixtures
// shared with the backend (tests/spec_validation/
// test_threshold_direction.py) — not invented data.
import { severityForSpec } from "./chart-theme";
import {
  sortedForDirection,
  thresholdLabel,
  type ActiveThreshold,
} from "./threshold";
import { resolveThreshold, type ThresholdStep } from "./threshold-engine";
import type { ThresholdDirection, WidgetThreshold } from "./spec-schema";

// `sorted` must already be sortedForDirection(thresholds, direction) —
// ascending for "above", descending for "below" — the order every
// existing threshold.ts helper (sortedForDirection, thresholdBands,
// GaugeWidget's legend) already produces and consumes.
export function toThresholdSteps(
  sorted: WidgetThreshold[],
  direction: ThresholdDirection,
): ThresholdStep[] {
  const sign = direction === "below" ? -1 : 1;
  return [
    { value: null, color: "" }, // base/"normal" step — selection only, never rendered
    ...sorted.map((t) => ({ value: sign * t.value, color: t.color })),
  ];
}

// Drop-in replacement for lib/threshold.ts's activeThreshold(): identical
// signature (minus the optional direction — callers here always already
// have it from resolveThresholdDirection) and identical return shape/
// values, with the ladder walk delegated to
// lib/threshold-engine.ts's resolveThreshold().
export function activeThresholdViaEngine(
  value: number | null | undefined,
  thresholds: WidgetThreshold[] | null | undefined,
  direction: ThresholdDirection,
): ActiveThreshold | null {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return null;
  }
  const sorted = sortedForDirection(thresholds, direction);
  if (sorted.length === 0) return null;

  const sign = direction === "below" ? -1 : 1;
  const steps = toThresholdSteps(sorted, direction);
  const { level } = resolveThreshold(sign * value, steps, "absolute");
  if (level === 0) return null; // base step active -> nothing breached

  const hit = sorted[level - 1];
  return {
    threshold: hit,
    severity: severityForSpec(hit.color) ?? "info",
    direction,
    label: thresholdLabel(hit, direction),
  };
}
