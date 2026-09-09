// ThresholdEngine — Grafana-aligned threshold resolution.
//
// Mirrors Grafana's field "Thresholds" model (Standard options >
// Thresholds, `ThresholdsConfig` in grafana/data): a ladder of steps
// sorted ascending by value, where the FIRST step is always the "base"
// (value: null, meaning -Infinity — it always matches) and every later
// step's `value` is the point at which that step's color takes over.
//
//   - "absolute" mode compares the raw value against each step's value
//     directly.
//   - "percentage" mode expresses each non-base step's value as 0-100
//     within a supplied [min, max] range; it is converted to an absolute
//     value (`min + (max - min) * value / 100`) before the same
//     comparison runs — this is Grafana's `getActiveThreshold` behavior.
//
// This module is intentionally separate from lib/threshold.ts, the
// direction-aware ladder StatCardWidget/GaugeWidget/LineChartWidget
// already share (WidgetThreshold[] with a resolved "above"/"below"
// direction inferred from severity + labels, no explicit base step, no
// percentage mode). See the bottom of this file for how the two relate
// and how StatCardWidget/GaugeWidget's existing ok/warn/danger logic maps
// onto resolveThreshold() below. Phase 3 only builds this engine; wiring
// widgets onto it is Phase 4.

export type ThresholdMode = "absolute" | "percentage";

export interface ThresholdStep {
  // null on the FIRST step only — Grafana's "base", meaning -Infinity.
  // Every later step must carry a finite number, and steps must already
  // be sorted ascending by value (base first).
  value: number | null;
  color: string;
}

export interface ThresholdRange {
  min: number;
  max: number;
}

export interface ResolvedThreshold {
  color: string;
  // Index into the `steps` array that is currently active. Lets a caller
  // that also carries its own label/severity-name array (e.g.
  // WidgetThreshold[]) look up the matching entry by position without
  // re-deriving "which rung of the ladder" itself.
  level: number;
}

/**
 * Resolve which threshold step governs `value`.
 *
 * `steps` must be sorted ascending by value with `steps[0].value === null`
 * (the base step). Absolute mode compares `value` against each step's
 * value directly; percentage mode first rescales every non-base step's
 * value from a 0-100 percentage into an absolute value using `range`,
 * then applies the same comparison.
 *
 * Returns the LAST step for which `value >= step.value` holds (the base
 * step always holds, so this never returns null/undefined).
 */
export function resolveThreshold(
  value: number,
  steps: ThresholdStep[],
  mode: ThresholdMode,
  range?: ThresholdRange,
): ResolvedThreshold {
  if (!steps || steps.length === 0) {
    throw new Error(
      "resolveThreshold: steps must contain at least one (base) entry",
    );
  }
  if (process.env.NODE_ENV !== "production" && steps[0].value !== null) {
    // eslint-disable-next-line no-console
    console.warn(
      "resolveThreshold: steps[0].value should be null (the Grafana-style base step)",
    );
  }

  const toAbsolute = (step: ThresholdStep): number | null => {
    if (step.value === null) return null; // base stays -Infinity
    if (mode !== "percentage") return step.value;
    if (!range) {
      throw new Error(
        "resolveThreshold: percentage mode requires a { min, max } range",
      );
    }
    return range.min + ((range.max - range.min) * step.value) / 100;
  };

  let level = 0;
  for (let i = 0; i < steps.length; i++) {
    const abs = toAbsolute(steps[i]);
    if (abs === null || value >= abs) level = i;
  }

  return { color: steps[level].color, level };
}

// ---------------------------------------------------------------------------
// How this replaces StatCardWidget.tsx / GaugeWidget.tsx's current
// ok/warn/danger derivation (see report for the full before/after — this
// is the short version for anyone reading the source later):
//
// Today (lib/threshold.ts, consumed identically by both widgets):
//   const direction = resolveThresholdDirection(thresholds);         // "above" | "below"
//   const active = activeThreshold(value, thresholds, direction);    // walks a WidgetThreshold[]
//   const color = active ? colorForSeverity(active.severity, theme.colors) : normalColor;
//
// A WidgetThreshold[] such as
//   [{ value: 80, color: "yellow", label: "偏高" }, { value: 90, color: "red", label: "過高" }]
// maps onto a Grafana-style ThresholdStep[] by (a) prepending an explicit
// base step for the "no threshold breached" state and (b) sorting for the
// resolved direction:
//   direction "above" (ascending):  [{ value: null, color: normalColor },
//                                     { value: 80, color: "yellow" },
//                                     { value: 90, color: "red" }]
//   direction "below" (a low-is-bad ladder, e.g. SpO2 [94 amber, 90 red])
//   needs one extra step: resolveThreshold only ever walks ascending, so a
//   "below" ladder is handled by negating both the reading and every step
//   value before calling resolveThreshold (value >= step  <=>  -value >= -step),
//   then reading `level`/`color` back unchanged — direction stays a
//   concern of the CALLER (mapping WidgetThreshold[] -> ThresholdStep[]),
//   exactly like Grafana's own thresholds have no notion of "direction"
//   at all (a Grafana panel author just enters the ladder in the sense
//   that makes red the danger color already sorted correctly).
// The two engines' outputs already carry the same content:
//   activeThreshold(...).severity  ~=  colorForSeverity lookup keyed by
//   resolveThreshold(...).color, so a StatCardWidget/GaugeWidget rewired
//   onto this module would do:
//     const steps = toThresholdSteps(widget.thresholds, direction, normalColor);
//     const { color, level } = resolveThreshold(value, steps, "absolute");
//   (`toThresholdSteps` — the WidgetThreshold[] -> ThresholdStep[] mapper —
//   is a Phase 4 task alongside the actual widget rewrites; it is not
//   included here since it touches widget-facing types this phase is not
//   supposed to touch.)
