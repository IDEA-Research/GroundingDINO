// Equivalence proof: lib/threshold.ts's activeThreshold() (the logic
// StatCardWidget/GaugeWidget used before this change) vs.
// lib/threshold-adapter.ts's activeThresholdViaEngine() (adapter +
// lib/threshold-engine.ts's resolveThreshold(), what they use after).
//
// Fixture data is NOT invented for this test: LIVE_SPO2 and HR are copied
// verbatim from tests/spec_validation/test_threshold_direction.py, the
// backend's own docstring calling LIVE_SPO2 "the live SpO2 ladder that
// motivated the field" — i.e. real production threshold shapes, one
// "below" ladder (SpO2, hex colors, Chinese labels) and one "above"
// ladder (HR, hex colors, no labels) — the two directions every widget
// actually renders.
//
// No jest/vitest is configured in frontend/ (see threshold-engine.test.ts
// for the same note) — this runs standalone against node:assert:
//
//   npx tsc lib/threshold.ts lib/threshold-engine.ts lib/threshold-adapter.ts \
//     lib/threshold-adapter.test.ts lib/chart-theme.ts lib/theme.ts \
//     --module commonjs --target es2020 --esModuleInterop --skipLibCheck \
//     --outDir <scratch-dir>
//   node <scratch-dir>/threshold-adapter.test.js

import assert from "node:assert/strict";
import { activeThreshold, resolveThresholdDirection } from "./threshold";
import { activeThresholdViaEngine } from "./threshold-adapter";
import type { WidgetThreshold } from "./spec-schema";

function test(name: string, fn: () => void): void {
  fn();
  console.log(`ok - ${name}`);
}

// Verbatim from tests/spec_validation/test_threshold_direction.py.
const LIVE_SPO2: WidgetThreshold[] = [
  { value: 94, color: "#f59e0b", label: "偏低 <95%" },
  { value: 90, color: "#ef4444", label: "危急 <90%" },
];
const HR: WidgetThreshold[] = [
  { value: 100, color: "#f59e0b" },
  { value: 120, color: "#ef4444" },
];

function assertSameResult(
  label: string,
  value: number | null,
  thresholds: WidgetThreshold[],
): void {
  const direction = resolveThresholdDirection(thresholds);
  const oldResult = activeThreshold(value, thresholds, direction);
  const newResult = activeThresholdViaEngine(value, thresholds, direction);

  if (oldResult === null || newResult === null) {
    assert.equal(
      newResult,
      oldResult,
      `${label}: expected both null, got old=${JSON.stringify(oldResult)} new=${JSON.stringify(newResult)}`,
    );
    return;
  }
  assert.deepEqual(
    { severity: newResult.severity, direction: newResult.direction, label: newResult.label, threshold: newResult.threshold },
    { severity: oldResult.severity, direction: oldResult.direction, label: oldResult.label, threshold: oldResult.threshold },
    `${label}: old vs new mismatch`,
  );
}

test("direction is resolved identically (below for SpO2, above for HR)", () => {
  assert.equal(resolveThresholdDirection(LIVE_SPO2), "below");
  assert.equal(resolveThresholdDirection(HR), "above");
});

test("LIVE_SPO2 (below): old and new agree on every value from the backend fixture + boundaries", () => {
  // 93/89/97 mirror test_active_threshold_honours_direction exactly;
  // 94/90 add the exact boundary values (>=/<= must include the edge).
  for (const v of [100, 95, 94.01, 94, 93.99, 93, 91, 90.01, 90, 89.99, 89, 80, 0, -10]) {
    assertSameResult(`SpO2 @ ${v}`, v, LIVE_SPO2);
  }
});

test("HR (above): old and new agree on every value from the backend fixture + boundaries", () => {
  for (const v of [0, 50, 99.99, 100, 100.01, 110, 119.99, 120, 120.01, 125, 200]) {
    assertSameResult(`HR @ ${v}`, v, HR);
  }
});

test("null/NaN/undefined readings never activate anything, in either implementation", () => {
  for (const bad of [null, undefined, Number.NaN]) {
    const direction = resolveThresholdDirection(LIVE_SPO2);
    assert.equal(activeThreshold(bad as number, LIVE_SPO2, direction), null);
    assert.equal(activeThresholdViaEngine(bad as number, LIVE_SPO2, direction), null);
  }
});

test("empty threshold list: both return null for any reading", () => {
  assertSameResult("empty @ 50", 50, []);
  assertSameResult("empty @ -1", -1, []);
});

test("single-threshold ladders (above and below) agree at and around the boundary", () => {
  const aboveOnly: WidgetThreshold[] = [{ value: 38, color: "red", label: "發燒" }];
  for (const v of [37, 37.99, 38, 38.01, 40]) assertSameResult(`fever @ ${v}`, v, aboveOnly);

  const belowOnly: WidgetThreshold[] = [{ value: 90, color: "red", label: "低血氧 <90%", direction: "below" }];
  for (const v of [95, 90.01, 90, 89.99, 80]) assertSameResult(`low-spo2 @ ${v}`, v, belowOnly);
});

console.log("threshold-adapter.test.ts: all tests passed (old activeThreshold === new activeThresholdViaEngine)");
