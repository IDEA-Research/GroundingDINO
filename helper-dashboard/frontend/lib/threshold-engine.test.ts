// Unit tests for resolveThreshold (lib/threshold-engine.ts).
//
// The project has no jest/vitest configured (no config file, no test
// runner in frontend/package.json devDependencies) — per Phase 3
// instructions this does not introduce one. This file is written against
// node:assert only, so it runs standalone:
//
//   npx tsc lib/threshold-engine.ts lib/threshold-engine.test.ts \
//     --module commonjs --target es2020 --esModuleInterop \
//     --outDir <scratch-dir>
//   node <scratch-dir>/threshold-engine.test.js
//
// It follows the *.test.ts naming convention so it slots straight into
// whatever runner a future phase adopts (jest/vitest both auto-discover
// this pattern) without renaming.

import assert from "node:assert/strict";
import { resolveThreshold, type ThresholdStep } from "./threshold-engine";

function test(name: string, fn: () => void): void {
  fn();
  console.log(`ok - ${name}`);
}

// ok(base) / warn(80) / danger(90) — the shape StatCardWidget/GaugeWidget
// produce today once mapped onto a Grafana-style step ladder (see the
// mapping note at the bottom of threshold-engine.ts).
const okWarnDanger: ThresholdStep[] = [
  { value: null, color: "green" },
  { value: 80, color: "yellow" },
  { value: 90, color: "red" },
];

test("base case: a value below every real step resolves to the base step", () => {
  const r = resolveThreshold(50, okWarnDanger, "absolute");
  assert.equal(r.color, "green");
  assert.equal(r.level, 0);
});

test("boundary: value exactly equal to a step activates that step (>=, not >)", () => {
  const r = resolveThreshold(80, okWarnDanger, "absolute");
  assert.equal(r.color, "yellow");
  assert.equal(r.level, 1);
  // one unit below the boundary must NOT activate it
  const below = resolveThreshold(79.99, okWarnDanger, "absolute");
  assert.equal(below.color, "green");
});

test("highest cleared step wins when a value clears more than one step", () => {
  const r = resolveThreshold(95, okWarnDanger, "absolute");
  assert.equal(r.color, "red");
  assert.equal(r.level, 2);
});

test("percentage mode rescales each step against [min, max] before comparing", () => {
  const steps: ThresholdStep[] = [
    { value: null, color: "green" },
    { value: 50, color: "red" }, // 50% of 0..200 -> absolute 100
  ];
  const below = resolveThreshold(99, steps, "percentage", { min: 0, max: 200 });
  assert.equal(below.color, "green");
  const at = resolveThreshold(100, steps, "percentage", { min: 0, max: 200 });
  assert.equal(at.color, "red");
});

test("percentage mode works with a negative-to-positive range", () => {
  const steps: ThresholdStep[] = [
    { value: null, color: "ok" },
    { value: 50, color: "danger" }, // 50% of -50..50 -> absolute 0
  ];
  assert.equal(
    resolveThreshold(-0.1, steps, "percentage", { min: -50, max: 50 }).color,
    "ok",
  );
  assert.equal(
    resolveThreshold(0, steps, "percentage", { min: -50, max: 50 }).color,
    "danger",
  );
});

test("single-step ladder (base only) always resolves to the base color", () => {
  const steps: ThresholdStep[] = [{ value: null, color: "blue" }];
  assert.equal(resolveThreshold(-1e9, steps, "absolute").color, "blue");
  assert.equal(resolveThreshold(1e9, steps, "absolute").color, "blue");
  assert.equal(resolveThreshold(0, steps, "absolute").level, 0);
});

test("throws on an empty steps array instead of silently misreporting", () => {
  assert.throws(() => resolveThreshold(1, [], "absolute"));
});

console.log("threshold-engine.test.ts: all tests passed");
