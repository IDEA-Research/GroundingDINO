"use client";

// ---------------------------------------------------------------------------
// Continuous value -> color interpolation for sequential ("heat") scales.
//
// `colorScaleAt` is a genuine per-value interpolation, not a fixed set of
// discrete buckets: every t in [0,1] maps to its own blended color, found
// by locating the two "stops" that bracket t and blending between them.
// With N stops there are N-1 continuous segments — e.g. for
// [ok, warn, danger], t=0 is pure ok, t=0.5 is pure warn, t=1 is pure
// danger, and every value in between (0.13, 0.5, 0.87, ...) gets its own
// unique blended color.
//
// The blend itself happens in HSL space rather than RGB. Interpolating
// green (#15803d) straight to amber (#b45309) in RGB space cuts a
// diagonal through the color cube and passes through a muddy olive/brown
// midpoint — both R and G land mid-range with no saturated hue at all.
// Converting to HSL and interpolating hue along the *shortest* arc
// (green ~142deg -> amber ~26deg -> red ~0deg, all trending downward)
// walks through true yellow/gold at the midpoint instead, which is the
// "Green-Yellow-Red" sequential look this scale is meant to have.
// ---------------------------------------------------------------------------

export type ColorStop = string; // hex color, e.g. "#15803d"

interface Hsl {
  h: number; // 0..360
  s: number; // 0..1
  l: number; // 0..1
}

function parseHex(hex: string): [number, number, number] {
  const m = /^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return [0, 0, 0];
  let body = m[1];
  if (body.length === 3) {
    body = body
      .split("")
      .map((c) => c + c)
      .join("");
  }
  return [
    parseInt(body.slice(0, 2), 16),
    parseInt(body.slice(2, 4), 16),
    parseInt(body.slice(4, 6), 16),
  ];
}

function rgbToHsl(r: number, g: number, b: number): Hsl {
  const rn = r / 255;
  const gn = g / 255;
  const bn = b / 255;
  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const l = (max + min) / 2;
  const d = max - min;
  if (d === 0) return { h: 0, s: 0, l };
  const s = d / (1 - Math.abs(2 * l - 1));
  let h: number;
  if (max === rn) h = ((gn - bn) / d) % 6;
  else if (max === gn) h = (bn - rn) / d + 2;
  else h = (rn - gn) / d + 4;
  h *= 60;
  if (h < 0) h += 360;
  return { h, s, l };
}

function hslToRgb({ h, s, l }: Hsl): [number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const hp = ((h % 360) + 360) % 360 / 60;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  let [r1, g1, b1] = [0, 0, 0];
  if (hp < 1) [r1, g1, b1] = [c, x, 0];
  else if (hp < 2) [r1, g1, b1] = [x, c, 0];
  else if (hp < 3) [r1, g1, b1] = [0, c, x];
  else if (hp < 4) [r1, g1, b1] = [0, x, c];
  else if (hp < 5) [r1, g1, b1] = [x, 0, c];
  else [r1, g1, b1] = [c, 0, x];
  const m = l - c / 2;
  return [
    Math.round((r1 + m) * 255),
    Math.round((g1 + m) * 255),
    Math.round((b1 + m) * 255),
  ];
}

function hexToHsl(hex: string): Hsl {
  const [r, g, b] = parseHex(hex);
  return rgbToHsl(r, g, b);
}

// Interpolate hue along the SHORT way around the wheel: 350deg -> 10deg
// blends forward through 0/360 (delta +20), not backwards through 180.
function lerpHue(h1: number, h2: number, t: number): number {
  let delta = h2 - h1;
  if (delta > 180) delta -= 360;
  if (delta < -180) delta += 360;
  return (((h1 + delta * t) % 360) + 360) % 360;
}

function lerpHsl(a: Hsl, b: Hsl, t: number): Hsl {
  return {
    h: lerpHue(a.h, b.h, t),
    s: a.s + (b.s - a.s) * t,
    l: a.l + (b.l - a.l) * t,
  };
}

function rgbAtScale(stops: ColorStop[], t: number): [number, number, number] {
  if (stops.length === 0) return [0, 0, 0];
  if (stops.length === 1) return parseHex(stops[0]);
  const clamped = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0));
  const scaled = clamped * (stops.length - 1);
  const lo = Math.floor(scaled);
  const hi = Math.min(lo + 1, stops.length - 1);
  const frac = scaled - lo;
  const hsl = lerpHsl(hexToHsl(stops[lo]), hexToHsl(stops[hi]), frac);
  return hslToRgb(hsl);
}

/**
 * Map t (0..1) to a CSS color by walking the ordered `stops` list and
 * interpolating (in HSL space) between the two stops bracketing t.
 */
export function colorScaleAt(stops: ColorStop[], t: number): string {
  const [r, g, b] = rgbAtScale(stops, t);
  return `rgb(${r}, ${g}, ${b})`;
}

/** Normalize a raw value into a clamped 0..1 fraction of a [min, max] domain. */
export function normalize(value: number, min: number, max: number): number {
  const range = max - min || 1;
  return Math.max(0, Math.min(1, (value - min) / range));
}

/**
 * The project's semantic low -> mid -> high sequential ramp: ok (green)
 * -> warn (amber) -> danger (red). Callers pass the live theme colors so
 * light/dark mode both resolve through the real CSS-variable-backed
 * tokens — this module never invents a hex value of its own for the
 * scale's endpoints.
 */
export function statusColorScale(colors: {
  ok: string;
  warn: string;
  danger: string;
}): ColorStop[] {
  return [colors.ok, colors.warn, colors.danger];
}

// Fixed contrast-overlay text colors (not scale endpoints): the same
// near-black / near-white pair chart-theme.ts's own scaleTextColorAt uses
// as its light/dark fallback ink, kept in sync here by value.
const CONTRAST_DARK_INK = "#0f172a";
const CONTRAST_LIGHT_INK = "#e8eef6";

/**
 * Legible in-cell text color for a given position on the scale: dark ink
 * over light/bright cells, light ink over dark/saturated cells, and the
 * theme's own ink color in the mid-luma band where either works.
 */
export function textColorForScale(
  stops: ColorStop[],
  t: number,
  themeInk: string,
): string {
  const [r, g, b] = rgbAtScale(stops, t);
  const luma = 0.299 * r + 0.587 * g + 0.114 * b;
  if (luma > 150) return CONTRAST_DARK_INK;
  if (luma < 90) return CONTRAST_LIGHT_INK;
  return themeInk;
}

/** Sample a scale at `count` evenly spaced points, for legend gradients. */
export function sampleScale(stops: ColorStop[], count = 12): string[] {
  const n = Math.max(2, count);
  return Array.from({ length: n }, (_, i) => colorScaleAt(stops, i / (n - 1)));
}
