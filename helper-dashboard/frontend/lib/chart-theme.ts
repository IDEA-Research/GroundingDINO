"use client";

// Chart theme — resolved at RUNTIME from the CSS variables in
// app/globals.css, so Recharts / hand-rolled SVG (which need literal color
// strings) follow the active theme without a second hand-maintained
// palette. `getChartTheme()` memoizes per theme and re-reads on the
// "hd:theme" event; `useChartTheme()` subscribes React components.
//
// The only hex literals in this file are SSR/no-DOM FALLBACKS and the
// viridis colormap constant (a perceptual data ramp, not a UI token).
//
// Grammar: accent = interactive; ok/warn/danger/info = STATUS only;
// data1..6 = series (HR data1 teal, SpO2 data2 violet, rSO2 data3 indigo).

import type { CSSProperties } from "react";
import { useSyncExternalStore } from "react";

import { THEME_EVENT, type ResolvedTheme } from "./theme";

export interface ChartColors {
  canvas: string;
  surface: string;
  surface2: string;
  surface3: string;
  edge: string;
  edgeStrong: string;
  grid: string;
  ink: string;
  inkDim: string;
  inkFaint: string;
  accent: string;
  ok: string;
  warn: string;
  danger: string;
  info: string;
  data1: string;
  data2: string;
  data3: string;
  data4: string;
  data5: string;
  data6: string;
}

const VAR_NAMES: Record<keyof ChartColors, string> = {
  canvas: "--canvas",
  surface: "--surface",
  surface2: "--surface-2",
  surface3: "--surface-3",
  edge: "--edge",
  edgeStrong: "--edge-strong",
  grid: "--grid",
  ink: "--ink",
  inkDim: "--ink-dim",
  inkFaint: "--ink-faint",
  accent: "--accent",
  ok: "--ok",
  warn: "--warn",
  danger: "--danger",
  info: "--info",
  data1: "--data-1",
  data2: "--data-2",
  data3: "--data-3",
  data4: "--data-4",
  data5: "--data-5",
  data6: "--data-6",
};

// Fallbacks (used on the server and when a variable is missing). They
// mirror globals.css; the live values always come from getComputedStyle.
const FALLBACK_LIGHT: ChartColors = {
  canvas: "#f5f7fa",
  surface: "#ffffff",
  surface2: "#f1f4f8",
  surface3: "#e6ebf2",
  edge: "#d9e0ea",
  edgeStrong: "#b8c4d3",
  grid: "#e7ecf2",
  ink: "#0f172a",
  inkDim: "#475569",
  inkFaint: "#596a80",
  accent: "#2563eb",
  ok: "#15803d",
  warn: "#b45309",
  danger: "#dc2626",
  info: "#475569",
  data1: "#0f766e",
  data2: "#7c3aed",
  data3: "#4338ca",
  data4: "#0369a1",
  data5: "#a21caf",
  data6: "#64748b",
};

const FALLBACK_DARK: ChartColors = {
  canvas: "#0f172a",
  surface: "#172033",
  surface2: "#1f2a40",
  surface3: "#283550",
  edge: "#2f3d57",
  edgeStrong: "#43557a",
  grid: "#223050",
  ink: "#e8eef6",
  inkDim: "#a6b3c8",
  inkFaint: "#8593a8",
  accent: "#60a5fa",
  ok: "#34d399",
  warn: "#fbbf24",
  danger: "#f87171",
  info: "#93a7c4",
  data1: "#2dd4bf",
  data2: "#a78bfa",
  data3: "#818cf8",
  data4: "#38bdf8",
  data5: "#e879f9",
  data6: "#94a3b8",
};

export const FALLBACK_COLORS: Record<ResolvedTheme, ChartColors> = {
  light: FALLBACK_LIGHT,
  dark: FALLBACK_DARK,
};

export const FONT_MONO =
  '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, "Courier New", monospace';

export interface ChartTheme {
  name: ResolvedTheme;
  colors: ChartColors;
  // Series palette (data1..6) — never green/amber/red.
  palette: string[];
  gridStroke: string;
  axisTick: { fontSize: number; fill: string; fontFamily: string };
  axisLine: { stroke: string };
  tooltipContentStyle: CSSProperties;
  tooltipLabelStyle: CSSProperties;
  legendStyle: CSSProperties;
}

function buildTheme(name: ResolvedTheme, colors: ChartColors): ChartTheme {
  return {
    name,
    colors,
    palette: [
      colors.data1,
      colors.data2,
      colors.data3,
      colors.data4,
      colors.data5,
      colors.data6,
    ],
    gridStroke: colors.grid,
    axisTick: { fontSize: 12, fill: colors.inkFaint, fontFamily: FONT_MONO },
    axisLine: { stroke: colors.edge },
    tooltipContentStyle: {
      background: colors.surface2,
      border: `1px solid ${colors.edgeStrong}`,
      borderRadius: 6,
      color: colors.ink,
      fontSize: 13,
      fontFamily: FONT_MONO,
      padding: "6px 10px",
      boxShadow: "none",
    },
    tooltipLabelStyle: { color: colors.inkDim },
    legendStyle: { fontSize: 12, color: colors.inkDim },
  };
}

const SERVER_THEME: ChartTheme = buildTheme("light", FALLBACK_LIGHT);

// "232 238 246" (globals.css triplet) -> "#e8eef6". Accepts commas too.
function tripletToHex(raw: string): string | null {
  const parts = raw
    .trim()
    .split(/[\s,]+/)
    .filter(Boolean)
    .map((p) => Number(p));
  if (parts.length < 3 || parts.some((n) => !Number.isFinite(n))) return null;
  const clamp = (n: number) => Math.max(0, Math.min(255, Math.round(n)));
  return (
    "#" +
    [parts[0], parts[1], parts[2]]
      .map((n) => clamp(n).toString(16).padStart(2, "0"))
      .join("")
  );
}

function currentThemeName(): ResolvedTheme {
  if (typeof document === "undefined") return "light";
  return document.documentElement.getAttribute("data-theme") === "dark"
    ? "dark"
    : "light";
}

function readColors(name: ResolvedTheme): ChartColors {
  const fallback = FALLBACK_COLORS[name];
  if (typeof window === "undefined" || typeof document === "undefined") {
    return fallback;
  }
  let style: CSSStyleDeclaration | null = null;
  try {
    style = window.getComputedStyle(document.documentElement);
  } catch {
    style = null;
  }
  if (!style) return fallback;
  const out = { ...fallback };
  (Object.keys(VAR_NAMES) as (keyof ChartColors)[]).forEach((key) => {
    const hex = tripletToHex(style!.getPropertyValue(VAR_NAMES[key]));
    if (hex) out[key] = hex;
  });
  return out;
}

let cache: { name: ResolvedTheme; theme: ChartTheme } | null = null;
let listening = false;
const listeners = new Set<() => void>();

function invalidate() {
  cache = null;
  listeners.forEach((l) => l());
}

function ensureListening() {
  if (listening || typeof window === "undefined") return;
  listening = true;
  window.addEventListener(THEME_EVENT, invalidate);
}

// Memoized per theme; identity is stable until the theme changes, so
// Recharts style props only change when colors actually do.
export function getChartTheme(): ChartTheme {
  if (typeof document === "undefined") return SERVER_THEME;
  ensureListening();
  const name = currentThemeName();
  if (cache && cache.name === name) return cache.theme;
  const theme = buildTheme(name, readColors(name));
  cache = { name, theme };
  return theme;
}

function subscribe(cb: () => void): () => void {
  ensureListening();
  listeners.add(cb);
  return () => {
    listeners.delete(cb);
  };
}

function getServerTheme(): ChartTheme {
  return SERVER_THEME;
}

export function useChartTheme(): ChartTheme {
  return useSyncExternalStore(subscribe, getChartTheme, getServerTheme);
}

// --- Series colors ----------------------------------------------------------

export function seriesPalette(colors: ChartColors = getChartTheme().colors) {
  return [
    colors.data1,
    colors.data2,
    colors.data3,
    colors.data4,
    colors.data5,
    colors.data6,
  ];
}

export function seriesColor(
  index: number,
  colors: ChartColors = getChartTheme().colors,
): string {
  const p = seriesPalette(colors);
  return p[((index % p.length) + p.length) % p.length];
}

// --- Vital mapping (HR -> data1, SpO2 -> data2, rSO2 -> data3) -------------

export type VitalKey = "data1" | "data2" | "data3";

export function vitalKeyFor(
  promql: string | null | undefined,
  title?: string | null,
): VitalKey | null {
  const hay = `${promql ?? ""} ${title ?? ""}`.toLowerCase();
  if (/rso2|rso₂|cerebral/.test(hay)) return "data3";
  if (/spo2|spo₂|oxygen_saturation|pulse_ox|血氧/.test(hay)) return "data2";
  if (/heart_rate|heart rate|pulse_rate|(^|[^a-z])hr([^a-z]|$)|心率|心跳/.test(hay))
    return "data1";
  return null;
}

// Returns null when the signal is not a recognized vital — callers then
// fall back to encoding.color / the palette.
export function vitalColorFor(
  promql: string | null | undefined,
  title?: string | null,
  colors: ChartColors = getChartTheme().colors,
): string | null {
  const key = vitalKeyFor(promql, title);
  return key ? colors[key] : null;
}

// --- Threshold severity (status colors) -------------------------------------

export type Severity = "danger" | "warn" | "ok" | "info";

const SEVERITY_RANK: Record<Severity, number> = {
  danger: 3,
  warn: 2,
  ok: 1,
  info: 0,
};

export function severityRank(sev: Severity): number {
  return SEVERITY_RANK[sev];
}

// Spec thresholds carry free-form LLM-authored colors ("red", "#f59e0b",
// "orange"...). They collapse onto the status grammar: red-family ->
// danger, yellow/amber/orange -> warn, green-family -> ok, else info.
// Threshold VALUES are never touched — only how the color is read.
export function severityForSpec(
  specColor: string | null | undefined,
): Severity | null {
  if (!specColor) return null;
  const s = specColor.trim().toLowerCase();
  if (/red|crimson|rose|scarlet|danger|critical|high/.test(s)) return "danger";
  if (/amber|orange|yellow|gold|warn|medium/.test(s)) return "warn";
  if (/green|emerald|lime|teal|ok|normal|good/.test(s)) return "ok";
  const rgb = parseHexColor(s);
  if (rgb) {
    const [r, g, b] = rgb;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    if (max - min < 24) return "info"; // grayish
    const h = hueOf(r, g, b);
    if (h < 20 || h >= 340) return "danger";
    if (h < 70) return "warn";
    if (h < 170) return "ok";
    return "info";
  }
  return "info";
}

// Finer ordering used ONLY by the threshold-direction inference. Mirrors
// backend/app/specs/threshold_semantics.color_severity EXACTLY: exact
// colour-name matches (red 3 > amber/orange 2 > yellow 1 > other 0), hex
// (#rgb/#rgba/#rrggbb/#rrggbbaa) ranked by hue with greys (sat < 0.25) = 0.
const SEVERITY_NAME_RANK: Record<string, number> = {
  red: 3,
  crimson: 3,
  rose: 3,
  danger: 3,
  critical: 3,
  amber: 2,
  orange: 2,
  warn: 2,
  warning: 2,
  yellow: 1,
  gold: 1,
};

export function severityScore(specColor: string | null | undefined): number {
  if (!specColor) return 0;
  const text = specColor.trim().toLowerCase();
  if (text in SEVERITY_NAME_RANK) return SEVERITY_NAME_RANK[text];
  const m = /^#([0-9a-f]{3,8})$/.exec(text);
  if (!m) return 0;
  let body = m[1];
  if (body.length === 3 || body.length === 4) {
    body = body
      .slice(0, 3)
      .split("")
      .map((ch) => ch + ch)
      .join("");
  } else if (body.length === 6 || body.length === 8) {
    body = body.slice(0, 6);
  } else {
    return 0;
  }
  const r = parseInt(body.slice(0, 2), 16) / 255;
  const g = parseInt(body.slice(2, 4), 16) / 255;
  const b = parseInt(body.slice(4, 6), 16) / 255;
  const mx = Math.max(r, g, b);
  const mn = Math.min(r, g, b);
  const delta = mx - mn;
  if (delta === 0) return 0;
  const sat = mx ? delta / mx : 0;
  if (sat < 0.25) return 0;
  const hue = hueOf(r, g, b);
  if (hue < 15 || hue >= 340) return 3;
  if (hue < 45) return 2;
  if (hue < 70) return 1;
  return 0;
}

export function colorForSeverity(
  sev: Severity,
  colors: ChartColors = getChartTheme().colors,
): string {
  switch (sev) {
    case "danger":
      return colors.danger;
    case "warn":
      return colors.warn;
    case "ok":
      return colors.ok;
    default:
      return colors.info;
  }
}

export function alarmColorForSpec(
  specColor: string | null | undefined,
  colors: ChartColors = getChartTheme().colors,
): string | null {
  const sev = severityForSpec(specColor);
  return sev ? colorForSeverity(sev, colors) : null;
}

export function parseHexColor(s: string): [number, number, number] | null {
  const m = /^#?([0-9a-f]{6})$/i.exec(s.trim());
  if (m) {
    const h = m[1];
    return [
      parseInt(h.slice(0, 2), 16),
      parseInt(h.slice(2, 4), 16),
      parseInt(h.slice(4, 6), 16),
    ];
  }
  const m3 = /^#?([0-9a-f]{3})$/i.exec(s.trim());
  if (m3) {
    const h = m3[1];
    return [
      parseInt(h[0] + h[0], 16),
      parseInt(h[1] + h[1], 16),
      parseInt(h[2] + h[2], 16),
    ];
  }
  return null;
}

function hueOf(r: number, g: number, b: number): number {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const d = max - min;
  if (d === 0) return 0;
  let h: number;
  if (max === r) h = ((g - b) / d) % 6;
  else if (max === g) h = (b - r) / d + 2;
  else h = (r - g) / d + 4;
  h *= 60;
  return h < 0 ? h + 360 : h;
}

// --- Heatmap color scales ---------------------------------------------------

function mixHex(a: string, b: string, t: number): string {
  const ca = parseHexColor(a) ?? [0, 0, 0];
  const cb = parseHexColor(b) ?? [0, 0, 0];
  const c = ca.map((v, i) => Math.round(v + (cb[i] - v) * t));
  return "#" + c.map((n) => n.toString(16).padStart(2, "0")).join("");
}

// Standard perceptual colormap (data ramp constant, not a UI token).
const VIRIDIS = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"];

// Sequential ramps start at the card's raised surface so an empty/low
// cell reads as "nothing" in both themes. "cool" is the default.
export function heatmapScale(
  name?: string,
  colors: ChartColors = getChartTheme().colors,
): string[] {
  if (name === "viridis") return VIRIDIS;
  if (name === "warm") {
    return [
      colors.surface2,
      mixHex(colors.surface2, colors.warn, 0.35),
      mixHex(colors.surface2, colors.warn, 0.7),
      colors.warn,
      mixHex(colors.warn, colors.surface, 0.45),
    ];
  }
  return [
    colors.surface2,
    mixHex(colors.surface2, colors.data4, 0.35),
    mixHex(colors.surface2, colors.data4, 0.7),
    colors.data4,
    mixHex(colors.data4, colors.surface, 0.45),
  ];
}

export function scaleColorAt(scale: string[], t: number): string {
  const [r, g, b] = scaleRgbAt(scale, t);
  return `rgb(${r},${g},${b})`;
}

// Legible in-cell text color for a given position on the ramp.
export function scaleTextColorAt(
  scale: string[],
  t: number,
  colors: ChartColors = getChartTheme().colors,
): string {
  const [r, g, b] = scaleRgbAt(scale, t);
  const luma = 0.299 * r + 0.587 * g + 0.114 * b;
  return luma > 150 ? FALLBACK_LIGHT.ink : luma < 90 ? FALLBACK_DARK.ink : colors.ink;
}

function scaleRgbAt(scale: string[], t: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0));
  const idx = clamped * (scale.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, scale.length - 1);
  const frac = idx - lo;
  const a = parseHexColor(scale[lo]) ?? [0, 0, 0];
  const b = parseHexColor(scale[hi]) ?? [0, 0, 0];
  return [
    Math.round(a[0] + (b[0] - a[0]) * frac),
    Math.round(a[1] + (b[1] - a[1]) * frac),
    Math.round(a[2] + (b[2] - a[2]) * frac),
  ];
}
