// Shared chart theme — the SINGLE place where chart-facing hex colors live.
//
// Recharts / hand-rolled SVG need literal color strings (CSS classes don't
// reach SVG presentation attributes reliably), so this module mirrors the
// CSS-variable tokens in app/globals.css. KEEP THE TWO FILES IN SYNC.
//
// Clinical grammar (IEC 60601-1-8 + bedside-monitor convention):
//   - HR = monitor green, SpO2 = cyan, cerebral rSO2 = violet
//   - red is ONLY the high-priority alarm color, amber ONLY medium
//   - the series palette never contains red / yellow / amber / orange

import type { CSSProperties } from "react";

export const CHART_COLORS = {
  canvas: "#05080c",
  surface: "#0a0f16",
  surface2: "#101721",
  surface3: "#17202d",
  edge: "#1c2635",
  edgeStrong: "#2a374a",
  grid: "#141c28",
  ink: "#e8eef6",
  inkDim: "#94a3b4",
  inkFaint: "#5f6c7e",
  vitalHr: "#00e07a",
  vitalSpo2: "#22d3ee",
  vitalRso2: "#a78bfa",
  alarmHigh: "#ff4554",
  alarmMed: "#ffb020",
  alarmInfo: "#7c93b2",
  ok: "#34d399",
} as const;

// Series palette (mirrors --chart-1..8). No red, no yellow.
export const CHART_PALETTE: string[] = [
  CHART_COLORS.vitalHr, // 1 green
  CHART_COLORS.vitalSpo2, // 2 cyan
  CHART_COLORS.vitalRso2, // 3 violet
  "#60a5fa", // 4 blue
  "#2dd4bf", // 5 teal
  "#f472b6", // 6 pink
  "#d88cf0", // 7 orchid
  "#8fa0c4", // 8 slate blue
];

export function seriesColor(index: number): string {
  return CHART_PALETTE[((index % CHART_PALETTE.length) + CHART_PALETTE.length) % CHART_PALETTE.length];
}

export const FONT_MONO =
  '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, "Courier New", monospace';

// --- Recharts style objects (previously copy-pasted per widget) ------------

export const gridStroke = CHART_COLORS.grid;

export const axisTick = {
  fontSize: 12,
  fill: CHART_COLORS.inkFaint,
  fontFamily: FONT_MONO,
};

export const axisLine = { stroke: CHART_COLORS.edge };

export const tooltipContentStyle: CSSProperties = {
  background: CHART_COLORS.surface2,
  border: `1px solid ${CHART_COLORS.edgeStrong}`,
  borderRadius: 6,
  color: CHART_COLORS.ink,
  fontSize: 12,
  fontFamily: FONT_MONO,
  padding: "6px 10px",
};

export const tooltipLabelStyle: CSSProperties = {
  color: CHART_COLORS.inkDim,
};

// --- Vital-color mapping ----------------------------------------------------

// Map a widget to its conventional bedside-monitor color from its PromQL
// and/or title. Returns null when the signal is not a recognized vital —
// callers then fall back to encoding.color / the palette.
export function vitalColorFor(
  promql: string,
  title?: string | null,
): string | null {
  const hay = `${promql} ${title ?? ""}`.toLowerCase();
  if (/rso2|rso₂|cerebral/.test(hay)) return CHART_COLORS.vitalRso2;
  if (/spo2|spo₂|oxygen_saturation|pulse_ox/.test(hay))
    return CHART_COLORS.vitalSpo2;
  if (/heart_rate|heart rate|pulse_rate|(^|[^a-z])hr([^a-z]|$)/.test(hay))
    return CHART_COLORS.vitalHr;
  return null;
}

// --- Threshold color normalization (IEC 60601-1-8) --------------------------

// Spec thresholds carry free-form LLM-authored colors ("red", "#f59e0b",
// "orange"...). On a clinical display those must collapse onto the alarm
// grammar: red-family -> high alarm, yellow/amber/orange -> medium alarm,
// green-family -> normal/ok, anything else -> informational blue-gray.
// Thresholds VALUES stay spec-driven — only the color is normalized.
export function alarmColorForSpec(
  specColor: string | null | undefined,
): string | null {
  if (!specColor) return null;
  const s = specColor.trim().toLowerCase();
  if (/red|crimson|rose|scarlet|danger|critical/.test(s))
    return CHART_COLORS.alarmHigh;
  if (/amber|orange|yellow|gold|warn/.test(s)) return CHART_COLORS.alarmMed;
  if (/green|emerald|lime|teal|ok|normal|good/.test(s)) return CHART_COLORS.ok;
  const rgb = parseHexColor(s);
  if (rgb) {
    const [r, g, b] = rgb;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    if (max - min < 24) return CHART_COLORS.alarmInfo; // grayish
    const h = hueOf(r, g, b);
    if (h < 20 || h >= 340) return CHART_COLORS.alarmHigh;
    if (h < 70) return CHART_COLORS.alarmMed;
    if (h < 170) return CHART_COLORS.ok;
    return CHART_COLORS.alarmInfo;
  }
  return CHART_COLORS.alarmInfo;
}

function parseHexColor(s: string): [number, number, number] | null {
  const m = /^#?([0-9a-f]{6})$/.exec(s);
  if (m) {
    const h = m[1];
    return [
      parseInt(h.slice(0, 2), 16),
      parseInt(h.slice(2, 4), 16),
      parseInt(h.slice(4, 6), 16),
    ];
  }
  const m3 = /^#?([0-9a-f]{3})$/.exec(s);
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

// Sequential data colormaps (not series colors — the no-red/yellow rule is
// about series and alarm colors; a continuous ramp is standard for heatmaps).
export const HEATMAP_SCALES: Record<string, string[]> = {
  warm: ["#101721", "#854d0e", "#ca8a04", "#facc15", "#fef08a"],
  cool: ["#101721", "#1e3a5f", "#2563eb", "#60a5fa", "#bfdbfe"],
  viridis: ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
};

// Default is the calm blue ramp — a yellow-heavy default reads as a
// medium-alarm wall on a clinical dark screen.
export function heatmapScale(name?: string): string[] {
  if (name && HEATMAP_SCALES[name]) return HEATMAP_SCALES[name];
  return HEATMAP_SCALES.cool;
}

export function scaleColorAt(scale: string[], t: number): string {
  const [r, g, b] = scaleRgbAt(scale, t);
  return `rgb(${r},${g},${b})`;
}

// Legible in-cell text color for a given position on the ramp.
export function scaleTextColorAt(scale: string[], t: number): string {
  const [r, g, b] = scaleRgbAt(scale, t);
  const luma = 0.299 * r + 0.587 * g + 0.114 * b;
  return luma > 150 ? CHART_COLORS.canvas : CHART_COLORS.ink;
}

function scaleRgbAt(scale: string[], t: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, t));
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
