/** @type {import('tailwindcss').Config} */

// Every color maps to a CSS-variable token defined in app/globals.css
// (R G B triplets, so Tailwind opacity modifiers like bg-alarm-high/10
// work). The hex mirror for SVG/Recharts is frontend/lib/chart-theme.ts.
const token = (name) => `rgb(var(${name}) / <alpha-value>)`;

module.exports = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./widget-toolkit/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: token("--canvas"),
        surface: {
          DEFAULT: token("--surface"),
          2: token("--surface-2"),
          3: token("--surface-3"),
        },
        edge: {
          DEFAULT: token("--edge"),
          strong: token("--edge-strong"),
        },
        grid: token("--grid"),
        ink: {
          DEFAULT: token("--ink"),
          dim: token("--ink-dim"),
          faint: token("--ink-faint"),
        },
        vital: {
          hr: token("--vital-hr"),
          spo2: token("--vital-spo2"),
          rso2: token("--vital-rso2"),
        },
        alarm: {
          high: token("--alarm-high"),
          med: token("--alarm-med"),
          info: token("--alarm-info"),
        },
        ok: token("--ok"),
        chart: {
          1: token("--chart-1"),
          2: token("--chart-2"),
          3: token("--chart-3"),
          4: token("--chart-4"),
          5: token("--chart-5"),
          6: token("--chart-6"),
          7: token("--chart-7"),
          8: token("--chart-8"),
        },
        ring: token("--ring"),
      },
      fontFamily: {
        sans: [
          "Inter",
          '"Noto Sans TC"',
          '"PingFang TC"',
          '"Microsoft JhengHei"',
          "system-ui",
          "-apple-system",
          "sans-serif",
        ],
        mono: [
          '"JetBrains Mono"',
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          '"Courier New"',
          "monospace",
        ],
      },
      fontSize: {
        // The ONE sanctioned sub-12px size — tiny badges/annotations only.
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
      },
    },
  },
  plugins: [],
};
