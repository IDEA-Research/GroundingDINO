/** @type {import('tailwindcss').Config} */

// Every color maps to a CSS-variable token defined in app/globals.css
// (R G B triplets, so Tailwind opacity modifiers like bg-danger/10 work).
// Light values live on :root, dark values on :root[data-theme="dark"], so no
// `dark:` variants are needed anywhere. Chart code resolves the same
// variables at runtime (lib/chart-theme.ts).
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
        accent: {
          DEFAULT: token("--accent"),
          ink: token("--accent-ink"),
        },
        focus: token("--focus"),
        // Status colors — meaning only (never for data series).
        ok: token("--ok"),
        warn: token("--warn"),
        danger: token("--danger"),
        info: token("--info"),
        // Data palette — series / vitals (HR data-1, SpO2 data-2, rSO2 data-3).
        data: {
          1: token("--data-1"),
          2: token("--data-2"),
          3: token("--data-3"),
          4: token("--data-4"),
          5: token("--data-5"),
          6: token("--data-6"),
        },
      },
      borderRadius: {
        sm: "var(--radius-sm)",
        DEFAULT: "var(--radius-sm)",
        md: "var(--radius)",
        lg: "var(--radius-lg)",
      },
      boxShadow: {
        card: "var(--shadow-card)",
        pop: "var(--shadow-pop)",
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
        // The ONE sanctioned sub-12px size — Latin-only badges, never CJK.
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
      },
      minHeight: {
        ctl: "2.25rem",
        "ctl-sm": "2rem",
        touch: "2.5rem",
      },
    },
  },
  plugins: [],
};
