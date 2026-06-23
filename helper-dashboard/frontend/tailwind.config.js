/** @type {import('tailwindcss').Config} */
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
        panel: "#0f172a",
        panel2: "#111827",
        muted: "#475569",
        accent: "#38bdf8",
      },
    },
  },
  plugins: [],
};
