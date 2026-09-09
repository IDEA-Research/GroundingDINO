// Self-hosted fonts (bundled at build time — the deployment LAN has no
// internet, so next/font/google is not an option). Inter for UI text with
// Traditional-Chinese system fallbacks; JetBrains Mono for numerals,
// PromQL, and logs.
import "@fontsource/inter/400.css";
import "@fontsource/inter/500.css";
import "@fontsource/inter/600.css";
import "@fontsource/inter/700.css";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/700.css";
import "./globals.css";
import type { Metadata, Viewport } from "next";
import React from "react";

export const metadata: Metadata = {
  title: "Helper Dashboard",
  description: "與 Helper 對話,建立 Prometheus 監控儀表板。",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f5f7fa" },
    { media: "(prefers-color-scheme: dark)", color: "#0f172a" },
  ],
};

// Runs BEFORE first paint (inline, blocking) so a stored dark preference
// never flashes the light palette. Mirrors lib/theme.ts: key "hd.theme"
// holds light | dark | system; anything else = system preference.
const THEME_BOOT_SCRIPT =
  "(function(){try{var t=null;try{t=window.localStorage.getItem('hd.theme');}catch(e){}" +
  "var m=false;try{m=window.matchMedia('(prefers-color-scheme: dark)').matches;}catch(e){}" +
  "var r=(t==='dark'||t==='light')?t:(m?'dark':'light');" +
  "var h=document.documentElement;h.setAttribute('data-theme',r);h.style.colorScheme=r;}catch(e){}})();";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-Hant" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOT_SCRIPT }} />
      </head>
      <body className="h-screen font-sans text-sm text-ink">{children}</body>
    </html>
  );
}
