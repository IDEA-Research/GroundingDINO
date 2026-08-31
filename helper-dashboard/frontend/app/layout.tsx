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
import type { Metadata } from "next";
import React from "react";

export const metadata: Metadata = {
  title: "Helper Dashboard",
  description: "與 Helper 對話,建立 Prometheus 監控儀表板。",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-Hant">
      <body className="h-screen font-sans">{children}</body>
    </html>
  );
}
