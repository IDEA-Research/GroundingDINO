"use client";

import React from "react";

// One grid cell holding a stat card with the gauge that shares its PromQL
// folded beneath it (compact, h-16). Both children are the widgets' own
// rendered elements, so BOTH keep their data-widget-id roots — the
// evaluator still counts two widgets. Rendering is delegated by
// DashboardPreview (lib/renderer.tsx) to avoid an import cycle.
export function PairedStatGauge({
  stat,
  gauge,
}: {
  stat: React.ReactNode;
  gauge: React.ReactNode;
}) {
  return (
    <div data-paired="stat-gauge" className="flex h-full min-h-0 flex-col gap-2">
      <div className="min-h-0 flex-1">{stat}</div>
      <div className="h-16 shrink-0 rounded-lg border border-edge bg-surface px-3 py-1 shadow-card">
        {gauge}
      </div>
    </div>
  );
}
