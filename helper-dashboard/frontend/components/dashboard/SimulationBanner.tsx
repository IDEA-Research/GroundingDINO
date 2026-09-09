"use client";

import React, { useContext } from "react";

import { Badge } from "@/components/ui";
import { DevModeContext } from "@/lib/dev-mode";

// Simulation is never silent: whenever any widget reports `simulated` or
// the datasource status says a scenario is active, this banner sits
// above the dashboard (sticky) so a screenshot can never pass synthetic
// data off as a real patient's.

export const SCENARIO_LABELS: Record<string, string> = {
  off: "關閉",
  normal: "正常",
  nodata: "無資料",
  stale: "資料過期",
  partial: "部分缺值",
  offline: "來源離線",
  error: "查詢錯誤",
};

export function scenarioLabel(scenario: string | null | undefined): string {
  if (!scenario) return "未知";
  return SCENARIO_LABELS[scenario] ?? scenario;
}

export function SimulationBanner({
  active,
  scenario,
}: {
  active: boolean;
  scenario: string | null;
}) {
  const { unlocked } = useContext(DevModeContext);
  if (!active) return null;
  return (
    <div
      role="status"
      data-simulation-banner
      className="sticky top-0 z-30 flex flex-wrap items-center gap-2 border-b border-warn/50 bg-warn/15 px-3 py-2 text-sm text-ink"
    >
      <Badge tone="warn">模擬</Badge>
      <span className="font-medium">模擬資料（測試用）</span>
      <span aria-hidden="true" className="text-ink-faint">
        ·
      </span>
      <span>
        情境：{scenarioLabel(scenario)}
        {unlocked && scenario ? (
          <span className="ml-1 font-mono text-xs text-ink-dim">({scenario})</span>
        ) : null}
      </span>
    </div>
  );
}
