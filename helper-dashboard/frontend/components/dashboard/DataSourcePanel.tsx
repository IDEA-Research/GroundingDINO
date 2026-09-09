"use client";

import React, { useContext, useEffect, useState } from "react";

import { Badge, Button, Disclosure, Drawer, Field, useToast } from "@/components/ui";
import { api, describeApiError } from "@/lib/api";
import { DevModeContext } from "@/lib/dev-mode";
import {
  SIMULATION_SCENARIOS,
  type Capabilities,
  type SimulationScenario,
} from "@/lib/spec-schema";
import type { BackendStatusState } from "@/lib/useBackendStatus";

import { formatAge, formatClock, type DatasourceState } from "./DataStatusStrip";
import { SCENARIO_LABELS, scenarioLabel } from "./SimulationBanner";

// 「檢查資料來源」 drawer: every hop between the browser and the bedside
// exporter in plain language, the last measurement, and the freshness
// budget. Hosts, error strings, metric names and the simulation switch
// live under 「技術細節」, rendered ONLY when the developer workspace is
// unlocked (the backend still gates the simulation route itself).

type Tone = "ok" | "warn" | "danger" | "neutral" | "info";

const METRIC_LABELS: Record<string, string> = {
  patient_heart_rate_bpm: "心率",
  patient_spo2_percentage: "血氧 SpO2",
  cerebral_rso2_left_percentage: "左側腦部血氧 rSO2",
  cerebral_rso2_right_percentage: "右側腦部血氧 rSO2",
};

export function metricLabel(name: string): string {
  return METRIC_LABELS[name] ?? name;
}

interface Hop {
  name: string;
  tone: Tone;
  text: string;
}

function hopsFor(backend: BackendStatusState, ds: DatasourceState): Hop[] {
  const hops: Hop[] = [];
  hops.push(
    backend.status === "up"
      ? { name: "伺服器", tone: "ok", text: "Helper Dashboard 後端連線正常。" }
      : backend.status === "down"
        ? { name: "伺服器", tone: "danger", text: "無法連線到後端伺服器；所有查詢都會失敗。" }
        : { name: "伺服器", tone: "neutral", text: "正在檢查後端連線…" },
  );
  const s = ds.status;
  if (!s) {
    hops.push({
      name: "資料來源",
      tone: ds.loading ? "neutral" : "warn",
      text: ds.loading
        ? "正在取得資料來源狀態…"
        : `無法取得資料來源狀態${ds.error ? `（${ds.error}）` : ""}。`,
    });
    return hops;
  }
  hops.push(
    s.prometheus.reachable
      ? { name: "Prometheus", tone: "ok", text: "時間序列資料庫可連線。" }
      : { name: "Prometheus", tone: "danger", text: "時間序列資料庫無法連線；儀表板無法取得任何資料。" },
  );
  hops.push(
    s.exporter.health === "up"
      ? { name: "量測輸出端", tone: "ok", text: "床邊量測輸出端正在被擷取。" }
      : s.exporter.health === "down"
        ? { name: "量測輸出端", tone: "danger", text: "床邊量測輸出端離線；資料不會再更新。" }
        : { name: "量測輸出端", tone: "warn", text: "無法判斷量測輸出端狀態。" },
  );
  const n = s.clinical_metrics_present.length;
  hops.push(
    n > 0
      ? {
          name: "臨床量測指標",
          tone: "ok",
          text: `已有 ${n} 項：${s.clinical_metrics_present.map(metricLabel).join("、")}。`,
        }
      : { name: "臨床量測指標", tone: "warn", text: "尚未看到任何臨床量測指標。" },
  );
  if (s.last_sample_ts) {
    const over =
      typeof s.age_seconds === "number" && s.age_seconds > s.freshness_budget_seconds;
    hops.push({
      name: "最後量測",
      tone: over ? "warn" : "ok",
      text: `${formatClock(s.last_sample_ts)}（${formatAge(s.age_seconds)}）${
        over ? `，已超過 ${s.freshness_budget_seconds} 秒的更新預算，資料視為過期。` : "，在更新預算內。"
      }`,
    });
  } else {
    hops.push({ name: "最後量測", tone: "warn", text: "尚無任何量測時間。" });
  }
  if (s.simulation?.active) {
    hops.push({
      name: "模擬",
      tone: "warn",
      text: `目前顯示的是模擬資料（情境：${scenarioLabel(s.simulation.scenario)}），不是真實病患量測。`,
    });
  }
  return hops;
}

const DOT: Record<Tone, string> = {
  ok: "bg-ok",
  warn: "bg-warn",
  danger: "bg-danger",
  neutral: "bg-ink-faint",
  info: "bg-info",
};

export function DataSourcePanel({
  open,
  onClose,
  backend,
  datasource,
  capabilities,
}: {
  open: boolean;
  onClose: () => void;
  backend: BackendStatusState;
  datasource: DatasourceState;
  capabilities: Capabilities | null;
}) {
  const { unlocked } = useContext(DevModeContext);
  const toast = useToast();
  const [scenario, setScenario] = useState<SimulationScenario>("off");
  const [applying, setApplying] = useState(false);

  const hops = hopsFor(backend, datasource);
  const s = datasource.status;

  // Keep the scenario selector in step with what the server reports so
  // 套用 never silently turns an active scenario off.
  const liveActive = s?.simulation?.active ?? false;
  const liveScenario = s?.simulation?.scenario ?? null;
  useEffect(() => {
    if (liveActive && liveScenario && (SIMULATION_SCENARIOS as string[]).includes(liveScenario)) {
      setScenario(liveScenario as SimulationScenario);
    } else if (!liveActive) {
      setScenario("off");
    }
  }, [liveActive, liveScenario]);

  async function applyScenario() {
    setApplying(true);
    try {
      const r = await api.simulationSet(scenario);
      toast.show({
        title: r.active ? "模擬已啟用" : "模擬已關閉",
        description: r.active ? `情境：${scenarioLabel(r.scenario)}` : undefined,
        tone: r.active ? "warn" : "ok",
        durationMs: 4000,
      });
      datasource.refresh();
    } catch (e) {
      toast.show({
        title: "無法切換模擬",
        description: describeApiError(e),
        tone: "danger",
        durationMs: 6000,
      });
    } finally {
      setApplying(false);
    }
  }

  return (
    <Drawer
      open={open}
      onClose={onClose}
      side="right"
      title="資料來源狀態"
      footer={
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-ink-dim">
            {datasource.lastFetched ? `更新於 ${formatClock(datasource.lastFetched)}` : ""}
          </span>
          <div className="flex gap-2">
            <Button type="button" variant="secondary" size="sm" onClick={() => datasource.refresh()}>
              重新檢查
            </Button>
            <Button type="button" variant="ghost" size="sm" onClick={onClose}>
              關閉
            </Button>
          </div>
        </div>
      }
    >
      <div className="space-y-4 p-3 text-sm">
        <p className="text-ink-dim">
          資料從床邊量測輸出端經 Prometheus 進入伺服器，再顯示在儀表板。任一環節中斷，畫面就會顯示「—」與原因。
        </p>

        <ol className="space-y-2">
          {hops.map((h) => (
            <li
              key={h.name}
              className="flex items-start gap-2 rounded-md border border-edge bg-surface px-3 py-2"
            >
              <span
                aria-hidden="true"
                className={`mt-1.5 inline-block h-2.5 w-2.5 shrink-0 rounded-full ${DOT[h.tone]}`}
              />
              <div className="min-w-0">
                <div className="font-medium text-ink">{h.name}</div>
                <div className="text-ink-dim">{h.text}</div>
              </div>
            </li>
          ))}
        </ol>

        {s ? (
          <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-ink-dim">
            <dt>更新預算</dt>
            <dd className="text-ink">{s.freshness_budget_seconds} 秒</dd>
            <dt>最後量測時間</dt>
            <dd className="tabular-nums text-ink">{formatClock(s.last_sample_ts)}</dd>
            <dt>距今</dt>
            <dd className="text-ink">{formatAge(s.age_seconds)}</dd>
          </dl>
        ) : null}

        {unlocked ? (
          <Disclosure summary="技術細節">
            <div className="space-y-3 pt-2">
              <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 font-mono text-xs text-ink-dim">
                <dt>prometheus</dt>
                <dd className="break-all text-ink">
                  {s?.prometheus.url_host_only ?? "—"} · reachable={String(s?.prometheus.reachable ?? "?")}
                </dd>
                <dt>exporter</dt>
                <dd className="break-all text-ink">
                  {s?.exporter.job ?? "—"} · {s?.exporter.health ?? "?"}
                  {s?.exporter.last_error_short ? ` · ${s.exporter.last_error_short}` : ""}
                </dd>
                <dt>metrics</dt>
                <dd className="break-all text-ink">
                  {s && s.clinical_metrics_present.length > 0
                    ? s.clinical_metrics_present.join(", ")
                    : "—"}
                </dd>
                <dt>simulation</dt>
                <dd className="text-ink">
                  {s ? `${s.simulation.active} · ${s.simulation.scenario ?? "—"}` : "—"}
                </dd>
                {datasource.error ? (
                  <>
                    <dt>error</dt>
                    <dd className="break-all text-danger">{datasource.error}</dd>
                  </>
                ) : null}
              </dl>

              {capabilities?.simulation_available ? (
                <div className="rounded-md border border-warn/40 bg-warn/10 p-2">
                  <div className="mb-2 flex items-center gap-2">
                    <Badge tone="warn">模擬</Badge>
                    <span className="text-xs text-ink-dim">
                      僅供測試；啟用後所有元件都會標示模擬。
                    </span>
                  </div>
                  <div className="flex flex-wrap items-end gap-2">
                    <div className="min-w-[10rem] flex-1">
                      <Field label="情境" htmlFor="hd-sim-scenario">
                        <select
                          id="hd-sim-scenario"
                          value={scenario}
                          onChange={(e) => setScenario(e.target.value as SimulationScenario)}
                          className="min-h-[36px] w-full rounded-md border border-edge bg-canvas px-2 text-sm text-ink focus:outline-none focus:ring-2 focus:ring-focus"
                        >
                          {SIMULATION_SCENARIOS.map((sc) => (
                            <option key={sc} value={sc}>
                              {SCENARIO_LABELS[sc] ?? sc} ({sc})
                            </option>
                          ))}
                        </select>
                      </Field>
                    </div>
                    <Button
                      type="button"
                      variant="secondary"
                      size="sm"
                      loading={applying}
                      onClick={applyScenario}
                    >
                      套用
                    </Button>
                  </div>
                </div>
              ) : null}

              <pre className="max-h-64 overflow-auto rounded-md border border-edge bg-canvas p-2 font-mono text-xs text-ink-dim">
                {s ? JSON.stringify(s, null, 2) : "// 尚無狀態"}
              </pre>
            </div>
          </Disclosure>
        ) : null}
      </div>
    </Drawer>
  );
}
