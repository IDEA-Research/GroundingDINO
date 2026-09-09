"use client";

import React, { useEffect, useState } from "react";

import { Button, Drawer, Skeleton, useToast } from "@/components/ui";
import { api, describeApiError } from "@/lib/api";
import type { DashboardSpec, Template } from "@/lib/spec-schema";

import { metricLabel } from "./DataSourcePanel";

// 「從範本建立」: lists GET /api/templates and instantiates one via
// POST /api/templates/{id}/instantiate (fresh dashboard_id, persisted
// server-side). Templates carry no clinical thresholds by design.

export function TemplatePicker({
  open,
  onClose,
  onCreated,
}: {
  open: boolean;
  onClose: () => void;
  onCreated: (dashboard: DashboardSpec, template: Template) => void;
}) {
  const toast = useToast();
  const [templates, setTemplates] = useState<Template[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setError(null);
    api
      .listTemplates()
      .then((r) => {
        if (!cancelled) setTemplates(Array.isArray(r.templates) ? r.templates : []);
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setTemplates([]);
          setError(describeApiError(e));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [open]);

  async function create(t: Template) {
    if (busyId) return;
    setBusyId(t.id);
    try {
      const r = await api.instantiateTemplate(t.id);
      toast.show({
        title: "已從範本建立",
        description: `「${t.name}」— 尚未儲存到清單，可用「儲存」保留。`,
        tone: "ok",
        durationMs: 5000,
      });
      onCreated(r.dashboard, t);
      onClose();
    } catch (e) {
      toast.show({
        title: "建立失敗",
        description: describeApiError(e),
        tone: "danger",
        durationMs: 6000,
      });
    } finally {
      setBusyId(null);
    }
  }

  return (
    <Drawer open={open} onClose={onClose} side="right" title="從範本建立">
      <div className="space-y-3 p-3 text-sm">
        <p className="text-ink-dim">
          範本只包含量測元件與版面，不含臨床門檻值；建立後可請 Helper 調整，或由臨床人員設定門檻。
        </p>

        {templates === null ? (
          <div className="space-y-2" aria-busy="true" aria-label="載入範本中">
            <Skeleton className="h-20 w-full rounded-lg" />
            <Skeleton className="h-20 w-full rounded-lg" />
            <Skeleton className="h-20 w-full rounded-lg" />
          </div>
        ) : error ? (
          <div className="rounded-md border border-danger/40 bg-danger/10 p-3 text-ink">
            無法取得範本清單：{error}
          </div>
        ) : templates.length === 0 ? (
          <div className="rounded-md border border-edge bg-surface p-3 text-ink-dim">
            目前沒有可用的範本。
          </div>
        ) : (
          <ul className="space-y-2">
            {templates.map((t) => (
              <li
                key={t.id}
                className="flex flex-col gap-2 rounded-lg border border-edge bg-surface p-3 sm:flex-row sm:items-start"
              >
                <div className="min-w-0 flex-1">
                  <div className="font-medium text-ink">{t.name}</div>
                  {t.description ? (
                    <div className="mt-0.5 text-ink-dim">{t.description}</div>
                  ) : null}
                  <div className="mt-1 text-xs text-ink-dim">
                    {t.widget_count} 個元件
                    {t.metrics.length > 0
                      ? ` · ${t.metrics.map(metricLabel).join("、")}`
                      : ""}
                  </div>
                </div>
                <Button
                  type="button"
                  variant="primary"
                  size="sm"
                  loading={busyId === t.id}
                  disabled={busyId !== null && busyId !== t.id}
                  onClick={() => create(t)}
                  className="shrink-0"
                >
                  建立
                </Button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </Drawer>
  );
}
