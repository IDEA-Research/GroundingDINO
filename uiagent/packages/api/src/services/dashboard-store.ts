import type { DashboardPayload, UISpec, WidgetCodePatch } from '@ui-agent/types';

type DashboardStoredRecord = DashboardPayload;

class DashboardStore {
  private store = new Map<string, DashboardStoredRecord>();
  private sequence = 0;

  save(input: { uiSpec: UISpec; widgetPatches: WidgetCodePatch[] }): DashboardStoredRecord {
    const dashboardId = `dash_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
    this.sequence += 1;
    const updatedAt = new Date().toISOString();
    const versionToken = `${updatedAt}#${this.sequence}`;

    const record: DashboardStoredRecord = {
      dashboardId,
      versionToken,
      updatedAt,
      uiSpec: input.uiSpec,
      widgetPatches: input.widgetPatches,
    };

    this.store.set(dashboardId, record);
    return record;
  }

  get(dashboardId: string): DashboardStoredRecord | null {
    return this.store.get(dashboardId) ?? null;
  }
}

export const dashboardStore = new DashboardStore();

