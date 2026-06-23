import { apiRequest } from "./base";
import type { DashboardSpec, MonitoringTaskModel, PatchSpec } from "../types/dashboard";

export async function createDashboard(prompt: string, context: Record<string, unknown> = {}) {
  return apiRequest<{ dashboard: DashboardSpec; task_model: MonitoringTaskModel }>("/api/agent/create-dashboard", {
    method: "POST",
    body: JSON.stringify({ prompt, context })
  });
}

export async function patchDashboard(dashboardId: string, prompt: string, currentDashboard: DashboardSpec) {
  return apiRequest<{ patch: PatchSpec; updated_dashboard: DashboardSpec; task_model: MonitoringTaskModel }>(
    "/api/agent/patch-dashboard",
    {
      method: "POST",
      body: JSON.stringify({
        dashboard_id: dashboardId,
        prompt,
        current_dashboard: currentDashboard
      })
    }
  );
}
