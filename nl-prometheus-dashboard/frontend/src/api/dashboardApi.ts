import { apiRequest } from "./base";
import type { DashboardSpec } from "../types/dashboard";

export async function getDashboard(dashboardId: string) {
  return apiRequest<DashboardSpec>(`/api/dashboard/${dashboardId}`);
}

export async function saveDashboard(dashboard: DashboardSpec) {
  return apiRequest<DashboardSpec>("/api/dashboard", {
    method: "POST",
    body: JSON.stringify(dashboard)
  });
}

export async function updateDashboard(dashboardId: string, dashboard: DashboardSpec) {
  return apiRequest<DashboardSpec>(`/api/dashboard/${dashboardId}`, {
    method: "PUT",
    body: JSON.stringify(dashboard)
  });
}

export async function listDashboardVersions(dashboardId: string) {
  return apiRequest<Array<Record<string, unknown>>>(`/api/dashboard/${dashboardId}/versions`);
}

export async function rollbackDashboard(dashboardId: string, versionId: string) {
  return apiRequest<DashboardSpec>(`/api/dashboard/${dashboardId}/rollback/${versionId}`, {
    method: "POST"
  });
}

