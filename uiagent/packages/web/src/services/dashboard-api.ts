import type {
  DashboardDataResponse,
  DashboardMetaResponse,
  GenerateDashboardResponse,
} from '@ui-agent/types';

function getApiBaseUrl(): string {
  const configured = (import.meta.env.VITE_API_URL as string | undefined)?.trim();
  if (configured) {
    return configured.endsWith('/') ? configured.slice(0, -1) : configured;
  }
  return './api';
}

const API_BASE_URL = getApiBaseUrl();

export async function generateDashboard(prompt: string): Promise<GenerateDashboardResponse> {
  const response = await fetch(`${API_BASE_URL}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt }),
  });

  const data = await response.json();
  if (!response.ok) {
    throw data;
  }
  return data as GenerateDashboardResponse;
}

export async function fetchDashboardMeta(dashboardId: string): Promise<DashboardMetaResponse> {
  const response = await fetch(`${API_BASE_URL}/dashboards/${encodeURIComponent(dashboardId)}/meta`);
  const data = await response.json();
  if (!response.ok) {
    throw data;
  }
  return data as DashboardMetaResponse;
}

export async function fetchDashboardData(dashboardId: string): Promise<DashboardDataResponse> {
  const response = await fetch(`${API_BASE_URL}/dashboards/${encodeURIComponent(dashboardId)}`);
  const data = await response.json();
  if (!response.ok) {
    throw data;
  }
  return data as DashboardDataResponse;
}

