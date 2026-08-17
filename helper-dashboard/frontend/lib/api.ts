import type {
  AnomalyAlertsResponse,
  BrowserEvaluationReport,
  ChatEnqueueResponse,
  ChatJobStatusResponse,
  ChatResponse,
  DashboardSpec,
  QuerySpec,
  SavedDashboardSummary,
} from "./spec-schema";

export interface WidgetQueryResponse {
  source: "prometheus" | "mock";
  points: { t: number; v: number }[];
  latest: number | null;
  table: { metric: Record<string, string>; value: number }[];
}

const API_BASE = "/api";

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = "";
    try {
      const data = (await res.json()) as { detail?: unknown };
      if (typeof data?.detail === "string") detail = data.detail;
    } catch {
      // ignore non-JSON error body
    }
    throw new Error(
      `${path} failed: ${res.status}${detail ? ` - ${detail}` : ""}`,
    );
  }
  return (await res.json()) as T;
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`${path} failed: ${res.status}`);
  }
  return (await res.json()) as T;
}

async function del<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { method: "DELETE" });
  if (!res.ok) {
    throw new Error(`${path} failed: ${res.status}`);
  }
  return (await res.json()) as T;
}

export const api = {
  async sendMessage(params: {
    session_id: string;
    message: string;
    current_dashboard_id?: string | null;
  }, opts?: {
    onProgress?: (p: { status: string; message?: string | null; pct?: number | null }) => void;
  }): Promise<ChatResponse> {
    const enq = await postJSON<ChatEnqueueResponse>("/chat/message", params);
    console.debug("[chat-api] enqueue response", enq);
    if (!enq || typeof enq.job_id !== "string" || !enq.job_id) {
      throw new Error("enqueue did not return a valid job_id");
    }
    const jobId = enq.job_id;

    const started = Date.now();
    const hardTimeoutMs = 5 * 60 * 1000;
    while (Date.now() - started < hardTimeoutMs) {
      console.debug("[chat-api] polling", { jobId });
      const st = await getJSON<ChatJobStatusResponse>(`/chat/message/${jobId}`);
      console.debug("[chat-api] polled status", st);
      opts?.onProgress?.({
        status: st.status,
        message: st.progress_message,
        pct: st.progress_pct,
      });
      if (st.status === "done") {
        if (!st.result) throw new Error("chat job done without result");
        return st.result;
      }
      if (st.status === "error") {
        throw new Error(st.error || "chat job failed");
      }
      await new Promise((r) => setTimeout(r, 1000));
    }
    throw new Error("chat job polling timeout");
  },

  getDashboard(id: string): Promise<{ spec: DashboardSpec }> {
    return getJSON(`/dashboard/${id}`);
  },

  // Execute one widget's PromQL against live Prometheus (backend proxies to
  // :9090). The response carries a `source` flag ("prometheus" | "mock") that
  // the renderer surfaces as a badge — a mock fallback is never shown as real.
  queryWidget(query: QuerySpec): Promise<WidgetQueryResponse> {
    return postJSON(`/dashboard/query`, query);
  },

  listDashboards(): Promise<{ dashboards: string[] }> {
    return getJSON(`/dashboard/`);
  },

  evaluate(id: string): Promise<{ report: BrowserEvaluationReport }> {
    return postJSON(`/evaluate/run`, { dashboard_id: id });
  },

  health(): Promise<{ status: string }> {
    return getJSON(`/health`);
  },

  saveDashboard(params: {
    dashboard_id: string;
    name: string;
    description?: string;
    tags?: string[];
  }): Promise<{ saved: SavedDashboardSummary }> {
    return postJSON(`/saved_dashboards/save`, params);
  },

  listSaved(): Promise<{ saved: SavedDashboardSummary[] }> {
    return getJSON(`/saved_dashboards/`);
  },

  deleteSaved(id: string): Promise<{ ok: boolean }> {
    return del(`/saved_dashboards/${id}`);
  },

  // Real anomaly alert events from the INC2/INC3 store. The decision_flow
  // widget consumes these — never the AlertListWidget mock.
  anomalyAlerts(): Promise<AnomalyAlertsResponse> {
    return getJSON(`/anomaly/alerts`);
  },
};
