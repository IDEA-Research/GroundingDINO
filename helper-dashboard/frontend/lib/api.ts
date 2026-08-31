import type {
  AnomalyAlertsResponse,
  BrowserEvaluationReport,
  ChatEnqueueResponse,
  ChatJobStatusResponse,
  ChatRequest,
  ChatResponse,
  DashboardSpec,
  LoadSavedResponse,
  QuerySpec,
  SavedDashboardEntry,
  SavedDashboardSummary,
} from "./spec-schema";

// Wire request for POST /api/dashboard/query: the widget's QuerySpec plus
// the shared refresh clock's `end` (contract C1, epoch seconds). The
// backend floors it to the widget's step so every widget renders the same
// aligned window.
export interface WidgetQueryRequest extends QuerySpec {
  end?: number;
}

export interface WidgetQueryResponse {
  source: "prometheus";
  // Explicit emptiness: "ok" = series present, "empty" = the query
  // succeeded but returned zero series/points. Errors never reach this
  // shape — the backend answers 400/502 and postJSON throws.
  data_status: "ok" | "empty";
  points: { t: number; v: number }[];
  latest: number | null;
  table: { metric: Record<string, string>; value: number }[];
  // How many series were collapsed (by MEAN) into the single rendered
  // series. >1 means the value shown is an average — the renderer
  // annotates it so nobody mistakes an averaged vital for a raw one.
  n_series: number;
  // Contract C1: the step-aligned end actually used (epoch seconds) and
  // the resolved window length. Optional so responses from a backend
  // predating the contract still parse.
  end?: number;
  range_seconds?: number;
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
  // Contract C2: `current_dashboard_id` rides on every message so the
  // backend re-points session memory at what the user is LOOKING at
  // before classification — patches then target the visible dashboard.
  async sendMessage(params: ChatRequest, opts?: {
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
  // :9090). There is NO mock fallback: a mock source is rejected with 400,
  // an unreachable Prometheus is a 502 (postJSON throws with the detail),
  // and `data_status` distinguishes real data from an honest empty result.
  queryWidget(query: WidgetQueryRequest): Promise<WidgetQueryResponse> {
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

  // Matches backend SaveRequest exactly (dashboard_id + name required,
  // description/tags optional). The response carries the FULL stored
  // entry (spec + summary), not the list-row summary shape.
  saveDashboard(params: {
    dashboard_id: string;
    name: string;
    description?: string;
    tags?: string[];
  }): Promise<{ saved: SavedDashboardEntry }> {
    return postJSON(`/saved_dashboards/save`, params);
  },

  listSaved(): Promise<{ saved: SavedDashboardSummary[] }> {
    return getJSON(`/saved_dashboards/`);
  },

  deleteSaved(id: string): Promise<{ ok: boolean }> {
    return del(`/saved_dashboards/${id}`);
  },

  // Full saved entry (including the snapshotted spec). Read-only — to
  // put a saved dashboard on screen, use loadSaved, which clones it.
  getSaved(id: string): Promise<{ saved: SavedDashboardEntry }> {
    return getJSON(`/saved_dashboards/${id}`);
  },

  // Contract C3: load a saved entry as a FRESH clone. The backend
  // deep-clones the snapshot under a new dashboard_id, persists the
  // clone, bumps used_count, and returns the clone's full spec —
  // chat patches then target the clone, never the shared snapshot.
  loadSaved(id: string): Promise<LoadSavedResponse> {
    return postJSON(`/saved_dashboards/${id}/load`, {});
  },

  // Real anomaly alert events from the INC2/INC3 store. The
  // decision_flow widget consumes these. (AlertListWidget is
  // query-driven via its own PromQL — e.g. ALERTS — not this feed.)
  anomalyAlerts(): Promise<AnomalyAlertsResponse> {
    return getJSON(`/anomaly/alerts`);
  },
};
