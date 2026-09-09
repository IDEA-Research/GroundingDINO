import type {
  AnomalyAlertsResponse,
  BrowserEvaluationReport,
  Capabilities,
  ChatEnqueueResponse,
  ChatJobStatusResponse,
  ChatRequest,
  ChatResponse,
  DashboardSpec,
  DatasourceStatus,
  DevRunDiffResponse,
  DevRunEventsResponse,
  DevRunReadResponse,
  DevRunStartRequest,
  DevRunStartResponse,
  DevRunsListResponse,
  DevfixConfig,
  FilterLabelName,
  InstantiateTemplateResponse,
  LabelValuesResponse,
  LoadSavedResponse,
  SavedDashboardEntry,
  SavedDashboardPatch,
  SavedDashboardSummary,
  SimulationScenario,
  SimulationState,
  TemplatesResponse,
  WidgetQueryRequest,
  WidgetQueryResponse,
} from "./spec-schema";
import { devHeaders } from "./dev-session";

// The wire types live in spec-schema.ts (single mirror of the backend);
// re-exported here so existing `import type { WidgetQueryRequest } from
// "./api"` call sites keep resolving.
export type { WidgetQueryRequest, WidgetQueryResponse } from "./spec-schema";

const API_BASE = "/api";

// Typed HTTP failure. `status` lets callers branch (403 = developer token
// missing/wrong, 409 = local run layer disabled / another run active,
// 502 = Prometheus unavailable) without parsing the message. The message
// keeps the historical `<path> failed: <status> - <detail>` shape because
// useQueryData surfaces it as errorDetail.
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string | null;
  readonly path: string;
  readonly body: unknown;

  constructor(path: string, status: number, detail: string | null, body?: unknown) {
    super(`${path} failed: ${status}${detail ? ` - ${detail}` : ""}`);
    this.name = "ApiError";
    this.path = path;
    this.status = status;
    this.detail = detail;
    this.body = body;
  }
}

export function isApiError(e: unknown): e is ApiError {
  return e instanceof ApiError;
}

// Short 繁中 explanation for the user UI; developer surfaces show
// `e.message` verbatim instead.
export function describeApiError(e: unknown): string {
  if (isApiError(e)) {
    if (e.status === 403) return "需要開發者權限（權杖不正確或未設定）";
    if (e.status === 404) return "找不到資料";
    // 409/422 details are authored 繁中 server-side (queue full, quota,
    // local runs off, validation); everything else stays generic so no
    // English framework text leaks into user toasts.
    if (e.status === 409) return e.detail || "目前無法執行此操作";
    if (e.status === 422) return e.detail || "資料格式不正確";
    if (e.status === 502) return "資料來源無法連線";
    if (e.status >= 500) return "伺服器發生錯誤";
    return `請求失敗（HTTP ${e.status}）`;
  }
  if (e instanceof TypeError) return "無法連線到伺服器";
  if (e instanceof SyntaxError) return "伺服器回應格式不正確";
  return e instanceof Error ? e.message : String(e);
}

interface RequestOptions {
  // Attach the developer token header (from sessionStorage) — only for
  // routes the backend gates with require_dev.
  dev?: boolean;
  // Explicit token override (used once, to VERIFY a freshly typed token
  // before storing it).
  token?: string | null;
}

async function request<T>(
  method: "GET" | "POST" | "PATCH" | "DELETE",
  path: string,
  body?: unknown,
  opts: RequestOptions = {},
): Promise<T> {
  const headers: Record<string, string> = {};
  if (body !== undefined) headers["content-type"] = "application/json";
  if (opts.dev || opts.token) Object.assign(headers, devHeaders(opts.token));
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: "no-store",
  });
  if (!res.ok) {
    let detail: string | null = null;
    let parsed: unknown = undefined;
    try {
      parsed = await res.json();
      const d = (parsed as { detail?: unknown })?.detail;
      if (typeof d === "string") detail = d;
      else if (d && typeof d === "object" && typeof (d as { message?: unknown }).message === "string") {
        // e.g. /api/dev/simulation 409 {"detail": {"message", "reason"}}
        detail = (d as { message: string }).message;
      } else if (d !== undefined) detail = JSON.stringify(d);
    } catch {
      // non-JSON error body
    }
    throw new ApiError(path, res.status, detail, parsed);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

function getJSON<T>(path: string, opts?: RequestOptions): Promise<T> {
  return request<T>("GET", path, undefined, opts);
}

function postJSON<T>(path: string, body: unknown, opts?: RequestOptions): Promise<T> {
  return request<T>("POST", path, body, opts);
}

function patchJSON<T>(path: string, body: unknown, opts?: RequestOptions): Promise<T> {
  return request<T>("PATCH", path, body, opts);
}

function del<T>(path: string, opts?: RequestOptions): Promise<T> {
  return request<T>("DELETE", path, undefined, opts);
}

function enc(s: string): string {
  return encodeURIComponent(s);
}

export const api = {
  // -------------------------------------------------------------------
  // Chat (job transport unchanged: enqueue -> poll)
  // -------------------------------------------------------------------

  // Contract C2: `current_dashboard_id` rides on every message so the
  // backend re-points session memory at what the user is LOOKING at
  // before classification — patches then target the visible dashboard.
  async sendMessage(params: ChatRequest, opts?: {
    onProgress?: (p: { status: string; message?: string | null; pct?: number | null }) => void;
  }): Promise<ChatResponse> {
    const enq = await postJSON<ChatEnqueueResponse>("/chat/message", params);
    if (!enq || typeof enq.job_id !== "string" || !enq.job_id) {
      throw new Error("enqueue did not return a valid job_id");
    }
    const jobId = enq.job_id;

    const started = Date.now();
    const hardTimeoutMs = 5 * 60 * 1000;
    while (Date.now() - started < hardTimeoutMs) {
      const st = await getJSON<ChatJobStatusResponse>(`/chat/message/${enc(jobId)}`);
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

  // -------------------------------------------------------------------
  // Dashboards & data
  // -------------------------------------------------------------------

  getDashboard(id: string): Promise<{ spec: DashboardSpec }> {
    return getJSON(`/dashboard/${enc(id)}`);
  },

  // Execute one widget's PromQL against live Prometheus (backend proxies to
  // :9090). There is NO mock fallback: a mock source is rejected with 400,
  // an unreachable Prometheus is a 502 (ApiError.status === 502), and
  // `data_status` distinguishes real data from an honest empty result.
  queryWidget(query: WidgetQueryRequest): Promise<WidgetQueryResponse> {
    return postJSON(`/dashboard/query`, query);
  },

  listDashboards(): Promise<{ dashboards: string[] }> {
    return getJSON(`/dashboard/`);
  },

  // Prometheus label values seen in the last `hours` (allow-listed labels
  // only: patient_id | device_model).
  labelValues(label: FilterLabelName, hours = 24): Promise<LabelValuesResponse> {
    return getJSON(`/dashboard/label-values?label=${enc(label)}&hours=${hours}`);
  },

  health(): Promise<{ status: string }> {
    return getJSON(`/health`);
  },

  // Public booleans only. Never authorizes anything.
  capabilities(): Promise<Capabilities> {
    return getJSON(`/capabilities`);
  },

  // Public: backend/Prometheus/exporter hop status + last measurement.
  datasourceStatus(): Promise<DatasourceStatus> {
    return getJSON(`/datasource/status`);
  },

  // -------------------------------------------------------------------
  // Templates
  // -------------------------------------------------------------------

  listTemplates(): Promise<TemplatesResponse> {
    return getJSON(`/templates`);
  },

  // Fresh dashboard_id, persisted server-side; render the returned spec.
  instantiateTemplate(id: string): Promise<InstantiateTemplateResponse> {
    return postJSON(`/templates/${enc(id)}/instantiate`, {});
  },

  // -------------------------------------------------------------------
  // Saved library
  // -------------------------------------------------------------------

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

  // Soft delete (deleted_at) — reversible via restoreSaved.
  deleteSaved(id: string): Promise<{ ok: boolean; restorable?: boolean }> {
    return del(`/saved_dashboards/${enc(id)}`);
  },

  restoreSaved(id: string): Promise<{ ok?: boolean; saved?: SavedDashboardEntry }> {
    return postJSON(`/saved_dashboards/${enc(id)}/restore`, {});
  },

  // Rename / re-describe / re-tag; saved_id stays stable.
  patchSaved(id: string, patch: SavedDashboardPatch): Promise<{ saved: SavedDashboardEntry }> {
    return patchJSON(`/saved_dashboards/${enc(id)}`, patch);
  },

  // New library entry "{name} 副本".
  duplicateSaved(id: string): Promise<{ saved: SavedDashboardEntry }> {
    return postJSON(`/saved_dashboards/${enc(id)}/duplicate`, {});
  },

  // Full saved entry (including the snapshotted spec). Read-only — to
  // put a saved dashboard on screen, use loadSaved, which clones it.
  getSaved(id: string): Promise<{ saved: SavedDashboardEntry }> {
    return getJSON(`/saved_dashboards/${enc(id)}`);
  },

  // Contract C3: load a saved entry as a FRESH clone. The backend
  // deep-clones the snapshot under a new dashboard_id, persists the
  // clone, bumps used_count, and returns the clone's full spec —
  // chat patches then target the clone, never the shared snapshot.
  loadSaved(id: string): Promise<LoadSavedResponse> {
    return postJSON(`/saved_dashboards/${enc(id)}/load`, {});
  },

  // Real anomaly alert events from the INC2/INC3 store. The
  // decision_flow widget consumes these.
  anomalyAlerts(): Promise<AnomalyAlertsResponse> {
    return getJSON(`/anomaly/alerts`);
  },

  // -------------------------------------------------------------------
  // Developer-gated routes (X-Developer-Token on every call)
  // -------------------------------------------------------------------

  evaluate(id: string): Promise<{ report: BrowserEvaluationReport }> {
    return postJSON(`/evaluate/run`, { dashboard_id: id }, { dev: true });
  },

  // Read-only preflight. Passing `token` verifies THAT token instead of
  // the stored one (used by the 解鎖 flow before anything is stored).
  devfixConfig(token?: string | null): Promise<DevfixConfig> {
    return getJSON(`/devfix/config`, token ? { token } : { dev: true });
  },

  devfixRuns(): Promise<DevRunsListResponse> {
    return getJSON(`/devfix/runs`, { dev: true });
  },

  devfixRun(id: string, after = 0): Promise<DevRunReadResponse> {
    return getJSON(`/devfix/runs/${enc(id)}?after=${after}`, { dev: true });
  },

  devfixEvents(id: string, after = 0): Promise<DevRunEventsResponse> {
    return getJSON(`/devfix/runs/${enc(id)}/events?after=${after}`, { dev: true });
  },

  devfixDiff(id: string): Promise<DevRunDiffResponse> {
    return getJSON(`/devfix/runs/${enc(id)}/diff`, { dev: true });
  },

  // Only `instruction` (+ optional title) ever crosses the wire — no
  // commands, paths, or model overrides. 409 when the local run layer is
  // off (ApiError.detail = 本機執行層未啟用) or the queue is full.
  devfixStart(body: DevRunStartRequest): Promise<DevRunStartResponse> {
    return postJSON(`/devfix/runs`, body, { dev: true });
  },

  // "cancelled" = removed from the queue; "cancelling" = SIGTERM sent, the
  // run lands on cancelled within ~8 s (keep polling).
  devfixCancel(
    id: string,
  ): Promise<{ run_id?: string; result?: "removed" | "signalled"; status?: "cancelled" | "cancelling" }> {
    return postJSON(`/devfix/runs/${enc(id)}/cancel`, {}, { dev: true });
  },

  devfixRetry(id: string): Promise<DevRunStartResponse> {
    return postJSON(`/devfix/runs/${enc(id)}/retry`, {}, { dev: true });
  },

  simulationGet(): Promise<SimulationState> {
    return getJSON(`/dev/simulation`, { dev: true });
  },

  simulationSet(scenario: SimulationScenario): Promise<SimulationState> {
    return postJSON(`/dev/simulation`, { scenario }, { dev: true });
  },
};
