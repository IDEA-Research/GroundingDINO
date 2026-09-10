// Spec schema (frontend mirror of backend/app/specs) plus every wire
// contract the shell imports (dashboard query, capabilities, datasource,
// templates, saved library, developer runs). Field names mirror the
// backend byte-for-byte — see docs/agent-ops DESIGN sections 3-4.
//
// Do NOT add fields like `raw_html`, `script`, `component`, `iframe`,
// `code`. The renderer will not honor them. The backend validator will
// reject them. Adding a new widget type is a Big guy task that must
// update both sides.

export type WidgetType =
  | "line_chart"
  | "stat_card"
  | "gauge"
  | "bar_gauge"
  | "table"
  | "alert_list"
  | "pie_chart"
  | "bar_chart"
  | "heatmap"
  | "decision_flow";

export const WIDGET_TYPES: WidgetType[] = [
  "line_chart",
  "stat_card",
  "gauge",
  "bar_gauge",
  "table",
  "alert_list",
  "pie_chart",
  "bar_chart",
  "heatmap",
  "decision_flow",
];

// Neonatal physiological signals a decision flow may branch on. Mirrors
// backend DECISION_FLOW_SIGNALS.
export type DecisionFlowSignal = "rSO2" | "SpO2" | "HR" | "MAP" | "FiO2";

// Node roles in a clinical decision flow. Mirrors backend FlowNodeKind.
export type FlowNodeKind =
  | "data_integrity_gate"
  | "decision"
  | "action"
  | "terminal"
  | "signal_lost";

export interface FlowNode {
  id: string;
  kind: FlowNodeKind;
  label: string;
  signal?: string | null;
}

export interface FlowEdge {
  from: string;
  to: string;
  condition: string;
}

export interface FlowStep {
  id: string;
  title: string;
  node: string;
  guidance: string;
}

export interface QueryInput {
  label: string;
  query: QuerySpec;
}

export interface DecisionFlowSpec {
  nodes: FlowNode[];
  edges: FlowEdge[];
  steps: FlowStep[];
  inputs: QueryInput[];
}

export interface QuerySpec {
  source: "prometheus" | "mock";
  promql: string;
  query_type: "instant" | "range";
  range?: string | null;
  step?: string | null;
  // Multi-query tables (WidgetSpec.queries): column key, display label and
  // unit for the column this query feeds. Ignored on `query`.
  key?: string | null;
  label?: string | null;
  unit?: string | null;
  // TableWidget cell display: a per-column threshold ladder + Grafana-style
  // colored-background mode (see CellDisplayMode below). Unlike key/label/
  // unit these are read on BOTH a plain `query` (the value column of a
  // legacy single-metric table) and each entry of `queries` (a multi-metric
  // table's columns) — TableWidget.tsx is the only reader, other widgets
  // ignore them. Frontend-only for now: the backend QuerySpec model is
  // `extra="forbid"` and does not carry these fields yet (the same
  // situation as StatCardWidget's `options.color_mode`, also absent from
  // backend's ALLOWED_OPTION_KEYS) — a spec produced by the chat/patch
  // pipeline will not set them until the backend is extended; hand-
  // authored/template/saved specs can already opt a column in. A column
  // with no `thresholds` (or `cellDisplayMode` "none"/absent) renders
  // exactly as before.
  thresholds?: WidgetThreshold[] | null;
  cellDisplayMode?: CellDisplayMode | null;
}

// TableWidget colored-background cell modes. "basic" fills the cell with
// the flat active-threshold color; "gradient" interpolates continuously
// within the breached band instead of snapping straight to full color.
// Gauge-cell and sparkline-cell modes are out of scope for now.
export type CellDisplayMode =
  | "none"
  | "colored-background-basic"
  | "colored-background-gradient";

export interface WidgetPosition {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface WidgetEncoding {
  unit?: string | null;
  legend?: string | null;
  color?: string | null;
}

// Which side of the threshold value is the alarmed side. Resolved by
// lib/threshold.ts (identical rules to the backend helper) when absent.
export type ThresholdDirection = "above" | "below";

export interface WidgetThreshold {
  value: number;
  color: string;
  label?: string | null;
  direction?: ThresholdDirection | null;
}

export type Threshold = WidgetThreshold;

export interface WidgetSpec {
  id: string;
  type: WidgetType;
  title: string;
  description?: string | null;
  // Optional ONLY for a table that carries `queries`; every other widget
  // has exactly one query. Read through `widgetQuery()` below.
  query?: QuerySpec | null;
  // Table only (max 6): one column per query, rows joined by `join_label`
  // (default patient_id) on the backend.
  queries?: QuerySpec[] | null;
  position: WidgetPosition;
  encoding?: WidgetEncoding;
  thresholds?: WidgetThreshold[];
  options?: Record<string, unknown>;
  // Present only for type === "decision_flow".
  decision_flow?: DecisionFlowSpec | null;
}

// The widget's primary query, or a synthesized one for `queries` tables so
// callers that only need promql/query_type keep working.
export function widgetQuery(widget: WidgetSpec): QuerySpec {
  if (widget.query) return widget.query;
  const first = widget.queries?.[0];
  return {
    source: first?.source ?? "prometheus",
    promql: first?.promql ?? "",
    query_type: first?.query_type ?? "instant",
    range: first?.range ?? null,
    step: first?.step ?? null,
  };
}

// Real alert event as read from the anomaly store (INC2/INC3). Mirrors the
// backend AlertEvent. The decision_flow widget renders THESE, never the
// hardcoded widget list (mock data is gone from every path).
export type AlertEventState =
  | "pending"
  | "firing"
  | "resolved"
  | "signal_lost"
  | "insufficient_baseline";

export interface AlertEventDTO {
  ts: string;
  rule_id: string;
  state: AlertEventState;
  value?: number | null;
  baseline?: number | null;
  threshold?: number | null;
  severity: string;
  signal_lost_reason?: string | null;
  would_page: boolean;
  paged: boolean;
  suppressed_reason?: string | null;
  non_diagnostic: boolean;
  message: string;
}

export interface AnomalyAlertsResponse {
  // null/empty when the store is present but has no events yet.
  events: AlertEventDTO[];
  // Loud degradation signal: true when the store could not be read.
  store_unavailable?: boolean;
}

export interface DashboardLayout {
  columns: number;
  row_height: number;
}

export interface DashboardVariable {
  name: string;
  label: string;
  query?: string | null;
  default?: string | null;
  values: string[];
}

export interface DashboardSpec {
  dashboard_id: string;
  title: string;
  description: string;
  layout: DashboardLayout;
  variables: DashboardVariable[];
  widgets: WidgetSpec[];
  refresh_interval: string;
}

export interface PatchSpec {
  patch_id: string;
  reason: string;
  target_dashboard_id: string;
  created_by: string;
  operations: PatchOperation[];
}

export type PatchOperation =
  | { op: "add_widget"; widget: WidgetSpec }
  | { op: "remove_widget"; widget_id: string }
  | {
      op: "update_widget";
      widget_id: string;
      fields: Partial<WidgetSpec>;
    }
  | { op: "update_dashboard"; fields: Partial<DashboardSpec> }
  | { op: "reorder_widgets"; order: string[] };

export interface ChatResponse {
  user_reply: string;
  intent_type: string;
  dashboard: DashboardSpec | null;
  patch: PatchSpec | null;
  warnings: string[];
  // Runtime provenance. "helpercode" when the hand-rolled CLI
  // succeeded, "mock", "mock_fallback" (auto mode fell back),
  // "heuristic" (deterministic provider). "opencode" is reserved
  // for the upcoming real OpenCode CLI devfix runtime. Developer
  // detail — the user UI hides it.
  runtime_used: string | null;
  fallback_reason: string | null;
  // Pre-output review transcript; empty when review is off.
  review_trail: ReviewTrailEvent[];
  // Present when rescue decided the user must clarify.
  clarification_questions: string[] | null;
  // Present when the orchestrator is asking "save this?" — carries
  // the dashboard_id that the next user message answers for.
  save_prompt_for: string | null;
  // Library entry the chat flow just saved (「已儲存為…」) or opened
  // (LoadSavedIntent) — the shell links its save-state chip to it.
  saved_id?: string | null;
}

// Chat request body sent to POST /api/chat/message (enqueue). Contract
// C2: current_dashboard_id names the dashboard the user is LOOKING at,
// so the backend re-points session memory before classification and
// patches apply to the visible dashboard.
export interface ChatRequest {
  session_id: string;
  message: string;
  current_dashboard_id?: string | null;
}

export interface ChatEnqueueResponse {
  job_id: string;
  status: string;
}

export interface ChatJobStatusResponse {
  job_id: string;
  status: "queued" | "running" | "done" | "error" | string;
  progress_message: string | null;
  progress_pct: number | null;
  result: ChatResponse | null;
  error: string | null;
}

export interface ReviewTrailEvent {
  stage: string;
  attempt?: number;
  decision?: string;
  rationale?: string;
  page_loaded?: boolean;
  missing?: string[];
  console_errors?: number;
  kind?: string;
  to?: string;
  error?: string;
}

// ---------------------------------------------------------------------------
// Dashboard query wire contract (POST /api/dashboard/query)
// ---------------------------------------------------------------------------

// Label-equality filters applied to result series BEFORE the backend
// collapses them (series lacking the label are dropped when set).
export interface QueryFilters {
  patient_id?: string | null;
  device_model?: string | null;
}

// One column query of a `queries` table.
export interface TableQuerySpec {
  key: string;
  promql: string;
  label?: string | null;
  unit?: string | null;
}

export interface WidgetQueryRequest {
  source: "prometheus" | "mock";
  // May be omitted when `queries` is present (table widgets only).
  promql?: string | null;
  query_type: "instant" | "range";
  range?: string | null;
  step?: string | null;
  // Shared refresh clock (contract C1, epoch seconds).
  end?: number;
  filters?: QueryFilters | null;
  queries?: TableQuerySpec[] | null;
  join_label?: string | null;
}

export interface Freshness {
  // Newest raw sample timestamp (epoch seconds) behind this answer; null
  // when the backend could not derive a metric name from the PromQL.
  sample_ts: number | null;
  age_seconds: number | null;
  stale: boolean;
  budget_seconds: number;
}

// Distinct label sets seen in the result (before collapse).
export interface SeriesLabelSet {
  patient_id?: string;
  device_model?: string;
}

export interface TableColumn {
  key: string;
  label: string;
  unit?: string | null;
}

// Legacy single-query table row (one Prometheus series).
export interface LegacyTableRow {
  metric: Record<string, string>;
  value: number;
}

// Joined row of a `queries` table: one row per join label value, one
// value + sample timestamp per column key. null = that column has no
// sample for this row (rendered as 「—」, never 0).
export interface JoinedTableRow {
  labels: Record<string, string>;
  values: Record<string, number | null>;
  ts: Record<string, number | null>;
}

export type TableRowWire = LegacyTableRow | JoinedTableRow;

export function isJoinedTableRow(row: TableRowWire): row is JoinedTableRow {
  return (
    typeof row === "object" &&
    row !== null &&
    "values" in row &&
    typeof (row as JoinedTableRow).values === "object"
  );
}

export interface WirePoint {
  t: number; // epoch seconds
  // null = gap (consecutive samples farther apart than 2 x step). Charts
  // must not bridge it.
  v: number | null;
}

export interface WidgetQueryResponse {
  source: "prometheus";
  // Explicit emptiness: "ok" = series present, "empty" = the query
  // succeeded but returned zero series/points. Staleness is a FLAG
  // (freshness.stale), never a data_status value.
  data_status: "ok" | "empty";
  points: WirePoint[];
  latest: number | null;
  table: TableRowWire[];
  // How many series were collapsed (by MEAN) into the single rendered
  // series. >1 means the value shown is an average — the renderer
  // annotates it so nobody mistakes an averaged vital for a raw one.
  n_series: number;
  // Contract C1: the step-aligned end actually used (epoch seconds) and
  // the resolved window length.
  end?: number;
  range_seconds?: number;
  // Additive (DESIGN section 4).
  freshness?: Freshness | null;
  step_seconds?: number | null;
  filters_applied?: QueryFilters | null;
  series_labels?: SeriesLabelSet[] | null;
  simulated?: boolean;
  sim_scenario?: string | null;
  // Present for `queries` tables.
  columns?: TableColumn[] | null;
  join_label?: string | null;
}

// GET /api/dashboard/label-values?label=patient_id&hours=24
export type FilterLabelName = "patient_id" | "device_model";

export interface LabelValues {
  values: string[];
  label?: FilterLabelName | string;
  hours?: number;
  simulated?: boolean;
  sim_scenario?: string | null;
}

export type LabelValuesResponse = LabelValues;

// ---------------------------------------------------------------------------
// Public capability + datasource status (booleans/timestamps only)
// ---------------------------------------------------------------------------

// GET /api/capabilities — booleans only; NEVER authorizes anything.
export interface Capabilities {
  dev_token_configured: boolean;
  devfix_local_runs: boolean;
  auto_fix_enabled: boolean;
  simulation_available: boolean;
  evaluate_enabled: boolean;
}

export type ExporterHealth = "up" | "down" | "unknown";

// GET /api/datasource/status
export interface DatasourceStatus {
  backend_ok: boolean;
  // Epoch seconds when the backend computed this answer.
  checked_at?: number;
  prometheus: {
    reachable: boolean;
    url_host_only: string | null;
  };
  exporter: {
    job: string;
    health: ExporterHealth;
    last_error_short: string | null;
    last_scrape?: string | null;
  };
  clinical_metrics_present: string[];
  last_sample_ts: number | null;
  age_seconds: number | null;
  // age_seconds > freshness_budget_seconds.
  stale?: boolean;
  freshness_budget_seconds: number;
  simulation: {
    active: boolean;
    scenario: string | null;
  };
}

// Dev-gated simulation (POST/GET /api/dev/simulation).
export type SimulationScenario =
  | "off"
  | "normal"
  | "nodata"
  | "stale"
  | "partial"
  | "offline"
  | "error";

export const SIMULATION_SCENARIOS: SimulationScenario[] = [
  "off",
  "normal",
  "nodata",
  "stale",
  "partial",
  "offline",
  "error",
];

export interface SimulationState {
  active: boolean;
  scenario: SimulationScenario | string;
  // env HELPER_DASHBOARD_ALLOW_SIMULATION=1 on the server.
  allowed?: boolean;
  since?: string | null;
  scenarios?: string[];
}

// ---------------------------------------------------------------------------
// Templates
// ---------------------------------------------------------------------------

export interface Template {
  id: string;
  name: string;
  description: string;
  widget_count: number;
  widget_types?: string[];
  metrics: string[];
}

export interface TemplatesResponse {
  templates: Template[];
}

// POST /api/templates/{id}/instantiate
export interface InstantiateTemplateResponse {
  template_id: string;
  name: string;
  dashboard: DashboardSpec;
}

// ---------------------------------------------------------------------------
// Saved library
// ---------------------------------------------------------------------------

export interface SavedDashboardSummary {
  saved_id: string;
  name: string;
  description: string;
  tags: string[];
  widget_types: string[];
  promqls: string[];
  used_count: number;
  // Soft delete: set by DELETE, cleared by POST /restore. The default list
  // excludes deleted entries; only a dev-header `include_deleted=1` list
  // carries them.
  deleted_at?: string | null;
}

// Full saved-library entry as returned by GET /api/saved_dashboards/{id}.
// Mirrors the backend SavedDashboard model (spec is the snapshot taken
// at save time).
export interface SavedDashboardEntry {
  saved_id: string;
  name: string;
  description: string;
  tags: string[];
  spec: DashboardSpec;
  summary: Record<string, unknown>;
  created_at: string;
  used_count: number;
  deleted_at?: string | null;
  updated_at?: string | null;
}

// One widget's content and/or grid-position update for PATCH
// /api/saved_dashboards/{id}. Mirrors backend `WidgetContentUpdate`
// (backend/app/api/saved_dashboards.py WidgetContentUpdate) field-for-
// field: `id` is the only required key, every other field is optional
// and independent — a drag/resize (DashboardGrid.tsx, Phase 2) sends
// only `position`, a panel edit (WidgetEditDrawer.tsx, via
// handleLayoutChange in page.tsx) sends the full widget content.
// `type`/`query`/`queries`/`decision_flow` are deliberately absent —
// this route never changes what a widget IS, only its position/style.
export interface WidgetContentUpdate {
  id: string;
  title?: string;
  description?: string | null;
  thresholds?: WidgetThreshold[];
  options?: Record<string, unknown>;
  position?: WidgetPosition;
}

// PATCH /api/saved_dashboards/{id}. `widgets`/`layout` are the Phase 2
// layout-engine additions (drag/resize persistence) folded into the SAME
// rename/re-tag route rather than a new endpoint — see DashboardGrid.tsx.
export interface SavedDashboardPatch {
  name?: string;
  description?: string;
  tags?: string[];
  widgets?: WidgetContentUpdate[];
  layout?: DashboardLayout;
}

// Response of POST /api/saved_dashboards/{saved_id}/load (contract C3).
// `dashboard` is a persisted CLONE of the saved snapshot under a fresh
// dashboard_id — render and patch the clone, never the snapshot.
export interface LoadSavedResponse {
  saved_id: string;
  name: string;
  dashboard: DashboardSpec;
}

export interface BrowserEvaluationReport {
  dashboard_id: string;
  page_loaded: boolean;
  widgets_rendered: string[];
  missing_widgets: string[];
  console_errors: string[];
  layout_errors: string[];
  prometheus_errors: string[];
  screenshot_path?: string | null;
  recommendation: string;
}

// ---------------------------------------------------------------------------
// Developer execution API (DESIGN section 3) — all dev-gated
// ---------------------------------------------------------------------------

// GET /api/devfix/config (read-only)
export interface DevfixConfig {
  project_root: string;
  worktree_parent: string;
  bin: string;
  bin_exists: boolean;
  model: string;
  timeout_seconds: number;
  runtime_mode: string;
  devfix_runtime: string;
  local_runs_enabled: boolean;
  // Why local runs are off (null when enabled): local_runs_disabled |
  // mock_mode | runtime_not_opencode_cli | binary_missing.
  local_runs_reason: string | null;
  auto_fix_enabled: boolean;
  daily_quota: {
    limit: number;
    used: number;
  };
  queue: {
    active_run_id: string | null;
    pending_run_ids: string[];
    max_pending: number;
  };
}

// meta.status vocabulary.
export type DevRunStatus =
  | "queued"
  | "running"
  | "waiting"
  | "success"
  | "failed"
  | "cancelled"
  | "timeout";

// Older run directories on disk may still carry the pre-redesign report
// vocabulary; normalizeDevRunStatus() folds them into DevRunStatus.
export type DevRunLegacyStatus = "resolved" | "rejected" | "needs_context";

export const DEV_RUN_STATUS_LABELS: Record<DevRunStatus | "idle", string> = {
  idle: "待命",
  queued: "排隊",
  running: "執行中",
  waiting: "等待處理",
  success: "成功",
  failed: "失敗",
  cancelled: "已取消",
  timeout: "逾時",
};

export function normalizeDevRunStatus(
  s: string | null | undefined,
): DevRunStatus | "idle" {
  switch (s) {
    case "queued":
    case "running":
    case "waiting":
    case "success":
    case "failed":
    case "cancelled":
    case "timeout":
      return s;
    case "resolved":
      return "success";
    case "rejected":
    case "needs_context":
      return "waiting";
    default:
      return "idle";
  }
}

export type DevRunSource = "auto" | "manual";

export type DevRunReportStatus =
  | "resolved"
  | "rejected"
  | "needs_context"
  | "failed"
  | "timeout"
  | "cancelled";

export interface DevRunTests {
  ran: boolean;
  passed: number | null;
  failed: number | null;
  summary: string | null;
  // Per-command parse (pytest / npm test / tsc / next build).
  commands?: {
    command: string;
    exit: number | null;
    passed: number | null;
    failed: number | null;
  }[];
}

export type DevRunAppliedStatus =
  | "applied"
  | "not_applied"
  | "violation"
  | "n/a";

export interface DevRunApplied {
  status: DevRunAppliedStatus;
  files: string[];
}

export type DevRunVerificationStatus = "passed" | "failed" | "none";

export interface DevRunVerification {
  status: DevRunVerificationStatus;
}

// meta.json (existing fields + DESIGN section 3 additive fields).
export interface DevRunMeta {
  run_id: string;
  ticket_id: string;
  status: DevRunStatus | DevRunLegacyStatus;
  started_at: string | null;
  finished_at: string | null;
  error_kind?: string | null;
  model: string;
  instruction_preview: string;
  source?: DevRunSource;
  title?: string | null;
  instruction?: string | null;
  parent_run_id?: string | null;
  attempt?: number;
  pid?: number | null;
  pgid?: number | null;
  queued_at?: string | null;
  files_read?: string[];
  files_changed?: string[];
  tests?: DevRunTests | null;
  applied?: DevRunApplied | null;
  verification?: DevRunVerification | null;
  failure_summary?: string | null;
  report_status?: DevRunReportStatus | null;
  safety_flags?: string[] | null;
  // Written on every terminal state (collect_changes); same rows as
  // GET /runs/{id}/diff `files`.
  diff_files: DevRunDiffFile[];
  // Big Guy's own report fields (handoff/A.md additive extras).
  report_summary?: string;
  actions_taken?: string[];
  tests_reported?: string[];
  // Auto-fix rolled the live tree back after a failed verification.
  rollback?: { restored: string[]; deleted: string[] };
}

// POST /api/devfix/runs {instruction, title?}
export interface DevRunStartRequest {
  instruction: string;
  title?: string;
}

export interface DevRunStartResponse {
  run_id: string;
  ticket_id: string;
  status: "queued" | "running";
}

// GET /api/devfix/runs
export interface DevRunsListResponse {
  runs: DevRunMeta[];
}

// GET /api/devfix/runs/{id}?after=N (unchanged shape + meta)
export interface DevRunReadResponse {
  meta: DevRunMeta;
  lines: string[];
  next_cursor: number;
}

// Scrubbed runtime event. The runtime defines the payload; these are the
// keys the log view relies on, everything else is passed through.
export interface DevRunEvent {
  ts?: string;
  type?: string;
  kind?: string;
  tool?: string;
  text?: string;
  message?: string;
  path?: string;
  command?: string;
  [key: string]: unknown;
}

// GET /api/devfix/runs/{id}/events?after=N
export interface DevRunEventsResponse {
  events: DevRunEvent[];
  next_cursor: number;
}

export type DevRunDiffStatus = "added" | "modified" | "deleted";

export interface DevRunDiffFile {
  path: string;
  additions: number;
  deletions: number;
  status: DevRunDiffStatus;
}

// GET /api/devfix/runs/{id}/diff
export interface DevRunDiffResponse {
  patch: string;
  files: DevRunDiffFile[];
}

// 409 body when the local run layer is disabled.
export interface LocalRunsDisabledDetail {
  detail: string;
  reason?: string;
}
