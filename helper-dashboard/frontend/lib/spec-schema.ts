// Spec schema (frontend mirror of backend/app/specs).
//
// Do NOT add fields like `raw_html`, `script`, `component`, `iframe`,
// `code`. The renderer will not honor them. The backend validator will
// reject them. Adding a new widget type is a Big guy task that must
// update both sides.

export type WidgetType =
  | "line_chart"
  | "stat_card"
  | "gauge"
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
}

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

export interface WidgetThreshold {
  value: number;
  color: string;
  label?: string | null;
}

export interface WidgetSpec {
  id: string;
  type: WidgetType;
  title: string;
  description?: string | null;
  query: QuerySpec;
  position: WidgetPosition;
  encoding?: WidgetEncoding;
  thresholds?: WidgetThreshold[];
  options?: Record<string, unknown>;
  // Present only for type === "decision_flow".
  decision_flow?: DecisionFlowSpec | null;
}

// Real alert event as read from the anomaly store (INC2/INC3). Mirrors the
// backend AlertEvent. The decision_flow widget renders THESE, never the
// AlertListWidget MOCK_ALERTS.
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
  // for the upcoming real OpenCode CLI devfix runtime.
  runtime_used: string | null;
  fallback_reason: string | null;
  // Pre-output review transcript; empty when review is off.
  review_trail: ReviewTrailEvent[];
  // Present when rescue decided the user must clarify.
  clarification_questions: string[] | null;
  // Present when the orchestrator is asking "save this?" — carries
  // the dashboard_id that the next user message answers for.
  save_prompt_for: string | null;
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

export interface SavedDashboardSummary {
  saved_id: string;
  name: string;
  description: string;
  tags: string[];
  widget_types: string[];
  promqls: string[];
  used_count: number;
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
