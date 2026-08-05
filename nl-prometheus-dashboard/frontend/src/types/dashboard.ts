export type WidgetType = "line_chart" | "stat_card" | "threshold_card" | "table";
export type QueryType = "instant" | "range";
export type ThresholdOperator = "gt" | "gte" | "lt" | "lte" | "eq";
export type ThresholdSeverity = "info" | "warning" | "critical";

export interface QuerySpec {
  metric: string;
  promql?: string | null;
  query_type: QueryType;
  label_matchers: Record<string, string>;
  time_range_seconds: number;
  step_seconds: number;
  start_time?: string | null;
  end_time?: string | null;
}

export interface ThresholdSpec {
  label: string;
  operator: ThresholdOperator;
  value: number;
  severity: ThresholdSeverity;
  message?: string | null;
}

export interface WidgetLayoutSpec {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface WidgetSpec {
  id: string;
  title: string;
  type: WidgetType;
  query: QuerySpec;
  unit?: string | null;
  thresholds: ThresholdSpec[];
  refresh_interval_ms?: number | null;
  layout: WidgetLayoutSpec;
  metadata: Record<string, string | number | boolean>;
}

export interface DashboardVariableSpec {
  name: string;
  label: string;
  type: "text" | "select";
  default?: string | null;
  options: string[];
  required: boolean;
  description?: string | null;
}

export interface DashboardSpec {
  id: string;
  title: string;
  description?: string | null;
  widgets: WidgetSpec[];
  variables: DashboardVariableSpec[];
  refresh_interval_ms: number;
  time_range_seconds: number;
  version: number;
  created_at: string;
  updated_at: string;
  metadata: Record<string, string | number | boolean>;
}

export interface PatchOperation {
  op: "add_widget" | "remove_widget" | "update_widget" | "update_dashboard_title" | "update_variable";
  widget_id?: string | null;
  widget?: WidgetSpec | null;
  title?: string | null;
  variable_name?: string | null;
  variable?: DashboardVariableSpec | null;
  updates?: Record<string, unknown> | null;
}

export interface PatchSpec {
  dashboard_id: string;
  operations: PatchOperation[];
  reason?: string | null;
}

export type AnalysisIntent =
  | "trend_monitoring"
  | "threshold_detection"
  | "latest_value_summary"
  | "tabular_review";

export interface TimeContextSpec {
  time_range_seconds: number;
  refresh_interval_ms: number;
  start_time?: string | null;
  end_time?: string | null;
}

export interface TaskConstraintsSpec {
  allowed_metrics: string[];
  max_time_range_seconds: number;
  min_step_interval_seconds: number;
}

export interface MonitoringTaskModel {
  id: string;
  domain: string;
  monitoring_goal: string;
  entities: string[];
  signals: string[];
  relationships: string[];
  analysis_intents: AnalysisIntent[];
  time_context: TimeContextSpec;
  constraints: TaskConstraintsSpec;
  metadata: Record<string, string | number | boolean>;
}

export interface DashboardPoint {
  timestamp: number;
  value: number;
}

export interface DashboardSeries {
  name: string;
  unit?: string | null;
  points: DashboardPoint[];
}

export interface WidgetQueryResponse {
  series: DashboardSeries[];
  metadata: Record<string, unknown>;
}
