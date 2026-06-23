/**
 * UI Spec Type Definitions
 * Core types for the UI Agent specification system
 */

import type { WidgetCodePatch } from './agent';

// ============================================
// Base Types
// ============================================

export type TimeRange = {
  start: string; // ISO 8601 or relative time (e.g., "now-1h")
  end: string;
  step?: string; // e.g., "15s", "1m"
};

export type Position = {
  row: number;
  col: number;
  colspan?: number;
  rowspan?: number;
};

// ============================================
// Data Source Types
// ============================================

export type PrometheusDataSource = {
  type: 'prometheus';
  query: string; // PromQL
  time_range: TimeRange;
  label_filters?: Record<string, string>;
};

export type DataSource = PrometheusDataSource;

// ============================================
// Widget Configuration Types
// ============================================

export type ChartType = 'line' | 'area' | 'bar' | 'scatter';
export type LegendPosition = 'top' | 'bottom' | 'left' | 'right' | 'none';
export type YAxisUnit = 'percent' | 'bytes' | 'seconds' | 'count' | 'custom';

export type TimeSeriesChartConfig = {
  title: string;
  chart_type: ChartType;
  y_axis_unit: YAxisUnit;
  y_axis_custom_unit?: string;
  legend_position: LegendPosition;
  show_grid?: boolean;
  smooth_curve?: boolean;
  color_scheme?: string[];
};

export type MetricCardConfig = {
  title: string;
  metric_type: 'current' | 'avg' | 'max' | 'min' | 'sum';
  unit: YAxisUnit;
  custom_unit?: string;
  threshold?: {
    warning: number;
    critical: number;
  };
  trend_enabled?: boolean;
};

export type GaugeConfig = {
  title: string;
  min: number;
  max: number;
  unit: YAxisUnit;
  custom_unit?: string;
  threshold?: {
    warning: number;
    critical: number;
  };
};

export type TableConfig = {
  title: string;
  columns: Array<{
    key: string;
    label: string;
    sortable?: boolean;
    format?: 'number' | 'bytes' | 'duration' | 'timestamp';
  }>;
  pagination?: {
    enabled: boolean;
    page_size: number;
  };
};

export type HeatmapConfig = {
  title: string;
  x_axis_label: string;
  y_axis_label: string;
  color_scale?: 'viridis' | 'plasma' | 'inferno' | 'magma';
};

export type BarChartConfig = {
  title: string;
  orientation: 'horizontal' | 'vertical';
  stacked?: boolean;
  y_axis_unit: YAxisUnit;
  custom_unit?: string;
};

export type AlertPanelConfig = {
  title: string;
  group_by?: string[];
  severity_filter?: Array<'critical' | 'warning' | 'info'>;
};

export type WidgetConfig =
  | TimeSeriesChartConfig
  | MetricCardConfig
  | GaugeConfig
  | TableConfig
  | HeatmapConfig
  | BarChartConfig
  | AlertPanelConfig;

// ============================================
// Widget Types
// ============================================

export type BuiltinWidgetType =
  | 'time_series_chart'
  | 'metric_card'
  | 'gauge'
  | 'table'
  | 'heatmap'
  | 'bar_chart'
  | 'alert_panel';

export type CustomWidgetType = `custom:${string}`;

export type WidgetType = BuiltinWidgetType | CustomWidgetType;

export type BaseWidget = {
  id: string;
  type: WidgetType;
  position: Position;
  data_source: DataSource;
};

export type TimeSeriesWidget = BaseWidget & {
  type: 'time_series_chart';
  config: TimeSeriesChartConfig;
};

export type MetricCardWidget = BaseWidget & {
  type: 'metric_card';
  config: MetricCardConfig;
};

export type GaugeWidget = BaseWidget & {
  type: 'gauge';
  config: GaugeConfig;
};

export type TableWidget = BaseWidget & {
  type: 'table';
  config: TableConfig;
};

export type HeatmapWidget = BaseWidget & {
  type: 'heatmap';
  config: HeatmapConfig;
};

export type BarChartWidget = BaseWidget & {
  type: 'bar_chart';
  config: BarChartConfig;
};

export type AlertPanelWidget = BaseWidget & {
  type: 'alert_panel';
  config: AlertPanelConfig;
};

export type CustomWidget = BaseWidget & {
  type: CustomWidgetType;
  config: Record<string, unknown>;
};

export type Widget =
  | TimeSeriesWidget
  | MetricCardWidget
  | GaugeWidget
  | TableWidget
  | HeatmapWidget
  | BarChartWidget
  | AlertPanelWidget
  | CustomWidget;

// ============================================
// Layout Types
// ============================================

export type LayoutType = 'grid' | 'flex' | 'stack';
export type Gap = 'none' | 'sm' | 'md' | 'lg' | 'xl';

export type Layout = {
  type: LayoutType;
  columns?: number; // For grid layout
  gap?: Gap;
};

// ============================================
// Action Types
// ============================================

export type ActionType =
  | 'time_range_picker'
  | 'time_options'
  | 'refresh'
  | 'export'
  | 'filter'
  | 'custom';

export type Action = {
  id: string;
  type: ActionType;
  label: string;
  icon?: string;
  config?: Record<string, unknown>;
};

// ============================================
// Metadata Types
// ============================================

export type IntentType =
  | 'show_metric_trend'
  | 'show_current_status'
  | 'show_comparison'
  | 'show_top_n'
  | 'show_alerts'
  | 'show_distribution';

export type Metadata = {
  title: string;
  description?: string;
  created_at: string; // ISO 8601
  intent: IntentType;
  tags?: string[];
  auto_refresh?: {
    enabled: boolean;
    interval: number; // seconds
  };
};

// ============================================
// Main UI Spec Type
// ============================================

export type UISpec = {
  version: string;
  metadata: Metadata;
  layout: Layout;
  widgets: Widget[];
  actions?: Action[];
};

// ============================================
// Validation Error Types
// ============================================

export type ValidationErrorType =
  | 'SCHEMA_VIOLATION'
  | 'INVALID_QUERY'
  | 'INVALID_TIME_RANGE'
  | 'MISSING_REQUIRED_FIELD'
  | 'INVALID_WIDGET_TYPE'
  | 'POSITION_CONFLICT';

export type ValidationError = {
  type: ValidationErrorType;
  message: string;
  path?: string;
  widget_id?: string;
};

export type ValidationResult = {
  valid: boolean;
  errors: ValidationError[];
  warnings?: string[];
};

// ============================================
// Dashboard API Types
// ============================================

export type DashboardSummary = {
  dashboardId: string;
  versionToken: string;
  updatedAt: string;
};

export type DashboardPayload = DashboardSummary & {
  uiSpec: UISpec;
  widgetPatches: WidgetCodePatch[];
};

export type GenerateDashboardResponse = {
  success: true;
  uiSpec: UISpec;
  warnings?: string[];
  retryCount: number;
  strategyUsed: string;
  reasoning?: string;
  coverageDiagnostics?: unknown;
  widgetPatches: WidgetCodePatch[];
  widget_patches: WidgetCodePatch[];
} & DashboardSummary;

export type DashboardMetaResponse = {
  success: true;
} & DashboardSummary;

export type DashboardDataResponse = {
  success: true;
} & DashboardPayload;
