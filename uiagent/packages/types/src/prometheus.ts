/**
 * Prometheus API Types
 */

// ============================================
// Query Result Types
// ============================================

export type PrometheusMetricType = 'counter' | 'gauge' | 'histogram' | 'summary';

export type PrometheusLabel = Record<string, string>;

export type PrometheusValue = [number, string]; // [timestamp, value]

export type PrometheusMetric = {
  metric: PrometheusLabel;
  value?: PrometheusValue;
  values?: PrometheusValue[];
};

export type PrometheusResultType = 'matrix' | 'vector' | 'scalar' | 'string';

export type PrometheusQueryResult = {
  resultType: PrometheusResultType;
  result: PrometheusMetric[];
};

export type PrometheusResponse = {
  status: 'success' | 'error';
  data?: PrometheusQueryResult;
  error?: string;
  errorType?: string;
  warnings?: string[];
};

// ============================================
// Time Series Data (Internal Format)
// ============================================

export type TimeSeriesDataPoint = {
  timestamp: number;
  value: number;
};

export type TimeSeries = {
  name: string;
  labels: PrometheusLabel;
  data: TimeSeriesDataPoint[];
};

export type TimeSeriesData = {
  series: TimeSeries[];
  query: string;
  time_range: {
    start: number;
    end: number;
  };
};

// ============================================
// Query Execution Types
// ============================================

export type QueryType = 'instant' | 'range';

export type QueryOptions = {
  timeout?: number; // milliseconds
  max_points?: number;
  step?: string;
};

export type QueryRequest = {
  query: string;
  type: QueryType;
  start?: string | number;
  end?: string | number;
  time?: string | number; // For instant queries
  step?: string;
  options?: QueryOptions;
};

// ============================================
// Security & Validation Types
// ============================================

export type QueryValidationResult = {
  valid: boolean;
  sanitized_query?: string;
  errors?: string[];
  warnings?: string[];
};

export type AllowedMetricPattern = {
  pattern: string | RegExp;
  description: string;
};

export type QuerySecurityPolicy = {
  allowed_functions: string[];
  forbidden_patterns: RegExp[];
  max_time_range_seconds: number;
  max_series_limit: number;
  allowed_metrics: AllowedMetricPattern[];
};
