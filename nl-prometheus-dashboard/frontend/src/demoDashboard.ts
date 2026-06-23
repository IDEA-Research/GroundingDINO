import type { DashboardSpec, WidgetSpec } from "./types/dashboard";

const now = new Date().toISOString();

function rangeQuery(metric: string) {
  return {
    metric,
    promql: null,
    query_type: "range" as const,
    label_matchers: {},
    time_range_seconds: 300,
    step_seconds: 5,
    start_time: null,
    end_time: null
  };
}

function instantQuery(metric: string) {
  return {
    ...rangeQuery(metric),
    query_type: "instant" as const
  };
}

const widgets: WidgetSpec[] = [
  {
    id: "demo_heart_rate_trend",
    title: "Heart Rate",
    type: "line_chart",
    query: rangeQuery("patient_heart_rate_bpm"),
    unit: "bpm",
    thresholds: [
      {
        label: "Low Heart Rate",
        operator: "lt",
        value: 60,
        severity: "critical",
        message: "Heart rate is below normal range"
      },
      {
        label: "High Heart Rate",
        operator: "gt",
        value: 100,
        severity: "warning",
        message: "Heart rate is above normal range"
      }
    ],
    refresh_interval_ms: 5000,
    layout: { x: 0, y: 0, w: 8, h: 3 },
    metadata: {}
  },
  {
    id: "demo_spo2_trend",
    title: "SpO2",
    type: "line_chart",
    query: rangeQuery("patient_spo2_percent"),
    unit: "%",
    thresholds: [
      {
        label: "Low SpO2",
        operator: "lt",
        value: 92,
        severity: "critical",
        message: "SpO2 is below normal range"
      }
    ],
    refresh_interval_ms: 5000,
    layout: { x: 0, y: 3, w: 8, h: 3 },
    metadata: {}
  },
  {
    id: "demo_blood_pressure_trend",
    title: "Blood Pressure",
    type: "line_chart",
    query: rangeQuery("patient_systolic_bp_mmhg"),
    unit: "mmHg",
    thresholds: [],
    refresh_interval_ms: 5000,
    layout: { x: 0, y: 6, w: 8, h: 3 },
    metadata: {
      secondary_metric: "patient_diastolic_bp_mmhg"
    }
  },
  {
    id: "demo_heart_rate_stat",
    title: "Heart Rate Now",
    type: "stat_card",
    query: instantQuery("patient_heart_rate_bpm"),
    unit: "bpm",
    thresholds: [
      {
        label: "High Heart Rate",
        operator: "gt",
        value: 100,
        severity: "warning",
        message: "Heart rate is above normal range"
      }
    ],
    refresh_interval_ms: 5000,
    layout: { x: 8, y: 0, w: 4, h: 2 },
    metadata: {}
  },
  {
    id: "demo_spo2_threshold",
    title: "SpO2 Alert",
    type: "threshold_card",
    query: instantQuery("patient_spo2_percent"),
    unit: "%",
    thresholds: [
      {
        label: "Low SpO2",
        operator: "lt",
        value: 92,
        severity: "critical",
        message: "SpO2 is below normal range"
      }
    ],
    refresh_interval_ms: 5000,
    layout: { x: 8, y: 2, w: 4, h: 2 },
    metadata: {}
  }
];

export const demoDashboard: DashboardSpec = {
  id: "demo_patient_vitals",
  title: "Patient Vitals Demo",
  description: "Runnable MVP dashboard using backend mock Prometheus time-series data.",
  widgets,
  variables: [],
  refresh_interval_ms: 5000,
  time_range_seconds: 300,
  version: 1,
  created_at: now,
  updated_at: now,
  metadata: {
    source: "demo"
  }
};
