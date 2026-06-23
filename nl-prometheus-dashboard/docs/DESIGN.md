# Design

## Current Goal

This repo is first a runnable live dashboard MVP. The working demo path is:

```text
Demo DashboardSpec
  -> DashboardRenderer
  -> WidgetRenderer
  -> Recharts LineChartWidget
  -> POST /api/query/widget
  -> mock Prometheus time-series data
  -> polling refresh every few seconds
```

The app must run without an OpenRouter key and without a real Prometheus server.

## Research Direction

The longer-term direction can still follow a model/spec based GenUI pattern:

```text
Natural language
  -> MonitoringTaskModel
  -> DashboardSpec
  -> deterministic renderer
  -> live Prometheus data
```

That is a design direction, not the main demo dependency. The current MVP focuses on `DashboardSpec -> React renderer -> backend query API -> mock data`.

## Why The MVP Avoids Generated UI Code

The LLM should not generate React code. UI behavior lives in fixed React components and the backend returns structured data. This keeps the graph demo testable:

- `DashboardSpec` controls which widgets render.
- `WidgetRenderer` dispatches by `widget.type`.
- `LineChartWidget` uses Recharts directly.
- `/api/query/widget` returns frontend-friendly series points.
- Mock mode produces changing values locally.

## Query API Contract

`POST /api/query/widget` returns data shaped for the frontend:

```json
{
  "series": [
    {
      "name": "Heart Rate",
      "unit": "bpm",
      "points": [
        { "timestamp": 1710000000000, "value": 82 },
        { "timestamp": 1710000005000, "value": 84 }
      ]
    }
  ],
  "metadata": {
    "source": "mock"
  }
}
```

The mock provider supports:

- `patient_heart_rate_bpm`
- `patient_spo2_percent`
- `patient_systolic_bp_mmhg`
- `patient_diastolic_bp_mmhg`

For the demo blood pressure widget, the primary metric is systolic and `metadata.secondary_metric` adds diastolic as a second line.

## Default Demo Dashboard

The frontend opens with a local demo `DashboardSpec`. It does not call the LLM first. The demo includes:

- Heart Rate line chart
- SpO2 line chart
- Blood Pressure line chart with systolic and diastolic lines
- Heart Rate stat card
- SpO2 threshold card

Each widget polls `/api/query/widget` using its refresh interval. The default refresh is 5 seconds.

## Implemented And Runnable Now

- React app entry via `frontend/src/App.tsx`.
- Default demo dashboard in `frontend/src/demoDashboard.ts`.
- `DashboardRenderer` and `WidgetRenderer`.
- Recharts line charts using `ResponsiveContainer`, `LineChart`, `Line`, `XAxis`, `YAxis`, `CartesianGrid`, `Tooltip`, and `Legend`.
- Stat and threshold cards driven by query API results.
- `POST /api/query/widget`.
- Mock time-series data in `QueryService` with changing values.
- `PROMETHEUS_MODE=mock` default behavior.
- Basic backend tests for schema, validator, patch apply, query mock data, and sample metrics.
- Frontend TypeScript build.

## Partially Implemented

- Natural-language create and patch routes exist, but they are not required for the default demo.
- `MonitoringTaskModel` exists as a structured intermediate representation, but the MVP demo starts from a static `DashboardSpec`.
- File-based dashboard versions, rollback, task model history, and patch logs exist in backend services, but there is no complete production audit UI.
- Real Prometheus mode has a client and validator, but mock mode is the default tested demo path.
- OpenRouter/local LLM adapters exist, but production LLM behavior is not the MVP acceptance path.

## Planned / TODO

- Stronger real Prometheus integration testing.
- Production storage instead of file storage.
- Auth/RBAC.
- UI for version history, patch logs, and rollback.
- Full PromQL AST validation and query cost budgets.
- SSE/WebSocket streaming.
- Branching/merging dashboard histories.
- Widget explanation UI.
