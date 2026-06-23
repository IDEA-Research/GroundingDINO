# Architecture

## MVP Runtime Flow

The currently runnable flow is:

```text
frontend/src/demoDashboard.ts
  -> DashboardRenderer
  -> WidgetRenderer
  -> widgets/LineChartWidget.tsx
  -> frontend/src/api/queryApi.ts
  -> POST /api/query/widget
  -> QueryService mock data
  -> Recharts graph refresh
```

This is the flow to verify first. LLM routes are optional for the MVP.

## Frontend Structure

```text
frontend/src/
  App.tsx
  main.tsx
  demoDashboard.ts
  pages/DashboardBuilder.tsx
  api/queryApi.ts
  components/DashboardRenderer.tsx
  components/WidgetRenderer.tsx
  components/widgets/LineChartWidget.tsx
  components/widgets/StatCardWidget.tsx
  components/widgets/ThresholdCardWidget.tsx
  components/widgets/TableWidget.tsx
  types/dashboard.ts
```

Responsibilities:

- `App.tsx`: app entry component.
- `DashboardBuilder.tsx`: hosts chat controls and the dashboard frame. It starts with the demo dashboard loaded.
- `demoDashboard.ts`: static MVP `DashboardSpec`.
- `DashboardRenderer.tsx`: renders dashboard title and widgets.
- `WidgetRenderer.tsx`: dispatches by `widget.type`.
- `LineChartWidget.tsx`: renders real Recharts line graphs.
- `useWidgetData.ts`: polls the backend query API.
- `types/dashboard.ts`: TypeScript dashboard and query response contracts.

## Backend Structure

```text
backend/app/
  main.py
  api/query_routes.py
  services/query_service.py
  specs/widget_spec.py
  specs/dashboard_spec.py
  prometheus/client.py
  prometheus/promql_validator.py
```

MVP responsibilities:

- `main.py`: FastAPI app and router registration.
- `api/query_routes.py`: receives `WidgetSpec` and returns chart-friendly series.
- `services/query_service.py`: validates the query and returns mock or real Prometheus-shaped data converted to frontend points.
- `specs/widget_spec.py`: defines `WidgetSpec` and `QuerySpec`.
- `prometheus/client.py`: used only when `PROMETHEUS_MODE` is not `mock`.
- `prometheus/promql_validator.py`: basic allowlist validation.

## Query Response Shape

The frontend expects:

```text
series[].name
series[].unit
series[].points[].timestamp
series[].points[].value
metadata.source
```

`metadata.source` is `mock` in the default MVP setup.

## Environment Defaults

The runnable default is:

```text
LLM_PROVIDER=mock
PROMETHEUS_MODE=mock
```

With these defaults:

- no OpenRouter API key is required
- no Prometheus server is required
- the demo dashboard should render immediately

## Optional / Not Required For MVP Demo

These modules exist, but the graph demo should not depend on them:

- `api/agent_routes.py`
- `agents/task_model_agent.py`
- `agents/dashboard_spec_agent.py`
- `agents/patch_agent.py`
- `llm/openrouter_client.py`
- `services/version_service.py`
- `services/task_model_service.py`

They are partial infrastructure for future natural-language generation and dashboard evolution.

## Implemented And Runnable Now

- Default dashboard renders on page load.
- Recharts line charts render real data series.
- Blood pressure chart renders systolic and diastolic lines.
- Stat card and threshold card read query results.
- Widgets poll `/api/query/widget`.
- Backend mock mode returns changing points.
- Backend can start with `uvicorn app.main:app` from inside `backend/`.

## Partially Implemented

- LLM-based create/patch dashboard endpoints.
- File-based versions, patch logs, rollback, and task model history.
- Real Prometheus client and basic validator.

## Planned / TODO

- Production persistence and auth.
- Complete audit/version UI.
- Strong Prometheus query validation and cost controls.
- Streaming transport.
- Branching/merging histories.
