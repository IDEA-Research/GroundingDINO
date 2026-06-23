# Natural Language Prometheus Dashboard MVP

Runnable MVP for a live patient monitoring dashboard.

The first screen works without an LLM key and without a Prometheus server:

```text
Demo DashboardSpec
  -> deterministic React renderer
  -> Recharts widgets
  -> POST /api/query/widget
  -> mock Prometheus time-series data
  -> polling graph refresh
```

Natural-language generation still exists as optional backend infrastructure, but it is not required for the demo dashboard.

## Quick Start

The repo folder is `nl-prometheus-dashboard`. A convenience symlink named `nlp-prom-dashboard` may also exist.

Backend:

```bash
cd nlp-prom-dashboard/backend
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd nlp-prom-dashboard/frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5174
```

You should immediately see:

- dashboard title
- Heart Rate line chart
- SpO2 line chart
- Blood Pressure line chart with systolic and diastolic lines
- Heart Rate stat card
- SpO2 threshold card
- changing mock data refreshed every few seconds

## Default Environment

```text
LLM_PROVIDER=mock
PROMETHEUS_MODE=mock
OPENROUTER_API_KEY=
PROMETHEUS_BASE_URL=http://localhost:9090
```

`PROMETHEUS_MODE=mock` is the runnable MVP default. Set it to another value only when using a real Prometheus server.

## Query API

`POST /api/query/widget` returns frontend-friendly data:

```json
{
  "series": [
    {
      "name": "Heart Rate",
      "unit": "bpm",
      "points": [
        { "timestamp": 1710000000000, "value": 82 }
      ]
    }
  ],
  "metadata": {
    "source": "mock"
  }
}
```

Supported mock metrics:

- `patient_heart_rate_bpm`
- `patient_spo2_percent`
- `patient_systolic_bp_mmhg`
- `patient_diastolic_bp_mmhg`

## Frontend Files

The runnable demo uses:

- `frontend/src/App.tsx`
- `frontend/src/main.tsx`
- `frontend/src/demoDashboard.ts`
- `frontend/src/pages/DashboardBuilder.tsx`
- `frontend/src/components/DashboardRenderer.tsx`
- `frontend/src/components/WidgetRenderer.tsx`
- `frontend/src/components/widgets/LineChartWidget.tsx`
- `frontend/src/components/widgets/StatCardWidget.tsx`
- `frontend/src/components/widgets/ThresholdCardWidget.tsx`
- `frontend/src/components/widgets/TableWidget.tsx`
- `frontend/src/api/queryApi.ts`
- `frontend/src/types/dashboard.ts`

## Tests

Backend:

```bash
python -m pytest backend/tests
```

Frontend:

```bash
npm --prefix frontend run build
```

## Status

Implemented and runnable now:

- default demo dashboard
- Recharts graph rendering
- `/api/query/widget`
- changing mock Prometheus data
- polling refresh
- stat card
- threshold card

Partially implemented:

- natural-language dashboard create/patch
- task model schema and agents
- file-based versioning, rollback, and patch logs
- real Prometheus client and basic validation

Planned:

- production auth/storage
- full audit UI
- stronger Prometheus validation
- streaming updates
