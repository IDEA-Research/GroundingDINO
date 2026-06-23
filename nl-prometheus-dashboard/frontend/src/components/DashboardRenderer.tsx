import type { DashboardSpec } from "../types/dashboard";
import { WidgetRenderer } from "./WidgetRenderer";

interface Props {
  dashboard: DashboardSpec;
  variables: Record<string, string>;
}

export function DashboardRenderer({ dashboard, variables }: Props) {
  return (
    <section className="dashboardSurface">
      <div className="dashboardTitle">
        <div>
          <h1>{dashboard.title}</h1>
          {dashboard.description && <p>{dashboard.description}</p>}
        </div>
        <span>v{dashboard.version}</span>
      </div>
      <div className="widgetGrid">
        {dashboard.widgets.map((widget) => (
          <WidgetRenderer
            key={widget.id}
            widget={widget}
            variables={variables}
            dashboardRefreshIntervalMs={dashboard.refresh_interval_ms}
          />
        ))}
      </div>
    </section>
  );
}

