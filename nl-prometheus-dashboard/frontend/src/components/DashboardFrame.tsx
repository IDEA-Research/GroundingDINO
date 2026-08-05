import { Activity, Database, RefreshCw } from "lucide-react";
import { DashboardRenderer } from "./DashboardRenderer";
import type { DashboardSpec } from "../types/dashboard";

interface Props {
  dashboard: DashboardSpec | null;
  variables: Record<string, string>;
}

export function DashboardFrame({ dashboard, variables }: Props) {
  return (
    <section className="dashboardFrame">
      <div className="frameToolbar">
        <div className="frameTitle">
          <Activity size={18} />
          <span>{dashboard?.title ?? "Live Data Frame"}</span>
        </div>
        <div className="frameMeta">
          <span>
            <Database size={15} />
            Prometheus
          </span>
          <span>
            <RefreshCw size={15} />
            {dashboard ? `${dashboard.refresh_interval_ms / 1000}s` : "5s"}
          </span>
          {dashboard && <span>v{dashboard.version}</span>}
        </div>
      </div>

      <div className="frameBody">
        {dashboard ? (
          <DashboardRenderer dashboard={dashboard} variables={variables} />
        ) : (
          <div className="emptyFrame">
            <Activity size={34} />
            <h1>No dashboard yet</h1>
            <p>The frame will render the live Prometheus widgets requested in the conversation.</p>
          </div>
        )}
      </div>
    </section>
  );
}
