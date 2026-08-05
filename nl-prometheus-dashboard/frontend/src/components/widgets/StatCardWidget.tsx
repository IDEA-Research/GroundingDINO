import type { WidgetSpec } from "../../types/dashboard";
import { latestValue, thresholdState } from "./dataUtils";
import { useWidgetData } from "./useWidgetData";

interface Props {
  widget: WidgetSpec;
  variables: Record<string, string>;
  refreshIntervalMs: number;
}

export function StatCardWidget({ widget, variables, refreshIntervalMs }: Props) {
  const { data, error, loading } = useWidgetData(widget, variables, refreshIntervalMs);
  const value = latestValue(data?.series ?? []);
  const threshold = thresholdState(value, widget.thresholds);

  return (
    <div className={`widget statWidget ${threshold ? `severity-${threshold.severity}` : ""}`}>
      <div className="widgetHeader">
        <h3>{widget.title}</h3>
      </div>
      {loading && <div className="muted">Loading</div>}
      {error && <div className="errorText">{error}</div>}
      {!loading && !error && (
        <>
          <div className="statValue">
            {value ?? "--"} <span>{widget.unit}</span>
          </div>
          {threshold && <div className="thresholdMessage">{threshold.message ?? threshold.label}</div>}
        </>
      )}
    </div>
  );
}

