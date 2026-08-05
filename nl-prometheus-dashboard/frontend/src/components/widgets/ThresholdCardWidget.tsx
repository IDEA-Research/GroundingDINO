import type { WidgetSpec } from "../../types/dashboard";
import { latestValue, thresholdState } from "./dataUtils";
import { useWidgetData } from "./useWidgetData";

interface Props {
  widget: WidgetSpec;
  variables: Record<string, string>;
  refreshIntervalMs: number;
}

export function ThresholdCardWidget({ widget, variables, refreshIntervalMs }: Props) {
  const { data, error, loading } = useWidgetData(widget, variables, refreshIntervalMs);
  const value = latestValue(data?.series ?? []);
  const threshold = thresholdState(value, widget.thresholds);

  return (
    <div className={`widget thresholdWidget ${threshold ? `severity-${threshold.severity}` : "severity-ok"}`}>
      <div className="widgetHeader">
        <h3>{widget.title}</h3>
      </div>
      {loading && <div className="muted">Loading</div>}
      {error && <div className="errorText">{error}</div>}
      {!loading && !error && (
        <>
          <div className="statusLabel">{threshold ? threshold.severity.toUpperCase() : "OK"}</div>
          <div className="statValue">
            {value ?? "--"} <span>{widget.unit}</span>
          </div>
          <div className="thresholdMessage">{threshold?.message ?? "Within configured thresholds"}</div>
        </>
      )}
    </div>
  );
}

