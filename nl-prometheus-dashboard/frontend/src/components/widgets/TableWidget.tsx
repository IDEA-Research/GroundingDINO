import type { WidgetSpec } from "../../types/dashboard";
import { latestValue } from "./dataUtils";
import { useWidgetData } from "./useWidgetData";

interface Props {
  widget: WidgetSpec;
  variables: Record<string, string>;
  refreshIntervalMs: number;
}

export function TableWidget({ widget, variables, refreshIntervalMs }: Props) {
  const { data, error, loading } = useWidgetData(widget, variables, refreshIntervalMs);

  return (
    <div className="widget">
      <div className="widgetHeader">
        <h3>{widget.title}</h3>
      </div>
      {loading && <div className="muted">Loading</div>}
      {error && <div className="errorText">{error}</div>}
      {!loading && !error && (
        <table>
          <thead>
            <tr>
              <th>Metric</th>
              <th>Labels</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {(data?.series ?? []).map((series, index) => (
              <tr key={index}>
                <td>{series.name}</td>
                <td>{data?.metadata.source === "mock" ? "mock" : "prometheus"}</td>
                <td>
                  {latestValue([series]) ?? "--"} {widget.unit}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
