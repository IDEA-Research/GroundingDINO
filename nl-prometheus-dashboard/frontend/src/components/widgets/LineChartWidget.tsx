import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { WidgetSpec } from "../../types/dashboard";
import { chartRows } from "./dataUtils";
import { useWidgetData } from "./useWidgetData";

interface Props {
  widget: WidgetSpec;
  variables: Record<string, string>;
  refreshIntervalMs: number;
}

export function LineChartWidget({ widget, variables, refreshIntervalMs }: Props) {
  const { data, error, loading } = useWidgetData(widget, variables, refreshIntervalMs);
  const rows = chartRows(data?.series ?? []);
  const colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"];

  return (
    <div className="widget">
      <div className="widgetHeader">
        <h3>{widget.title}</h3>
        <span>{widget.unit}</span>
      </div>
      {loading && <div className="muted">Loading</div>}
      {error && <div className="errorText">{error}</div>}
      {!loading && !error && (
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={rows}>
            <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
            <XAxis dataKey="time" minTickGap={24} />
            <YAxis width={48} />
            <Tooltip />
            <Legend />
            {(data?.series ?? []).map((series, index) => (
              <Line
                key={series.name}
                type="monotone"
                dataKey={series.name}
                stroke={colors[index % colors.length]}
                dot={false}
                strokeWidth={2}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
