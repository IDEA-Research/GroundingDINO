import type { DashboardSeries, ThresholdSpec } from "../../types/dashboard";

export function latestValue(series: DashboardSeries[]): number | null {
  const first = series[0];
  if (!first) {
    return null;
  }
  const point = first.points[first.points.length - 1];
  if (!point) {
    return null;
  }
  const parsed = Number(point.value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function thresholdState(value: number | null, thresholds: ThresholdSpec[]) {
  if (value === null) {
    return null;
  }
  return thresholds.find((threshold) => {
    if (threshold.operator === "gt") return value > threshold.value;
    if (threshold.operator === "gte") return value >= threshold.value;
    if (threshold.operator === "lt") return value < threshold.value;
    if (threshold.operator === "lte") return value <= threshold.value;
    return value === threshold.value;
  });
}

export function chartRows(series: DashboardSeries[]) {
  const byTimestamp = new Map<number, Record<string, string | number>>();
  for (const item of series) {
    for (const point of item.points) {
      const existing = byTimestamp.get(point.timestamp) ?? {
        timestamp: point.timestamp,
        time: new Date(point.timestamp).toLocaleTimeString()
      };
      existing[item.name] = point.value;
      byTimestamp.set(point.timestamp, existing);
    }
  }
  return [...byTimestamp.values()].sort((left, right) => Number(left.timestamp) - Number(right.timestamp));
}
