import React, { useEffect, useState } from 'react';
import type { MetricCardWidget, TimeSeriesData } from '@ui-agent/types';
import { fetchPrometheusData } from '../../services/prometheus';

interface MetricCardProps {
  widget: MetricCardWidget;
}

const MetricCard: React.FC<MetricCardProps> = ({ widget }) => {
  const { config, data_source } = widget;
  const [data, setData] = useState<TimeSeriesData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (data_source.type === 'prometheus') {
      fetchPrometheusData(data_source)
        .then(setData)
        .catch((err) => setError(err.message))
        .finally(() => setLoading(false));
    }
  }, [data_source]);

  if (loading) {
    return (
      <div className="card h-full flex items-center justify-center">
        <div className="text-gray-500">載入中...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card h-full">
        <h3 className="text-lg font-semibold text-gray-800 mb-2">{config.title}</h3>
        <div className="text-red-500 text-sm">錯誤: {error}</div>
      </div>
    );
  }

  // Calculate metric value based on metric_type
  const value = calculateMetric(data, config.metric_type);
  const formattedValue = formatValue(value, config.unit);

  // Determine status color based on threshold
  const statusColor = getStatusColor(value, config.threshold);

  return (
    <div className="card h-full">
      <h3 className="text-lg font-semibold text-gray-800 mb-2">{config.title}</h3>
      <div className={`text-4xl font-bold ${statusColor} mb-2`}>
        {formattedValue}
      </div>
      {config.trend_enabled && data && data.series.length > 0 && (
        <div className="text-sm text-gray-600">
          {getTrend(data.series[0].data)}
        </div>
      )}
    </div>
  );
};

function calculateMetric(data: TimeSeriesData | null, metricType: string): number {
  if (!data || data.series.length === 0) return 0;

  const allValues = data.series.flatMap(s => s.data.map(d => d.value));
  
  switch (metricType) {
    case 'avg':
      return allValues.reduce((a, b) => a + b, 0) / allValues.length;
    case 'sum':
      return allValues.reduce((a, b) => a + b, 0);
    case 'min':
      return Math.min(...allValues);
    case 'max':
      return Math.max(...allValues);
    case 'current':
    default:
      // Return the latest value
      const lastSeries = data.series[data.series.length - 1];
      return lastSeries.data[lastSeries.data.length - 1]?.value || 0;
  }
}

function formatValue(value: number, unit?: string): string {
  const rounded = Math.round(value * 100) / 100;
  
  switch (unit) {
    case 'percent':
      return `${rounded.toFixed(1)}%`;
    case 'bytes':
      return formatBytes(value);
    case 'seconds':
      return `${rounded.toFixed(2)}s`;
    default:
      return rounded.toLocaleString();
  }
}

function formatBytes(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unitIndex = 0;
  
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex++;
  }
  
  return `${value.toFixed(2)} ${units[unitIndex]}`;
}

function getStatusColor(value: number, threshold?: { warning: number; critical: number }): string {
  if (!threshold) return 'text-blue-600';
  
  if (value >= threshold.critical) return 'text-red-600';
  if (value >= threshold.warning) return 'text-yellow-600';
  return 'text-green-600';
}

function getTrend(dataPoints: Array<{ timestamp: number; value: number }>): string {
  if (dataPoints.length < 2) return '';
  
  const first = dataPoints[0].value;
  const last = dataPoints[dataPoints.length - 1].value;
  const change = ((last - first) / first) * 100;
  
  if (change > 0) {
    return `↑ ${change.toFixed(1)}%`;
  } else if (change < 0) {
    return `↓ ${Math.abs(change).toFixed(1)}%`;
  }
  return '→ 0%';
}

export default MetricCard;
