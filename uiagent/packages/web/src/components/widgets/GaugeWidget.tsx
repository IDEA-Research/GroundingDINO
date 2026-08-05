import React, { useEffect, useState } from 'react';
import type { GaugeWidget as GaugeWidgetType, TimeSeriesData } from '@ui-agent/types';
import { fetchPrometheusData } from '../../services/prometheus';

interface GaugeWidgetProps {
  widget: GaugeWidgetType;
}

const GaugeWidget: React.FC<GaugeWidgetProps> = ({ widget }) => {
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

  // Get the latest value
  const value = getLatestValue(data);
  const percentage = ((value - config.min) / (config.max - config.min)) * 100;
  const statusColor = getStatusColor(value, config.threshold);

  return (
    <div className="card h-full">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">{config.title}</h3>
      <div className="flex flex-col items-center justify-center">
        {/* Gauge visualization */}
        <div className="relative w-48 h-24 mb-4">
          <svg viewBox="0 0 200 100" className="w-full h-full">
            {/* Background arc */}
            <path
              d="M 20 80 A 80 80 0 0 1 180 80"
              fill="none"
              stroke="#e5e7eb"
              strokeWidth="20"
              strokeLinecap="round"
            />
            {/* Value arc */}
            <path
              d="M 20 80 A 80 80 0 0 1 180 80"
              fill="none"
              stroke={getArcColor(statusColor)}
              strokeWidth="20"
              strokeLinecap="round"
              strokeDasharray={`${percentage * 2.51}, 251`}
            />
            {/* Center text */}
            <text
              x="100"
              y="75"
              textAnchor="middle"
              className={`text-2xl font-bold ${statusColor}`}
              fill="currentColor"
            >
              {formatValue(value, config.unit)}
            </text>
          </svg>
        </div>
        {/* Min/Max labels */}
        <div className="flex justify-between w-48 text-sm text-gray-600">
          <span>{config.min}</span>
          <span>{config.max}</span>
        </div>
      </div>
    </div>
  );
};

function getLatestValue(data: TimeSeriesData | null): number {
  if (!data || data.series.length === 0) return 0;
  
  const lastSeries = data.series[data.series.length - 1];
  return lastSeries.data[lastSeries.data.length - 1]?.value || 0;
}

function formatValue(value: number, unit?: string): string {
  const rounded = Math.round(value * 100) / 100;
  
  switch (unit) {
    case 'percent':
      return `${rounded.toFixed(1)}%`;
    case 'bytes':
      return formatBytes(value);
    default:
      return rounded.toFixed(1);
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
  
  return `${value.toFixed(1)}${units[unitIndex]}`;
}

function getStatusColor(value: number, threshold?: { warning: number; critical: number }): string {
  if (!threshold) return 'text-blue-600';
  
  if (value >= threshold.critical) return 'text-red-600';
  if (value >= threshold.warning) return 'text-yellow-600';
  return 'text-green-600';
}

function getArcColor(statusColor: string): string {
  const colorMap: Record<string, string> = {
    'text-red-600': '#dc2626',
    'text-yellow-600': '#ca8a04',
    'text-green-600': '#16a34a',
    'text-blue-600': '#2563eb',
  };
  return colorMap[statusColor] || '#2563eb';
}

export default GaugeWidget;
