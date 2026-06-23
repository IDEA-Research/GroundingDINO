import React, { useEffect, useState } from 'react';
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { TimeSeriesWidget, TimeSeriesData } from '@ui-agent/types';
import { fetchPrometheusData } from '../../services/prometheus';

interface TimeSeriesChartProps {
  widget: TimeSeriesWidget;
}

const TimeSeriesChart: React.FC<TimeSeriesChartProps> = ({ widget }) => {
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
      <div className="card h-full flex items-center justify-center">
        <div className="text-red-500">錯誤: {error}</div>
      </div>
    );
  }

  if (!data || data.series.length === 0) {
    return (
      <div className="card h-full flex items-center justify-center">
        <div className="text-gray-500">無數據</div>
      </div>
    );
  }

  // Transform time series data to chart format
  const chartData = transformToChartData(data);

  const renderChart = () => {
    const commonProps = {
      data: chartData,
      margin: { top: 5, right: 30, left: 20, bottom: 5 },
    };

    const colors = config.color_scheme || ['#3b82f6', '#8b5cf6', '#ec4899', '#10b981', '#f59e0b'];

    switch (config.chart_type) {
      case 'area':
        return (
          <AreaChart {...commonProps}>
            {config.show_grid && <CartesianGrid strokeDasharray="3 3" />}
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            {config.legend_position !== 'none' && <Legend />}
            {data.series.map((series, idx) => (
              <Area
                key={series.name}
                type={config.smooth_curve ? 'monotone' : 'linear'}
                dataKey={series.name}
                stroke={colors[idx % colors.length]}
                fill={colors[idx % colors.length]}
                fillOpacity={0.3}
              />
            ))}
          </AreaChart>
        );
      case 'bar':
        return (
          <BarChart {...commonProps}>
            {config.show_grid && <CartesianGrid strokeDasharray="3 3" />}
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            {config.legend_position !== 'none' && <Legend />}
            {data.series.map((series, idx) => (
              <Bar
                key={series.name}
                dataKey={series.name}
                fill={colors[idx % colors.length]}
              />
            ))}
          </BarChart>
        );
      case 'line':
      default:
        return (
          <LineChart {...commonProps}>
            {config.show_grid && <CartesianGrid strokeDasharray="3 3" />}
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            {config.legend_position !== 'none' && <Legend />}
            {data.series.map((series, idx) => (
              <Line
                key={series.name}
                type={config.smooth_curve ? 'monotone' : 'linear'}
                dataKey={series.name}
                stroke={colors[idx % colors.length]}
                strokeWidth={2}
                dot={false}
              />
            ))}
          </LineChart>
        );
    }
  };

  return (
    <div className="card h-full">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">{config.title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        {renderChart()}
      </ResponsiveContainer>
    </div>
  );
};

// Helper function to transform TimeSeriesData to chart format
function transformToChartData(data: TimeSeriesData): any[] {
  if (data.series.length === 0) return [];

  // Get all unique timestamps
  const timestamps = new Set<number>();
  data.series.forEach(series => {
    series.data.forEach(point => timestamps.add(point.timestamp));
  });

  const sortedTimestamps = Array.from(timestamps).sort((a, b) => a - b);

  // Build chart data
  return sortedTimestamps.map(timestamp => {
    const point: any = {
      time: new Date(timestamp).toLocaleTimeString(),
      timestamp,
    };

    data.series.forEach(series => {
      const dataPoint = series.data.find(p => p.timestamp === timestamp);
      point[series.name] = dataPoint ? dataPoint.value : null;
    });

    return point;
  });
}

export default TimeSeriesChart;
