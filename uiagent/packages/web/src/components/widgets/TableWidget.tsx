import React, { useEffect, useState } from 'react';
import type { TableWidget as TableWidgetType, TimeSeriesData } from '@ui-agent/types';
import { fetchPrometheusData } from '../../services/prometheus';

interface TableWidgetProps {
  widget: TableWidgetType;
}

const TableWidget: React.FC<TableWidgetProps> = ({ widget }) => {
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

  // Transform time series data to table rows
  const rows = transformToTableRows(data, config.columns);

  return (
    <div className="card h-full overflow-auto">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">{config.title}</h3>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {config.columns.map((col) => (
                <th
                  key={col.key}
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  {col.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {rows.length === 0 ? (
              <tr>
                <td
                  colSpan={config.columns.length}
                  className="px-6 py-4 text-center text-gray-500"
                >
                  無數據
                </td>
              </tr>
            ) : (
              rows.map((row, idx) => (
                <tr key={idx} className="hover:bg-gray-50">
                  {config.columns.map((col) => (
                    <td key={col.key} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {formatCell(row[col.key], col.format)}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

function transformToTableRows(
  data: TimeSeriesData | null,
  columns: Array<{ key: string; label: string; format?: string }>
): any[] {
  if (!data || data.series.length === 0) return [];

  return data.series.map((series) => {
    const latestValue = series.data[series.data.length - 1]?.value || 0;
    
    // Create the base row with labels
    const row: any = {
      ...series.labels,
    };
    
    // Map the latest value to all non-label columns
    // This allows the table to display the value under different column names
    for (const col of columns) {
      if (col.key === 'instance') {
        row[col.key] = series.labels.instance || series.name;
      } else if (!series.labels[col.key]) {
        // If the column is not a label, map it to the latest value
        row[col.key] = latestValue;
      }
    }
    
    return row;
  });
}

function formatCell(value: any, format?: string): string {
  if (value === null || value === undefined) return '-';
  
  switch (format) {
    case 'number':
      return typeof value === 'number' ? value.toFixed(2) : value.toString();
    case 'percent':
      return typeof value === 'number' ? `${value.toFixed(1)}%` : value.toString();
    case 'bytes':
      return typeof value === 'number' ? formatBytes(value) : value.toString();
    default:
      return value.toString();
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

export default TableWidget;
