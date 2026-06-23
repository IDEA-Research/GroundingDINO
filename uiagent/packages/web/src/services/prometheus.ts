import type { PrometheusDataSource, TimeSeriesData } from '@ui-agent/types';

// Determine API base URL based on environment
const getApiUrl = (endpoint: string): string => {
  // Check if we're in a proxy environment (e.g., Kubeflow)
  const pathname = window.location.pathname;
  
  // If running under a proxy path like /notebook/.../proxy/4000/
  if (pathname.includes('/proxy/')) {
    // Replace port 4000 with 4001 for API server
    const apiPath = pathname.replace(/\/proxy\/\d+\//, '/proxy/4001/');
    return `${window.location.origin}${apiPath}${endpoint}`;
  }
  
  // For local development
  return `http://localhost:4001${endpoint}`;
};

export async function fetchPrometheusData(
  dataSource: PrometheusDataSource
): Promise<TimeSeriesData> {
  const apiUrl = getApiUrl('/api/query');
  const response = await fetch(apiUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ dataSource }),
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch Prometheus data: ${response.statusText}`);
  }

  const result = await response.json();
  return result.data;
}
