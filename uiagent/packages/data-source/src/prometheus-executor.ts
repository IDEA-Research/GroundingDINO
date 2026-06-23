/**
 * Prometheus Query Executor
 * Executes PromQL queries safely with validation and caching
 */

import type {
  PrometheusDataSource,
  TimeSeriesData,
  TimeSeries,
  TimeSeriesDataPoint,
  PrometheusResponse,
  QuerySecurityPolicy,
  QueryValidationResult,
} from '@ui-agent/types';

export interface PrometheusExecutorConfig {
  prometheusUrl: string;
  timeout?: number;
  maxRetries?: number;
  securityPolicy?: QuerySecurityPolicy;
}

export class PrometheusExecutor {
  private readonly config: Required<PrometheusExecutorConfig>;
  private readonly defaultPolicy: QuerySecurityPolicy = {
    allowed_functions: [
      'rate',
      'irate',
      'increase',
      'avg',
      'sum',
      'min',
      'max',
      'count',
      'topk',
      'bottomk',
      'histogram_quantile',
      'predict_linear',
    ],
    forbidden_patterns: [
      /\bdelete\b/i,
      /\bdrop\b/i,
      /\beval\b/i,
      /\bexec\b/i,
      /__/g, // Double underscore (internal metrics)
    ],
    max_time_range_seconds: 90 * 24 * 60 * 60, // 90 days
    max_series_limit: 10000,
    allowed_metrics: [],
  };

  constructor(config: PrometheusExecutorConfig) {
    this.config = {
      timeout: 30000,
      maxRetries: 3,
      securityPolicy: this.defaultPolicy,
      ...config,
    };
  }

  /**
   * Execute a Prometheus query and return time series data
   */
  async execute(dataSource: PrometheusDataSource): Promise<TimeSeriesData> {
    // Log the PromQL query
    console.log('\n=== PromQL Query ===');
    console.log('Query:', dataSource.query);
    console.log('Time Range:', dataSource.time_range);
    console.log('===================\n');

    // Validate query security
    const validation = this.validateQuery(dataSource.query);
    if (!validation.valid) {
      throw new SecurityError(
        `Query validation failed: ${validation.errors?.join(', ')}`
      );
    }

    // Parse and validate time range
    const { start, end } = this.parseTimeRange(dataSource.time_range);
    const step = this.parseStep(dataSource.time_range.step, start, end);

    // Validate time range duration
    const rangeDuration = (end - start) / 1000;
    if (rangeDuration > this.config.securityPolicy.max_time_range_seconds) {
      throw new RangeError(
        `Time range exceeds maximum allowed: ${rangeDuration}s > ${this.config.securityPolicy.max_time_range_seconds}s`
      );
    }

    // Execute query with retries
    const response = await this.executeWithRetry(
      dataSource.query,
      start,
      end,
      step
    );

    // Transform to our internal format
    return this.transformResponse(response, dataSource.query, start, end);
  }

  /**
   * Validate query against security policy
   */
  validateQuery(query: string): QueryValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Check for forbidden patterns
    for (const pattern of this.config.securityPolicy.forbidden_patterns) {
      if (pattern.test(query)) {
        errors.push(`Query contains forbidden pattern: ${pattern.source}`);
      }
    }

    // Check query length
    if (query.length > 10000) {
      errors.push('Query exceeds maximum length of 10000 characters');
    }

    // Basic syntax validation
    if (!this.isValidPromQLSyntax(query)) {
      errors.push('Query has invalid PromQL syntax');
    }

    // Check for allowed metrics (if policy specifies)
    if (this.config.securityPolicy.allowed_metrics.length > 0) {
      const metricName = this.extractMetricName(query);
      const isAllowed = this.config.securityPolicy.allowed_metrics.some(
        (pattern) => {
          if (typeof pattern.pattern === 'string') {
            return metricName === pattern.pattern;
          }
          return pattern.pattern.test(metricName);
        }
      );

      if (!isAllowed) {
        errors.push(`Metric ${metricName} is not in the allowed list`);
      }
    }

    return {
      valid: errors.length === 0,
      errors: errors.length > 0 ? errors : undefined,
      warnings: warnings.length > 0 ? warnings : undefined,
    };
  }

  /**
   * Execute query with automatic retries
   */
  private async executeWithRetry(
    query: string,
    start: number,
    end: number,
    step: string,
    attempt = 1
  ): Promise<PrometheusResponse> {
    try {
      const url = new URL('/api/v1/query_range', this.config.prometheusUrl);
      url.searchParams.append('query', query);
      url.searchParams.append('start', (start / 1000).toString());
      url.searchParams.append('end', (end / 1000).toString());
      url.searchParams.append('step', step);

      const controller = new AbortController();
      const timeoutId = setTimeout(
        () => controller.abort(),
        this.config.timeout
      );

      try {
        const response = await fetch(url.toString(), {
          method: 'GET',
          signal: controller.signal,
          headers: {
            Accept: 'application/json',
          },
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          throw new Error(
            `Prometheus API error: ${response.status} ${response.statusText}`
          );
        }

        const data = await response.json() as PrometheusResponse;

        if (data.status === 'error') {
          throw new Error(
            `Prometheus query error: ${data.error} (${data.errorType})`
          );
        }

        // Log the raw Prometheus response
        console.log('\n=== Prometheus Response ===');
        console.log('Status:', data.status);
        console.log('Result Type:', data.data?.resultType);
        console.log('Number of series:', data.data?.result?.length || 0);
        if (data.data?.result && data.data.result.length > 0) {
          console.log('Sample result:', JSON.stringify(data.data.result[0], null, 2));
        }
        console.log('===========================\n');

        return data;
      } finally {
        clearTimeout(timeoutId);
      }
    } catch (error) {
      if (attempt < this.config.maxRetries) {
        // Exponential backoff
        const delay = Math.min(1000 * Math.pow(2, attempt - 1), 10000);
        await new Promise((resolve) => setTimeout(resolve, delay));
        return this.executeWithRetry(query, start, end, step, attempt + 1);
      }
      throw error;
    }
  }

  /**
   * Transform Prometheus response to internal format
   */
  private transformResponse(
    response: PrometheusResponse,
    query: string,
    start: number,
    end: number
  ): TimeSeriesData {
    if (!response.data) {
      return {
        series: [],
        query,
        time_range: { start, end },
      };
    }

    const series: TimeSeries[] = [];

    for (const metric of response.data.result) {
      if (!metric.values) continue;

      const data: TimeSeriesDataPoint[] = metric.values.map(([ts, val]) => ({
        timestamp: ts * 1000, // Convert to milliseconds
        value: parseFloat(val),
      }));

      // Generate series name from labels
      const name = this.generateSeriesName(metric.metric);

      series.push({
        name,
        labels: metric.metric,
        data,
      });
    }

    // Check series limit
    if (series.length > this.config.securityPolicy.max_series_limit) {
      throw new Error(
        `Query returned ${series.length} series, exceeding limit of ${this.config.securityPolicy.max_series_limit}`
      );
    }

    // Log transformed data
    console.log('\n=== Transformed Time Series Data ===');
    console.log('Query:', query);
    console.log('Number of series:', series.length);
    console.log('Time range:', { start: new Date(start).toISOString(), end: new Date(end).toISOString() });
    
    if (series.length > 0) {
      console.log('\nSeries details:');
      series.forEach((s, index) => {
        console.log(`  [${index}] ${s.name}`);
        console.log(`      Labels:`, s.labels);
        console.log(`      Data points: ${s.data.length}`);
        if (s.data.length > 0) {
          const firstPoint = s.data[0];
          const lastPoint = s.data[s.data.length - 1];
          console.log(`      First: ${new Date(firstPoint.timestamp).toISOString()} = ${firstPoint.value}`);
          console.log(`      Last:  ${new Date(lastPoint.timestamp).toISOString()} = ${lastPoint.value}`);
        }
      });
    }
    console.log('====================================\n');

    return {
      series,
      query,
      time_range: { start, end },
    };
  }

  /**
   * Generate a human-readable series name from labels
   */
  private generateSeriesName(labels: Record<string, string>): string {
    const { __name__, ...otherLabels } = labels;
    const name = __name__ || 'unknown';

    if (Object.keys(otherLabels).length === 0) {
      return name;
    }

    const labelStr = Object.entries(otherLabels)
      .map(([k, v]) => `${k}="${v}"`)
      .join(',');

    return `${name}{${labelStr}}`;
  }

  /**
   * Parse time range
   */
  private parseTimeRange(timeRange: {
    start: string;
    end: string;
  }): { start: number; end: number } {
    return {
      start: this.parseTime(timeRange.start),
      end: this.parseTime(timeRange.end),
    };
  }

  /**
   * Parse time string to millisecond timestamp
   */
  private parseTime(timeStr: string): number {
    // Handle relative time
    if (timeStr.startsWith('now')) {
      const now = Date.now();
      if (timeStr === 'now') return now;

      const match = timeStr.match(/^now-(\d+)([smhd])$/);
      if (!match) {
        throw new Error(`Invalid relative time format: ${timeStr}`);
      }

      const value = parseInt(match[1], 10);
      const unit = match[2];

      const multipliers: Record<string, number> = {
        s: 1000,
        m: 60 * 1000,
        h: 60 * 60 * 1000,
        d: 24 * 60 * 60 * 1000,
      };

      return now - value * multipliers[unit];
    }

    // Handle ISO 8601 or Unix timestamp
    const timestamp = Date.parse(timeStr);
    if (isNaN(timestamp)) {
      // Try as Unix timestamp (seconds)
      const unixTs = parseInt(timeStr, 10);
      if (!isNaN(unixTs)) {
        return unixTs * 1000;
      }
      throw new Error(`Invalid time format: ${timeStr}`);
    }

    return timestamp;
  }

  /**
   * Calculate appropriate step based on time range
   */
  private parseStep(
    step: string | undefined,
    start: number,
    end: number
  ): string {
    if (step) return step;

    // Auto-calculate step to get ~250 data points
    const rangeSec = (end - start) / 1000;
    const targetPoints = 250;
    const stepSec = Math.ceil(rangeSec / targetPoints);

    if (stepSec < 15) return '15s';
    if (stepSec < 60) return `${stepSec}s`;
    if (stepSec < 3600) return `${Math.ceil(stepSec / 60)}m`;
    return `${Math.ceil(stepSec / 3600)}h`;
  }

  /**
   * Basic PromQL syntax validation
   */
  private isValidPromQLSyntax(query: string): boolean {
    // Check balanced parentheses, brackets, braces
    const checks = [
      this.checkBalance(query, '(', ')'),
      this.checkBalance(query, '[', ']'),
      this.checkBalance(query, '{', '}'),
    ];

    return checks.every((c) => c);
  }

  /**
   * Check if brackets are balanced
   */
  private checkBalance(str: string, open: string, close: string): boolean {
    let count = 0;
    for (const char of str) {
      if (char === open) count++;
      if (char === close) count--;
      if (count < 0) return false;
    }
    return count === 0;
  }

  /**
   * Extract metric name from query
   */
  private extractMetricName(query: string): string {
    // Simple extraction - gets the first metric name
    const match = query.match(/\b([a-zA-Z_:][a-zA-Z0-9_:]*)\b/);
    return match ? match[1] : 'unknown';
  }
}

/**
 * Custom error classes
 */
export class SecurityError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SecurityError';
  }
}

export class QueryError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'QueryError';
  }
}
