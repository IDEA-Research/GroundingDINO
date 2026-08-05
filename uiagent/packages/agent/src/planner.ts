import type { PlannerResult } from '@ui-agent/types';

const BUILTIN_WIDGET_CAPABILITIES: Record<string, string[]> = {
  time_series_chart: [
    'trend',
    'time_series',
    'multi_series',
    'line',
    'area',
    'bar',
    'percentage',
  ],
  metric_card: ['single_value', 'current_value', 'aggregation', 'threshold', 'trend_hint'],
  gauge: ['single_value', 'percentage', 'min_max', 'threshold'],
  table: ['top_n', 'list', 'comparison', 'multi_column'],
};

function containsAny(text: string, keywords: string[]): boolean {
  return keywords.some((k) => text.includes(k));
}

function inferRequiredCapabilities(query: string): string[] {
  const q = query.toLowerCase();
  const caps = new Set<string>();

  if (containsAny(q, ['趨勢', '過去', 'trend', 'time series'])) caps.add('trend');
  if (containsAny(q, ['top', '排名', '列表', 'table'])) caps.add('top_n');
  if (containsAny(q, ['使用率', '%', 'percent', '儀表'])) caps.add('percentage');
  if (containsAny(q, ['即時', 'current', '現在'])) caps.add('single_value');
  if (containsAny(q, ['動畫', '閃爍', 'pulse', 'heatmap', '蜂巢', 'honeycomb'])) caps.add('custom_visual');
  if (containsAny(q, ['互動', '點擊', 'drilldown', '展開'])) caps.add('advanced_interaction');

  if (caps.size === 0) caps.add('single_value');
  return Array.from(caps);
}

function matchBestBaseWidget(required: string[]): { baseWidget: string; score: number; coverage: string[] } {
  let best = { baseWidget: 'metric_card', score: 0, coverage: [] as string[] };

  for (const [widget, caps] of Object.entries(BUILTIN_WIDGET_CAPABILITIES)) {
    const coverage = required.filter((r) => caps.includes(r));
    if (coverage.length > best.score) {
      best = { baseWidget: widget, score: coverage.length, coverage };
    }
  }

  return best;
}

export function planGenerationStrategy(userQuery: string): PlannerResult {
  const normalized = userQuery.toLowerCase();

  // 常見純趨勢需求：直接使用既有 time_series_chart，避免不必要 patch 生成
  if (
    containsAny(normalized, ['過去', 'trend', '趨勢']) &&
    containsAny(normalized, ['cpu', '記憶體', 'memory', '網路', 'network', '磁碟', 'disk'])
  ) {
    return {
      strategy: { mode: 'spec_only' },
      reasoning: '此需求可由既有 time_series_chart 完整滿足，採用 spec_only。',
      widget_analysis: {
        required_capabilities: ['trend', 'time_series'],
        existing_coverage: ['trend', 'time_series'],
        gaps: [],
      },
    };
  }

  const required = inferRequiredCapabilities(userQuery);
  const { baseWidget, score, coverage } = matchBestBaseWidget(required);
  const gaps = required.filter((r) => !coverage.includes(r));

  if (gaps.length === 0) {
    return {
      strategy: { mode: 'spec_only' },
      reasoning: '現有 widget 能完整覆蓋需求能力。',
      widget_analysis: {
        required_capabilities: required,
        existing_coverage: coverage,
        gaps,
      },
    };
  }

  if (score > 0 && gaps.length <= 2) {
    return {
      strategy: { mode: 'inherit_widget', base_widget: baseWidget },
      reasoning: `需求可由 ${baseWidget} 延伸，僅需小幅功能擴展。`,
      widget_analysis: {
        required_capabilities: required,
        existing_coverage: coverage,
        gaps,
      },
    };
  }

  return {
    strategy: { mode: 'new_widget' },
    reasoning: '需求包含現有 widget 無法覆蓋的核心能力，需建立全新 widget。',
    widget_analysis: {
      required_capabilities: required,
      existing_coverage: coverage,
      gaps,
    },
  };
}
