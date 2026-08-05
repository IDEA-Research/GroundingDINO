import type { GenerationStrategy, UISpec, WidgetType } from '@ui-agent/types';

export type RiskLevel = 'low' | 'medium' | 'high';

export type WidgetCapability = {
  configKeys: string[];
  supportedActions: string[];
};

export type CapabilityCatalog = Record<string, WidgetCapability>;

export type CoverageDiagnostics = {
  unsupportedWidgetTypes: string[];
  unsupportedConfigKeys: Array<{
    widgetId: string;
    widgetType: string;
    key: string;
  }>;
  missingActionsOrControls: string[];
  coverageScore: number;
  riskLevel: RiskLevel;
};

const SUPPORTED_ACTIONS = [
  'time_range_picker',
  'time_options',
  'refresh',
  'export',
  'filter',
] as const;

export const BUILTIN_CAPABILITY_CATALOG: CapabilityCatalog = {
  metric_card: {
    configKeys: ['title', 'metric_type', 'unit', 'custom_unit', 'threshold', 'trend_enabled'],
    supportedActions: [...SUPPORTED_ACTIONS],
  },
  time_series_chart: {
    configKeys: [
      'title',
      'chart_type',
      'y_axis_unit',
      'y_axis_custom_unit',
      'legend_position',
      'show_grid',
      'smooth_curve',
      'color_scheme',
    ],
    supportedActions: [...SUPPORTED_ACTIONS],
  },
  gauge: {
    configKeys: ['title', 'min', 'max', 'unit', 'custom_unit', 'threshold'],
    supportedActions: [...SUPPORTED_ACTIONS],
  },
  table: {
    configKeys: ['title', 'columns', 'pagination'],
    supportedActions: [...SUPPORTED_ACTIONS],
  },
};

function toRiskLevel(score: number, hasHighRiskSignals: boolean): RiskLevel {
  if (hasHighRiskSignals || score < 0.7) return 'high';
  if (score < 1) return 'medium';
  return 'low';
}

function normalizeScore(score: number): number {
  if (score <= 0) return 0;
  if (score >= 1) return 1;
  return Number(score.toFixed(3));
}

export function canRenderUISpec(
  spec: UISpec,
  capabilityCatalog: CapabilityCatalog = BUILTIN_CAPABILITY_CATALOG
): CoverageDiagnostics {
  const unsupportedWidgetTypes = new Set<string>();
  const unsupportedConfigKeys: CoverageDiagnostics['unsupportedConfigKeys'] = [];
  const missingActionsOrControls = new Set<string>();

  let totalChecks = 0;
  let passedChecks = 0;

  for (const widget of spec.widgets) {
    totalChecks += 1;

    const widgetType = widget.type as string;
    const cap = capabilityCatalog[widgetType];
    if (!cap) {
      unsupportedWidgetTypes.add(widgetType);
      continue;
    }
    passedChecks += 1;

    const config = (widget.config ?? {}) as Record<string, unknown>;
    const allowedKeys = new Set(cap.configKeys);
    for (const key of Object.keys(config)) {
      totalChecks += 1;
      if (!allowedKeys.has(key)) {
        unsupportedConfigKeys.push({
          widgetId: widget.id,
          widgetType,
          key,
        });
        continue;
      }
      passedChecks += 1;
    }
  }

  const actions = spec.actions ?? [];
  const globalAllowedActions = new Set(SUPPORTED_ACTIONS);
  const declaredWidgetTypes = new Set(
    spec.widgets
      .map((w) => w.type as WidgetType)
      .filter((t) => capabilityCatalog[t as string] !== undefined)
      .map(String)
  );

  for (const action of actions) {
    totalChecks += 1;
    const actionType = String(action.type);

    if (!globalAllowedActions.has(actionType as (typeof SUPPORTED_ACTIONS)[number])) {
      missingActionsOrControls.add(actionType);
      continue;
    }

    const supportedByAnyWidget = Array.from(declaredWidgetTypes).some((widgetType) =>
      capabilityCatalog[widgetType]?.supportedActions.includes(actionType)
    );

    if (!supportedByAnyWidget && declaredWidgetTypes.size > 0) {
      missingActionsOrControls.add(actionType);
      continue;
    }

    passedChecks += 1;
  }

  const highRiskSignals = unsupportedWidgetTypes.size > 0 || missingActionsOrControls.size > 0;
  const rawScore = totalChecks === 0 ? 1 : passedChecks / totalChecks;
  const coverageScore = normalizeScore(rawScore);
  const riskLevel = toRiskLevel(coverageScore, highRiskSignals);

  return {
    unsupportedWidgetTypes: Array.from(unsupportedWidgetTypes),
    unsupportedConfigKeys,
    missingActionsOrControls: Array.from(missingActionsOrControls),
    coverageScore,
    riskLevel,
  };
}

export function strategyFromCoverage(
  diagnostics: CoverageDiagnostics,
  fallbackBaseWidget = 'metric_card'
): GenerationStrategy {
  if (diagnostics.coverageScore === 1) {
    return { mode: 'spec_only' };
  }

  if (diagnostics.coverageScore >= 0.7 && diagnostics.riskLevel !== 'high') {
    return { mode: 'inherit_widget', base_widget: fallbackBaseWidget };
  }

  return { mode: 'new_widget' };
}

