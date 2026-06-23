import React from 'react';
import MetricCard from '../components/widgets/MetricCard';
import TimeSeriesChart from '../components/widgets/TimeSeriesChart';
import GaugeWidget from '../components/widgets/GaugeWidget';
import TableWidget from '../components/widgets/TableWidget';

export type WidgetComponent = React.ComponentType<{ widget: any }>;

type GeneratedWidgetModule = {
  default?: WidgetComponent;
  widgetType?: string;
};

const generatedWidgetModules = import.meta.glob<GeneratedWidgetModule>(
  '../components/widgets/generated/*.tsx',
  { eager: true }
);

class WidgetRegistry {
  private builtinWidgets = new Map<string, WidgetComponent>();

  private customWidgets = new Map<string, WidgetComponent>();

  constructor() {
    this.builtinWidgets.set('metric_card', MetricCard as WidgetComponent);
    this.builtinWidgets.set('time_series_chart', TimeSeriesChart as WidgetComponent);
    this.builtinWidgets.set('gauge', GaugeWidget as WidgetComponent);
    this.builtinWidgets.set('table', TableWidget as WidgetComponent);

    for (const [, mod] of Object.entries(generatedWidgetModules)) {
      if (!mod?.default) continue;

      const explicitType = typeof mod.widgetType === 'string' ? mod.widgetType : '';
      const type = explicitType;
      if (!type) continue;

      this.customWidgets.set(type, mod.default);
    }
  }

  registerCustom(type: string, component: WidgetComponent): void {
    this.customWidgets.set(type, component);
  }

  isCustomRegistered(type: string): boolean {
    return this.customWidgets.has(type);
  }

  getFallbackBaseType(type: string): string | null {
    if (!type.startsWith('custom:')) return null;
    const inheritMatch = type.match(/^custom:(.+)_enhanced$/);
    if (!inheritMatch) return null;
    return inheritMatch[1] ?? null;
  }

  getWidget(type: string): WidgetComponent | null {
    if (type.startsWith('custom:')) {
      const custom = this.customWidgets.get(type);
      if (custom) return custom;

      // Fallback for inherit mode naming: custom:<base>_enhanced
      const inheritMatch = type.match(/^custom:(.+)_enhanced$/);
      if (inheritMatch) {
        const baseType = inheritMatch[1];
        return this.builtinWidgets.get(baseType) ?? null;
      }

      return null;
    }
    return this.builtinWidgets.get(type) ?? this.customWidgets.get(type) ?? null;
  }
}

export const widgetRegistry = new WidgetRegistry();
