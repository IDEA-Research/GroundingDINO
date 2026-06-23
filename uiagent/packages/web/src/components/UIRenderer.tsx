import React from 'react';
import type { UISpec, Widget } from '@ui-agent/types';
import { widgetRegistry } from '../services/widget-registry';

interface UIRendererProps {
  uiSpec: UISpec;
}

const UIRenderer: React.FC<UIRendererProps> = ({ uiSpec }) => {
  const { metadata, layout, widgets, actions } = uiSpec;

  const renderWidget = (widget: Widget) => {
    const Component = widgetRegistry.getWidget(widget.type);
    const isCustomType = widget.type.startsWith('custom:');
    const isCustomRegistered = isCustomType
      ? widgetRegistry.isCustomRegistered(widget.type)
      : true;
    const fallbackBaseType = isCustomType
      ? widgetRegistry.getFallbackBaseType(widget.type)
      : null;
    const isUsingFallback =
      isCustomType && !isCustomRegistered && Boolean(fallbackBaseType);

    if (!Component) {
      return (
        <div key={widget.id} className="card">
          <p className="text-gray-500">
            不支援的 widget 類型: {widget.type}
          </p>
        </div>
      );
    }

    return (
      <div key={widget.id} className="space-y-2">
        {isUsingFallback && (
          <div className="rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-800">
            目前使用預設 widget（{fallbackBaseType}）進行降級顯示，
            custom widget 尚未成功載入，功能可能不完整。
          </div>
        )}
        <Component widget={widget} />
      </div>
    );
  };

  const getGridStyle = (widget: Widget) => {
    const { row, col, colspan = 1, rowspan = 1 } = widget.position;
    return {
      gridColumn: `${col} / span ${colspan}`,
      gridRow: `${row} / span ${rowspan}`,
    };
  };

  return (
    <div className="space-y-6">
      {/* Dashboard Header */}
      <div className="card">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">
          {metadata.title}
        </h2>
        {metadata.description && (
          <p className="text-gray-600">{metadata.description}</p>
        )}
        {metadata.tags && metadata.tags.length > 0 && (
          <div className="flex gap-2 mt-3">
            {metadata.tags.map((tag) => (
              <span
                key={tag}
                className="px-3 py-1 bg-blue-100 text-blue-800 text-sm rounded-full"
              >
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Widgets Grid */}
      <div
        className="grid gap-4"
        style={{
          gridTemplateColumns: `repeat(${layout.columns || 12}, minmax(0, 1fr))`,
          gridAutoRows: 'minmax(80px, auto)',
        }}
      >
        {widgets.map((widget) => (
          <div key={widget.id} style={getGridStyle(widget)}>
            {renderWidget(widget)}
          </div>
        ))}
      </div>

      {/* Actions / Controls */}
      {actions && actions.length > 0 && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-800 mb-3">控制項</h3>
          <div className="flex flex-wrap gap-2">
            {actions.map((action) => {
              const presets =
                action.type === 'time_range_picker' || action.type === 'time_options'
                  ? (((action.config as { presets?: string[] } | undefined)?.presets ||
                      []) as string[])
                  : [];

              return (
                <div key={action.id} className="rounded-md border border-gray-200 bg-gray-50 p-2">
                  <div className="text-sm font-medium text-gray-700">{action.label}</div>
                  {presets.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-2">
                      {presets.map((preset) => (
                        <span
                          key={`${action.id}-${preset}`}
                          className="inline-flex items-center rounded-full bg-blue-100 px-2 py-1 text-xs text-blue-800"
                        >
                          {preset}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* JSON View (for debugging) */}
      <details className="card">
        <summary className="cursor-pointer font-semibold text-gray-700 mb-2">
          查看 UI Spec JSON
        </summary>
        <pre className="bg-gray-50 p-4 rounded-lg overflow-auto text-sm">
          {JSON.stringify(uiSpec, null, 2)}
        </pre>
      </details>
    </div>
  );
};

export default UIRenderer;
