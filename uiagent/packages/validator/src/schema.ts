/**
 * JSON Schema for UI Spec validation
 */

import type { JSONSchemaType } from 'ajv';
import type { UISpec } from '@ui-agent/types';

export const uiSpecSchema: JSONSchemaType<UISpec> = {
  type: 'object',
  required: ['version', 'metadata', 'layout', 'widgets'],
  properties: {
    version: {
      type: 'string',
      pattern: '^\\d+\\.\\d+$',
    },
    metadata: {
      type: 'object',
      required: ['title', 'created_at', 'intent'],
      properties: {
        title: {
          type: 'string',
          minLength: 1,
          maxLength: 200,
        },
        description: {
          type: 'string',
          maxLength: 1000,
          nullable: true,
        },
        created_at: {
          type: 'string',
          format: 'date-time',
        },
        intent: {
          type: 'string',
          enum: [
            'show_metric_trend',
            'show_current_status',
            'show_comparison',
            'show_top_n',
            'show_alerts',
            'show_distribution',
          ],
        },
        tags: {
          type: 'array',
          items: {
            type: 'string',
          },
          nullable: true,
        },
        auto_refresh: {
          type: 'object',
          required: ['enabled', 'interval'],
          properties: {
            enabled: {
              type: 'boolean',
            },
            interval: {
              type: 'number',
              minimum: 5,
              maximum: 3600,
            },
          },
          nullable: true,
        },
      },
    },
    layout: {
      type: 'object',
      required: ['type'],
      properties: {
        type: {
          type: 'string',
          enum: ['grid', 'flex', 'stack'],
        },
        columns: {
          type: 'number',
          minimum: 1,
          maximum: 24,
          nullable: true,
        },
        gap: {
          type: 'string',
          enum: ['none', 'sm', 'md', 'lg', 'xl'],
          nullable: true,
        },
      },
    },
    widgets: {
      type: 'array',
      minItems: 1,
      maxItems: 20,
      items: {
        type: 'object',
        required: ['id', 'type', 'position', 'data_source', 'config'],
        properties: {
          id: {
            type: 'string',
            minLength: 1,
          },
          type: {
            type: 'string',
            enum: [
              'time_series_chart',
              'metric_card',
              'gauge',
              'table',
              'heatmap',
              'bar_chart',
              'alert_panel',
            ],
          },
          position: {
            type: 'object',
            required: ['row', 'col'],
            properties: {
              row: {
                type: 'number',
                minimum: 1,
              },
              col: {
                type: 'number',
                minimum: 1,
              },
              colspan: {
                type: 'number',
                minimum: 1,
                nullable: true,
              },
              rowspan: {
                type: 'number',
                minimum: 1,
                nullable: true,
              },
            },
          },
          data_source: {
            type: 'object',
            required: ['type', 'query', 'time_range'],
            properties: {
              type: {
                type: 'string',
                const: 'prometheus',
              },
              query: {
                type: 'string',
                minLength: 1,
                maxLength: 10000,
              },
              time_range: {
                type: 'object',
                required: ['start', 'end'],
                properties: {
                  start: {
                    type: 'string',
                  },
                  end: {
                    type: 'string',
                  },
                  step: {
                    type: 'string',
                    nullable: true,
                  },
                },
              },
              label_filters: {
                type: 'object',
                nullable: true,
              },
            },
          },
          config: {
            type: 'object',
            // Config validation is widget-type specific
            // We'll validate this separately based on widget type
          },
        },
      } as any, // Type assertion needed due to complex discriminated union
    },
    actions: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'type', 'label'],
        properties: {
          id: {
            type: 'string',
          },
          type: {
            type: 'string',
            enum: ['time_range_picker', 'refresh', 'export', 'filter', 'custom'],
          },
          label: {
            type: 'string',
          },
          icon: {
            type: 'string',
            nullable: true,
          },
          config: {
            type: 'object',
            nullable: true,
          },
        },
      },
      nullable: true,
    },
  },
};
