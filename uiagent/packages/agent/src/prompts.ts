/**
 * AI Agent Prompt Templates
 * System prompts and few-shot examples for generating UI specs
 */

import type { UISpec, PromptTemplate, FewShotExample } from '@ui-agent/types';

export const SYSTEM_PROMPT = `你是一個專業的 Prometheus 監控系統 UI 規格產生器。

你的任務：
1. 理解使用者用自然語言描述的監控需求
2. 分析需要查詢什麼資料（metric、時間範圍、聚合方式）
3. 選擇最適合的視覺化方式（圖表類型）
4. 產生符合 UI Spec Schema 的 JSON

核心原則：
- 只輸出有效的 JSON，不要有額外的解釋文字
- PromQL 必須正確且安全（禁止寫入、刪除操作）
- 時間範圍必須明確（使用 "now-Xh" 格式或 ISO 8601）
- Widget 類型選擇要符合資料特性

可用的 Widget 類型：
- time_series_chart: 時序趨勢圖（適合：CPU、記憶體、網路流量的趨勢）
- metric_card: 單一指標卡片（適合：當前值、總數、百分比）
- gauge: 儀表盤（適合：使用率、百分比）
- table: 資料表格（適合：Top N、列表）
- bar_chart: 長條圖（適合：分類資料比較）
- heatmap: 熱力圖（適合：多維度分佈）

PromQL 安全規則：
- 只允許讀取類查詢
- 常用函數：rate(), irate(), avg(), sum(), max(), min(), topk(), bottomk()
- 禁止使用：eval(), __開頭的內部 metric

時間格式：
- 相對時間：now, now-1h, now-24h, now-7d
- 絕對時間：ISO 8601 (2026-01-05T00:00:00Z)

輸出格式要求：
- 必須是有效的 JSON
- version 固定為 "1.0"
- 每個 widget 必須有唯一的 id
- position 的 row 和 col 從 1 開始
- 預設使用 12 欄 grid layout`;

export const FEW_SHOT_EXAMPLES: FewShotExample[] = [
  {
    user_query: '顯示過去 1 小時的 CPU 使用率',
    expected_output: {
      version: '1.0',
      metadata: {
        title: 'CPU 使用率監控',
        description: '過去 1 小時的 CPU 使用率趨勢',
        created_at: new Date().toISOString(),
        intent: 'show_metric_trend',
      },
      layout: {
        type: 'grid',
        columns: 12,
        gap: 'md',
      },
      widgets: [
        {
          id: 'cpu-usage-chart',
          type: 'time_series_chart',
          position: { row: 1, col: 1, colspan: 12, rowspan: 6 },
          config: {
            title: 'CPU 使用率',
            chart_type: 'line',
            y_axis_unit: 'percent',
            legend_position: 'bottom',
            show_grid: true,
          },
          data_source: {
            type: 'prometheus',
            query:
              '100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',
            time_range: {
              start: 'now-1h',
              end: 'now',
              step: '30s',
            },
          },
        },
      ],
    },
    reasoning:
      '使用者想看 CPU 使用率的趨勢，所以選擇 time_series_chart。使用 irate 計算瞬時變化率，並從 100 減去 idle 時間來得到使用率。',
  },
  {
    user_query: '我想看現在記憶體使用了多少',
    expected_output: {
      version: '1.0',
      metadata: {
        title: '記憶體使用狀態',
        description: '當前記憶體使用情況',
        created_at: new Date().toISOString(),
        intent: 'show_current_status',
      },
      layout: {
        type: 'grid',
        columns: 12,
        gap: 'md',
      },
      widgets: [
        {
          id: 'mem-used-card',
          type: 'metric_card',
          position: { row: 1, col: 1, colspan: 4, rowspan: 3 },
          config: {
            title: '已使用記憶體',
            metric_type: 'current',
            unit: 'bytes',
            threshold: {
              warning: 0.8,
              critical: 0.9,
            },
          },
          data_source: {
            type: 'prometheus',
            query: 'node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes',
            time_range: {
              start: 'now-5m',
              end: 'now',
            },
          },
        },
        {
          id: 'mem-usage-gauge',
          type: 'gauge',
          position: { row: 1, col: 5, colspan: 4, rowspan: 3 },
          config: {
            title: '記憶體使用率',
            min: 0,
            max: 100,
            unit: 'percent',
            threshold: {
              warning: 80,
              critical: 90,
            },
          },
          data_source: {
            type: 'prometheus',
            query:
              '(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100',
            time_range: {
              start: 'now-5m',
              end: 'now',
            },
          },
        },
        {
          id: 'mem-total-card',
          type: 'metric_card',
          position: { row: 1, col: 9, colspan: 4, rowspan: 3 },
          config: {
            title: '總記憶體',
            metric_type: 'current',
            unit: 'bytes',
          },
          data_source: {
            type: 'prometheus',
            query: 'node_memory_MemTotal_bytes',
            time_range: {
              start: 'now-5m',
              end: 'now',
            },
          },
        },
      ],
    },
    reasoning:
      '使用者想看當前狀態，所以使用 metric_card 和 gauge。提供三個指標：已使用量、使用率和總量，讓使用者全面了解記憶體狀況。',
  },
  {
    user_query: '顯示網路流量最高的 5 台機器',
    expected_output: {
      version: '1.0',
      metadata: {
        title: 'Top 5 網路流量',
        description: '網路流量最高的 5 台機器',
        created_at: new Date().toISOString(),
        intent: 'show_top_n',
      },
      layout: {
        type: 'grid',
        columns: 12,
        gap: 'md',
      },
      widgets: [
        {
          id: 'top-network-table',
          type: 'table',
          position: { row: 1, col: 1, colspan: 12, rowspan: 6 },
          config: {
            title: 'Top 5 網路流量機器',
            columns: [
              { key: 'instance', label: '機器', sortable: true },
              {
                key: 'rx_bytes',
                label: '接收流量',
                sortable: true,
                format: 'bytes',
              },
              {
                key: 'tx_bytes',
                label: '發送流量',
                sortable: true,
                format: 'bytes',
              },
              {
                key: 'total_bytes',
                label: '總流量',
                sortable: true,
                format: 'bytes',
              },
            ],
            pagination: {
              enabled: false,
              page_size: 5,
            },
          },
          data_source: {
            type: 'prometheus',
            query:
              'topk(5, sum by (instance) (rate(node_network_receive_bytes_total[5m]) + rate(node_network_transmit_bytes_total[5m])))',
            time_range: {
              start: 'now-5m',
              end: 'now',
            },
          },
        },
      ],
    },
    reasoning:
      '使用者想看 Top N 排名，所以使用 table widget。使用 topk() 函數取前 5 名，並用 rate() 計算網路流量的變化率。',
  },
  {
    user_query: '過去 24 小時磁碟使用率的變化',
    expected_output: {
      version: '1.0',
      metadata: {
        title: '磁碟使用率趨勢',
        description: '過去 24 小時磁碟使用率變化',
        created_at: new Date().toISOString(),
        intent: 'show_metric_trend',
      },
      layout: {
        type: 'grid',
        columns: 12,
        gap: 'md',
      },
      widgets: [
        {
          id: 'disk-usage-chart',
          type: 'time_series_chart',
          position: { row: 1, col: 1, colspan: 12, rowspan: 6 },
          config: {
            title: '磁碟使用率',
            chart_type: 'area',
            y_axis_unit: 'percent',
            legend_position: 'right',
            show_grid: true,
            smooth_curve: true,
          },
          data_source: {
            type: 'prometheus',
            query:
              '(1 - node_filesystem_avail_bytes{fstype!~"tmpfs|fuse.*"} / node_filesystem_size_bytes{fstype!~"tmpfs|fuse.*"}) * 100',
            time_range: {
              start: 'now-24h',
              end: 'now',
              step: '5m',
            },
          },
        },
      ],
    },
    reasoning:
      '24 小時的趨勢圖，使用 area chart 更能看出變化。過濾掉 tmpfs 等臨時檔案系統。step 設為 5m 以獲得合適的資料點數量。',
  },
];

export const PROMETHEUS_MONITORING_TEMPLATE: PromptTemplate = {
  id: 'prometheus-monitoring',
  name: 'Prometheus Monitoring UI Generator',
  system_prompt: SYSTEM_PROMPT,
  few_shot_examples: FEW_SHOT_EXAMPLES,
  constraints: [
    '輸出必須是有效的 JSON',
    'PromQL 只能使用安全的讀取操作',
    '時間範圍不超過 90 天',
    'Widget 數量不超過 20 個',
    '每個 widget id 必須唯一',
  ],
};

/**
 * 建構完整的 prompt 給 LLM
 */
export function buildPrompt(userQuery: string, context?: any): string {
  const examples = FEW_SHOT_EXAMPLES.map(
    (ex, i) =>
      `範例 ${i + 1}:
使用者: ${ex.user_query}
輸出: ${JSON.stringify(ex.expected_output, null, 2)}`
  ).join('\n\n');

  return `${SYSTEM_PROMPT}

參考範例：
${examples}

現在請根據以下使用者需求產生 UI Spec：
使用者: ${userQuery}

輸出（只輸出 JSON，不要有其他文字）:`;
}
