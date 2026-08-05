import OpenAI from 'openai';
import path from 'node:path';
import { promises as fs } from 'node:fs';
import { exec as execCb } from 'node:child_process';
import { promisify } from 'node:util';
import { buildPrompt, buildWidgetCodePrompt } from '@ui-agent/agent';
import type {
  GenerationStrategy,
  UISpec,
  ValidationError,
  WidgetCodePatch,
} from '@ui-agent/types';

const exec = promisify(execCb);

// 延遲初始化 OpenAI client，確保環境變數已載入
let openai: OpenAI | null = null;

function getOpenAIClient(): OpenAI {
  if (!openai) {
    openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY || '',
    });
  }
  return openai;
}

type OpenCodeLikeResponse = {
  output_text?: string;
  output?: Array<{
    content?: Array<{
      type?: string;
      text?: string;
    }>;
  }>;
};

function extractResponseText(response: OpenCodeLikeResponse): string {
  if (response.output_text?.trim()) return response.output_text.trim();

  const chunks: string[] = [];
  for (const item of response.output ?? []) {
    for (const content of item.content ?? []) {
      if (content.type === 'output_text' && content.text) {
        chunks.push(content.text);
      }
    }
  }

  return chunks.join('\n').trim();
}

function sanitizeWidgetTypeForFilename(widgetType: string): string {
  return widgetType.replace(/[^a-zA-Z0-9_-]/g, '_');
}

function getGeneratedWidgetFilePath(widgetType: string): string {
  const fileName = `${sanitizeWidgetTypeForFilename(widgetType)}.tsx`;
  return path.resolve(process.cwd(), '..', 'web', 'src', 'components', 'widgets', 'generated', fileName);
}

async function persistGeneratedWidget(widgetType: string, code: string): Promise<string> {
  const filePath = getGeneratedWidgetFilePath(widgetType);
  await fs.mkdir(path.dirname(filePath), { recursive: true });

  const normalized = code.trim();
  const source = `import React from 'react';

export const widgetType = ${JSON.stringify(widgetType)};

${normalized}
`;

  await fs.writeFile(filePath, source, 'utf-8');
  return filePath;
}

async function compileGeneratedWidgetsOnce(): Promise<{ success: boolean; output: string }> {
  try {
    const { stdout, stderr } = await exec('npm run build', {
      cwd: path.resolve(process.cwd(), '..', 'web'),
      timeout: 240_000,
      maxBuffer: 1024 * 1024 * 5,
    });
    return { success: true, output: `${stdout}\n${stderr}`.trim() };
  } catch (error) {
    const e = error as { stdout?: string; stderr?: string; message?: string };
    const output = [e.message, e.stdout, e.stderr].filter(Boolean).join('\n').trim();
    return { success: false, output };
  }
}

async function generateWidgetCodeWithOpenCode(
  client: OpenAI,
  prompt: string,
  systemInstruction: string
): Promise<string> {
  const responsesApi = (
    client as unknown as {
      responses?: {
        create: (params: Record<string, unknown>) => Promise<OpenCodeLikeResponse>;
      };
    }
  ).responses;

  if (!responsesApi?.create) {
    throw new Error('Responses API unavailable in current OpenAI SDK version');
  }

  const response = await responsesApi.create({
    model: process.env.OPEN_CODE_MODEL || 'gpt-5-codex',
    instructions: systemInstruction,
    input: prompt,
    reasoning: { effort: 'medium' },
    text: { verbosity: 'medium' },
    max_output_tokens: 2200,
  });

  return extractResponseText(response);
}

/**
 * 使用 AI 生成 UI Spec
 */
export async function generateUISpec(userPrompt: string): Promise<UISpec> {
  try {
    // Build the complete prompt with examples
    const fullPrompt = buildPrompt(userPrompt);

    // 顯示 prompt 資訊
    console.log('\n=== PROMPT 資訊 ===');
    console.log('System Prompt:');
    console.log(fullPrompt);
    console.log('\nUser Prompt:');
    console.log(userPrompt);
    console.log('==================\n');

    // Check if we have an API key
    if (!process.env.OPENAI_API_KEY) {
      console.warn('No OpenAI API key found, using mock data');
      return generateMockUISpec(userPrompt);
    }

    // Get OpenAI client (初始化會在這時才發生)
    const client = getOpenAIClient();

    // Call OpenAI API
    const completion = await client.chat.completions.create({
      model: 'gpt-4',
      messages: [
        {
          role: 'system',
          content: fullPrompt,
        },
        {
          role: 'user',
          content: userPrompt,
        },
      ],
      temperature: 0.3,
      max_tokens: 2000,
    });

    const response = completion.choices[0]?.message?.content;
    if (!response) {
      throw new Error('No response from OpenAI');
    }

    // 顯示 LLM response
    console.log('\n=== LLM RESPONSE ===');
    console.log(response);
    console.log('====================\n');

    // Parse the JSON response
    const uiSpec = JSON.parse(response) as UISpec;
    return uiSpec;
  } catch (error) {
    console.error('Error calling OpenAI API:', error);
    console.log('Falling back to mock data');
    return generateMockUISpec(userPrompt);
  }
}

/**
 * 使用 AI 修正 UI Spec validation 錯誤
 */
export async function fixUISpec(
  originalPrompt: string,
  invalidUISpec: UISpec,
  validationErrors: ValidationError[]
): Promise<UISpec> {
  try {
    // Check if we have an API key
    if (!process.env.OPENAI_API_KEY) {
      console.warn('No OpenAI API key found, cannot fix UI spec automatically');
      throw new Error('Cannot fix UI spec without API key');
    }

    // Get OpenAI client
    const client = getOpenAIClient();

    // Build the fix prompt
    const errorDetails = validationErrors
      .map((err) => {
        let detail = `- ${err.type}: ${err.message}`;
        if (err.path) detail += `\n  路徑: ${err.path}`;
        if (err.widget_id) detail += `\n  Widget ID: ${err.widget_id}`;
        return detail;
      })
      .join('\n');

    const fixPrompt = `之前生成的 UI Spec 有以下驗證錯誤，請修正這些錯誤並返回正確的 UI Spec JSON：

驗證錯誤：
${errorDetails}

原始使用者需求：
${originalPrompt}

錯誤的 UI Spec：
${JSON.stringify(invalidUISpec, null, 2)}

請修正以上錯誤，確保：
1. 所有必要欄位都存在且格式正確
2. intent 必須是以下其中之一：show_metric_trend, show_current_status, show_comparison, show_top_n, show_alerts, show_distribution
3. 所有 widget 的配置都符合 schema
4. 時間範圍和查詢都是有效的

只返回修正後的完整 UI Spec JSON，不要包含任何額外說明。`;

    console.log('\n=== FIX PROMPT ===');
    console.log(fixPrompt);
    console.log('==================\n');

    // Call OpenAI API to fix the spec
    const completion = await client.chat.completions.create({
      model: 'gpt-4',
      messages: [
        {
          role: 'system',
          content: buildPrompt(''),
        },
        {
          role: 'user',
          content: fixPrompt,
        },
      ],
      temperature: 0.2, // Lower temperature for more consistent fixes
      max_tokens: 2000,
    });

    const response = completion.choices[0]?.message?.content;
    if (!response) {
      throw new Error('No response from OpenAI when fixing UI spec');
    }

    console.log('\n=== FIX RESPONSE ===');
    console.log(response);
    console.log('====================\n');

    // Parse the fixed JSON response
    const fixedUISpec = JSON.parse(response) as UISpec;
    return fixedUISpec;
  } catch (error) {
    console.error('Error fixing UI Spec:', error);
    throw error;
  }
}

/**
 * 依策略生成 widget code patches
 */
export async function generateWidgetPatches(
  userPrompt: string,
  strategy: GenerationStrategy
): Promise<WidgetCodePatch[]> {
  if (strategy.mode === 'spec_only') return [];

  const targetType =
    strategy.mode === 'inherit_widget'
      ? `custom:${strategy.base_widget}_enhanced`
      : 'custom:generated_widget';

  // 無 API key 時回傳 mock patch（可在前端註冊並驗證流程）
  if (!process.env.OPENAI_API_KEY) {
    return [
      {
        widget_type: targetType,
        base_widget:
          strategy.mode === 'inherit_widget' ? strategy.base_widget : undefined,
        description: 'Mock widget patch for local development',
        code: `
module.exports.default = function GeneratedWidget({ widget }) {
  return React.createElement(
    'div',
    { className: 'card h-full' },
    React.createElement('h3', { className: 'text-lg font-semibold text-gray-800 mb-2' }, widget?.config?.title || 'Generated Widget'),
    React.createElement('p', { className: 'text-sm text-gray-600' }, '這是動態生成的 custom widget（mock）')
  );
};
`.trim(),
      },
    ];
  }

  const client = getOpenAIClient();
  const prompt = buildWidgetCodePrompt({
    userQuery: userPrompt,
    strategy,
    targetWidgetType: targetType,
  });

  const systemInstruction =
    '你是資深前端工程師。請輸出可直接放在 .tsx 檔案中的 React 元件程式碼，且必須以 export default function GeneratedWidget({ widget }: { widget: any }) { ... } 為核心。不要加入 markdown code fence，也不要輸出額外說明文字。';

  let code = '';
  const maxRepairAttempts = Number(process.env.WIDGET_CODE_REPAIR_MAX_ATTEMPTS || '5');
  let compileOutput = '';
  let compileSuccess = false;
  let generatedFilePath = '';
  let attempts = 0;

  let workingPrompt = prompt;

  while (attempts < maxRepairAttempts) {
    attempts += 1;
    try {
      code = await generateWidgetCodeWithOpenCode(client, workingPrompt, systemInstruction);
    } catch (openCodeError) {
      console.warn('Open Code SDK generation failed, fallback to chat.completions:', openCodeError);

      const completion = await client.chat.completions.create({
        model: 'gpt-4',
        messages: [
          {
            role: 'system',
            content: systemInstruction,
          },
          {
            role: 'user',
            content: workingPrompt,
          },
        ],
        temperature: 0.2,
        max_tokens: 1800,
      });

      code = completion.choices[0]?.message?.content?.trim() ?? '';
    }

    if (!code) {
      compileOutput = 'No response when generating widget patch';
      continue;
    }

    generatedFilePath = await persistGeneratedWidget(targetType, code);
    const compileResult = await compileGeneratedWidgetsOnce();
    compileOutput = compileResult.output;
    compileSuccess = compileResult.success;

    if (compileSuccess) {
      break;
    }

    workingPrompt = `${prompt}\n\n以下是上次編譯錯誤，請直接修正元件程式碼並回傳完整可編譯版本：\n${compileOutput}`;
  }

  if (!code) throw new Error('No response when generating widget patch');
  if (!compileSuccess) {
    throw new Error(`Generated widget failed to compile after ${attempts} attempts.\n${compileOutput}`);
  }

  return [
    {
      widget_type: targetType,
      base_widget: strategy.mode === 'inherit_widget' ? strategy.base_widget : undefined,
      description: 'AI generated widget patch',
      code,
      generated_file_path: generatedFilePath,
      compile_success: compileSuccess,
      compile_output: compileOutput,
      compile_attempts: attempts,
    },
  ];
}

/**
 * 生成 mock UI Spec (用於測試或沒有 API key 時)
 */
function generateMockUISpec(userPrompt: string): UISpec {
  // Determine intent based on keywords
  const prompt = userPrompt.toLowerCase();
  const isCPU = prompt.includes('cpu') || prompt.includes('處理器');
  const isMemory = prompt.includes('記憶體') || prompt.includes('memory');
  const isNetwork = prompt.includes('網路') || prompt.includes('network');
  const isTopN = prompt.includes('top') || prompt.includes('最高');
  const isTrend = prompt.includes('趨勢') || prompt.includes('trend') || prompt.includes('過去');

  if (isTopN) {
    return createTopNUISpec(userPrompt);
  }

  if (isTrend) {
    return createTrendUISpec(userPrompt);
  }

  if (isCPU) {
    return createCPUUISpec();
  }

  if (isMemory) {
    return createMemoryUISpec();
  }

  // Default: comprehensive dashboard
  return createDefaultUISpec();
}

function createCPUUISpec(): UISpec {
  return {
    version: '1.0',
    metadata: {
      title: 'CPU 使用率監控',
      description: '即時 CPU 使用狀況',
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
        id: 'cpu-card',
        type: 'metric_card',
        position: { row: 1, col: 1, colspan: 4, rowspan: 2 },
        config: {
          title: 'CPU 使用率',
          metric_type: 'avg',
          unit: 'percent',
          threshold: { warning: 70, critical: 90 },
          trend_enabled: true,
        },
        data_source: {
          type: 'prometheus',
          query: '100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',
          time_range: { start: 'now-5m', end: 'now' },
        },
      },
      {
        id: 'cpu-chart',
        type: 'time_series_chart',
        position: { row: 1, col: 5, colspan: 8, rowspan: 4 },
        config: {
          title: 'CPU 使用率趨勢',
          chart_type: 'line',
          y_axis_unit: 'percent',
          legend_position: 'bottom',
          show_grid: true,
        },
        data_source: {
          type: 'prometheus',
          query: '100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',
          time_range: { start: 'now-1h', end: 'now', step: '30s' },
        },
      },
    ],
  };
}

function createMemoryUISpec(): UISpec {
  return {
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
        id: 'mem-gauge',
        type: 'gauge',
        position: { row: 1, col: 1, colspan: 4, rowspan: 3 },
        config: {
          title: '記憶體使用率',
          min: 0,
          max: 100,
          unit: 'percent',
          threshold: { warning: 80, critical: 90 },
        },
        data_source: {
          type: 'prometheus',
          query: '(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100',
          time_range: { start: 'now-5m', end: 'now' },
        },
      },
      {
        id: 'mem-chart',
        type: 'time_series_chart',
        position: { row: 1, col: 5, colspan: 8, rowspan: 3 },
        config: {
          title: '記憶體使用趨勢',
          chart_type: 'area',
          y_axis_unit: 'bytes',
          legend_position: 'bottom',
          show_grid: true,
        },
        data_source: {
          type: 'prometheus',
          query: 'node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes',
          time_range: { start: 'now-1h', end: 'now', step: '30s' },
        },
      },
    ],
  };
}

function createTopNUISpec(prompt: string): UISpec {
  return {
    version: '1.0',
    metadata: {
      title: 'Top N 排名',
      description: prompt,
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
        id: 'top-table',
        type: 'table',
        position: { row: 1, col: 1, colspan: 12, rowspan: 6 },
        config: {
          title: 'Top 排名列表',
          columns: [
            { key: 'instance', label: '機器', sortable: true },
            { key: 'value', label: '數值', sortable: true, format: 'number' },
          ],
          pagination: { enabled: false, page_size: 10 },
        },
        data_source: {
          type: 'prometheus',
          query: 'topk(10, node_cpu_seconds_total)',
          time_range: { start: 'now-5m', end: 'now' },
        },
      },
    ],
  };
}

function createTrendUISpec(prompt: string): UISpec {
  return {
    version: '1.0',
    metadata: {
      title: '趨勢分析',
      description: prompt,
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
        id: 'trend-chart',
        type: 'time_series_chart',
        position: { row: 1, col: 1, colspan: 12, rowspan: 6 },
        config: {
          title: '趨勢圖表',
          chart_type: 'line',
          y_axis_unit: 'percent',
          legend_position: 'bottom',
          show_grid: true,
          smooth_curve: true,
        },
        data_source: {
          type: 'prometheus',
          query: 'rate(node_cpu_seconds_total[5m])',
          time_range: { start: 'now-24h', end: 'now', step: '5m' },
        },
      },
    ],
  };
}

function createDefaultUISpec(): UISpec {
  return {
    version: '1.0',
    metadata: {
      title: '系統監控儀表板',
      description: '完整的系統監控概覽',
      created_at: new Date().toISOString(),
      intent: 'show_current_status',
      tags: ['monitoring', 'system'],
    },
    layout: {
      type: 'grid',
      columns: 12,
      gap: 'md',
    },
    widgets: [
      {
        id: 'cpu-card',
        type: 'metric_card',
        position: { row: 1, col: 1, colspan: 3, rowspan: 2 },
        config: {
          title: 'CPU 使用率',
          metric_type: 'avg',
          unit: 'percent',
          threshold: { warning: 70, critical: 90 },
        },
        data_source: {
          type: 'prometheus',
          query: '100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',
          time_range: { start: 'now-5m', end: 'now' },
        },
      },
      {
        id: 'memory-gauge',
        type: 'gauge',
        position: { row: 1, col: 4, colspan: 3, rowspan: 2 },
        config: {
          title: '記憶體使用率',
          min: 0,
          max: 100,
          unit: 'percent',
          threshold: { warning: 80, critical: 90 },
        },
        data_source: {
          type: 'prometheus',
          query: '(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100',
          time_range: { start: 'now-5m', end: 'now' },
        },
      },
      {
        id: 'disk-card',
        type: 'metric_card',
        position: { row: 1, col: 7, colspan: 3, rowspan: 2 },
        config: {
          title: '磁碟使用率',
          metric_type: 'max',
          unit: 'percent',
          threshold: { warning: 80, critical: 95 },
        },
        data_source: {
          type: 'prometheus',
          query: 'max((1 - node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100)',
          time_range: { start: 'now-5m', end: 'now' },
        },
      },
      {
        id: 'network-card',
        type: 'metric_card',
        position: { row: 1, col: 10, colspan: 3, rowspan: 2 },
        config: {
          title: '網路流量',
          metric_type: 'sum',
          unit: 'bytes',
        },
        data_source: {
          type: 'prometheus',
          query: 'sum(rate(node_network_receive_bytes_total[5m]))',
          time_range: { start: 'now-5m', end: 'now' },
        },
      },
      {
        id: 'cpu-trend',
        type: 'time_series_chart',
        position: { row: 3, col: 1, colspan: 6, rowspan: 4 },
        config: {
          title: 'CPU 使用率趨勢',
          chart_type: 'area',
          y_axis_unit: 'percent',
          legend_position: 'bottom',
          show_grid: true,
        },
        data_source: {
          type: 'prometheus',
          query: '100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',
          time_range: { start: 'now-1h', end: 'now', step: '30s' },
        },
      },
      {
        id: 'memory-trend',
        type: 'time_series_chart',
        position: { row: 3, col: 7, colspan: 6, rowspan: 4 },
        config: {
          title: '記憶體使用趨勢',
          chart_type: 'line',
          y_axis_unit: 'bytes',
          legend_position: 'bottom',
          show_grid: true,
        },
        data_source: {
          type: 'prometheus',
          query: 'node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes',
          time_range: { start: 'now-1h', end: 'now', step: '30s' },
        },
      },
    ],
  };
}
