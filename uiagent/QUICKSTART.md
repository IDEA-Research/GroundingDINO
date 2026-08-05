# 快速開始指南

本指南將幫助您快速了解和使用 UI Agent 系統。

## 5 分鐘快速體驗

### 1. 安裝依賴

```bash
# 安裝 pnpm（如果尚未安裝）
npm install -g pnpm

# 安裝專案依賴
pnpm install

# 建置所有套件
pnpm build
```

### 2. 驗證 UI Spec

建立一個簡單的 UI Spec：

```typescript
// test-spec.ts
import { validateUISpec } from '@ui-agent/validator';

const spec = {
  version: '1.0',
  metadata: {
    title: '我的第一個儀表板',
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
      id: 'cpu-usage',
      type: 'metric_card',
      position: { row: 1, col: 1, colspan: 4, rowspan: 2 },
      config: {
        title: 'CPU 使用率',
        metric_type: 'avg',
        unit: 'percent',
      },
      data_source: {
        type: 'prometheus',
        query: 'avg(rate(node_cpu_seconds_total[5m])) * 100',
        time_range: {
          start: 'now-5m',
          end: 'now',
        },
      },
    },
  ],
};

const result = validateUISpec(spec);

if (result.valid) {
  console.log('✅ UI Spec 有效！');
  console.log('Spec:', JSON.stringify(spec, null, 2));
} else {
  console.error('❌ 驗證失敗：');
  result.errors.forEach(err => console.error(`  - ${err.message}`));
}
```

執行：

```bash
npx ts-node test-spec.ts
```

### 3. 使用 AI Agent 產生 UI Spec

```typescript
// ai-generate.ts
import { buildPrompt } from '@ui-agent/agent';
import { validateUISpec } from '@ui-agent/validator';

// 1. 建構 Prompt
const userQuery = '顯示過去 1 小時的 CPU 使用率';
const prompt = buildPrompt(userQuery);

console.log('Prompt 已建立，請將以下內容傳送給 LLM:');
console.log('---');
console.log(prompt);
console.log('---');

// 2. 假設從 LLM 獲得回應（實際應呼叫 OpenAI/Claude API）
const mockLLMResponse = `{
  "version": "1.0",
  "metadata": {
    "title": "CPU 使用率監控",
    "description": "過去 1 小時的 CPU 使用率趨勢",
    "created_at": "${new Date().toISOString()}",
    "intent": "show_metric_trend"
  },
  "layout": {
    "type": "grid",
    "columns": 12,
    "gap": "md"
  },
  "widgets": [
    {
      "id": "cpu-chart",
      "type": "time_series_chart",
      "position": { "row": 1, "col": 1, "colspan": 12, "rowspan": 6 },
      "config": {
        "title": "CPU 使用率",
        "chart_type": "line",
        "y_axis_unit": "percent",
        "legend_position": "bottom",
        "show_grid": true
      },
      "data_source": {
        "type": "prometheus",
        "query": "100 - (avg(irate(node_cpu_seconds_total{mode=\\"idle\\"}[5m])) * 100)",
        "time_range": {
          "start": "now-1h",
          "end": "now",
          "step": "30s"
        }
      }
    }
  ]
}`;

// 3. 解析並驗證
try {
  const generatedSpec = JSON.parse(mockLLMResponse);
  const validation = validateUISpec(generatedSpec);
  
  if (validation.valid) {
    console.log('\n✅ AI 產生的 UI Spec 通過驗證！');
    if (validation.warnings) {
      console.warn('\n⚠️ 警告：');
      validation.warnings.forEach(w => console.warn(`  - ${w}`));
    }
  } else {
    console.error('\n❌ AI 產生的 Spec 無效：');
    validation.errors.forEach(e => console.error(`  - ${e.message}`));
  }
} catch (err) {
  console.error('❌ JSON 解析失敗:', err);
}
```

### 4. 執行 Prometheus 查詢

```typescript
// prometheus-query.ts
import { PrometheusExecutor } from '@ui-agent/data-source';

async function testQuery() {
  const executor = new PrometheusExecutor({
    prometheusUrl: 'http://localhost:9090', // 替換為您的 Prometheus URL
    timeout: 30000,
  });

  try {
    const data = await executor.execute({
      type: 'prometheus',
      query: 'up',
      time_range: {
        start: 'now-5m',
        end: 'now',
        step: '30s',
      },
    });

    console.log('✅ 查詢成功！');
    console.log(`查詢: ${data.query}`);
    console.log(`時間範圍: ${new Date(data.time_range.start).toISOString()} ~ ${new Date(data.time_range.end).toISOString()}`);
    console.log(`序列數量: ${data.series.length}`);
    
    data.series.forEach((series, i) => {
      console.log(`\n序列 ${i + 1}: ${series.name}`);
      console.log(`  標籤:`, series.labels);
      console.log(`  資料點數量: ${series.data.length}`);
      console.log(`  最新值: ${series.data[series.data.length - 1]?.value}`);
    });
  } catch (error) {
    console.error('❌ 查詢失敗:', error);
  }
}

testQuery();
```

執行（需要本地 Prometheus 實例）：

```bash
npx ts-node prometheus-query.ts
```

## 常見使用場景

### 場景 1: 驗證手寫的 UI Spec

```typescript
import { validateUISpec } from '@ui-agent/validator';
import * as fs from 'fs';

const specJson = fs.readFileSync('./examples/ui-spec-example.json', 'utf-8');
const spec = JSON.parse(specJson);

const result = validateUISpec(spec);
console.log(result.valid ? '✅ 有效' : '❌ 無效', result);
```

### 場景 2: 整合 OpenAI API

```typescript
import OpenAI from 'openai';
import { buildPrompt } from '@ui-agent/agent';
import { validateUISpec } from '@ui-agent/validator';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

async function generateUISpec(userQuery: string) {
  // 1. 建構 Prompt
  const prompt = buildPrompt(userQuery);
  
  // 2. 呼叫 OpenAI
  const response = await openai.chat.completions.create({
    model: 'gpt-4',
    messages: [
      {
        role: 'user',
        content: prompt,
      },
    ],
    temperature: 0.3,
  });
  
  // 3. 解析回應
  const content = response.choices[0].message.content;
  const spec = JSON.parse(content);
  
  // 4. 驗證
  const validation = validateUISpec(spec);
  
  if (!validation.valid) {
    throw new Error(`Invalid spec: ${validation.errors.map(e => e.message).join(', ')}`);
  }
  
  return spec;
}

// 使用
generateUISpec('顯示過去 24 小時的記憶體使用趨勢')
  .then(spec => console.log('Generated spec:', spec))
  .catch(err => console.error('Error:', err));
```

### 場景 3: 建立自訂安全策略

```typescript
import { PrometheusExecutor } from '@ui-agent/data-source';

const executor = new PrometheusExecutor({
  prometheusUrl: 'http://localhost:9090',
  securityPolicy: {
    // 只允許特定函數
    allowed_functions: ['rate', 'avg', 'sum'],
    
    // 禁止的模式
    forbidden_patterns: [
      /delete/i,
      /drop/i,
      /process_/i, // 禁止查詢進程相關 metrics
    ],
    
    // 最大時間範圍：7 天
    max_time_range_seconds: 7 * 24 * 60 * 60,
    
    // 最大序列數
    max_series_limit: 1000,
    
    // 白名單 metrics
    allowed_metrics: [
      { pattern: /^node_cpu_/, description: 'CPU metrics' },
      { pattern: /^node_memory_/, description: 'Memory metrics' },
    ],
  },
});
```

## 下一步

1. **探索範例**：查看 [`examples/ui-spec-example.json`](./examples/ui-spec-example.json)
2. **閱讀架構文件**：深入了解系統設計 [`plans/ui-agent-architecture.md`](./plans/ui-agent-architecture.md)
3. **實作 Renderer**：開發 React UI Renderer 來渲染 UI Spec
4. **整合後端**：建立 API 服務整合所有組件

## 常見問題

### Q: 如何處理 AI 產生錯誤的 Spec？

A: 使用嚴格的驗證和重試機制：

```typescript
async function generateWithRetry(query: string, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    const spec = await generateUISpec(query);
    const validation = validateUISpec(spec);
    
    if (validation.valid) {
      return spec;
    }
    
    console.log(`Attempt ${i + 1} failed, retrying...`);
  }
  
  throw new Error('Failed to generate valid spec after retries');
}
```

### Q: 如何自訂 Widget 類型？

A: 在 `@ui-agent/types` 中擴充類型定義，並在 Renderer 中實作對應組件。

### Q: 支援哪些 LLM？

A: 系統與 LLM 無關，只要能輸出 JSON 即可。推薦：
- OpenAI GPT-4
- Anthropic Claude
- 本地部署的 Llama 2/3

### Q: 如何確保 PromQL 安全？

A: 系統內建多層防護：
1. Schema 驗證禁止危險操作
2. 查詢執行器白名單機制
3. 時間範圍與複雜度限制
4. 支援自訂安全策略

## 獲取幫助

- 📖 [完整文件](./README.md)
- 🐛 [回報問題](https://github.com/your-org/ui-agent/issues)
- 💬 [討論區](https://github.com/your-org/ui-agent/discussions)
