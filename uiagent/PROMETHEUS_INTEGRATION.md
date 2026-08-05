# Prometheus 整合說明

## 概述

Prometheus 已經成功整合到 UI Agent 系統中。以下是整合的關鍵組件和使用方式。

## 安裝與編譯

由於新增了 `@ui-agent/data-source` package，需要重新編譯專案：

```bash
# 使用 conda 環境中的 pnpm
conda activate uiagent  # 或你的環境名稱
pnpm install
pnpm build
```

這會按照正確的順序編譯所有 packages：
1. `@ui-agent/types`
2. `@ui-agent/data-source` (新增)
3. `@ui-agent/validator`
4. `@ui-agent/agent`
5. `@ui-agent/api`
6. `@ui-agent/web`

## 架構

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   前端      │ HTTP │ API Server  │ HTTP │ Prometheus  │
│  Widgets    │─────▶│/api/query   │─────▶│   Server    │
└─────────────┘      └─────────────┘      └─────────────┘
                            │
                            ▼
                    ┌─────────────────┐
                    │ PrometheusExecutor│
                    │  - 查詢驗證      │
                    │  - 安全檢查      │
                    │  - 數據轉換      │
                    └─────────────────┘
```

## 關鍵組件

### 1. PrometheusExecutor (`packages/data-source/src/prometheus-executor.ts`)

負責執行 Prometheus 查詢的核心組件：

- **安全驗證**：檢查 PromQL 查詢是否符合安全政策
- **查詢執行**：支援自動重試機制
- **數據轉換**：將 Prometheus 響應轉換為內部時間序列格式
- **錯誤處理**：提供詳細的錯誤信息

功能特性：
```typescript
- 允許的函數白名單 (rate, irate, avg, sum, topk 等)
- 禁止的模式黑名單 (delete, drop, eval 等)
- 最大時間範圍限制 (90 天)
- 最大序列數限制 (10000)
```

### 2. API 端點 (`packages/api/src/index.ts`)

#### POST `/api/query`

執行 Prometheus 查詢的 API 端點。

**請求格式：**
```json
{
  "dataSource": {
    "type": "prometheus",
    "query": "rate(node_cpu_seconds_total[5m])",
    "time_range": {
      "start": "now-1h",
      "end": "now",
      "step": "30s"
    }
  }
}
```

**響應格式：**
```json
{
  "success": true,
  "data": {
    "series": [
      {
        "name": "node_cpu_seconds_total{instance=\"localhost:9100\"}",
        "labels": {
          "instance": "localhost:9100",
          "mode": "idle"
        },
        "data": [
          { "timestamp": 1234567890000, "value": 0.95 }
        ]
      }
    ],
    "query": "rate(node_cpu_seconds_total[5m])",
    "time_range": { "start": 1234567890000, "end": 1234571490000 }
  }
}
```

### 3. 前端服務 (`packages/web/src/services/prometheus.ts`)

提供統一的 Prometheus 數據獲取介面：

```typescript
import { fetchPrometheusData } from '@/services/prometheus';

const data = await fetchPrometheusData(dataSource);
```

### 4. Widget 組件

所有 Widget 組件已更新為使用真實的 Prometheus 數據：

#### TimeSeriesChart
- 支援 line、area、bar 三種圖表類型
- 自動從 Prometheus 獲取時間序列數據
- 支援多個序列同時顯示
- 載入狀態與錯誤處理

#### MetricCard
- 顯示單一指標值
- 支援 avg、sum、min、max、current 等聚合方式
- 根據閥值顯示不同顏色狀態
- 可選的趨勢顯示

#### GaugeWidget
- 儀表盤視覺化
- 即時顯示當前值
- 支援警告和危險閥值
- SVG 繪製的圓弧儀表

#### TableWidget
- 表格形式顯示數據
- 支援多種格式化選項（number, percent, bytes）
- 自動提取 label 信息
- 每個序列一行

## 配置

### 環境變數

在 `packages/api/.env` 中設定 Prometheus 服務器地址：

```bash
PROMETHEUS_URL=http://localhost:9090
```

### 安全政策

可在初始化 PrometheusExecutor 時自定義安全政策：

```typescript
const executor = new PrometheusExecutor({
  prometheusUrl: 'http://localhost:9090',
  securityPolicy: {
    allowed_functions: ['rate', 'irate', 'avg'],
    forbidden_patterns: [/\bdelete\b/i],
    max_time_range_seconds: 7776000, // 90 days
    max_series_limit: 10000,
  }
});
```

## 使用範例

### 1. 查詢 CPU 使用率

```typescript
const dataSource = {
  type: 'prometheus',
  query: '100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',
  time_range: {
    start: 'now-1h',
    end: 'now',
    step: '30s'
  }
};
```

### 2. 查詢記憶體使用率

```typescript
const dataSource = {
  type: 'prometheus',
  query: '(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100',
  time_range: {
    start: 'now-5m',
    end: 'now'
  }
};
```

### 3. Top N 查詢

```typescript
const dataSource = {
  type: 'prometheus',
  query: 'topk(10, rate(node_cpu_seconds_total[5m]))',
  time_range: {
    start: 'now-5m',
    end: 'now'
  }
};
```

## 時間範圍格式

支援以下格式：

- **相對時間**: `now`, `now-5m`, `now-1h`, `now-1d`
- **ISO 8601**: `2024-01-01T00:00:00Z`
- **Unix 時間戳**: `1704067200` (秒)

時間單位：
- `s`: 秒
- `m`: 分鐘
- `h`: 小時
- `d`: 天

## 錯誤處理

系統會在以下情況拋出錯誤：

1. **SecurityError**: 查詢違反安全政策
2. **RangeError**: 時間範圍超出限制
3. **QueryError**: PromQL 語法錯誤
4. **NetworkError**: 無法連接到 Prometheus 服務器

前端 Widgets 會自動顯示錯誤信息。

## 日誌輸出

PrometheusExecutor 會記錄以下信息：

```
=== PromQL Query ===
Query: rate(node_cpu_seconds_total[5m])
Time Range: { start: 'now-1h', end: 'now' }
===================
```

這些日誌可以幫助調試查詢問題。

## 測試

確保 Prometheus 服務器正在運行：

```bash
# 檢查 Prometheus 是否可訪問
curl http://localhost:9090/api/v1/query?query=up

# 編譯專案
conda activate uiagent
pnpm build

# 啟動 API server
cd packages/api
pnpm dev

# 在另一個終端啟動前端
cd packages/web
pnpm dev
```

然後在瀏覽器中訪問應用並輸入查詢，例如：
- "顯示 CPU 使用率"
- "記憶體使用情況"
- "過去一小時的網路流量趨勢"

## 效能優化

1. **Step 自動計算**: 如果未指定 step，系統會自動計算以生成約 250 個數據點
2. **重試機制**: 失敗的查詢會自動重試最多 3 次，使用指數退避策略
3. **超時設置**: 默認查詢超時為 30 秒

## TypeScript 配置變更

為了支援 `@ui-agent/data-source` package，已更新以下配置：

### `packages/api/package.json`
```json
{
  "dependencies": {
    "@ui-agent/data-source": "workspace:*"
  }
}
```

### `packages/api/tsconfig.json`
```json
{
  "references": [
    { "path": "../data-source" },
    { "path": "../agent" }
  ]
}
```

## 未來改進

- [ ] 添加查詢結果緩存
- [ ] 支援查詢模板
- [ ] 實作查詢歷史記錄
- [ ] 添加查詢性能監控
- [ ] 支援多個 Prometheus 數據源
- [ ] Widget 自動刷新
- [ ] 查詢參數化支援
