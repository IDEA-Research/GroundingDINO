# 專案結構說明

本文件說明 UI Agent 專案的目錄結構和各個組件的職責。

## 目錄結構

```
ui-agent/
├── packages/                    # Monorepo 套件
│   ├── types/                  # 類型定義套件
│   │   ├── src/
│   │   │   ├── ui-spec.ts     # UI Spec 核心類型
│   │   │   ├── prometheus.ts  # Prometheus 相關類型
│   │   │   ├── agent.ts       # AI Agent 類型
│   │   │   └── index.ts       # 統一匯出
│   │   ├── package.json
│   │   └── tsconfig.json
│   │
│   ├── validator/              # UI Spec 驗證器
│   │   ├── src/
│   │   │   ├── schema.ts      # JSON Schema 定義
│   │   │   ├── validator.ts   # 驗證邏輯實作
│   │   │   └── index.ts
│   │   ├── package.json
│   │   └── tsconfig.json
│   │
│   ├── data-source/            # 資料來源連接器
│   │   ├── src/
│   │   │   ├── prometheus-executor.ts  # Prometheus 查詢執行器
│   │   │   └── index.ts
│   │   ├── package.json
│   │   └── tsconfig.json
│   │
│   ├── agent/                  # AI Agent
│   │   ├── src/
│   │   │   ├── prompts.ts     # Prompt 模板與範例
│   │   │   └── index.ts
│   │   ├── package.json
│   │   └── tsconfig.json
│   │
│   ├── renderer/               # React UI Renderer（待實作）
│   │   ├── src/
│   │   │   ├── components/    # Widget 組件
│   │   │   ├── hooks/         # React Hooks
│   │   │   └── index.tsx
│   │   └── package.json
│   │
│   └── api/                    # 後端 API 服務（待實作）
│       ├── src/
│       │   ├── routes/        # API 路由
│       │   ├── controllers/   # 控制器
│       │   └── index.ts
│       └── package.json
│
├── examples/                   # 範例與演示
│   └── ui-spec-example.json   # 完整的 UI Spec 範例
│
├── plans/                      # 設計文件
│   └── ui-agent-architecture.md  # 完整架構設計
│
├── package.json               # 根 package.json
├── turbo.json                 # Turborepo 設定
├── tsconfig.base.json         # 基礎 TypeScript 設定
├── README.md                  # 主要文件
├── QUICKSTART.md              # 快速開始指南
├── PROJECT_STRUCTURE.md       # 本文件
└── .gitignore                 # Git 忽略設定
```

## 套件說明

### 1. @ui-agent/types

**職責**: 定義整個系統的 TypeScript 類型

**核心類型**:
- `UISpec` - 完整的 UI 規格結構
- `Widget` - 所有 Widget 類型的聯合類型
- `PrometheusDataSource` - Prometheus 查詢介面
- `ValidationResult` - 驗證結果

**依賴**: 無

**被依賴**: 所有其他套件

### 2. @ui-agent/validator

**職責**: 驗證 UI Spec 的有效性和安全性

**核心功能**:
- JSON Schema 驗證
- Widget 位置衝突檢查
- 時間範圍合理性驗證
- PromQL 安全性檢查
- Widget ID 唯一性驗證

**依賴**: `@ui-agent/types`, `ajv`, `ajv-formats`

**主要 API**:
```typescript
validateUISpec(spec: unknown): ValidationResult
```

### 3. @ui-agent/data-source

**職責**: 執行 Prometheus 查詢並轉換資料格式

**核心功能**:
- 安全的 PromQL 執行
- 查詢結果快取
- 自動重試機制
- 時間範圍解析
- 資料格式轉換

**依賴**: `@ui-agent/types`, `node-fetch`

**主要 API**:
```typescript
class PrometheusExecutor {
  execute(dataSource: PrometheusDataSource): Promise<TimeSeriesData>
  validateQuery(query: string): QueryValidationResult
}
```

### 4. @ui-agent/agent

**職責**: 提供 AI Agent 的 Prompt 模板和範例

**核心功能**:
- System Prompt 定義
- Few-shot 範例庫
- Prompt 建構工具

**依賴**: `@ui-agent/types`

**主要 API**:
```typescript
buildPrompt(userQuery: string, context?: any): string
```

**包含的範例**:
- CPU 使用率趨勢
- 記憶體使用狀態
- Top N 網路流量
- 磁碟使用率變化

### 5. @ui-agent/renderer (待實作)

**職責**: 根據 UI Spec 渲染實際 React 組件

**規劃功能**:
- Widget Factory
- Layout Engine
- 資料載入與快取
- 互動功能實作

**技術棧**:
- React 18+
- React Query
- Apache ECharts / Chart.js
- Tailwind CSS

### 6. @ui-agent/api (待實作)

**職責**: 提供 RESTful API 服務

**規劃 API**:
- `POST /api/v1/spec/generate` - 產生 UI Spec
- `POST /api/v1/spec/validate` - 驗證 UI Spec
- `POST /api/v1/data/execute` - 執行查詢

**技術棧**:
- Express.js / FastAPI
- OpenAPI 3.0
- JWT 認證

## 資料流程

```
使用者輸入
    ↓
┌─────────────────┐
│  AI Agent       │ (@ui-agent/agent)
│  buildPrompt()  │
└─────────────────┘
    ↓ Prompt
┌─────────────────┐
│  LLM API        │ (OpenAI/Claude)
└─────────────────┘
    ↓ JSON
┌─────────────────┐
│  Validator      │ (@ui-agent/validator)
│  validateUISpec()│
└─────────────────┘
    ↓ Valid Spec
┌─────────────────┐
│  Renderer       │ (@ui-agent/renderer)
│  UIRenderer     │
└─────────────────┘
    ↓ Needs Data
┌─────────────────┐
│  Data Source    │ (@ui-agent/data-source)
│  execute()      │
└─────────────────┘
    ↓ Time Series
┌─────────────────┐
│  Prometheus     │
└─────────────────┘
```

## 開發工作流程

### 1. 開發新功能

```bash
# 在對應的套件目錄下開發
cd packages/validator

# 監聽模式編譯
pnpm dev

# 在其他終端測試
pnpm test
```

### 2. 建置所有套件

```bash
# 根目錄執行
pnpm build

# 或只建置特定套件
pnpm --filter @ui-agent/types build
```

### 3. 新增套件

```bash
# 建立新套件目錄
mkdir -p packages/new-package/src

# 建立 package.json
cd packages/new-package
pnpm init

# 更新依賴
pnpm add @ui-agent/types --workspace
```

## 套件依賴關係

```
@ui-agent/types (基礎類型)
    ↓
    ├── @ui-agent/validator
    ├── @ui-agent/data-source
    ├── @ui-agent/agent
    │
    ↓
@ui-agent/renderer (整合所有套件)
    │
    ↓
@ui-agent/api (後端服務)
```

## 檔案命名規範

- **類型定義**: `*.ts` (不含實作邏輯)
- **實作檔案**: `*.ts`
- **測試檔案**: `*.test.ts` 或 `*.spec.ts`
- **配置檔案**: `*.config.js` 或 `*.json`
- **範例檔案**: `*.example.ts` 或 `examples/*.json`

## 程式碼風格

- TypeScript strict mode
- 使用 `const` 而非 `let`
- 優先使用 functional programming
- 所有公開 API 必須有 JSDoc 註解
- 類型優先於 `any`

## 測試策略

### 單元測試
- 每個套件獨立測試
- 使用 Vitest
- 覆蓋率目標: 80%+

### 整合測試
- 測試多個套件的互動
- 使用真實的 UI Spec 範例

### E2E 測試 (待實作)
- 使用 Playwright
- 測試完整的使用者流程

## 版本管理

採用 Semantic Versioning (semver):
- `1.0.0` - 穩定版本
- `1.1.0` - 新增功能
- `1.1.1` - Bug 修復

## 發布流程

```bash
# 1. 更新版本
pnpm changeset

# 2. 建置
pnpm build

# 3. 測試
pnpm test

# 4. 發布
pnpm changeset publish
```

## 下一步開發重點

1. **完成 React Renderer**
   - 實作 Widget Factory
   - 開發基礎 Widget 組件
   - 整合 React Query

2. **建立後端 API**
   - 實作 OpenAPI 規格
   - 整合 LLM API
   - 新增認證機制

3. **效能優化**
   - 實作查詢快取
   - 優化大量資料渲染
   - 新增 Service Worker

4. **擴充功能**
   - 支援更多資料源
   - 新增更多 Widget 類型
   - 實作儀表板模板庫
