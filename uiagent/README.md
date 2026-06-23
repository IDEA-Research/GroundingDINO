# UI Agent - AI-Powered Prometheus Monitoring Dashboard Generator

使用自然語言描述來自動生成 Prometheus 監控儀表板的 AI 系統。

## 🌟 功能特色

- **自然語言輸入**：用中文或英文描述你想要的監控需求
- **AI 智能生成**：自動產生符合需求的 UI 規格
- **即時預覽**：立即看到生成的監控儀表板
- **多種視覺化**：支援時序圖、儀表盤、指標卡片、表格等
- **自動驗證**：確保生成的規格符合標準
- **可擴展架構**：模組化設計，易於擴展新功能

## 🎯 快速開始

### 前置需求

**方法 A (推薦)：使用 Anaconda**
- Anaconda 或 Miniconda
- 下載: https://docs.conda.io/en/latest/miniconda.html

**方法 B：直接安裝**
- Node.js >= 18.0.0
- pnpm >= 8.0.0 或 npm >= 9.0.0

### 🚀 方法 A: 使用 Anaconda (推薦，一鍵啟動)

**生產模式（推薦用於 Kubeflow/JupyterHub proxy 環境）**

```bash
# 1. 設定 Conda 環境（僅需執行一次）
./setup-conda.sh

# 2. 啟動系統（生產構建，避免 proxy 路徑問題）
./start-with-conda.sh
```

**開發模式（僅用於本機開發）**

```bash
# 啟動開發服務器（支援熱模組替換）
./start-dev.sh
```

就這麼簡單！腳本會自動：
- ✅ 啟動 conda 環境
- ✅ 安裝 npm 依賴
- ✅ 建置所有 packages
- ✅ 建立環境變數檔案
- ✅ 同時啟動 API 和 Web 服務

💡 **關於 Proxy 環境**：
- 在 Kubeflow/JupyterHub 等 proxy 環境下，建議使用 `./start-with-conda.sh`（生產模式）
- 這避免了 Vite 開發模式的動態路徑問題
- 詳見 [KUBEFLOW_PROXY.md](KUBEFLOW_PROXY.md)

### 📦 方法 B: 傳統安裝方式

```bash
# 1. 安裝所有依賴
pnpm install  # 或 npm install

# 2. (可選) 設定 OpenAI API Key
cp packages/api/.env.example packages/api/.env
# 編輯 packages/api/.env 加入你的 OPENAI_API_KEY

# 3. 啟動系統
pnpm dev      # 或 npm run dev
```

### 訪問服務

無論使用哪種方法，服務都會在：
- 📊 Web UI: http://localhost:3000
- 🔌 API Server: http://localhost:3001

## 📚 專案結構

```
uiagent/
├── packages/
│   ├── types/           # TypeScript 型別定義
│   ├── validator/       # UI Spec 驗證器
│   ├── agent/          # AI Prompt 模板
│   ├── data-source/    # Prometheus 資料源執行器
│   ├── api/            # 後端 API 服務
│   └── web/            # 前端 Web UI
├── examples/           # 範例檔案
└── plans/             # 架構設計文件
```

## 🚀 使用範例

1. 開啟 Web UI (http://localhost:3000)
2. 在輸入框中輸入自然語言描述，例如：
   - "顯示過去 1 小時的 CPU 使用率"
   - "我想看現在記憶體使用了多少"
   - "顯示網路流量最高的 5 台機器"
   - "過去 24 小時磁碟使用率的變化"
3. 點擊「生成儀表板」
4. 立即看到生成的監控儀表板

## 📦 Package 說明

### @ui-agent/types
核心型別定義，包括：
- UI Spec 結構
- Widget 類型
- 資料源定義
- 驗證錯誤類型

### @ui-agent/validator
UI Spec 驗證器，確保生成的規格符合 schema。

### @ui-agent/agent
AI Agent 的 prompt 模板和範例，用於引導 LLM 生成正確的 UI Spec。

### @ui-agent/data-source
Prometheus 資料源執行器，負責執行 PromQL 查詢。

### @ui-agent/api
後端 API 服務：
- 接收使用者 prompt
- 呼叫 OpenAI API 生成 UI Spec
- 驗證並返回結果
- 提供 fallback mock 資料

### @ui-agent/web
前端 Web UI：
- 使用者輸入介面
- UI Spec 渲染器
- 各種視覺化元件

## 🔧 開發

### 建置所有 packages

```bash
pnpm build
```

### 執行測試

```bash
pnpm test
```

### 型別檢查

```bash
pnpm type-check
```

### Linting

```bash
pnpm lint
```

## 🎨 支援的 Widget 類型

- **time_series_chart** - 時序圖表（折線圖、面積圖、長條圖）
- **metric_card** - 指標卡片（顯示單一數值）
- **gauge** - 儀表盤（百分比視覺化）
- **table** - 資料表格（Top N 排名）
- **bar_chart** - 長條圖（分類比較）
- **heatmap** - 熱力圖（多維度分佈）
- **alert_panel** - 告警面板（顯示告警）

## 🌐 環境變數

### API Server

```bash
OPENAI_API_KEY=your_api_key  # OpenAI API 金鑰 (可選)
PORT=3001                    # API 服務埠號
```

## 📖 文件

- [快速開始指南](QUICKSTART.md)
- [專案結構說明](PROJECT_STRUCTURE.md)
- [架構設計](plans/ui-agent-architecture.md)
- [API 文件](packages/api/README.md)
- [Web UI 文件](packages/web/README.md)

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📄 授權

MIT License
