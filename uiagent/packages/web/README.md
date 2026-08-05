# UI Agent Web Interface

這是 UI Agent 的 Web 介面，讓使用者可以透過自然語言描述來生成 Prometheus 監控儀表板。

## 功能特色

- 🎯 自然語言輸入：用中文或英文描述你想要的監控儀表板
- 🤖 AI 驅動生成：自動產生符合需求的 UI Spec
- 📊 即時渲染：立即顯示生成的監控儀表板
- 🎨 多種視覺化元件：支援圖表、儀表盤、卡片、表格等

## 快速開始

### 1. 安裝依賴

```bash
cd packages/web
pnpm install
```

### 2. 啟動開發伺服器

```bash
pnpm dev
```

應用程式將在 http://localhost:3000 啟動

### 3. 確保 API 服務運行

Web UI 需要連接到 API 服務來生成 UI Spec。請確保 API 服務已在 http://localhost:3001 運行。

## 使用範例

在 prompt 輸入框中輸入以下任何描述：

- "顯示過去 1 小時的 CPU 使用率"
- "我想看現在記憶體使用了多少"
- "顯示網路流量最高的 5 台機器"
- "過去 24 小時磁碟使用率的變化"

系統會自動生成相應的監控儀表板。

## 架構說明

- `src/components/PromptInput.tsx` - 使用者輸入介面
- `src/components/UIRenderer.tsx` - UI Spec 渲染器
- `src/components/widgets/` - 各種視覺化元件
  - `MetricCard.tsx` - 指標卡片元件
  - `TimeSeriesChart.tsx` - 時序圖表元件
  - `GaugeWidget.tsx` - 儀表盤元件
  - `TableWidget.tsx` - 表格元件

## 技術棧

- React 18
- TypeScript
- Vite
- Tailwind CSS
- Recharts (圖表庫)
