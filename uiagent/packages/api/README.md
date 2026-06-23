# UI Agent API Server

這是 UI Agent 的後端 API 服務，負責接收 prompt 並生成 UI Spec。

## 功能特色

- 🤖 AI 驅動：使用 OpenAI GPT-4 生成 UI Spec
- 🔄 Fallback 機制：沒有 API key 時使用智能 mock 資料
- ✅ 自動驗證：生成的 UI Spec 會自動經過驗證
- 🚀 高效能：基於 Express.js 的輕量級服務

## 快速開始

### 1. 安裝依賴

```bash
cd packages/api
pnpm install
```

### 2. 設定環境變數

```bash
cp .env.example .env
# 編輯 .env 檔案，加入你的 OpenAI API Key (可選)
```

### 3. 啟動伺服器

```bash
# 開發模式 (支援熱重載)
pnpm dev

# 建置並啟動
pnpm build
pnpm start
```

伺服器將在 http://localhost:3001 啟動

## API 端點

### POST /api/generate

生成 UI Spec

**Request:**
```json
{
  "prompt": "顯示過去 1 小時的 CPU 使用率"
}
```

**Response:**
```json
{
  "success": true,
  "uiSpec": { /* UISpec object */ },
  "warnings": []
}
```

### GET /api/health

健康檢查端點

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2026-01-05T19:00:00.000Z"
}
```

## 環境變數

- `OPENAI_API_KEY` - OpenAI API 金鑰 (可選，未設定時使用 mock 資料)
- `PORT` - 伺服器埠號 (預設: 3001)

## Mock 資料模式

如果沒有設定 OpenAI API key，系統會根據 prompt 關鍵字智能生成對應的 UI Spec：

- 包含 "CPU" → 生成 CPU 監控儀表板
- 包含 "記憶體" → 生成記憶體監控儀表板
- 包含 "Top" → 生成 Top N 排名表格
- 包含 "趨勢" → 生成趨勢圖表
- 其他 → 生成完整的系統監控儀表板
