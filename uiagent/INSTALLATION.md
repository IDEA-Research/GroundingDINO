# 安裝指引

本文件提供詳細的安裝步驟來設定 UI Agent 系統。

## 前置需求

### 方法 1: 使用 Anaconda (推薦)

- **Anaconda** 或 **Miniconda**
- 下載連結: https://docs.conda.io/en/latest/miniconda.html

### 方法 2: 直接安裝

- **Node.js** >= 18.0.0
- **pnpm** >= 8.0.0 (推薦) 或 **npm** >= 9.0.0

## 安裝步驟

### 方法 A: 使用 Anaconda 環境 (推薦)

#### 1. 設定 Conda 環境

```bash
# 執行設定腳本
./setup-conda.sh
```

這個腳本會：
- 建立名為 `uiagent` 的 Conda 環境
- 安裝 Python 3.11、Node.js 20.10.0 和 npm
- 自動設定所有必要工具

#### 2. 啟動系統

```bash
# 使用 Conda 環境啟動（一鍵完成所有設定）
./start-with-conda.sh
```

這個腳本會：
- 啟動 `uiagent` conda 環境
- 自動安裝 npm 依賴（首次執行）
- 建置所有 packages
- 建立環境變數檔案
- 啟動 API 和 Web 服務

就這麼簡單！系統將在：
- Web UI: http://localhost:3000
- API Server: http://localhost:3001

---

### 方法 B: 直接安裝（不使用 Conda）

### 1. 安裝依賴套件

在專案根目錄執行：

```bash
# 如果使用 pnpm (推薦)
pnpm install

# 如果使用 npm
npm install
```

這會自動安裝所有 workspace packages 的依賴。

### 2. 設定 API 服務

```bash
# 複製環境變數範例檔案
cp packages/api/.env.example packages/api/.env

# 編輯 .env 檔案 (可選)
# 如果你有 OpenAI API Key，可以加入：
# OPENAI_API_KEY=sk-your-api-key-here
```

**注意**：即使沒有 OpenAI API Key，系統也能正常運作，會使用智能 mock 資料。

### 3. 建置所有 packages

```bash
# 使用 turbo (推薦)
pnpm build

# 或使用 npm
npm run build
```

## 啟動系統

### 方法 1：同時啟動所有服務 (推薦)

```bash
# 使用 turbo 同時啟動 API 和 Web
pnpm dev
```

### 方法 2：分別啟動服務

**終端 1 - 啟動 API 服務：**
```bash
cd packages/api
pnpm dev
# 或
npm run dev
```

API 服務將在 `http://localhost:3001` 啟動

**終端 2 - 啟動 Web UI：**
```bash
cd packages/web
pnpm dev
# 或
npm run dev
```

Web UI 將在 `http://localhost:3000` 啟動

## 驗證安裝

1. 開啟瀏覽器訪問 `http://localhost:3000`
2. 你應該看到 UI Agent 的輸入介面
3. 嘗試輸入一個 prompt，例如："顯示過去 1 小時的 CPU 使用率"
4. 點擊「生成儀表板」
5. 系統應該生成並顯示監控儀表板

## API 端點測試

你也可以直接測試 API：

```bash
# 健康檢查
curl http://localhost:3001/api/health

# 生成 UI Spec
curl -X POST http://localhost:3001/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "顯示 CPU 使用率"}'
```

## 常見問題

### Q: 安裝時出現 pnpm 未找到

**A:** 安裝 pnpm：
```bash
npm install -g pnpm
```

### Q: API 服務無法啟動

**A:** 檢查：
1. Port 3001 是否被佔用
2. 依賴是否正確安裝：`cd packages/api && pnpm install`
3. 查看終端的錯誤訊息

### Q: Web UI 無法連接到 API

**A:** 確保：
1. API 服務正在運行 (http://localhost:3001)
2. 檢查 `packages/web/vite.config.ts` 中的 proxy 設定
3. 查看瀏覽器的 Console 錯誤訊息

### Q: TypeScript 錯誤

**A:** 執行型別檢查和建置：
```bash
pnpm type-check
pnpm build
```

## 開發模式 vs 生產模式

### 開發模式
- 支援熱重載
- 包含完整的錯誤訊息
- 使用 `pnpm dev`

### 生產模式
```bash
# 建置所有 packages
pnpm build

# 啟動 API 服務
cd packages/api
pnpm start

# 啟動 Web UI (需要另外設定靜態檔案伺服器)
cd packages/web
pnpm preview
```

## 下一步

安裝完成後，請參考：
- [README.md](README.md) - 專案概覽
- [QUICKSTART.md](QUICKSTART.md) - 快速開始指南
- [packages/web/README.md](packages/web/README.md) - Web UI 文件
- [packages/api/README.md](packages/api/README.md) - API 文件
