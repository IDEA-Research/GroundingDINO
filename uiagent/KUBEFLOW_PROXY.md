# Kubeflow Proxy 環境配置說明

## 問題背景

### 核心問題

在 Kubeflow 或 JupyterHub 的 nginx proxy 環境下運行 Vite 開發服務器時，會遇到以下問題：

1. **路徑錯誤**：Vite 生成的資源使用絕對路徑（從根路徑開始），但在 proxy 環境下需要包含 proxy 前綴
   - ❌ 錯誤: `https://kflow2.cgu.edu.tw/@vite/client`
   - ✅ 正確: `https://kflow2.cgu.edu.tw/notebook/d000018238/a2/proxy/4000/@vite/client`

2. **動態注入的虛擬模組**：
   - `/@vite/client` - Vite 的客戶端腳本
   - `/@react-refresh` - React Fast Refresh 支援
   - 這些模組在運行時動態注入，難以通過靜態轉換修正

3. **HMR WebSocket 連接問題**：
   - 開發模式的熱更新需要 WebSocket 連接
   - Proxy 環境下 WebSocket 路徑可能不正確

## 解決方案

### 🎯 方案 1: 生產構建模式（最推薦）

在 proxy 環境下，**最可靠的方式是使用生產構建**，避免開發模式的複雜性：

```bash
# 1. 構建應用
pnpm build

# 2. 進入構建目錄
cd packages/web/dist

# 3. 啟動靜態文件服務器
python3 -m http.server 4000
```

**優點**：
- ✅ 無 HMR/WebSocket 問題
- ✅ 所有路徑都是相對的
- ✅ 載入速度更快
- ✅ 更接近生產環境

**缺點**：
- ❌ 修改代碼後需要重新構建
- ❌ 無法使用熱更新

### 🔧 方案 2: 配置 Base Path（開發模式）

如果必須使用開發模式，需要配置正確的 base path：

```bash
# 編輯 packages/web/.env.development
VITE_BASE_PATH=/notebook/d000018238/a2/proxy/4000/
```

然後啟動開發服務器：

```bash
# 從專案根目錄執行
pnpm dev

# 或指定 host（重要！）
cd packages/web
vite --host 0.0.0.0
```

**重要設定**：
1. `VITE_BASE_PATH` 必須以 `/` 開頭和結尾
2. 必須加上 `--host 0.0.0.0` 讓伺服器接受來自 proxy 的請求
3. 修改配置後**必須重啟**開發服務器

**已知限制**：
- ⚠️ HMR 可能仍無法正常工作
- ⚠️ 某些動態注入的模組可能有路徑問題
- ⚠️ 需要 proxy 正確轉發 WebSocket 連接

### 🌐 方案 3: SSH Tunnel（開發體驗最佳）

使用 SSH tunnel 將遠端的 localhost:4000 映射到本機：

```bash
# 在本機執行（替換成實際的 server 位址）
ssh -L 4000:localhost:4000 user@kflow2.cgu.edu.tw

# 然後在 server 上啟動開發服務器
pnpm dev

# 在本機瀏覽器訪問
# http://localhost:4000
```

**優點**：
- ✅ 完整的開發體驗（HMR 正常運作）
- ✅ 無需處理 proxy 路徑問題
- ✅ 調試方便

**缺點**：
- ❌ 需要 SSH 存取權限
- ❌ 需要保持 SSH 連接

## 配置文件說明

### vite.config.ts 變更

已更新的配置包含：

1. **動態 base path**：
   ```typescript
   const basePath = process.env.VITE_BASE_PATH || '';
   base: basePath,
   ```

2. **增強的路徑轉換**：
   - 處理 `@react-refresh`、`@vite/client`、`@id/` 等虛擬模組
   - 支援 inline script 中的 import 語句

3. **Host 配置**：
   ```typescript
   server: {
     host: '0.0.0.0',  // 接受來自任何 host 的請求
   }
   ```

### .env.development 配置

已提供兩種模式的清晰說明：

- **模式 1**：`VITE_BASE_PATH=`（空字串，用於本地開發）
- **模式 2**：`VITE_BASE_PATH=/notebook/.../proxy/4000/`（用於 proxy 環境）

## 故障排除

### 1. 檢查實際請求路徑

開啟瀏覽器開發者工具 → Network 分頁，檢查失敗的請求：

```
❌ GET https://kflow2.cgu.edu.tw/@vite/client [404]
✅ GET https://kflow2.cgu.edu.tw/notebook/.../proxy/4000/@vite/client [200]
```

### 2. 查看轉換日誌

啟動開發服務器後，終端會顯示路徑轉換日誌：

```
[RelativePathPlugin] Transforming HTML...
[RelativePathPlugin] Rewriting: from "/@react-refresh" -> from "./@react-refresh"
```

### 3. 確認環境變數已載入

在 `vite.config.ts` 中暫時加入 console.log：

```typescript
console.log('VITE_BASE_PATH:', process.env.VITE_BASE_PATH);
```

### 4. 清除緩存

```bash
# 清除 Vite 緩存
rm -rf packages/web/node_modules/.vite

# 清除瀏覽器緩存（或使用無痕模式）
```

### 5. 檢查 Proxy 配置

確認 Kubeflow/JupyterHub 的 proxy 配置正確轉發請求到 port 4000。

## 快速檢查清單

開發模式（方案 2）：
- [ ] 設定 `VITE_BASE_PATH` 環境變數
- [ ] 使用 `--host 0.0.0.0` 啟動
- [ ] 重啟開發服務器
- [ ] 清除瀏覽器緩存
- [ ] 檢查 Network 分頁的請求路徑
- [ ] 確認 proxy 轉發設定正確

生產模式（方案 1）：
- [ ] 執行 `pnpm build`
- [ ] 啟動靜態文件服務器
- [ ] 透過 proxy URL 訪問

## 建議

對於 **Kubeflow/JupyterHub proxy 環境**，我們建議：

1. **開發時**：使用 SSH tunnel（方案 3）獲得最佳開發體驗
2. **測試時**：使用生產構建（方案 1）確保穩定性
3. **部署時**：使用生產構建 + Nginx/CDN

避免在 proxy 環境下直接使用 Vite 開發模式，因為它的動態特性與 proxy 路徑重寫衝突。
