# MongoDB 資料查看功能說明

## 📋 功能概述

已在 Web UI 中新增 MongoDB 資料庫查看功能，可以直接在瀏覽器中查看所有儲存在 MongoDB 中的影片分析記錄。

## 🎯 新增的功能

### 1. **後端 API** (web_app.py)

新增了以下 API 端點：

#### `/api/mongodb/videos`
- **功能**: 取得所有影片分析記錄
- **方法**: GET
- **回應格式**:
```json
{
    "success": true,
    "data": [
        {
            "_id": "ObjectId字串",
            "video_name": "影片名稱",
            "video_path": "影片路徑",
            "session_id": "UUID",
            "timestamp": "ISO時間格式",
            "total_frames": 13,
            "status": "completed"
        }
    ],
    "count": 4
}
```

### 2. **前端 UI** (templates/index.html)

#### MongoDB 資料查看區塊
- 在頁面標題下方新增「📊 查看 MongoDB 資料庫」按鈕
- 點擊後展開 MongoDB 資料查看面板
- 顯示所有儲存的影片分析記錄

#### 新增的 JavaScript 函數
1. `toggleMongoDBView()` - 切換 MongoDB 視圖顯示/隱藏
2. `loadMongoDBData()` - 載入 MongoDB 資料
3. `displayMongoDBVideos(videos)` - 顯示影片記錄列表
4. `viewMongoDBDetails(sessionId)` - 查看特定影片的詳細醫療數值

## 🚀 使用方法

### 步驟 1: 重啟 Flask 應用程式

由於新增了路由，需要重啟 Flask：

```bash
# 停止現有的 Flask 進程
pkill -f "python.*web_app.py"

# 重新啟動 Flask
python web_app.py
```

### 步驟 2: 開啟 Web 介面

在瀏覽器中訪問：
```
http://localhost:3001
```

### 步驟 3: 查看 MongoDB 資料

1. 在頁面頂部點擊「📊 查看 MongoDB 資料庫」按鈕
2. MongoDB 資料面板會自動展開並載入資料
3. 你會看到所有儲存的影片記錄，包括：
   - 影片名稱
   - Session ID
   - 處理狀態（✅ 已完成 / ⏳ 處理中）
   - 檔案路徑
   - 處理時間
   - 總幀數

### 步驟 4: 查看詳細資料

點擊任一影片記錄的「📊 查看詳細資料」按鈕，系統會：
1. 從 MongoDB 載入該影片的所有醫療數值
2. 在下方的「🩺 醫療數值列表」區塊顯示
3. 自動滾動到該區塊

## 📊 資料顯示格式

### 影片記錄卡片
```
┌─────────────────────────────────────────────┐
│ 📹 影片名稱                    [✅ 已完成]  │
│ 🆔 Session ID: xxxxxxxxxx                   │
│                                              │
│ ┌──────────┬──────────┬──────────┐          │
│ │ 📂 路徑  │ ⏰ 時間  │ 🎞️ 幀數 │          │
│ └──────────┴──────────┴──────────┘          │
│                                              │
│ [📊 查看詳細資料]                           │
└─────────────────────────────────────────────┘
```

### 醫療數值顯示
每個時間點的醫療數值會以卡片形式顯示：
- 時間標記（分:秒）
- 螢幕編號
- 螢幕截圖預覽
- 醫療數值標籤（心率、血壓、血氧等）
- 設備型號

## 🔧 技術細節

### 資料流程
1. **前端**: 點擊按鈕 → `toggleMongoDBView()`
2. **前端**: 呼叫 `loadMongoDBData()`
3. **API**: GET `/api/mongodb/status` - 檢查連接
4. **API**: GET `/api/mongodb/videos` - 取得影片列表
5. **前端**: `displayMongoDBVideos()` - 渲染畫面
6. **用戶**: 點擊「查看詳細資料」
7. **API**: GET `/api/mongodb/medical_values/{session_id}` - 取得醫療數值
8. **前端**: `displayMedicalValues()` - 顯示數值

### MongoDB Collections 結構
```
medical_monitor_db
├── video_analysis (影片分析主記錄)
├── frame_results (畫面結果)
└── screen_analysis (螢幕分析與醫療數值)
```

## ✅ 功能特點

1. **即時載入**: 自動從 MongoDB 讀取最新資料
2. **狀態顯示**: 清楚標示每個影片的處理狀態
3. **詳細資訊**: 一鍵查看完整的醫療數值記錄
4. **圖片預覽**: 直接顯示螢幕截圖
5. **響應式設計**: 自動適應不同螢幕尺寸

## 🐛 疑難排解

### 問題：點擊按鈕沒有反應
**解決**: 確認 Flask 已重啟並載入新的路由

### 問題：顯示「MongoDB 未連接」
**解決**: 
```bash
# 檢查 MongoDB 狀態
pgrep -f mongod

# 如果沒有運行，執行：
python -c "from mongo_manager import start_local_mongodb; start_local_mongodb()"
```

### 問題：顯示「資料庫中沒有記錄」
**解決**: 這是正常的，表示還沒有處理過影片。上傳並處理一個影片後，資料就會出現。

## 📝 測試建議

1. 上傳一個測試影片
2. 等待處理完成
3. 點擊「查看 MongoDB 資料庫」
4. 應該會看到剛才處理的影片記錄
5. 點擊「查看詳細資料」確認醫療數值顯示正確

## 🔄 未來改進方向

- [ ] 新增搜尋和過濾功能
- [ ] 新增日期範圍篩選
- [ ] 新增匯出功能（CSV/PDF）
- [ ] 新增資料統計圖表
- [ ] 新增批量刪除功能