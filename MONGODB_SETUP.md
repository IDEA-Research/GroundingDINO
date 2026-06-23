# MongoDB 自動啟動設置說明

## 概述

本專案已整合自動化的 MongoDB 解決方案，可以在 conda 環境中自動啟動本地 MongoDB 實例，並將醫療設備檢測數據儲存到數據庫中。圖片檔案存放在檔案系統，MongoDB 中只儲存檔案路徑。

## 安裝步驟

### 1. 安裝 MongoDB 到 conda 環境

```bash
# 安裝 MongoDB 伺服器
conda install -c conda-forge mongodb

# 安裝 Python MongoDB 驅動
pip install pymongo

# 驗證安裝
mongod --version
```

### 2. 安裝專案依賴

```bash
# 安裝所有依賴
pip install -r requirements.txt
```

### 3. 啟動應用程式

```bash
# 啟動 Web 應用程式（會自動啟動 MongoDB）
python web_app.py
```

## 功能特點

### ✅ 自動化特點
- **自動啟動**: 應用程式啟動時自動啟動 MongoDB
- **自動配置**: 自動創建配置檔案和數據目錄
- **自動索引**: 自動創建針對醫療數據優化的索引
- **自動停止**: 應用程式關閉時自動停止 MongoDB

### ✅ 數據管理
- **檔案系統儲存**: 圖片檔案存放在本地檔案系統
- **路徑儲存**: MongoDB 中只儲存圖片檔案路徑
- **結構化數據**: 醫療數值以結構化格式儲存
- **查詢優化**: 支援複雜的醫療數據查詢

### ✅ 支援的醫療設備
- **Philips MP20**: heart_rate, blood_pressure, spo2, respiration_rate
- **Dräger C500**: FiO2, Pinsp, PEEP, PIP, Plateau, 呼吸頻率, I:E
- **SOMANETICS INVOS**: rSO2_right, rSO2_left

## 目錄結構

啟動後會自動創建以下目錄結構：

```
GroundingDINO/
├── mongo_manager.py              # MongoDB 管理模組
├── mongodb_data/                 # MongoDB 數據目錄
│   ├── db/                      # 數據庫檔案
│   └── logs/                    # MongoDB 日誌
├── mongodb_config/              # MongoDB 配置
│   └── mongod.conf             # 配置檔案
├── video_screen_analysis/       # 分析結果和圖片
│   └── [video_name]/
│       ├── frames/              # 原始影片畫面
│       └── screens/             # 裁切的螢幕圖片
├── uploads/                     # 上傳的影片檔案
└── templates/                   # 設備識別模板
```

## API 端點

### MongoDB 狀態檢查
```
GET /api/mongodb/status
```

**回應範例：**
```json
{
  "mongodb_connected": true,
  "database": "medical_monitor_db",
  "port": 27017
}
```

### 取得醫療數值
```
GET /api/mongodb/medical_values/<session_id>
```

**回應範例：**
```json
{
  "success": true,
  "data": [
    {
      "minute": 5,
      "screen_number": 1,
      "medical_values": {
        "heart_rate": 92,
        "blood_pressure": 141,
        "spo2": 98
      },
      "model": "Philips MP20",
      "original_image_url": "./video_screen_analysis/video_name/frames/frame_minute_005.jpg",
      "screen_image_url": "./video_screen_analysis/video_name/screens/frame_minute_005_screen_01.jpg"
    }
  ],
  "count": 1
}
```

### 進階查詢
```
POST /api/mongodb/query
```

**請求範例：**
```json
{
  "device_model": "Philips MP20",
  "heart_rate_range": [100, 200],
  "date_range": ["2024-10-26T00:00:00", "2024-10-27T00:00:00"]
}
```

### 設備統計
```
GET /api/mongodb/stats
```

**回應範例：**
```json
{
  "success": true,
  "device_stats": [
    {
      "_id": "Philips MP20",
      "total_detections": 150,
      "avg_heart_rate": 85.5,
      "avg_spo2": 97.2,
      "latest_detection": "2024-10-26T08:00:00Z"
    }
  ]
}
```

## 使用範例

### 1. JavaScript 查詢範例

```javascript
// 查詢心率異常的記錄
fetch('/api/mongodb/query', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        'device_model': 'Philips MP20',
        'heart_rate_range': [100, 150]
    })
})
.then(response => response.json())
.then(data => {
    console.log('異常心率記錄:', data.data);
});

// 取得設備統計
fetch('/api/mongodb/stats')
.then(response => response.json())
.then(data => {
    console.log('設備統計:', data.device_stats);
});
```

### 2. Python 查詢範例

```python
import requests

# 查詢特定會話的醫療數值
session_id = "your-session-id"
response = requests.get(f'http://localhost:3000/api/mongodb/medical_values/{session_id}')
data = response.json()

if data['success']:
    for item in data['data']:
        print(f"第 {item['minute']} 分鐘:")
        print(f"  設備: {item['model']}")
        print(f"  醫療數值: {item['medical_values']}")
        print(f"  螢幕圖片: {item['screen_image_url']}")
```

## 數據庫架構

### VideoAnalysis Collection
```json
{
  "_id": ObjectId,
  "video_path": "path/to/video.mp4",
  "video_name": "video_name",
  "session_id": "uuid-string",
  "timestamp": ISODate,
  "total_frames": 120,
  "status": "completed"
}
```

### FrameResult Collection
```json
{
  "_id": ObjectId,
  "video_analysis_id": ObjectId,
  "minute": 5,
  "original_image_path": "./video_screen_analysis/video_name/frames/frame_minute_005.jpg",
  "screens_detected": 2,
  "processed_at": ISODate
}
```

### ScreenAnalysis Collection
```json
{
  "_id": ObjectId,
  "frame_result_id": ObjectId,
  "screen_number": 1,
  "screen_image_path": "./video_screen_analysis/video_name/screens/frame_minute_005_screen_01.jpg",
  "detected_model": "Philips MP20",
  "medical_values": {
    "heart_rate": 92,
    "blood_pressure": 141,
    "spo2": 98
  },
  "success": true,
  "analyzed_at": ISODate
}
```

## 故障排除

### MongoDB 啟動失敗
```bash
# 檢查 MongoDB 是否已安裝
mongod --version

# 如果未安裝，請執行：
conda install -c conda-forge mongodb
```

### 端口衝突
如果 27017 端口被佔用，可以修改 `mongo_manager.py` 中的端口設定：
```python
mongo_manager = LocalMongoDBManager(port=27018)
```

### 權限問題
確保應用程式有權限創建目錄和檔案：
```bash
# 檢查目錄權限
ls -la mongodb_data/
```

## 進階配置

### 自定義 MongoDB 配置
可以修改 `mongodb_config/mongod.conf` 來自定義 MongoDB 設定：

```yaml
storage:
  dbPath: ./mongodb_data/db
  journal:
    enabled: true

systemLog:
  destination: file
  path: ./mongodb_data/logs/mongod.log
  logAppend: true
  logRotate: reopen

net:
  port: 27017
  bindIp: 127.0.0.1

processManagement:
  fork: false

security:
  authorization: disabled
```

### 性能優化
對於大量數據，可以調整以下設定：

```python
# 在 mongo_manager.py 中調整連接池大小
self.client = MongoClient(
    f'mongodb://localhost:{self.port}',
    maxPoolSize=50,
    serverSelectionTimeoutMS=5000
)
```

## 安全考慮

### 生產環境建議
1. **啟用認證**: 在生產環境中啟用 MongoDB 認證
2. **網路限制**: 限制 MongoDB 只能從本地存取
3. **備份策略**: 定期備份 MongoDB 數據
4. **日誌監控**: 監控 MongoDB 日誌檔案

### 數據備份
```bash
# 備份數據庫
mongodump --db medical_monitor_db --out backup/

# 還原數據庫
mongorestore --db medical_monitor_db backup/medical_monitor_db/
```

## 效能監控

### MongoDB 效能查詢
```javascript
// 查看數據庫統計
db.stats()

// 查看 collection 統計
db.screen_analysis.stats()

// 查看索引使用情況
db.screen_analysis.aggregate([{$indexStats: {}}])
```

這個整合方案提供了完整的醫療設備檢測數據管理解決方案，結合了 MongoDB 的查詢能力和檔案系統的圖片儲存優勢。