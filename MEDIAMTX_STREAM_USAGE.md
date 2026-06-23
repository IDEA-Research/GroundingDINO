# MediaMTX RTSP 流處理使用說明

## 概述

系統現在支持從 MediaMTX 接收 RTSP 流進行即時處理，並將分析結果保存到 MongoDB，同時在 Prometheus 上顯示即時數據。

## 架構流程

```
MediaMTX 伺服器
    ↓ (發布 RTSP 流)
rtsp://localhost:8554/stream1
    ↓ (OpenCV 讀取)
process_camera_stream(rtsp_url="rtsp://...")
    ↓ (處理並保存)
MongoDB (帶時間戳: analyzed_at, processed_at)
    ↓ (讀取最新數據)
medical_mongodb_exporter.py (每 15 秒)
    ↓ (轉換為 metrics)
Prometheus (http://localhost:9090)
    ↓ (查詢和可視化)
Grafana (http://localhost:3001)
```

## 使用步驟

### 1. 啟動 MediaMTX

```bash
# 使用 Docker 啟動 MediaMTX
docker run --rm -it \
  -p 8554:8554 \
  -p 1935:1935 \
  -p 8888:8888 \
  bluenviron/mediamtx

# 或從源碼安裝
# 參考: https://github.com/bluenviron/mediamtx
```

### 2. 發布測試視頻到 MediaMTX

```bash
# 使用 ffmpeg 將視頻文件推送到 MediaMTX
ffmpeg -re -stream_loop -1 -i test_video.mp4 \
  -c copy -f rtsp rtsp://localhost:8554/stream1

# 或使用其他 RTSP 源
```

### 3. 啟動 MongoDB 和監控服務

```bash
# 啟動 MongoDB
python3 -c "from mongo_manager import start_local_mongodb; start_local_mongodb()"

# 啟動監控服務（如果使用）
./start_medical_monitoring.sh
```

### 4. 運行視頻處理程序

#### 基本 RTSP 流處理

```bash
python3 video_screen_digit_extractor.py \
  --camera \
  --rtsp_url rtsp://localhost:8554/stream1 \
  --api_key YOUR_OPENAI_API_KEY \
  --target_data medical_values \
  --interval 5
```

#### 自定義參數

```bash
python3 video_screen_digit_extractor.py \
  --camera \
  --rtsp_url rtsp://localhost:8554/stream1 \
  --api_key YOUR_API_KEY \
  --target_data medical_values \
  --interval 3 \
  --output_dir stream_analysis \
  --session_id my_custom_session_001
```

#### 不使用 MongoDB（僅本地處理）

```bash
python3 video_screen_digit_extractor.py \
  --camera \
  --rtsp_url rtsp://localhost:8554/stream1 \
  --api_key YOUR_API_KEY \
  --no_mongodb
```

## 參數說明

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--camera` | 啟用攝像頭/流處理模式 | - |
| `--rtsp_url` | RTSP 流 URL | None |
| `--camera_index` | 本地攝像頭索引（當 rtsp_url 為空時使用） | 0 |
| `--interval` | 處理間隔（秒） | 2 |
| `--max_duration` | 最大執行時間（秒） | None（無限制） |
| `--output_dir` | 輸出目錄 | `video_screen_analysis` |
| `--session_id` | 自定義會話 ID | 自動生成 UUID |
| `--no_mongodb` | 不保存到 MongoDB | False（預設保存） |
| `--target_data` | 目標數據類型 | `all` |

## 數據記錄

### 時間戳記錄

每筆數據都會記錄以下時間戳：

- **`timestamp`**: 視頻分析會話開始時間
- **`processed_at`**: 每幀處理完成時間
- **`analyzed_at`**: GPT 分析完成時間

### MongoDB 數據結構

```json
{
  "video_analysis": {
    "session_id": "uuid",
    "timestamp": "2024-01-01T12:00:00",
    "video_path": "RTSP: rtsp://..._uuid",
    "status": "completed"
  },
  "frame_results": {
    "minute": 5.0,
    "processed_at": "2024-01-01T12:00:05",
    "original_image_path": "..."
  },
  "screen_analysis": {
    "analyzed_at": "2024-01-01T12:00:05.123",
    "medical_values": {
      "PIP": 25.5,
      "PEEP": 8.0,
      "MAP": 15.0,
      "FiO2": 40
    }
  }
}
```

## Prometheus 監控

### 查看即時數據

```bash
# 查看醫療指標
curl http://localhost:8000/metrics | grep patient_

# 查看設備統計
curl http://localhost:8000/metrics | grep device_

# 查看異常警報
curl http://localhost:8000/metrics | grep medical_alerts
```

### Prometheus 查詢範例

```promql
# 最新的心率數據
patient_heart_rate_bpm

# 特定會話的數據
patient_heart_rate_bpm{session_id="your-session-id"}

# 最近 5 分鐘的平均血氧
avg_over_time(patient_spo2_percentage[5m])
```

## Grafana 監控面板

訪問 `http://localhost:3001` 查看即時監控面板：

- **患者生理指標**: 心率、血氧、血壓等
- **呼吸機參數**: PIP、PEEP、MAP、FiO2 等
- **異常警報**: 實時異常檢測和記錄
- **設備統計**: 各設備的使用情況

## 調試技巧

### 1. 檢查 RTSP 流是否可訪問

```bash
# 使用 ffplay 測試 RTSP 流
ffplay rtsp://localhost:8554/stream1

# 使用 OpenCV 測試
python3 -c "import cv2; cap = cv2.VideoCapture('rtsp://localhost:8554/stream1'); print('可訪問' if cap.isOpened() else '無法訪問')"
```

### 2. 檢查 MongoDB 連接

```bash
python3 -c "from mongo_manager import mongo_manager; mongo_manager.start_mongodb(); print('MongoDB 狀態:', mongo_manager.is_mongodb_running())"
```

### 3. 查看實時日誌

```bash
# 查看處理進度
tail -f stream_analysis/stream_*/frames/*.jpg

# 查看 MongoDB 中的數據
python3 -c "from mongo_manager import mongo_manager; mongo_manager.start_mongodb(); results = list(mongo_manager.db.screen_analysis.find().sort('analyzed_at', -1).limit(5)); print(results)"
```

## 常見問題

### Q: RTSP 流連接失敗

**A**: 檢查：
1. MediaMTX 是否正在運行
2. RTSP URL 是否正確
3. 防火牆是否阻止了端口

### Q: 數據沒有出現在 Prometheus

**A**: 檢查：
1. MongoDB 是否有數據（`analyzed_at` 時間戳在最近 10 分鐘內）
2. `medical_mongodb_exporter.py` 是否正在運行
3. 查詢是否使用了正確的標籤（session_id, device_model）

### Q: 處理速度太慢

**A**: 可以：
1. 增加 `--interval` 參數（例如改為 5 或 10 秒）
2. 使用 GPU 模式（移除 `--cpu-only`）
3. 優化 GPT 調用（減少不必要的分析）

## 範例腳本

### 完整的啟動腳本

```bash
#!/bin/bash

# 啟動 MongoDB
python3 -c "from mongo_manager import start_local_mongodb; start_local_mongodb()"

# 等待 MongoDB 啟動
sleep 3

# 啟動監控服務
./start_medical_monitoring.sh &

# 等待監控服務啟動
sleep 5

# 啟動 RTSP 流處理
python3 video_screen_digit_extractor.py \
  --camera \
  --rtsp_url rtsp://localhost:8554/stream1 \
  --api_key YOUR_API_KEY \
  --target_data medical_values \
  --interval 5 \
  --session_id test_session_$(date +%Y%m%d_%H%M%S)
```

## 相關文檔

- [CAMERA_MODE_USAGE.md](CAMERA_MODE_USAGE.md) - 攝像頭模式使用說明
- [MONGODB_SETUP.md](MONGODB_SETUP.md) - MongoDB 設置指南
- [README_MONITORING.md](README_MONITORING.md) - 監控系統說明


