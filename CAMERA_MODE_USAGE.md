# 攝影機即時模式使用說明

## 功能概述

[`video_screen_digit_extractor.py`](video_screen_digit_extractor.py) 現在支援兩種模式：
1. **影片檔案模式** - 處理預錄的影片檔案
2. **攝影機即時模式** - 即時處理攝影機串流

## 攝影機即時模式

### 基本使用

```bash
# 基本攝影機模式
python video_screen_digit_extractor.py --camera --api_key YOUR_OPENAI_API_KEY

# 指定目標資料類型
python video_screen_digit_extractor.py --camera --api_key YOUR_API_KEY --target_data medical_values

# 自定義處理間隔
python video_screen_digit_extractor.py --camera --api_key YOUR_API_KEY --interval 5
```

### 參數說明

#### 必要參數
- `--camera` 或 `-cam`: 啟用攝影機即時模式
- `--api_key` 或 `-k`: OpenAI API 金鑰

#### 可選參數
- `--target_data` 或 `-t`: 指定要抓取的資料類型
  - `all` (預設): 抓取所有數字資料
  - `digits`: 只抓取單獨數字字符 (0-9)
  - `numbers`: 只抓取完整數字 (如 123, 45.67)
  - `medical_values`: 只抓取醫療相關數值
  - 自定義關鍵詞: 如 `heart_rate,blood_pressure`

- `--camera_index`: 攝影機索引 (預設 0)
- `--interval` 或 `-i`: 處理間隔秒數 (預設 2)
- `--max_duration`: 最大執行時間（秒）
- `--model` 或 `-m`: OpenAI 模型名稱 (預設 gpt-4o)
- `--cpu-only`: 只使用 CPU 進行 GroundingDINO 推理

### 使用範例

#### 1. 基本醫療數值監控
```bash
python video_screen_digit_extractor.py \
  --camera \
  --api_key sk-your-api-key \
  --target_data medical_values \
  --interval 3
```

#### 2. 監控特定數值
```bash
python video_screen_digit_extractor.py \
  --camera \
  --api_key sk-your-api-key \
  --target_data "heart_rate,blood_pressure,temperature" \
  --interval 5
```

#### 3. 高頻數字監控
```bash
python video_screen_digit_extractor.py \
  --camera \
  --api_key sk-your-api-key \
  --target_data numbers \
  --interval 1 \
  --max_duration 300
```

#### 4. 使用不同攝影機
```bash
python video_screen_digit_extractor.py \
  --camera \
  --api_key sk-your-api-key \
  --camera_index 1 \
  --target_data all
```

## 操作說明

### 啟動程式
1. 確保攝影機已連接並可正常使用
2. 準備好 OpenAI API 金鑰
3. 執行命令啟動程式

### 即時操作
- 程式會開啟攝影機視窗顯示即時畫面
- 按 **'q'** 鍵退出程式
- 程式會根據設定的間隔時間自動處理畫面
- 檢測到的數字會即時顯示在終端機中

### 輸出格式
```
=== 處理第 1 張畫面 ===
  檢測到 2 個螢幕
    分析螢幕 1...
      螢幕 1 結果:
        醫療數值: ["120", "80", "98.6"]
    分析螢幕 2...
      螢幕 2 結果:
        完整數字: ["72", "36.5"]
```

## 注意事項

1. **API 使用量**: 每次分析都會呼叫 OpenAI API，請注意使用量
2. **處理間隔**: 建議設定適當的間隔時間 (2-5秒) 以平衡即時性和 API 成本
3. **攝影機權限**: 確保程式有存取攝影機的權限
4. **網路連線**: 需要穩定的網路連線以呼叫 OpenAI API
5. **硬體需求**: GroundingDINO 模型需要一定的運算資源

## 故障排除

### 攝影機無法開啟
- 檢查攝影機是否已連接
- 確認沒有其他程式正在使用攝影機
- 嘗試不同的 `--camera_index` 值

### API 錯誤
- 檢查 API 金鑰是否正確
- 確認 OpenAI 帳戶有足夠額度
- 檢查網路連線

### 模型載入失敗
- 確認 GroundingDINO 模型檔案存在
- 檢查配置檔案路徑是否正確
- 如果 GPU 記憶體不足，使用 `--cpu-only` 參數

## 與影片模式的差異

| 功能 | 影片模式 | 攝影機模式 |
|------|----------|------------|
| 輸入來源 | 影片檔案 | 即時攝影機 |
| 處理方式 | 每分鐘一張 | 可自定義間隔 |
| 輸出方式 | 保存檔案 | 即時顯示 |
| 互動性 | 無 | 可即時退出 |
| 適用場景 | 批次分析 | 即時監控 |