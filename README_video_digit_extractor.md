# 影片數字抓取程式

這個程式可以從影片檔案每分鐘抓取一張圖片，然後使用 OpenAI GPT 模型識別其中的數字。

## 功能特色

- 使用 OpenCV 讀取影片檔案
- 每分鐘自動抓取一張圖片
- 使用 OpenAI GPT-4o 模型識別圖片中的數字（GPT-5 尚未發布）
- 生成詳細的分析報告
- 支援多種影片格式（.mkv, .mp4, .avi 等）

## 安裝依賴

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python video_digit_extractor.py --video_path your_video.mkv --api_key your_openai_api_key
```

### 參數說明

- `--video_path` 或 `-v`: 影片檔案路徑（必需）
- `--api_key` 或 `-k`: OpenAI API 金鑰（必需）
- `--output_dir` 或 `-o`: 輸出目錄（預設: video_analysis）

### 範例

```bash
# 處理 medSample 資料夾中的影片
python video_digit_extractor.py -v medSample/med20250812-9.mkv -k sk-your-api-key-here

# 指定輸出目錄
python video_digit_extractor.py -v medSample/med20250812-9.mkv -k sk-your-api-key-here -o my_analysis
```

## 輸出結果

程式會在指定的輸出目錄中創建以下檔案：

```
video_analysis/
└── med20250812-9/
    ├── frames/
    │   ├── frame_minute_000.jpg
    │   ├── frame_minute_001.jpg
    │   └── ...
    ├── digit_results.json
    └── report.txt
```

### 檔案說明

- `frames/`: 包含從影片抓取的所有圖片
- `digit_results.json`: 詳細的 JSON 格式分析結果
- `report.txt`: 人類可讀的摘要報告

## 注意事項

1. **API 金鑰**: 需要有效的 OpenAI API 金鑰
2. **模型**: 目前使用 gpt-4o 模型，因為 GPT-5 尚未發布
3. **費用**: 每張圖片的分析會消耗 OpenAI API 額度
4. **速度**: 程式會在每次 API 調用之間暫停 1 秒以避免速率限制

## 系統需求

- Python 3.7+
- OpenCV
- OpenAI Python SDK
- 足夠的磁碟空間存儲抓取的圖片

## 故障排除

### 常見錯誤

1. **影片無法開啟**: 檢查影片檔案路徑和格式
2. **API 錯誤**: 確認 OpenAI API 金鑰有效且有足夠額度
3. **依賴項錯誤**: 確保已安裝所有必需的 Python 套件

### 支援的影片格式

- .mkv
- .mp4
- .avi
- .mov
- .wmv
- 其他 OpenCV 支援的格式