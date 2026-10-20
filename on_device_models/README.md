# 本機微調螢幕 OCR 模型（on-device）

`ocr_server.py` 在 Jetson 上載入兩包微調 LoRA，提供 OpenAI 相容 API（`127.0.0.1:18765/v1`）。
web_app 的 provider `ondevice` 會連到這個服務；前端（上傳、資料集、即時監測、詳細頁重新分析）都能選。

| 前端模型 ID | 基底模型 | adapter | 說明 |
|---|---|---|---|
| `ondevice-glm46v-fix4-vote` | GLM-4.6V-Flash | `screen-ocr-models-2026-10-12/glm46v-flash-fix4/adapter_s42,s1,s7` | 三個 adapter 逐欄位多數決，最準，跑 3 次最慢 |
| `ondevice-glm46v-r9` | GLM-4.6V-Flash | `glm46v-flash-screen-ocr-r9/adapter` | 舊版單一 adapter |
| `ondevice-paddleocr-vl` | PaddleOCR-VL | `screen-ocr-models-2026-10-12/paddleocr-vl/adapter` | 0.96B，最快 |

## 目錄
- `*.zip` / `*.sha256`：原始交付檔；解壓後是 `glm46v-flash-screen-ocr-r9/`、`screen-ocr-models-2026-10-12/`
- `base_models/`：從 Hugging Face 下載的基底模型（`zai-org/GLM-4.6V-Flash`、`PaddlePaddle/PaddleOCR-VL`）
- 以上都不進 git（見 `.gitignore`）

## 環境
conda env `ondevice_ocr`（Python 3.10）：Jetson torch 2.8.0（`pypi.jetson-ai-lab.io/jp6/cu126`）、
transformers 5.19、peft 0.21、numpy<2、fastapi、uvicorn。
注意：專案 `.venv` 在 PATH 前面，請用絕對路徑 `/home/cluster/miniforge3/envs/ondevice_ocr/bin/python`。

## 啟動
```bash
./start_ocr_server.sh                      # 基底模型在第一次請求時才載入
./start_ocr_server.sh --preload paddle glm # 啟動時先載入
curl 127.0.0.1:18765/health
```
常駐：`sudo cp ../systemd/ondevice-ocr.service /etc/systemd/system/ && sudo systemctl enable --now ondevice-ocr`

## 行為
- 伺服器固定使用 `prompt_with_rules.txt`、關閉 thinking、小於 0.8 MP 的裁切圖放大到 0.8 MP、
  像素上限 1,003,520、greedy 解碼——與訓練一致，用戶端送來的文字提示會被忽略。
- 回傳 `{"device", "values", "per_adapter"}`；`video_screen_digit_extractor.py` 轉成既有的
  `{"model", "medical_values", "numbers", ...}` 再寫入 MongoDB。
- GPU 一次只跑一個請求（同一基底模型的請求排隊）。
- PaddleOCR-VL 的 generation_config 預設 `use_cache=false`，伺服器強制開啟，否則每張圖慢 7 倍以上。
