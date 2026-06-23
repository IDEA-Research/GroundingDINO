# 使用說明

您的GroundingDINO到YOLO格式轉換器已經創建完成並支援多螢幕檢測！

## 已創建的檔案

1. **`convert_to_yolo_format.py`** - 主要轉換程式
2. **`run_conversion.py`** - 簡化使用介面
3. **`test_conversion.py`** - 測試腳本
4. **`test_multi_detection.py`** - 多螢幕檢測測試腳本
5. **`YOLO_CONVERSION_README.md`** - 詳細說明文件

## 測試結果

✅ 測試已通過！程式可以正常工作：
- 單張圖片轉換：成功
- 資料夾批次轉換：成功
- **多螢幕檢測：成功** (新功能)
- 生成的標注格式與您的範例完全一致

## 新功能：IoU過濾去除重複檢測

程式現在支援智能的IoU（Intersection over Union）過濾來去除重複檢測：

### 🔍 IoU過濾機制
- **自動檢測重疊**: 計算所有檢測框之間的IoU值
- **保留最大面積**: 在重疊的檢測中保留面積最大的框
- **可調整閾值**: 可以設定IoU閾值來控制過濾敏感度

### 📊 過濾效果示例
```
原始檢測 → IoU過濾後:
frame_000020: 7 → 3 (移除4個重複)
frame_000061: 6 → 3 (移除3個重複)  
frame_000062: 7 → 1 (移除6個重複)
frame_000008: 12 → 5 (移除7個重複)
```

### ⚙️ IoU閾值說明
- **0.3**: 較嚴格過濾，移除更多重疊檢測
- **0.5**: 平衡設置（預設）
- **0.7**: 較寬鬆過濾，保留更多檢測
- **1.0**: 不進行IoU過濾

## 新功能：多螢幕檢測

現在程式支援兩種檢測模式：
1. **保留所有檢測** (預設) - 檢測到多個螢幕時，會為每個螢幕生成一行標注
2. **只保留最高信心度** - 只保留信心度最高的一個螢幕檢測

### 多螢幕檢測範例

當檢測到多個螢幕時，標注檔案會包含多行：
```
1 0.214877 0.409024 0.414382 0.711548
0 0.217370 0.390690 0.313146 0.545913
2 0.217030 0.390244 0.310359 0.537965
2 0.215475 0.407241 0.411047 0.710152
0 0.213802 0.396738 0.367825 0.651365
0 0.215235 0.406567 0.411441 0.710352
```

## 快速開始

### 方法1：使用簡化介面（推薦初學者）

```bash
source .venv/bin/activate  # 先激活虛擬環境
python run_conversion.py
```

程式會詢問：
1. 處理單張圖片還是整個資料夾
2. **是否保留所有檢測到的螢幕**

### 方法2：直接使用命令列

#### 處理單張圖片（保留所有檢測，使用IoU過濾）：
```bash
source .venv/bin/activate
python convert_to_yolo_format.py \
    --config_file groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --checkpoint_path weights/groundingdino_swinb_cogcoor.pth \
    --input medSample/frame_000019.jpg \
    --text_prompt "screen . monitor . display ." \
    --output_dir my_annotations \
    --iou_threshold 0.5 \
    --keep_all
```

#### 處理整個資料夾（自訂IoU閾值）：
```bash
source .venv/bin/activate
python convert_to_yolo_format.py \
    --config_file groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --checkpoint_path weights/groundingdino_swinb_cogcoor.pth \
    --input medSample/ \
    --text_prompt "screen . monitor . display ." \
    --output_dir my_annotations \
    --iou_threshold 0.3
    # 更嚴格的IoU過濾，移除更多重複檢測
```

## 輸出格式

程式會生成：

1. **標注檔案** (例如 `frame_000019.txt`)：
   
   **多螢幕檢測模式**：
   ```
   1 0.214877 0.409024 0.414382 0.711548
   0 0.217370 0.390690 0.313146 0.545913
   2 0.217030 0.390244 0.310359 0.537965
   ```
   
   **單螢幕檢測模式**：
   ```
   1 0.214877 0.409024 0.414382 0.711548
   ```
   
   格式：`class_id x_center y_center width height`

2. **類別檔案** (`classes.txt`)：
   ```
   screen
   monitor
   display
   born_screen1
   born_screen2
   born_screen3
   ```

## 重要特點

- ✅ **智能IoU過濾** - 自動去除重複檢測，保留面積最大的檢測框
- ✅ **支援多螢幕檢測** - 可檢測同一張圖片中的多個螢幕
- ✅ **可選檢測模式** - 保留所有檢測或只保留最高信心度檢測
- ✅ **可調IoU閾值** - 靈活控制重複檢測過濾的嚴格程度
- ✅ 座標自動歸一化到0-1範圍
- ✅ 與您提供的YOLO標注格式完全一致
- ✅ 自動生成類別對應檔案
- ✅ 支援批次處理

## 檢測結果示例

根據測試結果，程式能成功檢測到多個螢幕：
- `frame_000019.jpg`: 檢測到 6 個螢幕
- `frame_000008.jpg`: 檢測到 12 個螢幕
- `frame_000020.jpg`: 檢測到 7 個螢幕

每個檢測都會顯示信心度和類別分類，例如：
```
Detection 1: monitor (confidence: 0.457) -> class 1
Detection 2: screen (confidence: 0.441) -> class 0
Detection 3: display (confidence: 0.287) -> class 2
```

## 自訂設定

### 調整檢測靈敏度
降低閾值可以檢測到更多螢幕：
```bash
--box_threshold 0.2 --text_threshold 0.2
```

### 自訂檢測對象
修改 `--text_prompt` 參數：
```bash
# 檢測電腦螢幕
--text_prompt "computer screen . monitor . display ."

# 檢測手機螢幕
--text_prompt "phone screen . mobile screen . smartphone display ."
```

## 注意事項

⚠️ **重要**：使用前請先激活虛擬環境：
```bash
source .venv/bin/activate
```

如果您遇到任何問題，請查看 `YOLO_CONVERSION_README.md` 獲取更詳細的說明。
