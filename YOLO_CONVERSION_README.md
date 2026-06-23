# GroundingDINO to YOLO Format Converter

這個工具可以將GroundingDINO的檢測結果轉換為YOLO標注格式。

## 功能特色

- 將GroundingDINO的bounding box轉換為YOLO格式 (歸一化的 x_center, y_center, width, height)
- **智能IoU過濾** - 自動去除重複檢測，保留面積最大的檢測框
- **支援多螢幕檢測** - 可以檢測並標注同一張圖片中的多個螢幕
- 可選擇保留所有檢測框或只保留最高信心度的檢測框
- 支援單張圖片或整個資料夾的批次處理
- 自動生成classes.txt檔案
- 生成的標注檔案格式與YOLO標準完全相容

## 檔案說明

- `convert_to_yolo_format.py`: 主要轉換程式
- `run_conversion.py`: 簡化的使用介面
- `YOLO_CONVERSION_README.md`: 說明文件

## 安裝需求

確保您已經安裝了GroundingDINO及其相關依賴：

```bash
pip install torch torchvision
pip install -e .
```

## 使用方法

### 方法1: 使用簡化介面 (推薦)

```bash
python run_conversion.py
```

按照提示選擇：
1. 處理單張圖片或整個資料夾
2. 輸入圖片/資料夾路徑
3. 設定輸出資料夾

### 方法2: 直接使用主程式

#### 處理單張圖片：

```bash
python convert_to_yolo_format.py \
    --config_file groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --checkpoint_path weights/groundingdino_swinb_cogcoor.pth \
    --input /path/to/your/image.jpg \
    --text_prompt "screen . monitor . display ." \
    --output_dir yolo_annotations \
    --box_threshold 0.3 \
    --text_threshold 0.25
```

#### 處理整個資料夾：

```bash
python convert_to_yolo_format.py \
    --config_file groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --checkpoint_path weights/groundingdino_swinb_cogcoor.pth \
    --input /path/to/your/image/folder \
    --text_prompt "screen . monitor . display ." \
    --output_dir yolo_annotations \
    --box_threshold 0.3 \
    --text_threshold 0.25
```

## 參數說明

- `--config_file`: GroundingDINO配置檔案路徑
- `--checkpoint_path`: GroundingDINO模型權重檔案路徑
- `--input`: 輸入圖片檔案或資料夾路徑
- `--text_prompt`: 檢測文字提示 (例如: "screen . monitor . display .")
- `--output_dir`: 輸出標注檔案的資料夾
- `--box_threshold`: 檢測框閾值 (預設: 0.3)
- `--text_threshold`: 文字閾值 (預設: 0.25)
- `--iou_threshold`: IoU閾值，用於去除重複檢測 (預設: 0.5，範圍: 0.0-1.0)
- `--classes_file`: 自訂類別檔案路徑 (可選)
- `--cpu_only`: 只使用CPU (可選)
- `--keep_all`: 保留所有檢測框而非只保留最高信心度的檢測框 (可選)

## 輸出格式

程式會在輸出資料夾中生成：

1. **標注檔案**: 每張圖片對應一個`.txt`檔案
   - 格式: `class_id x_center y_center width height`
   - **多螢幕檢測範例** (使用 `--keep_all` 參數):
     ```
     1 0.214877 0.409024 0.414382 0.711548
     0 0.217370 0.390690 0.313146 0.545913
     2 0.217030 0.390244 0.310359 0.537965
     ```
   - **單螢幕檢測範例** (預設行為):
     ```
     1 0.214877 0.409024 0.414382 0.711548
     ```

2. **classes.txt**: 類別對應檔案
   ```
   screen
   monitor
   display
   born_screen1
   born_screen2
   born_screen3
   ```

## IoU過濾功能

程式內建智能IoU（Intersection over Union）過濾功能，可以有效去除重複檢測：

### 工作原理
1. **計算重疊度**: 對所有檢測框計算兩兩之間的IoU值
2. **識別重複**: IoU值超過設定閾值的檢測被視為重複
3. **保留最大**: 在重複檢測中保留面積最大的檢測框
4. **移除其他**: 其他重複檢測被自動移除

### IoU閾值設定
- **0.3**: 嚴格過濾，移除更多可能的重複檢測
- **0.5**: 平衡設置（預設），適合大多數情況
- **0.7**: 寬鬆過濾，只移除高度重疊的檢測
- **1.0**: 關閉IoU過濾功能

### 過濾效果示例
```
原始檢測: 6個螢幕
IoU過濾處理:
  - 檢測2被抑制 (IoU: 0.580)
  - 檢測3被抑制 (IoU: 0.566)  
  - 檢測4被抑制 (IoU: 0.987)
  - 檢測5被抑制 (IoU: 0.813)
  - 檢測6被抑制 (IoU: 0.986)
最終結果: 1個唯一檢測
```

## 自訂類別

如果您有自己的類別檔案，可以使用 `--classes_file` 參數指定：

```bash
python convert_to_yolo_format.py \
    --classes_file /path/to/your/classes.txt \
    [其他參數...]
```

## 範例

假設您有以下檔案結構：

```
images/
├── frame_000019.jpg
├── frame_000020.jpg
└── frame_000021.jpg
```

執行：

```bash
# 多螢幕檢測模式 (保留所有檢測)
python convert_to_yolo_format.py \
    --config_file groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --checkpoint_path weights/groundingdino_swinb_cogcoor.pth \
    --input images/ \
    --text_prompt "screen . monitor . display ." \
    --output_dir annotations/ \
    --keep_all

# 單螢幕檢測模式 (只保留最高信心度)
python convert_to_yolo_format.py \
    --config_file groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --checkpoint_path weights/groundingdino_swinb_cogcoor.pth \
    --input images/ \
    --text_prompt "screen . monitor . display ." \
    --output_dir annotations/
```

會生成：

```
annotations/
├── frame_000019.txt    # 可能包含多個檢測框
├── frame_000020.txt    # 可能包含多個檢測框
├── frame_000021.txt    # 可能包含多個檢測框
└── classes.txt
```

**多螢幕檢測範例**：
- `frame_000019.txt` 可能包含 6 個檢測框
- `frame_000008.txt` 可能包含 12 個檢測框  
- 每個檢測框對應圖片中的一個螢幕

每個`.txt`檔案包含該圖片中所有檢測到的螢幕的YOLO格式標注（如果使用 `--keep_all`），或只包含信心度最高的一個檢測（預設行為）。

## 注意事項

1. 確保GroundingDINO模型檔案存在於指定路徑
2. **多螢幕檢測**: 使用 `--keep_all` 參數可以保留所有檢測到的螢幕
3. **單螢幕檢測**: 不使用 `--keep_all` 參數只會保留信心度最高的檢測
4. 座標會自動歸一化到0-1範圍
5. 支援的圖片格式：jpg, jpeg, png, bmp, tiff, tif
6. **重要**: 使用前請先激活虛擬環境 `source .venv/bin/activate`

## 故障排除

如果遇到問題，請檢查：

1. CUDA/GPU是否可用 (如果使用GPU)
2. GroundingDINO相關依賴是否正確安裝
3. 模型檔案路徑是否正確
4. 輸入圖片路徑是否存在
