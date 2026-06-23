# GroundingDINO C++ 擴展載入問題解決方案

## 問題描述

在運行 GroundingDINO 時出現以下警告：
```
/home/jovyan/GroundingDINO/groundingdino/models/GroundingDINO/ms_deform_attn.py:31: UserWarning: Failed to load custom C++ ops. Running on CPU mode Only!
```

## 問題原因

GroundingDINO 的 C++ 擴展模組 `groundingdino._C` 無法載入，原因是動態連結器找不到 PyTorch 的共享庫檔案 `libc10.so`。

## 解決方案

### 1. 自動環境設定

已在以下檔案中添加自動環境設定代碼：

- `web_app.py` - Web 應用程式
- `video_screen_digit_extractor.py` - 主要處理腳本
- `setup_environment.py` - 獨立的環境設定工具

### 2. 環境設定代碼

```python
# 設定 GroundingDINO 環境
import torch
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
if torch_lib_path not in current_ld_path:
    if current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{current_ld_path}"
    else:
        os.environ['LD_LIBRARY_PATH'] = torch_lib_path
    print(f"已設定 LD_LIBRARY_PATH 包含 PyTorch 庫: {torch_lib_path}")

# 測試 C++ 擴展載入
try:
    from groundingdino import _C
    print("✅ GroundingDINO C++ 擴展載入成功，將使用 GPU 加速")
except ImportError as e:
    print(f"⚠️  GroundingDINO C++ 擴展載入失敗，將使用 CPU 模式: {e}")
```

### 3. 手動測試

如果需要手動測試環境設定，可以運行：

```bash
python setup_environment.py
```

## 驗證結果

修復後，運行任何 GroundingDINO 相關腳本時會看到：
```
✅ GroundingDINO C++ 擴展載入成功，將使用 GPU 加速
```

而不是之前的警告訊息。

## 技術細節

### 問題分析
1. GroundingDINO 使用 PyTorch 的 C++ 擴展機制
2. 擴展模組需要連結到 PyTorch 的共享庫
3. 系統的 `LD_LIBRARY_PATH` 沒有包含 PyTorch 庫路徑

### 解決方法
1. 動態獲取 PyTorch 庫路徑：`torch.__file__` + `/lib`
2. 將該路徑添加到 `LD_LIBRARY_PATH` 環境變數
3. 在導入 GroundingDINO 模組前設定環境

### 影響的檔案
- `groundingdino/models/GroundingDINO/ms_deform_attn.py` - 原始警告來源
- `groundingdino/models/GroundingDINO/csrc/vision.cpp` - C++ 擴展源碼
- `setup.py` - 編譯設定

## 效能提升

修復後，GroundingDINO 將能夠：
- 使用 GPU 加速的多尺度可變形注意力機制
- 顯著提升推理速度
- 減少 CPU 負載

## 相容性

此解決方案適用於：
- Linux 系統
- CUDA 環境
- PyTorch 2.0+
- GroundingDINO 0.1.0

## 注意事項

1. 此修復會在每次導入時自動執行
2. 不會影響其他 PyTorch 應用程式
3. 如果 CUDA 不可用，仍會回退到 CPU 模式
4. 環境變數設定僅在當前 Python 程序中有效

## 最終結果

修復完成後，GroundingDINO 將能夠：
- ✅ 成功載入 C++ 擴展模組
- ✅ 使用 GPU 加速的多尺度可變形注意力機制
- ✅ 模型正確載入到 GPU (cuda:0)
- ✅ 顯著提升推理速度
- ✅ 減少 CPU 負載

## 驗證方法

```bash
# 測試環境設定
python setup_environment.py

# 測試模型載入
python -c "
from inference_screen_crop import load_model
model = load_model(
    'groundingdino/config/GroundingDINO_SwinT_OGC.py',
    'groundingdino_swint_ogc.pth',
    cpu_only=False
)
print(f'模型設備: {next(model.parameters()).device}')
"
```

預期輸出：
```
✅ GroundingDINO C++ 擴展載入成功，將使用 GPU 加速
模型已載入到設備: cuda
模型設備: cuda:0
```