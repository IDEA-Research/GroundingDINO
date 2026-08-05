# 開發與故障排除日誌 (Development & Troubleshooting Log)

**建立日期**: 2025-12-02
**專案位置**: `/etc/GroundDino/GroundingDINO/GroundingDINO`

此文件記錄了專案設置過程中的所有問題、修改步驟、安裝指令及解決方案。

---

## 1. 初始執行與權限設定
- **問題**: 嘗試執行 `./run.sh` 時出現 `Permission denied`。
- **解決方案**:
  - 執行 `chmod +x run.sh` 賦予執行權限。

## 2. Conda 環境配置 (CPU 版本)
- **問題 1**: `run.sh` 中的 Conda 路徑錯誤 (`/opt/conda/...`)。
- **解決方案**:
  - 修改 `run.sh`，將路徑更新為 `/home/cluster/miniforge3/etc/profile.d/conda.sh`。
- **問題 2**: 缺少指定的 Conda 環境 `webapp_mongo`。
- **解決方案**:
  - 建立新環境：
    ```bash
    conda create -n webapp_mongo python=3.9 -y
    ```

## 3. 依賴套件安裝 (CPU 版本)
- **安裝基礎套件**:
  ```bash
  conda install -n webapp_mongo -c conda-forge flask pymongo requests opencv -y
  ```
- **安裝 PyTorch 與 OpenAI**:
  ```bash
  /home/cluster/miniforge3/envs/webapp_mongo/bin/pip install torch torchvision openai
  ```
- **安裝 GroundingDINO**:
  - **問題**: 使用 `pip install -e .` 安裝失敗，出現建置隔離 (Build Isolation) 與找不到 `torch` 的問題。
  - **解決方案**: 改用 `setup.py install` 直接安裝：
    ```bash
    /home/cluster/miniforge3/envs/webapp_mongo/bin/python setup.py install
    ```
  - **狀態**: 成功安裝，但在執行時顯示 `Failed to load custom C++ ops. Running on CPU mode Only!`

## 4. MongoDB 資料庫設定
- **問題 1**: 找不到 `mongod` 指令。
- **解決方案**: 安裝 MongoDB：
  ```bash
  conda install -n webapp_mongo -c conda-forge mongodb -y
  ```
- **問題 2**: MongoDB 啟動失敗，日誌顯示資料目錄版本不相容 (`This version of MongoDB is too recent...`)。
- **解決方案**:
  - 將舊的資料目錄備份並重置：
    ```bash
    mv mongodb_data mongodb_data_backup_old
    ```
  - 讓程式重新初始化 `mongodb_data` 目錄。

## 5. 網路埠號衝突 (Port Conflict)
- **問題**: Web 應用程式無法啟動，顯示 Port 3000 已被佔用 (`Address already in use`)。
- **解決方案**:
  - 修改 `web_app.py`，將連接埠從 `3000` 改為 `3001`：
    ```python
    app.run(host='0.0.0.0', port=3001, debug=True)
    ```

## 6. GPU 支援啟用 (JetPack 6 / CUDA 12.6)

### 6.1 建立 GPU 環境
- **目標**: 啟用 GroundingDINO 的 CUDA 加速功能。
- **系統資訊**:
  - JetPack 版本: R36.4.3
  - CUDA 版本: 12.6
  - cuDNN 版本: 9.3.0 (系統已安裝)
  - GPU: NVIDIA Orin

### 6.2 建立新的 Conda 環境
```bash
conda create -n webapp_mongo_gpu python=3.10 -y
conda install -n webapp_mongo_gpu -c conda-forge flask pymongo requests opencv mongodb -y
```

### 6.3 安裝 CUDA 版 PyTorch
- **問題**: 標準 PyTorch 不支援 JetPack 6 的 ARM64 架構。
- **解決方案**: 使用 NVIDIA 提供的 JetPack 6 專用 PyTorch wheel：
  ```bash
  pip install https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+3bcc3cddb5.nv24.07.16234504-cp310-cp310-linux_aarch64.whl
  ```

### 6.4 安裝 cuDNN 與 cusparseLt
- **問題**: PyTorch 需要 cuDNN 8 和 cusparseLt 庫。
- **解決方案**:
  ```bash
  conda install -n webapp_mongo_gpu -c conda-forge cudnn=8 cusparselt -y
  ```

### 6.5 降級 NumPy
- **問題**: PyTorch 2.4.0 與 NumPy 2.x 不相容。
- **解決方案**:
  ```bash
  conda install -n webapp_mongo_gpu "numpy<2" -y
  ```

### 6.6 編譯 Torchvision
- **問題**: 沒有預編譯的 JetPack 6 Torchvision wheel。
- **解決方案**: 從源碼編譯：
  ```bash
  git clone --branch v0.19.0 --depth 1 https://github.com/pytorch/vision torchvision_build
  conda install -n webapp_mongo_gpu -c conda-forge cmake ninja -y
  cd torchvision_build
  export LD_LIBRARY_PATH=/home/cluster/miniforge3/envs/webapp_mongo_gpu/lib:$LD_LIBRARY_PATH
  export CMAKE_PREFIX_PATH=/home/cluster/miniforge3/envs/webapp_mongo_gpu
  python setup.py install
  ```

### 6.7 安裝 GroundingDINO 依賴
```bash
pip install transformers addict yapf timm supervision pycocotools openai
```

### 6.8 編譯 CUDA 版 GroundingDINO
```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/home/cluster/miniforge3/envs/webapp_mongo_gpu/lib:$LD_LIBRARY_PATH
cd /etc/GroundDino/GroundingDINO/GroundingDINO/
python setup.py install
```
- **結果**: 成功編譯 `groundingdino._C.cpython-310-aarch64-linux-gnu.so`

### 6.9 動態連結器問題
- **問題**: Python 啟動後修改 `LD_LIBRARY_PATH` 無效，導致 `_C` 模組無法載入所需的 `libc10.so`。
- **嘗試的解決方案**:
  1. 在 `run.sh` 中設置 `LD_LIBRARY_PATH` ✅ (部分有效)
  2. 在 `web_app.py` 開頭設置環境變數 ❌ (太晚)
  3. 使用 `os.execve` 重啟 Python 進程 ❌ (複雜且不穩定)
  4. 使用 `ctypes.CDLL` 預載入庫 ⚠️ (有幫助但不完全)

- **當前狀態**: 
  - ✅ 在 shell 中可以成功導入 `_C`
  - ❌ 在 Flask 應用中仍然無法載入 (可能是導入順序問題)

### 6.10 Python 模組搜索路徑衝突 ✅ 已解決
- **核心問題**: 當在 `/etc/GroundDino/GroundingDINO/GroundingDINO/` 目錄中運行 Python 時，當前目錄會被添加到 `sys.path[0]`，導致 Python 優先導入本地的 `groundingdino` 源碼目錄（沒有編譯的 `_C.so`），而不是安裝在 site-packages 中的編譯版本。
- **解決方案**: 將源碼目錄重命名：
  ```bash
  mv groundingdino groundingdino_source
  ```
- **結果**: ✅ 成功！GroundingDINO C++ 擴展現在可以正常載入。

## 7. 最終系統狀態 (Final Status) ✅

### GPU 環境 (webapp_mongo_gpu) - ✅ 成功運行
- **服務狀態**: 🟢 運行中
- **Web URL**: `http://127.0.0.1:3001` 或 `http://192.168.1.141:3001`
- **MongoDB**: ✅ 已連線 (資料庫: `medical_monitor_db`)
- **PyTorch**: ✅ CUDA 12.2 可用 (torch 2.4.0a0+3bcc3cddb5.nv24.07)
- **GPU**: NVIDIA Orin (61.37 GB 記憶體)
- **GroundingDINO**: ✅ **GPU 模式已啟用** (C++ 擴展載入成功)
- **執行指令**:
  ```bash
  cd /etc/GroundDino/GroundingDINO/GroundingDINO/ && ./run.sh
  ```

### 關鍵修改檔案
1. **`run.sh`**: 
   - 更新 conda 路徑為 `/home/cluster/miniforge3/etc/profile.d/conda.sh`
   - 設置 `LD_LIBRARY_PATH` 包含 PyTorch 和 conda 庫路徑
   - 切換至 `webapp_mongo_gpu` 環境
   
2. **`web_app.py`**: 
   - 添加 ctypes 預載入 PyTorch 庫（`libc10.so`, `libtorch_cpu.so`, `libtorch_cuda.so`）
   - 禁用 Flask reloader (`use_reloader=False`)
   - 修改 port 為 3001
   - 更新配置檔案路徑為 `groundingdino_source/config/...`
   
3. **`video_screen_digit_extractor.py`**: 
   - 實作延遲導入機制避免過早載入 groundingdino
   - 更新配置檔案路徑為 `groundingdino_source/config/...`
   
4. **源碼目錄**: 
   - 重命名 `groundingdino` → `groundingdino_source` 避免與安裝的套件衝突
   - 這是**最關鍵**的修改，解決了模組搜索路徑優先級問題

5. **`test_gpu_mode.py`**: 
   - 新增測試腳本用於驗證 GPU 模式

### 驗證指令
```bash
# 驗證 PyTorch CUDA
export LD_LIBRARY_PATH=/home/cluster/miniforge3/envs/webapp_mongo_gpu/lib/python3.10/site-packages/torch/lib:/home/cluster/miniforge3/envs/webapp_mongo_gpu/lib:$LD_LIBRARY_PATH
/home/cluster/miniforge3/envs/webapp_mongo_gpu/bin/python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 驗證 GroundingDINO C++ 擴展
/home/cluster/miniforge3/envs/webapp_mongo_gpu/bin/python -c "from groundingdino import _C; print('GroundingDINO GPU mode: Success!')"
```

### 環境配置總結
```bash
# 完整的環境設置流程
conda create -n webapp_mongo_gpu python=3.10 -y
conda install -n webapp_mongo_gpu -c conda-forge flask pymongo requests opencv mongodb cmake ninja cudnn=8 cusparselt "numpy<2" -y
pip install https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+3bcc3cddb5.nv24.07.16234504-cp310-cp310-linux_aarch64.whl
pip install transformers addict yapf timm supervision pycocotools openai

# 編譯 Torchvision
git clone --branch v0.19.0 --depth 1 https://github.com/pytorch/vision torchvision_build
cd torchvision_build
export LD_LIBRARY_PATH=/home/cluster/miniforge3/envs/webapp_mongo_gpu/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=/home/cluster/miniforge3/envs/webapp_mongo_gpu
python setup.py install

# 編譯 GroundingDINO
cd /etc/GroundDino/GroundingDINO/GroundingDINO/
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/home/cluster/miniforge3/envs/webapp_mongo_gpu/lib:$LD_LIBRARY_PATH
python setup.py install

# 重命名源碼目錄避免衝突
mv groundingdino groundingdino_source
```

## 8. GPU 模式驗證結果 ✅

運行測試腳本 `test_gpu_mode.py` 的結果：

```
PyTorch 版本: 2.4.0a0+3bcc3cddb5.nv24.07
CUDA 可用: True
CUDA 版本: 12.2
cuDNN 版本: 8907
GPU 數量: 1
GPU 名稱: Orin
GPU 記憶體: 61.37 GB

✅ GroundingDINO C++ 擴展載入成功！
✅ GPU 加速模式已啟用
✅ 模型載入成功！
✅ 模型設備: cuda:0
✅ 模型成功載入到 GPU！
```

**Web 應用狀態**: 🟢 正常運行於 `http://127.0.0.1:3001`

## 9. 快速啟動指南

### 啟動服務
```bash
cd /etc/GroundDino/GroundingDINO/GroundingDINO/
./run.sh
```

### 測試 GPU 模式
```bash
cd /etc/GroundDino/GroundingDINO/GroundingDINO/
export LD_LIBRARY_PATH=/home/cluster/miniforge3/envs/webapp_mongo_gpu/lib/python3.10/site-packages/torch/lib:/home/cluster/miniforge3/envs/webapp_mongo_gpu/lib:$LD_LIBRARY_PATH
/home/cluster/miniforge3/envs/webapp_mongo_gpu/bin/python test_gpu_mode.py
```

### 停止服務
```bash
pkill -f "python.*web_app.py"
```

## 10. 關鍵學習點

1. **JetPack 6 需要特殊的 PyTorch wheel**: 不能使用標準的 PyPI 版本，必須使用 NVIDIA 提供的預編譯版本。

2. **動態連結器路徑**: `LD_LIBRARY_PATH` 必須在 Python 進程啟動**之前**設置，啟動後修改無效。

3. **模組搜索路徑優先級**: Python 會優先導入當前目錄中的模組，即使 site-packages 中有同名的已安裝套件。

4. **ctypes.CDLL 預載入**: 使用 `RTLD_GLOBAL` 模式預載入共享庫可以解決部分動態連結問題。

5. **Flask debug reloader**: 會重新啟動進程並可能丟失環境變數，使用 `use_reloader=False` 可以避免。

## 11. 故障排除備忘

### 問題 1: `cannot import name '_C'` 錯誤
**解決方法**:
1. 確認 `LD_LIBRARY_PATH` 包含 PyTorch 庫路徑
2. 確認當前目錄沒有 `groundingdino` 源碼目錄
3. 確認 `_C.so` 文件存在於 site-packages 中
4. 使用 `ldd` 檢查 `_C.so` 的依賴庫是否都能找到

### 問題 2: 模型在 CPU 而非 GPU 上運行
**解決方法**:
1. 確認 `cpu_only=False` 參數
2. 確認 `torch.cuda.is_available()` 返回 `True`
3. 檢查模型載入後的設備：`next(model.parameters()).device`

### 問題 3: `CUDA error: CUBLAS_STATUS_ALLOC_FAILED` ⚠️
**症狀**: 模型成功載入到 GPU，但在推理時出現記憶體分配錯誤。

**原因**: 
- Jetson Orin 使用統一記憶體架構，GPU 和 CPU 共享記憶體
- 可能是 CUDA 上下文初始化問題
- 可能需要清理 GPU 快取

**解決方法**:
1. **重啟應用** (最簡單):
   ```bash
   pkill -f "python.*web_app.py"
   cd /etc/GroundDino/GroundingDINO/GroundingDINO && ./run.sh
   ```

2. **清理 CUDA 快取**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. **減少批次大小或模型大小** (如果問題持續):
   - 在 `web_app.py` 中調整 `frame_interval_seconds` 增加處理間隔
   - 考慮使用較小的模型

4. **檢查是否有其他 CUDA 程序**:
   ```bash
   fuser -v /dev/nvidia*
   ```

5. **重啟系統** (如果以上都無效):
   ```bash
   sudo reboot
   ```

**臨時解決方案**: 如果 GPU 模式不穩定，可以回退到 CPU 模式：
- 修改 `web_app.py` 第284行：`cpu_only=True`

