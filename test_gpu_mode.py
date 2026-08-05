#!/usr/bin/env python3
"""
測試 GroundingDINO GPU 模式
"""
import os
import sys

# 設置 LD_LIBRARY_PATH
conda_env_path = os.path.dirname(os.path.dirname(sys.executable))
torch_lib_path = os.path.join(conda_env_path, 'lib', 'python3.10', 'site-packages', 'torch', 'lib')
conda_lib_path = os.path.join(conda_env_path, 'lib')
os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{conda_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

import torch

print("=" * 60)
print("PyTorch 環境測試")
print("=" * 60)
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
    print(f"GPU 數量: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        print(f"GPU 名稱: {torch.cuda.get_device_name(0)}")
        print(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print("\n" + "=" * 60)
print("GroundingDINO C++ 擴展測試")
print("=" * 60)

try:
    from groundingdino import _C
    print("✅ GroundingDINO C++ 擴展載入成功！")
    print("✅ GPU 加速模式已啟用")
except ImportError as e:
    print(f"❌ GroundingDINO C++ 擴展載入失敗: {e}")
    print("⚠️  將使用 CPU 模式")

print("\n" + "=" * 60)
print("GroundingDINO 模型載入測試")
print("=" * 60)

try:
    from inference_screen_crop import load_model
    
    config_file = "groundingdino_source/config/GroundingDINO_SwinT_OGC.py"
    checkpoint_path = "groundingdino_swint_ogc.pth"
    
    if os.path.exists(config_file) and os.path.exists(checkpoint_path):
        print("正在載入模型...")
        model = load_model(config_file, checkpoint_path, cpu_only=False)
        print(f"✅ 模型載入成功！")
        device = next(model.parameters()).device
        print(f"模型設備: {device}")
        if device.type == 'cuda':
            print(f"✅ 模型成功載入到 GPU！")
        else:
            print(f"⚠️  模型在 CPU 上運行")
    else:
        print(f"⚠️  找不到配置檔案或權重檔案")
        print(f"   config: {config_file} (存在: {os.path.exists(config_file)})")
        print(f"   checkpoint: {checkpoint_path} (存在: {os.path.exists(checkpoint_path)})")
        
except Exception as e:
    import traceback
    print(f"❌ 模型載入失敗: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)

