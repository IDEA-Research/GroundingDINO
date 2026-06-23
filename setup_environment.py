#!/usr/bin/env python3
"""
GroundingDINO 環境設定腳本
解決 C++ 擴展載入問題
"""

import os
import sys
import torch

def setup_pytorch_lib_path():
    """設定 PyTorch 庫路徑到 LD_LIBRARY_PATH"""
    # 獲取 PyTorch 庫目錄
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
    
    # 獲取當前的 LD_LIBRARY_PATH
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    
    # 如果 PyTorch 庫路徑不在 LD_LIBRARY_PATH 中，則添加它
    if torch_lib_path not in current_ld_path:
        if current_ld_path:
            new_ld_path = f"{torch_lib_path}:{current_ld_path}"
        else:
            new_ld_path = torch_lib_path
        
        os.environ['LD_LIBRARY_PATH'] = new_ld_path
        print(f"已設定 LD_LIBRARY_PATH: {new_ld_path}")
    else:
        print("LD_LIBRARY_PATH 已包含 PyTorch 庫路徑")

def test_cpp_extension():
    """測試 C++ 擴展是否可以正常載入"""
    try:
        from groundingdino import _C
        print("✅ C++ 擴展載入成功!")
        print(f"可用函數: {[func for func in dir(_C) if not func.startswith('_')]}")
        return True
    except ImportError as e:
        print(f"❌ C++ 擴展載入失敗: {e}")
        return False

if __name__ == "__main__":
    print("=== GroundingDINO 環境設定 ===")
    print(f"Python 執行檔: {sys.executable}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    # 設定庫路徑
    setup_pytorch_lib_path()
    
    # 測試 C++ 擴展
    success = test_cpp_extension()
    
    if success:
        print("🎉 環境設定完成，GroundingDINO 可以使用 GPU 加速!")
    else:
        print("⚠️  C++ 擴展載入失敗，將使用 CPU 模式")
    