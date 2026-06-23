#!/usr/bin/env python3
"""
測試攝影機模式的簡單腳本
"""

import cv2
import sys
import os

def test_camera_access():
    """測試攝影機是否可以正常存取"""
    print("測試攝影機存取...")
    
    # 嘗試開啟攝影機
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 無法開啟攝影機 0")
        return False
    
    # 嘗試讀取一張畫面
    ret, frame = cap.read()
    
    if not ret:
        print("❌ 無法讀取攝影機畫面")
        cap.release()
        return False
    
    print(f"✅ 攝影機正常運作，畫面大小: {frame.shape}")
    cap.release()
    return True

def test_dependencies():
    """測試相依套件是否正常"""
    print("測試相依套件...")
    
    try:
        import cv2
        print(f"✅ OpenCV 版本: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV 未安裝")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch 版本: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch 未安裝")
        return False
    
    try:
        from openai import OpenAI
        print("✅ OpenAI SDK 已安裝")
    except ImportError:
        print("❌ OpenAI SDK 未安裝")
        return False
    
    return True

def main():
    print("=== 攝影機模式測試 ===\n")
    
    # 測試相依套件
    if not test_dependencies():
        print("\n❌ 相依套件測試失敗")
        return
    
    print()
    
    # 測試攝影機存取
    if not test_camera_access():
        print("\n❌ 攝影機存取測試失敗")
        print("提示: 請確認攝影機已連接且未被其他程式使用")
        return
    
    print("\n✅ 所有測試通過！")
    print("\n使用範例:")
    print("python video_screen_digit_extractor.py --camera --api_key YOUR_API_KEY --target_data medical_values")

if __name__ == "__main__":
    main()