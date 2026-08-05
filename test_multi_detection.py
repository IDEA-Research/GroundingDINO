#!/usr/bin/env python3
"""
測試多螢幕檢測功能
"""

import os
import subprocess
import sys

def test_multi_detection():
    """測試多螢幕檢測功能"""
    print("測試多螢幕檢測功能...")
    
    cmd = [
        "python3", "convert_to_yolo_format.py",
        "--config_file", "groundingdino/config/GroundingDINO_SwinB_cfg.py",
        "--checkpoint_path", "weights/groundingdino_swinb_cogcoor.pth",
        "--input", "medSample/",
        "--text_prompt", "screen . monitor . display .",
        "--output_dir", "test_multi_output",
        "--box_threshold", "0.2",  # 降低閾值來檢測更多螢幕
        "--text_threshold", "0.2",
        "--keep_all"  # 保留所有檢測
    ]
    
    print(f"執行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)
    print(f"Return code: {result.returncode}")
    
    return result.returncode == 0

def test_single_detection():
    """測試單螢幕檢測功能（只保留最高信心度）"""
    print("\n測試單螢幕檢測功能...")
    
    cmd = [
        "python3", "convert_to_yolo_format.py",
        "--config_file", "groundingdino/config/GroundingDINO_SwinB_cfg.py",
        "--checkpoint_path", "weights/groundingdino_swinb_cogcoor.pth",
        "--input", "medSample/",
        "--text_prompt", "screen . monitor . display .",
        "--output_dir", "test_single_output",
        "--box_threshold", "0.2",
        "--text_threshold", "0.2"
        # 不加 --keep_all，所以只保留最高信心度的檢測
    ]
    
    print(f"執行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)
    print(f"Return code: {result.returncode}")
    
    return result.returncode == 0

def compare_outputs():
    """比較兩種模式的輸出差異"""
    print("\n比較輸出差異:")
    
    multi_dir = "test_multi_output"
    single_dir = "test_single_output"
    
    if not os.path.exists(multi_dir) or not os.path.exists(single_dir):
        print("測試輸出資料夾不存在")
        return
    
    # 比較檔案數量
    multi_files = [f for f in os.listdir(multi_dir) if f.endswith('.txt') and f != 'classes.txt']
    single_files = [f for f in os.listdir(single_dir) if f.endswith('.txt') and f != 'classes.txt']
    
    print(f"多檢測模式標注檔案數量: {len(multi_files)}")
    print(f"單檢測模式標注檔案數量: {len(single_files)}")
    
    # 比較每個檔案的內容
    for filename in multi_files:
        if filename in single_files:
            multi_path = os.path.join(multi_dir, filename)
            single_path = os.path.join(single_dir, filename)
            
            with open(multi_path, 'r') as f:
                multi_content = f.read().strip().split('\n')
            
            with open(single_path, 'r') as f:
                single_content = f.read().strip().split('\n')
            
            print(f"\n{filename}:")
            print(f"  多檢測模式: {len([line for line in multi_content if line.strip()])} 個檢測框")
            print(f"  單檢測模式: {len([line for line in single_content if line.strip()])} 個檢測框")
            
            if len(multi_content) > 1:
                print(f"  多檢測內容:")
                for i, line in enumerate(multi_content):
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            print(f"    檢測框{i+1}: class={parts[0]}, x={parts[1]}, y={parts[2]}, w={parts[3]}, h={parts[4]}")
            
            print(f"  單檢測內容:")
            for line in single_content:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        print(f"    檢測框: class={parts[0]}, x={parts[1]}, y={parts[2]}, w={parts[3]}, h={parts[4]}")

def main():
    print("多螢幕檢測功能測試")
    print("="*50)
    
    # 檢查必要檔案
    if not os.path.exists("groundingdino/config/GroundingDINO_SwinB_cfg.py"):
        print("錯誤: 找不到配置檔案")
        return
    
    if not os.path.exists("weights/groundingdino_swinb_cogcoor.pth"):
        print("錯誤: 找不到模型權重檔案")
        return
    
    if not os.path.exists("medSample/"):
        print("錯誤: 找不到測試圖片資料夾")
        return
    
    # 測試多檢測模式
    success_multi = test_multi_detection()
    
    # 測試單檢測模式
    success_single = test_single_detection()
    
    # 比較輸出
    if success_multi and success_single:
        compare_outputs()
    
    print(f"\n測試結果:")
    print(f"多檢測模式測試: {'✓' if success_multi else '✗'}")
    print(f"單檢測模式測試: {'✓' if success_single else '✗'}")

if __name__ == "__main__":
    main()
