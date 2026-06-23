#!/usr/bin/env python3
"""
簡單的使用範例腳本
"""

import os
import subprocess
import sys

# 配置參數
CONFIG_FILE = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
CHECKPOINT_PATH = "weights/groundingdino_swinb_cogcoor.pth"
TEXT_PROMPT = "screen"
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.25

def run_conversion(input_path, output_dir, keep_all=True):
    """執行轉換"""
    cmd = [
        "python3", "convert_to_yolo_format.py",
        "--config_file", CONFIG_FILE,
        "--checkpoint_path", CHECKPOINT_PATH,
        "--input", input_path,
        "--text_prompt", TEXT_PROMPT,
        "--output_dir", output_dir,
        "--box_threshold", str(BOX_THRESHOLD),
        "--text_threshold", str(TEXT_THRESHOLD)
    ]
    
    if keep_all:
        cmd.append("--keep_all")
    
    print(f"執行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("轉換成功!")
        print(result.stdout)
    else:
        print("轉換失敗!")
        print(result.stderr)

def main():
    print("GroundingDINO to YOLO 轉換工具")
    print("="*50)
    
    # 檢查必要文件
    if not os.path.exists(CONFIG_FILE):
        print(f"錯誤: 找不到配置文件 {CONFIG_FILE}")
        return
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"錯誤: 找不到模型權重文件 {CHECKPOINT_PATH}")
        return
    
    # 選擇輸入
    print("\n請選擇處理模式:")
    print("1. 處理單張圖片")
    print("2. 處理整個資料夾")
    
    choice = input("\n請輸入選擇 (1 或 2): ").strip()
    
    # 選擇檢測模式
    print("\n請選擇檢測模式:")
    print("1. 保留所有檢測到的螢幕 (推薦)")
    print("2. 只保留信心度最高的一個螢幕")
    
    detection_choice = input("\n請輸入選擇 (1 或 2): ").strip()
    keep_all = detection_choice != "2"
    
    if choice == "1":
        # 單張圖片模式
        image_path = input("請輸入圖片路徑: ").strip()
        if not os.path.exists(image_path):
            print(f"錯誤: 找不到圖片 {image_path}")
            return
        
        output_dir = input("請輸入輸出資料夾 (預設: yolo_annotations): ").strip()
        if not output_dir:
            output_dir = "yolo_annotations"
        
        run_conversion(image_path, output_dir, keep_all)
        
    elif choice == "2":
        # 資料夾模式
        input_dir = input("請輸入圖片資料夾路徑: ").strip()
        if not os.path.exists(input_dir):
            print(f"錯誤: 找不到資料夾 {input_dir}")
            return
        
        output_dir = input("請輸入輸出資料夾 (預設: yolo_annotations): ").strip()
        if not output_dir:
            output_dir = "yolo_annotations"
        
        run_conversion(input_dir, output_dir, keep_all)
        
    else:
        print("無效的選擇!")

if __name__ == "__main__":
    main()
