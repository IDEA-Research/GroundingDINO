#!/usr/bin/env python3
"""
測試GroundingDINO to YOLO轉換器
"""

import os
import subprocess
import sys

def test_single_image():
    """測試單張圖片轉換"""
    print("測試單張圖片轉換...")
    
    cmd = [
        "python", "convert_to_yolo_format.py",
        "--config_file", "groundingdino/config/GroundingDINO_SwinB_cfg.py",
        "--checkpoint_path", "weights/groundingdino_swinb_cogcoor.pth",
        "--input", "medSample/frame_000019.jpg",
        "--text_prompt", "screen . monitor . display .",
        "--output_dir", "test_output_single",
        "--box_threshold", "0.3",
        "--text_threshold", "0.25"
    ]
    
    print(f"執行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)
    print(f"Return code: {result.returncode}")
    
    return result.returncode == 0

def test_directory():
    """測試資料夾批次轉換"""
    print("\n測試資料夾批次轉換...")
    
    cmd = [
        "python", "convert_to_yolo_format.py",
        "--config_file", "groundingdino/config/GroundingDINO_SwinB_cfg.py",
        "--checkpoint_path", "weights/groundingdino_swinb_cogcoor.pth",
        "--input", "medSample/",
        "--text_prompt", "screen . monitor . display .",
        "--output_dir", "test_output_batch",
        "--box_threshold", "0.3",
        "--text_threshold", "0.25"
    ]
    
    print(f"執行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)
    print(f"Return code: {result.returncode}")
    
    return result.returncode == 0

def check_output(output_dir):
    """檢查輸出結果"""
    print(f"\n檢查輸出資料夾: {output_dir}")
    
    if not os.path.exists(output_dir):
        print(f"輸出資料夾 {output_dir} 不存在")
        return False
    
    files = os.listdir(output_dir)
    print(f"輸出檔案: {files}")
    
    # 檢查是否有classes.txt
    if "classes.txt" in files:
        print("✓ classes.txt 檔案已生成")
        with open(os.path.join(output_dir, "classes.txt"), 'r') as f:
            print("類別內容:")
            print(f.read())
    else:
        print("✗ classes.txt 檔案未找到")
    
    # 檢查標注檔案
    txt_files = [f for f in files if f.endswith('.txt') and f != 'classes.txt']
    print(f"標注檔案數量: {len(txt_files)}")
    
    for txt_file in txt_files:
        print(f"\n檢查 {txt_file}:")
        with open(os.path.join(output_dir, txt_file), 'r') as f:
            content = f.read().strip()
            print(f"內容: {content}")
            
            # 驗證格式
            lines = content.split('\n')
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:])
                            print(f"  ✓ 格式正確: class={class_id}, x={x_center:.4f}, y={y_center:.4f}, w={width:.4f}, h={height:.4f}")
                        except ValueError:
                            print(f"  ✗ 格式錯誤: {line}")
                    else:
                        print(f"  ✗ 欄位數量錯誤: {line}")
    
    return True

def main():
    print("GroundingDINO to YOLO 轉換器測試")
    print("="*50)
    
    # 檢查必要檔案
    if not os.path.exists("groundingdino/config/GroundingDINO_SwinB_cfg.py"):
        print("錯誤: 找不到配置檔案")
        return
    
    if not os.path.exists("weights/groundingdino_swinb_cogcoor.pth"):
        print("錯誤: 找不到模型權重檔案")
        return
    
    if not os.path.exists("medSample/frame_000019.jpg"):
        print("錯誤: 找不到測試圖片")
        return
    
    # 測試單張圖片
    success1 = test_single_image()
    if success1:
        check_output("test_output_single")
    
    # 測試資料夾
    success2 = test_directory()
    if success2:
        check_output("test_output_batch")
    
    print(f"\n測試結果:")
    print(f"單張圖片測試: {'✓' if success1 else '✗'}")
    print(f"資料夾測試: {'✓' if success2 else '✗'}")

if __name__ == "__main__":
    main()
