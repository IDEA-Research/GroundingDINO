#!/usr/bin/env python3
"""
測試IoU過濾功能
"""

import os
import subprocess
import sys

def test_iou_filtering():
    """測試IoU過濾功能"""
    print("測試IoU過濾功能...")
    
    # 測試不同的IoU閾值
    iou_thresholds = [0.3, 0.5, 0.7]
    
    for iou_threshold in iou_thresholds:
        print(f"\n--- 測試 IoU閾值: {iou_threshold} ---")
        
        cmd = [
            "python3", "convert_to_yolo_format.py",
            "--config_file", "groundingdino/config/GroundingDINO_SwinB_cfg.py",
            "--checkpoint_path", "weights/groundingdino_swinb_cogcoor.pth",
            "--input", "medSample/frame_000019.jpg",
            "--text_prompt", "screen . monitor . display .",
            "--output_dir", f"test_iou_{iou_threshold}",
            "--box_threshold", "0.2",  # 降低閾值來獲得更多檢測
            "--text_threshold", "0.2",
            "--iou_threshold", str(iou_threshold),
            "--keep_all"
        ]
        
        print(f"執行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ 執行成功")
            # 檢查輸出檔案
            output_file = f"test_iou_{iou_threshold}/frame_000019.txt"
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    lines = f.read().strip().split('\n')
                    detection_count = len([line for line in lines if line.strip()])
                    print(f"  檢測到 {detection_count} 個螢幕")
            else:
                print("  未找到輸出檔案")
        else:
            print("✗ 執行失敗")
            print(result.stderr)

def compare_with_without_iou():
    """比較有無IoU過濾的結果"""
    print("\n" + "="*60)
    print("比較有無IoU過濾的結果")
    print("="*60)
    
    # 不使用IoU過濾（設置非常高的閾值）
    print("\n1. 不使用IoU過濾 (iou_threshold=1.0):")
    cmd_no_iou = [
        "python3", "convert_to_yolo_format.py",
        "--config_file", "groundingdino/config/GroundingDINO_SwinB_cfg.py",
        "--checkpoint_path", "weights/groundingdino_swinb_cogcoor.pth",
        "--input", "medSample/frame_000019.jpg",
        "--text_prompt", "screen . monitor . display .",
        "--output_dir", "test_no_iou",
        "--box_threshold", "0.2",
        "--text_threshold", "0.2",
        "--iou_threshold", "1.0",  # 不會過濾任何檢測
        "--keep_all"
    ]
    
    result = subprocess.run(cmd_no_iou, capture_output=True, text=True)
    if result.returncode == 0:
        with open("test_no_iou/frame_000019.txt", 'r') as f:
            lines = f.read().strip().split('\n')
            no_iou_count = len([line for line in lines if line.strip()])
            print(f"  檢測到 {no_iou_count} 個螢幕")
    
    # 使用IoU過濾
    print("\n2. 使用IoU過濾 (iou_threshold=0.5):")
    cmd_with_iou = [
        "python3", "convert_to_yolo_format.py",
        "--config_file", "groundingdino/config/GroundingDINO_SwinB_cfg.py",
        "--checkpoint_path", "weights/groundingdino_swinb_cogcoor.pth",
        "--input", "medSample/frame_000019.jpg",
        "--text_prompt", "screen . monitor . display .",
        "--output_dir", "test_with_iou",
        "--box_threshold", "0.2",
        "--text_threshold", "0.2",
        "--iou_threshold", "0.5",
        "--keep_all"
    ]
    
    result = subprocess.run(cmd_with_iou, capture_output=True, text=True)
    if result.returncode == 0:
        with open("test_with_iou/frame_000019.txt", 'r') as f:
            lines = f.read().strip().split('\n')
            with_iou_count = len([line for line in lines if line.strip()])
            print(f"  檢測到 {with_iou_count} 個螢幕")
        
        if 'no_iou_count' in locals():
            removed_count = no_iou_count - with_iou_count
            print(f"\n📊 IoU過濾效果:")
            print(f"  原始檢測: {no_iou_count} 個")
            print(f"  過濾後: {with_iou_count} 個")
            print(f"  移除重複: {removed_count} 個")

def show_detection_details():
    """顯示檢測詳細信息"""
    print("\n" + "="*60)
    print("檢測詳細信息")
    print("="*60)
    
    # 比較不同檔案的檢測結果
    test_dirs = ["test_no_iou", "test_with_iou", "test_iou_0.3", "test_iou_0.7"]
    
    for test_dir in test_dirs:
        txt_file = f"{test_dir}/frame_000019.txt"
        if os.path.exists(txt_file):
            print(f"\n{test_dir}:")
            with open(txt_file, 'r') as f:
                lines = f.read().strip().split('\n')
                for i, line in enumerate(lines):
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id, x, y, w, h = parts
                            area = float(w) * float(h)
                            print(f"  Box {i+1}: class={class_id}, center=({x}, {y}), size=({w}, {h}), area={area:.6f}")

def main():
    print("IoU過濾功能測試")
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
    
    # 執行測試
    test_iou_filtering()
    compare_with_without_iou()
    show_detection_details()
    
    print("\n" + "="*60)
    print("測試完成！請檢查生成的輸出檔案來驗證IoU過濾效果。")

if __name__ == "__main__":
    main()
