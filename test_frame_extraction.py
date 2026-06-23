#!/usr/bin/env python3
"""
測試影片圖片抓取功能
"""

import cv2
import os
from pathlib import Path

def test_frame_extraction():
    """測試從影片抓取圖片的功能"""
    
    video_path = "medSample/med20250812-9.mkv"
    output_dir = "test_frames"
    
    print("=== 測試影片圖片抓取功能 ===")
    
    if not os.path.exists(video_path):
        print(f"❌ 影片檔案不存在: {video_path}")
        return False
    
    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 開啟影片
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 無法開啟影片: {video_path}")
        return False
    
    # 取得影片資訊
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_minutes = (total_frames / fps) / 60
    
    print(f"影片資訊:")
    print(f"  檔案: {video_path}")
    print(f"  FPS: {fps:.1f}")
    print(f"  總幀數: {total_frames}")
    print(f"  總時長: {duration_minutes:.1f} 分鐘")
    
    # 每分鐘的幀數
    frames_per_minute = int(fps * 60)
    
    extracted_frames = []
    current_frame = 0
    minute = 0
    
    print(f"\n開始抓取圖片（每分鐘一張）...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 每分鐘抓取一張
        if current_frame % frames_per_minute == 0:
            filename = f"test_frame_minute_{minute:03d}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # 保存圖片
            success = cv2.imwrite(filepath, frame)
            if success:
                extracted_frames.append((filepath, minute))
                print(f"  ✅ 已保存: {filename} (第 {minute} 分鐘)")
            else:
                print(f"  ❌ 保存失敗: {filename}")
            
            minute += 1
        
        current_frame += 1
    
    cap.release()
    
    print(f"\n=== 測試結果 ===")
    print(f"成功抓取 {len(extracted_frames)} 張圖片")
    print(f"圖片保存在: {output_dir}/")
    
    # 檢查檔案大小
    for filepath, minute in extracted_frames:
        if os.path.exists(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  {os.path.basename(filepath)}: {size_kb:.1f} KB")
    
    return len(extracted_frames) > 0

if __name__ == "__main__":
    success = test_frame_extraction()
    if success:
        print("\n✅ 圖片抓取功能測試成功！")
    else:
        print("\n❌ 圖片抓取功能測試失敗！")