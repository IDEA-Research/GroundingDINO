#!/usr/bin/env python3
"""
影片數字抓取程式

使用 OpenCV 從影片每分鐘抓取圖片，然後使用 OpenAI GPT-5 模型識別其中的數字。

使用方法:
    python video_digit_extractor.py --video_path video.mkv --api_key your_openai_api_key
"""

import cv2
import os
import argparse
import json
import base64
from datetime import datetime
from pathlib import Path
import time
from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    print("請安裝 OpenAI Python SDK: pip install openai")
    exit(1)


def extract_frames_every_minute(video_path, output_dir):
    """從影片每分鐘抓取一張圖片"""
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"影片檔案不存在: {video_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 開啟影片
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"無法開啟影片: {video_path}")
    
    # 取得影片資訊
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_minutes = (total_frames / fps) / 60
    
    print(f"影片資訊: FPS={fps:.1f}, 總時長={duration_minutes:.1f}分鐘")
    
    # 每分鐘的幀數
    frames_per_minute = int(fps * 60)
    
    extracted_frames = []
    current_frame = 0
    minute = 0
    
    print("開始抓取圖片...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 每分鐘抓取一張
        if current_frame % frames_per_minute == 0:
            filename = f"frame_minute_{minute:03d}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            cv2.imwrite(filepath, frame)
            extracted_frames.append((filepath, minute))
            
            print(f"  已保存: {filename} (第 {minute} 分鐘)")
            minute += 1
        
        current_frame += 1
    
    cap.release()
    print(f"圖片抓取完成，共 {len(extracted_frames)} 張")
    
    return extracted_frames


def encode_image_to_base64(image_path):
    """將圖片編碼為 base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_digits_with_gpt5(client, image_path):
    """使用LLM從圖片中識別數字"""
    
    try:
        base64_image = encode_image_to_base64(image_path)
        
        response = client.chat.completions.create(
            model="gpt-4o",  # 注意：GPT-5 尚未發布，使用 gpt-4o
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "請找出這張圖片中所有的數字，以 JSON 格式回傳：{\"digits\": [\"數字1\", \"數字2\", ...], \"description\": \"數字的位置描述\"}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=512
        )
        
        content = response.choices[0].message.content
        
        # 嘗試解析 JSON
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            result = {"digits": [], "description": content}
        
        return {"success": True, "result": result}
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def process_video(video_path, api_key, output_dir="video_analysis"):
    """處理影片的主要函數"""
    
    print(f"開始處理影片: {video_path}")
    
    # 建立輸出目錄
    video_name = Path(video_path).stem
    analysis_dir = os.path.join(output_dir, video_name)
    frames_dir = os.path.join(analysis_dir, "frames")
    
    # 初始化 OpenAI 客戶端
    client = OpenAI(api_key=api_key)
    
    # 1. 抓取圖片
    print("\n=== 抓取影片圖片 ===")
    extracted_frames = extract_frames_every_minute(video_path, frames_dir)
    
    if not extracted_frames:
        print("未能抓取任何圖片")
        return
    
    # 2. 分析每張圖片
    print("\n=== 分析圖片中的數字 ===")
    results = []
    
    for i, (image_path, minute) in enumerate(extracted_frames, 1):
        print(f"\n處理第 {i}/{len(extracted_frames)} 張圖片 (第 {minute} 分鐘)")
        
        analysis = extract_digits_with_gpt5(client, image_path)
        
        result_entry = {
            "minute": minute,
            "image_path": image_path,
            "analysis": analysis
        }
        
        results.append(result_entry)
        
        if analysis["success"]:
            digits = analysis["result"].get("digits", [])
            print(f"  找到數字: {digits}")
        else:
            print(f"  錯誤: {analysis['error']}")
        
        # 避免 API 限制
        time.sleep(1)
    
    # 3. 保存結果
    print("\n=== 保存結果 ===")
    results_file = os.path.join(analysis_dir, "digit_results.json")
    
    final_result = {
        "video_path": video_path,
        "timestamp": datetime.now().isoformat(),
        "total_frames": len(results),
        "results": results
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    
    print(f"結果已保存到: {results_file}")
    
    # 生成簡單報告
    report_file = os.path.join(analysis_dir, "report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"影片數字識別報告\n")
        f.write(f"影片: {video_path}\n")
        f.write(f"分析時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"總圖片數: {len(results)}\n\n")
        
        for result in results:
            f.write(f"第 {result['minute']} 分鐘:\n")
            if result['analysis']['success']:
                digits = result['analysis']['result'].get('digits', [])
                f.write(f"  數字: {digits}\n")
            else:
                f.write(f"  錯誤: {result['analysis']['error']}\n")
            f.write("\n")
    
    print(f"報告已保存到: {report_file}")
    print("\n✅ 處理完成！")


def main():
    parser = argparse.ArgumentParser(description="影片數字抓取程式")
    parser.add_argument("--video_path", "-v", required=True, help="影片檔案路徑")
    parser.add_argument("--api_key", "-k", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API 金鑰")
    parser.add_argument("--output_dir", "-o", default="video_analysis", help="輸出目錄")
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("錯誤: 未設定 API Key。請設定 OPENAI_API_KEY 環境變數或使用 --api_key 參數。")
        return
    
    if not os.path.exists(args.video_path):
        print(f"錯誤: 影片檔案不存在: {args.video_path}")
        return
    
    try:
        process_video(args.video_path, args.api_key, args.output_dir)
    except Exception as e:
        print(f"錯誤: {e}")


if __name__ == "__main__":
    main()