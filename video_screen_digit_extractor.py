#!/usr/bin/env python3
"""
影片螢幕數字抓取程式

使用 GroundingDINO 檢測影片中的螢幕，然後使用 OpenAI GPT 模型識別螢幕中的數字。
每10秒從影片中擷取一幀進行分析。

使用方法:
    python video_screen_digit_extractor.py --video_path video.mkv --provider openrouter --api_key your_openrouter_api_key
"""

import cv2
import os
import math
import argparse
import json
import base64
from datetime import datetime
from pathlib import Path
import time
import sys
import torch
import numpy as np
from PIL import Image
import traceback
import uuid
import threading
import concurrent.futures
from dotenv import load_dotenv

# 載入 .env 檔案
load_dotenv()


# 添加當前目錄到路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 設定 GroundingDINO 環境
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
if torch_lib_path not in current_ld_path:
    if current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{current_ld_path}"
    else:
        os.environ['LD_LIBRARY_PATH'] = torch_lib_path
    print(f"已設定 LD_LIBRARY_PATH 包含 PyTorch 庫: {torch_lib_path}")

# GroundingDINO 會自動檢測是否有 C++ 擴展並選擇合適的模式
print("🔄 初始化 GroundingDINO...")

# 延遲導入 GroundingDINO 相關模組 (在需要時才導入)
_inference_screen_crop_module = None

def _get_inference_module():
    """延遲導入 inference_screen_crop 模組"""
    global _inference_screen_crop_module
    if _inference_screen_crop_module is None:
        import inference_screen_crop
        _inference_screen_crop_module = inference_screen_crop
    return _inference_screen_crop_module

def load_model(*args, **kwargs):
    return _get_inference_module().load_model(*args, **kwargs)

def load_image(*args, **kwargs):
    return _get_inference_module().load_image(*args, **kwargs)

def get_grounding_output(*args, **kwargs):
    return _get_inference_module().get_grounding_output(*args, **kwargs)

def merge_overlapping_boxes(*args, **kwargs):
    return _get_inference_module().merge_overlapping_boxes(*args, **kwargs)

def crop_and_save_screens(*args, **kwargs):
    return _get_inference_module().crop_and_save_screens(*args, **kwargs)

try:
    from openai import OpenAI
except ImportError:
    print("請安裝 OpenAI Python SDK: pip install openai")
    # exit(1) # Don't exit here, might use other providers


class VideoScreenDigitExtractor:
    """影片螢幕數字抓取器"""
    
    def __init__(self, api_key: str, config_file: str, checkpoint_path: str,
                 provider: str = "openrouter", base_url: str = None,
                 model: str = None, cpu_only: bool = False, target_data: str = "all",
                 frame_interval_seconds: int = 10,
                 rtsp_url: str = None,
                 camera_name: str = None):
        """
        初始化抓取器
        
        Args:
            api_key: API 金鑰
            config_file: GroundingDINO 配置檔案路徑
            checkpoint_path: GroundingDINO 模型權重路徑
            provider: 模型廠商 (openrouter, local)
            model: 使用的模型名稱
            cpu_only: 是否只使用 CPU
            target_data: 要抓取的目標資料類型
            frame_interval_seconds: 幀擷取間隔（秒），預設為10秒 (用於影片檔案分析)
            rtsp_url: RTSP 串流 URL (用於即時監測)
            camera_name: IP 攝影機的自訂名稱 (用於即時監測)
        """
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.config_file = config_file
        self.checkpoint_path = checkpoint_path
        self.cpu_only = cpu_only
        self.grounding_model = None
        self.target_data = target_data
        self.frame_interval_seconds = frame_interval_seconds # GPT 分析的幀間隔 (影片模式)
        self.rtsp_url = rtsp_url
        self.camera_name = camera_name

        self._stop_event = threading.Event() # 用於停止串流捕獲的事件
        self._is_running = False # 標誌位，指示串流捕獲是否正在運行
        self._capture_thread = None # 用於保存串流捕獲線程
        self.last_error = None # 用於保存最後發生的錯誤訊息
        self.gpu_lock = threading.Lock() # 用於同步 GPU 推論，避免 Jetson OOM
        self._analysis_executor = None # 用於背景非同步分析的執行緒池
        
        self.init_client()

    def init_client(self):
        """初始化 API 客戶端"""
        if "OpenAI" not in globals():
            raise ImportError("請安裝 openai 套件")
            
        if self.provider == "openrouter":
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
            )
        elif self.provider == "local":
            self.client = OpenAI(
                base_url=self.base_url or "http://192.168.1.237:1234/v1",
                api_key=self.api_key or "sk-no-key-required",
            )
        else:
            raise ValueError(f"不支援的廠商: {self.provider}")
        
    def load_grounding_model(self):
        """載入 GroundingDINO 模型"""
        if self.grounding_model is None:
            print("正在載入 GroundingDINO 模型...")
            self.grounding_model = load_model(
                self.config_file, 
                self.checkpoint_path, 
                cpu_only=self.cpu_only
            )
            print("GroundingDINO 模型載入完成")
        return self.grounding_model

    def start_rtsp_capture(self, session_id, rtsp_url, camera_name, capture_interval_seconds=60, output_dir="video_screen_analysis", save_to_mongodb=True):
        """在獨立線程中啟動 RTSP 串流捕獲和分析（每隔指定秒數進行一輪，對齊絕對系統時間）"""
        if self._is_running:
            print("🚫 串流捕獲已在運行中。")
            return False

        if not rtsp_url:
            print("❌ 缺少 RTSP URL。")
            return False

        # 初始化非同步分析的執行緒池
        if self._analysis_executor is None:
            self._analysis_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=3,
                thread_name_prefix="RTSPAnalysis"
            )

        self._stop_event.clear()  # 重置停止事件
        self._is_running = True
        self._capture_thread = threading.Thread(
            target=self._run_rtsp_capture_loop,
            args=(session_id, rtsp_url, camera_name, capture_interval_seconds, output_dir, save_to_mongodb)
        )
        self._capture_thread.daemon = True
        self._capture_thread.start()
        print(f"🚀 已在背景啟動 RTSP 串流捕獲: {camera_name} ({rtsp_url}), 間隔: {capture_interval_seconds}秒, 會話ID: {session_id}")
        return True

    def _run_rtsp_capture_loop(self, session_id, rtsp_url, camera_name, capture_interval_seconds, output_dir, save_to_mongodb):
        """RTSP 串流捕獲的實際循環邏輯 (在獨立線程中運行，對齊絕對系統時間)"""
        print(f"[RTSP 捕獲線程] 啟動: {rtsp_url}，擷取間隔: {capture_interval_seconds} 秒")
        frame_count = 0
        stream_start_time = time.time()

        # 檢查是否使用絕對時間對齊
        use_alignment = capture_interval_seconds > 0
        if use_alignment:
            # 計算出第一個對齊的擷取時間點 (例如：若間隔 60 秒，則對齊到下一個整分鐘)
            next_capture_time = math.ceil(time.time() / capture_interval_seconds) * capture_interval_seconds
        else:
            next_capture_time = time.time()

        # 導入 MongoDB 管理器
        mongo_manager = None
        if save_to_mongodb:
            try:
                from mongo_manager import mongo_manager
                if mongo_manager.db is None:
                    print("[RTSP 捕獲線程] 正在啟動 MongoDB...")
                    mongo_manager.start_mongodb()
            except Exception as e:
                print(f"[RTSP 捕獲線程] ⚠️ 無法導入 MongoDB 管理器: {e}")
                save_to_mongodb = False

        try:
            while not self._stop_event.is_set():
                current_time = time.time()

                # 檢查是否到達了下一個對齊的擷取時間點
                if use_alignment and current_time < next_capture_time:
                    # 還沒到時間，精細等待 0.1 秒，以確保高精度的時間對齊
                    time.sleep(0.1)
                    continue

                # 確定這次擷取的基準時間戳記
                capture_timestamp = next_capture_time if use_alignment else current_time

                print(f"\n[RTSP 捕獲線程] === 正在連接並擷取畫面 (第 {frame_count + 1} 次) ===")
                if use_alignment:
                    print(f"[RTSP 捕獲線程] 🎯 對齊時間點: {datetime.fromtimestamp(capture_timestamp).strftime('%Y-%m-%d %H:%M:%S')}")

                # 立即計算「下一個」對齊的擷取時間點（單純遞增，不跳過任何整點，確保排隊分析）
                if use_alignment:
                    next_capture_time += capture_interval_seconds

                frame = None
                cap = cv2.VideoCapture(rtsp_url)
                # 檢查 OpenCV 版本是否支援 CAP_PROP_FFMPEG_CAPTURE_MODE
                if hasattr(cv2, 'CAP_PROP_FFMPEG_CAPTURE_MODE'):
                    cap.set(cv2.CAP_PROP_FFMPEG_CAPTURE_MODE, 1) # Force FFmpeg to use TCP
                else:
                    # 備選方案：透過環境變數設定 FFmpeg 傳輸協議
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                
                if hasattr(cv2, 'CAP_PROP_RTMP_TIMEOUT_MS'):
                    cap.set(cv2.CAP_PROP_RTMP_TIMEOUT_MS, 3000) # 設置連接超時
                
                if not cap.isOpened():
                    print(f"[RTSP 捕獲線程] ⚠️ 無法連接 RTSP 流: {rtsp_url}，稍後重試...")
                    cap.release()
                    time.sleep(5)
                    continue

                # 讀取幾幀以確保畫面穩定
                for _ in range(5): 
                    ret, temp_frame = cap.read()
                    if ret:
                        frame = temp_frame
                    else:
                        print(f"[RTSP 捕獲線程] ⚠️ 無法從流中讀取幀。")
                        break
                
                cap.release()

                if frame is None:
                    print("[RTSP 捕獲線程] ⚠️ 未能獲取有效畫面")
                    time.sleep(1)
                    continue

                time_seconds = capture_timestamp - stream_start_time
                filename = f"stream_frame_{session_id[:8]}_{frame_count:05d}_{int(time_seconds):04d}s.jpg"
                analysis_base_dir = os.path.join(output_dir, f"stream_{session_id[:8]}")
                frames_dir = os.path.join(analysis_base_dir, "frames")
                os.makedirs(frames_dir, exist_ok=True)
                filepath = os.path.join(frames_dir, filename)
                cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"[RTSP 捕獲線程] 已保存: {filename}")

                # 核心改變：將整套分析與資料庫寫入流程提交到背景非同步執行緒池
                if self._analysis_executor:
                    self._analysis_executor.submit(
                        self._async_analyze_frame_task,
                        filepath,
                        time_seconds,
                        analysis_base_dir,
                        capture_timestamp,
                        save_to_mongodb,
                        rtsp_url,
                        session_id,
                        camera_name,
                        mongo_manager
                    )
                    print(f"[RTSP 捕獲線程] 🚀 已將 {filename} 提交至背景分析執行緒池。")
                else:
                    print(f"[RTSP 捕獲線程] ⚠️ 背景執行緒池未啟動，改為同步處理。")
                    self._async_analyze_frame_task(
                        filepath,
                        time_seconds,
                        analysis_base_dir,
                        capture_timestamp,
                        save_to_mongodb,
                        rtsp_url,
                        session_id,
                        camera_name,
                        mongo_manager
                    )

                frame_count += 1

        except Exception as e:
            print(f"[RTSP 捕獲線程] ❌ 發生未預期的錯誤: {e}")
            print(traceback.format_exc())
        finally:
            self._is_running = False
            print(f"[RTSP 捕獲線程] 停止捕獲 RTSP 流: {rtsp_url}")

    def _async_analyze_frame_task(self, filepath, time_seconds, analysis_base_dir, capture_timestamp, save_to_mongodb, rtsp_url, session_id, camera_name, mongo_manager):
        """背景非同步分析單個畫面的任務"""
        try:
            print(f"[背景分析] 🎯 開始處理畫面: {os.path.basename(filepath)} (對齊時間: {datetime.fromtimestamp(capture_timestamp).strftime('%Y-%m-%d %H:%M:%S')})")
            
            # 呼叫現有的分析邏輯 (其中 detect_screens_in_image 已被 self.gpu_lock 保護)
            frame_result = self.process_single_video_frame(filepath, time_seconds, analysis_base_dir)
            
            # 檢查是否有 API 錯誤
            has_api_error = False
            for sa in frame_result.get('screen_analyses', []):
                if not sa['analysis']['success']:
                    self.last_error = sa['analysis']['error']
                    has_api_error = True
                    break
            
            if not has_api_error:
                self.last_error = None # 如果成功，清除之前的錯誤

            frame_result['capture_timestamp'] = datetime.fromtimestamp(capture_timestamp)

            if save_to_mongodb and mongo_manager and mongo_manager.db is not None:
                try:
                    video_path_for_db = f"RTSP:{rtsp_url}" # 將 RTSP URL 作為 video_path 儲存
                    mongo_manager.save_analysis_result(
                        video_path_for_db, session_id, [frame_result], source="stream",
                        llm_model=self.model, camera_name=camera_name
                    )
                    print(f"[背景分析] ✅ [{datetime.fromtimestamp(capture_timestamp).strftime('%H:%M:%S')}] 已成功分析並保存到 MongoDB")
                except Exception as e:
                    print(f"[背景分析] ⚠️ MongoDB 保存失敗: {e}")

        except Exception as e:
            print(f"[背景分析] ❌ 處理對齊時間點 {datetime.fromtimestamp(capture_timestamp).strftime('%H:%M:%S')} 時發生錯誤: {e}")
            print(traceback.format_exc())

    def stop_capture(self):
        """停止 RTSP 串流捕獲線程"""
        if self._analysis_executor:
            print("[RTSP 捕獲線程] 正在關閉背景分析線程池...")
            self._analysis_executor.shutdown(wait=False)
            self._analysis_executor = None

        if self._is_running:
            self._stop_event.set()
            if self._capture_thread:
                self._capture_thread.join(timeout=10)  # 等待線程結束，設置超時
                if self._capture_thread.is_alive():
                    print("⚠️ 停止捕獲線程超時，可能未能完全停止。")
            self._is_running = False
            print("🛑 串流捕獲已停止。")
            return True
        print("🚫 串流捕獲未在運行。")
        return False
    
    def process_video_realtime(self, video_path, output_dir="video_screen_analysis"):
        """即時處理影片，使用 seek 直接跳到每10秒開頭"""
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"影片檔案不存在: {video_path}")
        
        # 載入 GroundingDINO 模型
        self.load_grounding_model()
        
        # 建立輸出目錄結構
        video_name = Path(video_path).stem
        analysis_dir = os.path.join(output_dir, video_name)
        frames_dir = os.path.join(analysis_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # 開啟影片
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"無法開啟影片: {video_path}")
        
        # 取得影片資訊
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps
        duration_minutes = duration_seconds / 60
        total_intervals = int(duration_seconds / self.frame_interval_seconds) + (1 if duration_seconds % self.frame_interval_seconds > 0 else 0)
        
        print(f"影片資訊: FPS={fps:.1f}, 總時長={duration_minutes:.1f}分鐘 ({duration_seconds:.1f}秒)")
        print(f"目標資料類型: {self.target_data}")
        print(f"擷取間隔: {self.frame_interval_seconds}秒")
        print(f"將處理 {total_intervals} 個時間點（每{self.frame_interval_seconds}秒）")
        print("開始使用 seek模式處理影片...")
        
        results = []
        
        try:
            for interval in range(total_intervals):
                # 計算目標時間（毫秒）
                target_time_ms = interval * self.frame_interval_seconds * 1000
                
                # 使用 seek 直接跳到指定時間
                cap.set(cv2.CAP_PROP_POS_MSEC, target_time_ms)
                
                # 讀取該時間點的畫面
                ret, frame = cap.read()
                if not ret:
                    print(f"無法讀取第 {interval * self.frame_interval_seconds} 秒的畫面，可能已到達影片結尾")
                    break
                
                time_seconds = interval * self.frame_interval_seconds
                minutes = time_seconds // 60
                seconds = time_seconds % 60
                print(f"\n=== 處理第 {interval + 1} 個畫面 (時間: {minutes}:{seconds:02d}) ===")
                
                # 保存當前畫面
                filename = f"frame_{time_seconds:04d}s.jpg"
                filepath = os.path.join(frames_dir, filename)
                cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])  # Fixed: Explicitly set quality to 95
                
                print(f"  已保存: {filename}")
                
                # 立即進行螢幕檢測和數字識別
                frame_result = self.process_single_video_frame(filepath, time_seconds, analysis_dir)
                results.append(frame_result)
                
        except KeyboardInterrupt:
            print("\n使用者中斷處理")
        finally:
            cap.release()
            print(f"\n影片處理完成，共處理 {len(results)} 張畫面")
        
        # 保存最終結果
        self.save_video_results(video_path, results, analysis_dir)
        
        return results
    
    def merge_duplicate_devices(self, screen_analyses):
        """合併同一 frame 中的重複設備，保留最完整的數據"""
        if len(screen_analyses) <= 1:
            return screen_analyses
        
        # 按設備類型分組
        device_groups = {}
        for analysis in screen_analyses:
            if analysis['analysis']['success']:
                result_data = analysis['analysis']['result']
                device_model = result_data.get('model', 'unknown')
                
                if device_model not in device_groups:
                    device_groups[device_model] = []
                device_groups[device_model].append(analysis)
            else:
                # 分析失敗的直接保留
                if 'failed' not in device_groups:
                    device_groups['failed'] = []
                device_groups['failed'].append(analysis)
        
        merged_analyses = []
        
        for device_model, analyses in device_groups.items():
            if device_model == 'failed':
                # 分析失敗的直接添加
                merged_analyses.extend(analyses)
            elif len(analyses) == 1:
                # 只有一個，直接添加
                merged_analyses.append(analyses[0])
            else:
                # 有多個相同設備，需要合併
                print(f"    發現 {len(analyses)} 個 {device_model} 設備，正在合併...")
                best_analysis = self.select_best_device_analysis(analyses, device_model)
                merged_analyses.append(best_analysis)
        
        return merged_analyses
    
    def select_best_device_analysis(self, analyses, device_model):
        """從多個相同設備分析中選擇最佳的"""
        best_analysis = None
        best_score = -1
        
        for analysis in analyses:
            score = self.calculate_analysis_completeness_score(analysis, device_model)
            print(f"      螢幕 {analysis['screen_number']} 完整度評分: {score}")
            
            if score > best_score:
                best_score = score
                best_analysis = analysis
        
        # 檢查數值是否一致
        self.check_values_consistency(analyses, device_model)
        
        print(f"      選擇螢幕 {best_analysis['screen_number']} 作為最佳結果")
        return best_analysis
    
    def calculate_analysis_completeness_score(self, analysis, device_model):
        """計算分析結果的完整度評分"""
        if not analysis['analysis']['success']:
            return 0
        
        result_data = analysis['analysis']['result']
        medical_values = result_data.get('medical_values', {})
        
        # 根據設備類型定義期望的醫療數值
        expected_values = self.get_expected_medical_values(device_model)
        
        score = 0
        total_possible = len(expected_values)
        
        if total_possible == 0:
            # 如果沒有定義期望值，則根據實際獲得的數值數量評分
            return len(medical_values)
        
        for expected_key in expected_values:
            if expected_key in medical_values:
                value = medical_values[expected_key]
                if value is not None and value != "":
                    if isinstance(value, (int, float)) and value > 0:
                        score += 1
                    elif isinstance(value, str) and value.strip():
                        score += 1
        
        # 額外獎勵：有更多數字檢測到
        numbers_count = len(result_data.get('numbers', []))
        score += numbers_count * 0.1
        
        # 額外獎勵：有描述內容
        if result_data.get('description'):
            score += 0.5
        
        return score
    
    def get_expected_medical_values(self, device_model):
        """根據設備類型返回期望的醫療數值欄位"""
        device_expectations = {
            'Philips MP20': ['heart_rate', 'blood_pressure', 'spo2', 'respiration_rate'],
            'Philips intellivue mp20 patient monitor': ['heart_rate', 'blood_pressure', 'spo2', 'respiration_rate'],
            'Dräger C500': ['PIP', 'PEEP', 'MAP', 'FiO2', 'VT', 'RR', 'MV'],
            'Dräger Infinity C500': ['PIP', 'PEEP', 'MAP', 'FiO2', 'VT', 'RR', 'MV'],
            'Dräger Evita V600': ['PIP', 'PEEP', 'MAP', 'FiO2'],
            'Dräger PRT-V600 neo': ['PIP', 'PEEP', 'MAP', 'FiO2'],
            'SOMANETICS INVOS': ['rSO2_right', 'rSO2_left'],
            'SOMANETICS INVOS cerebral/somatic oximeter monitor': ['rSO2_right', 'rSO2_left'],
            'Medtronic_INVOS_5100C': ['rso2_left', 'rso2_left_avg', 'rso2_right', 'rso2_right_avg'],
            #----測試新增設備----
            'Osypka Medical': ['CO', 'CI', 'HR', 'ICON', 'TFC', 'SVRI', 'SVV'],
            'Osypka Aesculon': ['CO', 'CI', 'HR', 'ICON', 'TFC', 'SVRI', 'SVV'],
            'Osypka Icon': ['CO', 'CI', 'HR', 'ICON', 'TFC', 'SVRI', 'SVV']
        }
        
        # 嘗試精確匹配
        if device_model in device_expectations:
            return device_expectations[device_model]
        
        # 嘗試部分匹配
        for key, values in device_expectations.items():
            if device_model.lower() in key.lower() or key.lower() in device_model.lower():
                return values
        
        return []
    
    def check_values_consistency(self, analyses, device_model):
        """檢查相同設備的數值是否一致"""
        if len(analyses) <= 1:
            return
        
        print(f"      檢查 {device_model} 數值一致性:")
        
        # 收集所有成功分析的醫療數值
        all_medical_values = []
        for analysis in analyses:
            if analysis['analysis']['success']:
                medical_values = analysis['analysis']['result'].get('medical_values', {})
                all_medical_values.append({
                    'screen_number': analysis['screen_number'],
                    'values': medical_values
                })
        
        if len(all_medical_values) <= 1:
            return
        
        # 檢查每個醫療數值欄位
        all_keys = set()
        for mv in all_medical_values:
            print(mv)
            all_keys.update(mv['values'].keys())
        
        for key in all_keys:
            values = []
            screens = []
            for mv in all_medical_values:
                if key in mv['values'] and mv['values'][key] is not None:
                    values.append(mv['values'][key])
                    screens.append(mv['screen_number'])
            
            if len(set(values)) > 1:
                print(f"        ⚠️  {key} 數值不一致:")
                for i, (screen, value) in enumerate(zip(screens, values)):
                    print(f"          螢幕 {screen}: {value}")
            else:
                if values:
                    print(f"        ✅ {key}: {values[0]} (一致)")
    
    def process_single_video_frame(self, image_path, time_seconds, analysis_dir):
        """處理單一影片畫面"""
        frame_result = {
            "time_seconds": time_seconds,
            "image_path": image_path,
            "screens_detected": 0,
            "screen_analyses": []
        }
        
        try:
            # 檢測螢幕
            t0 = time.time() # 測量GroundingDINO 時間
            print("detect screens by using grounding dino")
            screen_paths = self.detect_screens_in_image(image_path, analysis_dir)
            dino_time = time.time() - t0
            print(f"[效能分析] GroundingDINO 偵測耗時: {dino_time:.2f} 秒")
            frame_result["screens_detected"] = len(screen_paths)
            
            if not screen_paths:
                print("  未檢測到螢幕")
                return frame_result
            
            print(f"  檢測到 {len(screen_paths)} 個螢幕")
            
            # 分析每個螢幕中的數字 (並行處理)
            screen_analyses = []
            
            # 定義單個分析任務
            def analyze_single_screen(idx, path):
                # print(f"    [並行] 開始分析螢幕 {idx}: {os.path.basename(path)}")
                t_start = time.time()
                try:
                    analysis_result = self.extract_digits_from_screen(path)
                    cost = time.time() - t_start
                    print(f"[效能分析] LLM 分析螢幕 {idx} 耗時: {cost:.2f} 秒")
                    
                    if analysis_result["success"]:
                        res = analysis_result["result"]
                        model_name = res.get('model', 'unknown')
                        print(f"      螢幕 {idx} 檢測到設備: {model_name}")
                    else:
                        print(f"      螢幕 {idx} 分析失敗: {analysis_result['error']}")
                        
                    return {
                        "screen_number": idx,
                        "screen_path": path,
                        "analysis": analysis_result
                    }
                except Exception as e:
                    print(f"      螢幕 {idx} 執行發生異常: {str(e)}")
                    return {
                        "screen_number": idx,
                        "screen_path": path,
                        "analysis": {"success": False, "error": str(e)}
                    }

            # 使用 ThreadPoolExecutor 並行執行
            # 限制最大 worker 數量，避免瞬間請求過多
            max_workers = min(len(screen_paths), 10)
            print(f"  🚀 啟動 {max_workers} 個線程同時進行 GPT 分析...")
            
            t_parallel_start = time.time() # 記錄並行開始時間
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交任務
                future_to_idx = {
                    executor.submit(analyze_single_screen, j, screen_path): j
                    for j, screen_path in enumerate(screen_paths, 1)
                }
                
                # 收集結果
                for future in concurrent.futures.as_completed(future_to_idx):
                    try:
                        result = future.result()
                        screen_analyses.append(result)
                    except Exception as exc:
                        idx = future_to_idx[future]
                        print(f"      螢幕 {idx} 任務產生異常: {exc}")

            parallel_total_time = time.time() - t_parallel_start
            print(f"[效能分析] GPT 多路並行總耗時: {parallel_total_time:.2f} 秒 (處理 {len(screen_paths)} 個螢幕)")

            # 確保結果按螢幕編號排序
            screen_analyses.sort(key=lambda x: x['screen_number'])
            
            # 合併重複設備，保留最完整的數據
            if len(screen_analyses) > 1:
                print(f"  正在檢查重複設備...")
                merged_analyses = self.merge_duplicate_devices(screen_analyses)
                frame_result["screen_analyses"] = merged_analyses
                
                # 更新螢幕檢測數量為合併後的數量
                frame_result["screens_detected"] = len(merged_analyses)
                
                if len(merged_analyses) < len(screen_analyses):
                    print(f"  合併完成: {len(screen_analyses)} -> {len(merged_analyses)} 個螢幕")
            else:
                frame_result["screen_analyses"] = screen_analyses
            
            # 顯示最終結果
            print(f"  最終結果: {len(frame_result['screen_analyses'])} 個螢幕")
            for analysis in frame_result["screen_analyses"]:
                if analysis["analysis"]["success"]:
                    result_data = analysis["analysis"]["result"]
                    self.display_filtered_results(result_data, analysis["screen_number"])
                
        except Exception as e:
            print(traceback.format_exc())
            print(f"  處理畫面時發生錯誤: {str(e)}")
        
        return frame_result
    
    def save_video_results(self, video_path, results, analysis_dir):
        """保存影片分析結果"""
        results_file = os.path.join(analysis_dir, "screen_digit_results.json")
        
        final_result = {
            "video_path": video_path,
            "timestamp": datetime.now().isoformat(),
            "total_frames": len(results),
            "model_used": self.model,
            "target_data": self.target_data,
            "results": results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        print(f"結果已保存到: {results_file}")
        
        # 生成報告
        self.generate_report(final_result, analysis_dir)
    
    def detect_screens_in_image(self, image_path, output_dir):
        """使用 GroundingDINO 檢測圖片中的螢幕"""
        
        # 載入模型與執行 GPU 偵測 (使用 GPU 鎖同步，避免 OOM)
        with self.gpu_lock:
            model = self.load_grounding_model()
            print("groundingdino is loaded")
            
            # 載入圖片
            image_pil, image = load_image(image_path)
            
            # 檢測螢幕
            text_prompt = "screen . monitor . display . medical monitor ."
            box_threshold = 0.25 #0.3
            text_threshold = 0.2 #0.25
            iou_threshold = 0.3 #0.2，0.3，0.4 測試
            
            boxes_filt, pred_phrases = get_grounding_output(
                model, image, text_prompt, box_threshold, text_threshold, 
                cpu_only=self.cpu_only, token_spans=None
            )
        
        if len(boxes_filt) == 0:
            return []
        
        print(f"  檢測到 {len(boxes_filt)} 個螢幕區域")
        
        # 合併重疊的框
        if len(boxes_filt) > 1:
            merged_boxes, merged_labels = merge_overlapping_boxes(
                boxes_filt, pred_phrases, iou_threshold
            )
            print(f"  合併後: {len(merged_boxes)} 個獨特螢幕")
        else:
            merged_boxes, merged_labels = boxes_filt, pred_phrases
        
        # 裁切螢幕區域
        screen_dir = os.path.join(output_dir, "screens")
        os.makedirs(screen_dir, exist_ok=True)
        # 使用 Path.stem 來取得不含副檔名的檔名
        prefix = Path(image_path).stem
        cropped_images = crop_and_save_screens(
            image_pil, merged_boxes, merged_labels, screen_dir, prefix=prefix
        )
        
        # 返回裁切的螢幕圖片路徑
        screen_paths = []
        for cropped_img, filename in cropped_images:
            screen_path = os.path.join(screen_dir, filename)
            screen_paths.append(screen_path)
        
        return screen_paths
    
    def encode_image_to_base64(self, image_path, max_edge=1024):
        """將圖片編碼為 base64 (加入地端模型所需的 Resize 邏輯)"""
        import io
        from PIL import Image
        
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        
        # 如果圖片大於 max_edge，則進行等比例縮放
        if max(w, h) > max_edge:
            scale = max_edge / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    def get_analysis_prompt(self,device_prompts):
        """根據目標資料類型生成分析提示詞"""
        base_prompt = """請仔細分析這張螢幕圖片，找出其中的數字資訊。請以 JSON 格式回傳：
{
    "model":"Philips mp30",
    "numbers": ["完整數字1", "完整數字2", ...],
    "medical_values": {"heart_beat":92, "blood pressure":141, ...},
    "description": "螢幕內容描述"
}

下面是一些機器的描述，請在讀數字前先決定是那一台機器然後根據說明來決定如何決定數字的類別, 然後根據機器描述中規定的 keyword產生 medical_values 的值
如果找不到一樣的，model 欄位請寫unknown
"""+device_prompts+"""

請特別注意：
- digits: 單獨的數字字符 (0-9)
- numbers: 完整的數字（如 123, 45.67 等）
- medical_values: 醫療相關數值（如心率、血壓、溫度等）"""
        return base_prompt + "\n\n請識別所有可見的數字，無論大小或位置。"
    def get_templates(self):
        template_dir = 'device_templates'
        t = ''
        for fn in os.listdir(template_dir):
            if fn.split('.')[-1] == 'txt':
                f = open(template_dir+'/'+fn)
                t = t + f.read()+'\n'
        return t

    def extract_json_from_response(self, content):
        """
        從 GPT 回應中提取 JSON 內容
        處理各種可能的格式：
        1. 純 JSON
        2. Markdown code block 包裹的 JSON
        3. <|begin_of_box|> 標籤包裹的 JSON
        4. 前面有說明文字的 JSON
        """
        import re
        
        # 嘗試 1: 直接解析
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            pass
        
        # 嘗試 2: 尋找特定標籤包裹的內容
        # 匹配 ```json ... ```, ``` ... ```, 或 <|begin_of_box|> ... <|end_of_box|>
        patterns = [
            r'```(?:json)?\s*\n?(.*?)\n?```',
            r'<\|begin_of_box\|>\s*\n?(.*?)\n?<\|end_of_box\|>'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue
        
        # 嘗試 3: 尋找最外層的大括號內容
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx + 1]
                return json.loads(json_str.strip())
        except (json.JSONDecodeError, ValueError):
            pass

        # 嘗試 4: 尋找任何看起來像 JSON 的內容 (舊的 Regex 備用)
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\} [^{}]*)*\}'
        json_matches = re.findall(json_pattern, content, re.DOTALL)
        
        for match in sorted(json_matches, key=len, reverse=True):
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
        
        # 如果都失敗了，返回 None
        return None
    
    def extract_digits_from_screen(self, screen_path):
        """使用 LLM 從螢幕圖片中識別數字"""
        
        try:
            base64_image = self.encode_image_to_base64(screen_path)
            device_prompts = self.get_templates()
            # 根據目標資料類型調整提示詞
            prompt = self.get_analysis_prompt(device_prompts)
            
            content = ""
            
            if self.provider in ["openrouter", "local"]:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
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
                )
                content = response.choices[0].message.content
                
                # 顯示 Token 使用量
                if hasattr(response, 'usage') and response.usage:
                    u = response.usage
                    print(f"\n[Token 使用量] Prompt: {u.prompt_tokens}, Completion: {u.completion_tokens}, Total: {u.total_tokens}")
            
            # 顯示原始回應以便調試
            print(f"============ {self.provider} 原始回應 ============")
            print(content)
            print("=" * 50)
            
            # 使用新的 JSON 提取方法
            result = self.extract_json_from_response(content)
            
            if result is None:
                print("⚠️ 無法從回應中提取有效的 JSON")
                result = {
                    "digits": [],
                    "numbers": [],
                    "medical_values": {},
                    "description": content
                }
            else:
                print("✅ 成功提取並解析 JSON")
            
            return {"success": True, "result": result}
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ LLM 請求或解析失敗: {error_msg}")
            # 如果是 AuthenticationError，特別標註
            if "AuthenticationError" in error_msg or "401" in error_msg:
                error_msg = "API Key 驗證失敗，請檢查您輸入的 API Key 是否正確且有效。"
            elif "400" in error_msg and "model" in error_msg.lower():
                error_msg = f"模型名稱錯誤或不支援: {self.model}"
            
            return {"success": False, "error": error_msg}
    
    def process_video(self, video_path, output_dir="video_screen_analysis"):
        """處理整個影片，即時檢測螢幕並識別數字"""
        
        print(f"開始處理影片: {video_path}")
        
        # 這裡根據是否有配置 rtsp_url 來決定是處理本地影片還是啟動 RTSP 串流
        if self.rtsp_url:
            print(f"初始化時提供了 RTSP URL ({self.rtsp_url})，將以串流模式運行。")
            return self.process_video_realtime(video_path, output_dir)
        else:
            return self.process_video_realtime(video_path, output_dir)
    
    def display_filtered_results(self, result_data, screen_number):
        """根據目標資料類型顯示過濾後的結果"""
        print(f"      螢幕 {screen_number} 結果:")
        
        if self.target_data == "all":
            # 顯示所有資料
            if result_data.get('digits'):
                print(f"        數字字符: {result_data['digits']}")
            if result_data.get('numbers'):
                print(f"        完整數字: {result_data['numbers']}")
            if result_data.get('medical_values'):
                print(f"        醫療數值: {result_data['medical_values']}")
            if result_data.get('description'):
                print(f"        描述: {result_data['description']}")
        
        elif self.target_data == "digits":
            if result_data.get('digits'):
                print(f"        數字字符: {result_data['digits']}")
            else:
                print("        未找到數字字符")
        
        elif self.target_data == "numbers":
            if result_data.get('numbers'):
                print(f"        完整數字: {result_data['numbers']}")
            else:
                print("        未找到完整數字")
        
        elif self.target_data == "medical_values":
            if result_data.get('medical_values'):
                print(f"        醫療數值: {result_data['medical_values']}")
            else:
                print("        未找到醫療數值")
        
        else:
            # 自定義目標資料
            target_types = [t.strip() for t in self.target_data.split(',')]
            found_any = False
            
            for target_type in target_types:
                if target_type in ['digits', 'numbers', 'medical_values']:
                    if result_data.get(target_type):
                        print(f"        {target_type}: {result_data[target_type]}")
                        found_any = True
                elif target_type.lower() in str(result_data.get('description', '')).lower():
                    print(f"        找到目標 '{target_type}': {result_data.get('description', '')}")
                    found_any = True
            
            if not found_any:
                print(f"        未找到目標資料: {self.target_data}")
    
    def process_camera_stream(self, camera_index=0, rtsp_url=None, capture_interval_seconds=60,
                              max_duration=None, output_dir="video_screen_analysis",
                              save_to_mongodb=True, session_id=None, camera_name=None):
        """
        即時處理攝影機串流或 RTSP 流 (主要用於命令行測試或單獨運行)
        """
        print(f"開始即時處理模式 (命令行或單獨運行)..")
        
        if session_id is None:
            session_id = str(uuid.uuid4())
        if camera_name is None:
            camera_name = f"Camera_{session_id[:8]}"
        
        print(f"會話 ID: {session_id}, 攝影機名稱: {camera_name}")
        
        if rtsp_url:
            self._stop_event.clear()
            self._is_running = True # 標記為運行中，但沒有線程
            try:
                self._run_rtsp_capture_loop(session_id, rtsp_url, camera_name, capture_interval_seconds,
                                            output_dir, save_to_mongodb)
            finally:
                self._is_running = False
        else:
            print("❌ 命令行攝影機模式目前只支援 RTSP URL。")

    
    def process_single_frame(self, image_path, frame_number):
        """處理單一畫面"""
        try:
            # 檢測螢幕
            screen_paths = self.detect_screens_in_image(image_path, "temp_analysis")
            
            if not screen_paths:
                print("  未檢測到螢幕")
                return
            
            print(f"  檢測到 {len(screen_paths)} 個螢幕")
            
            # 分析每個螢幕
            for i, screen_path in enumerate(screen_paths, 1):
                print(f"    分析螢幕 {i}...")
                
                analysis = self.extract_digits_from_screen(screen_path)
                
                if analysis["success"]:
                    result_data = analysis["result"]
                    self.display_filtered_results(result_data, i)
                else:
                    print(f"      螢幕 {i} 分析失敗: {analysis['error']}")
            
            # 清理臨時檔案
            if os.path.exists("temp_analysis"):
                import shutil
                shutil.rmtree("temp_analysis")
                
        except Exception as e:
            print(traceback.format_exc())
            print(f"  處理畫面時發生錯誤: {str(e)}")
    
    def generate_report(self, analysis_result, output_dir):
        """生成摘要報告"""
        report_file = os.path.join(output_dir, "screen_analysis_report.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("影片螢幕數字識別報告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"影片檔案: {analysis_result['video_path']}\n")
            f.write(f"分析時間: {analysis_result['timestamp']}\n")
            f.write(f"總圖片數: {analysis_result['total_frames']}\n")
            f.write(f"使用模型: {analysis_result['model_used']}\n\n")
            
            # 統計資訊
            total_screens = sum(r['screens_detected'] for r in analysis_result['results'])
            successful_analyses = 0
            total_digits = 0
            total_numbers = 0
            total_medical_values = 0
            
            for result in analysis_result['results']:
                for screen_analysis in result['screen_analyses']:
                    if screen_analysis['analysis']['success']:
                        successful_analyses += 1
                        result_data = screen_analysis['analysis']['result']
                        total_digits += len(result_data.get('digits', []))
                        total_numbers += len(result_data.get('numbers', []))
                        total_medical_values += len(result_data.get('medical_values', []))
            
            f.write("統計資訊:\n")
            f.write(f"  檢測到的螢幕總數: {total_screens}\n")
            f.write(f"  成功分析的螢幕: {successful_analyses}\n")
            f.write(f"  總數字字符數: {total_digits}\n")
            f.write(f"  總完整數字數: {total_numbers}\n")
            f.write(f"  總醫療數值數: {total_medical_values}\n\n")
            
            # 詳細結果
            f.write("詳細結果:\n")
            f.write("-" * 30 + "\n")
            
            for result in analysis_result['results']:
                minutes = result['time_seconds'] // 60
                seconds = result['time_seconds'] % 60
                f.write(f"\n時間點 {minutes}:{seconds:02d} ({result['time_seconds']}秒):\n")
                f.write(f"  檢測到 {result['screens_detected']} 個螢幕\n")
                
                for screen_analysis in result['screen_analyses']:
                    screen_num = screen_analysis['screen_number']
                    f.write(f"\n  螢幕 {screen_num}:\n")
                    
                    if screen_analysis['analysis']['success']:
                        result_data = screen_analysis['analysis']['result']
                        f.write(f"    數字字符: {result_data.get('digits', [])}\n")
                        f.write(f"    完整數字: {result_data.get('numbers', [])}\n")
                        f.write(f"    醫療數值: {result_data.get('medical_values', [])}\n")
                        if result_data.get('description'):
                            f.write(f"    描述: {result_data['description']}\n")
                    else:
                        f.write(f"    錯誤: {screen_analysis['analysis']['error']}\n")
        
        print(f"報告已保存到: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="影片螢幕數字抓取程式")
    
    # 模式選擇
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--video_path", "-v", help="影片檔案路徑")
    mode_group.add_argument("--camera", "-cam", action="store_true", help="使用攝影機即時模式")
    
    # 共用參數
    parser.add_argument("--provider", default="openrouter", choices=["openrouter", "local"], help="API 提供者 (openrouter, local)")
    parser.add_argument("--base_url", default=None, help="自訂 API Base URL (用於地端模型)")
    parser.add_argument("--api_key", "-k", default=None, help="OpenRouter API 金鑰（openrouter 必填）")
    parser.add_argument("--config_file", "-c",
                       default="groundingdino_source/config/GroundingDINO_SwinT_OGC.py",
                       help="GroundingDINO 配置檔案路徑")
    parser.add_argument("--checkpoint_path", "-p",
                       default="groundingdino_swint_ogc.pth",
                       help="GroundingDINO 模型權重路徑")
    parser.add_argument("--model", "-m", required=True, help="模型名稱（OpenRouter 或地端模型 ID）")
    parser.add_argument("--cpu-only", action="store_true", help="只使用 CPU")
    parser.add_argument("--target_data", "-t", default="all",
                       help="要抓取的目標資料類型 (all, digits, numbers, medical_values, 或自定義關鍵詞)")
    
    # 影片模式參數
    parser.add_argument("--output_dir", "-o", default="video_screen_analysis", help="輸出目錄 (僅影片模式)")
    
    # 攝影機模式參數
    parser.add_argument("--camera_index", default=0, type=int, help="攝影機索引 (預設 0)")
    parser.add_argument("--rtsp_url", default=None, type=str, help="RTSP 流 URL (例如: rtsp://localhost:8554/stream1)，優先於 camera_index")
    parser.add_argument("--max_duration", default=None, type=int, help="最大執行時間（秒）")
    parser.add_argument("--no_mongodb", action="store_true", help="不保存到 MongoDB")
    parser.add_argument("--session_id", default=None, type=str, help="自定義會話 ID")
    parser.add_argument("--camera_name", default=None, type=str, help="IP 攝影機的自訂名稱")
    parser.add_argument("--interval", "-i", default=60, type=int, help="處理間隔秒數 (預設 60 秒)")
    
    args = parser.parse_args()
    
    # 檢查檔案
    if args.video_path and not os.path.exists(args.video_path):
        print(f"錯誤: 影片檔案不存在: {args.video_path}")
        return
    
    if not os.path.exists(args.config_file):
        print(f"錯誤: 配置檔案不存在: {args.config_file}")
        return
    
    if not os.path.exists(args.checkpoint_path):
        print(f"錯誤: 模型權重檔案不存在: {args.checkpoint_path}")
        print("請下載 GroundingDINO 模型權重或更新路徑")
        return

    api_key = args.api_key
    base_url = args.base_url
    if args.provider == "local":
        api_key = api_key or "lm-studio"
        base_url = base_url or "http://192.168.1.237:1234/v1"
    elif not api_key or not str(api_key).strip():
        print("錯誤: openrouter 須使用 --api_key 提供 API 金鑰")
        return
    else:
        api_key = str(api_key).strip()
    
    try:
        # 建立抓取器
        extractor = VideoScreenDigitExtractor(
            api_key=api_key,
            config_file=args.config_file,
            checkpoint_path=args.checkpoint_path,
            provider=args.provider,
            base_url=base_url,
            model=args.model,
            cpu_only=args.cpu_only,
            target_data=args.target_data,
            rtsp_url=args.rtsp_url,
            camera_name=args.camera_name
        )
        
        if args.camera:
            # 攝影機模式或 RTSP 流模式 (由 Web UI 呼叫，直接啟動)
            extractor.start_rtsp_capture(
                session_id=args.session_id if args.session_id else str(uuid.uuid4()), # 命令行允許自訂 session_id
                rtsp_url=args.rtsp_url,
                camera_name=args.camera_name if args.camera_name else f"Camera_{args.session_id[:8] if args.session_id else str(uuid.uuid4())[:8]}", # 命令行模式下自動生成或使用提供的值
                capture_interval_seconds=args.interval,
                save_to_mongodb=not args.no_mongodb
            )
            # 因為是在背景線程運行，這裡直接返回，不會有結果返回
            print("RTSP 串流模式已在背景啟動。")
            while extractor._is_running and not extractor._stop_event.is_set():
                time.sleep(1)
            print("RTSP 串流模式結束運行。")
        else:
            # 影片模式
            result = extractor.process_video(args.video_path, args.output_dir)
            print(f"\n✅ 影片處理完成！")
            print(f"📁 結果保存在: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ 程式執行錯誤: {str(e)}")


if __name__ == "__main__":
    main()
