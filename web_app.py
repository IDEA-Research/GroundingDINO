#!/usr/bin/env python3
"""
影片螢幕數字抓取程式 - Web 介面 (含 MongoDB 自動啟動)
使用 Flask 提供 Web UI 來上傳影片並即時顯示處理狀態
"""

import os
import sys
import ctypes
import glob

# 在導入任何其他模組之前預載入 PyTorch 庫
conda_env_path = os.path.dirname(os.path.dirname(sys.executable))
torch_lib_path = os.path.join(conda_env_path, 'lib', 'python3.10', 'site-packages', 'torch', 'lib')
conda_lib_path = os.path.join(conda_env_path, 'lib')

# 預載入必要的共享庫
try:
    # 載入 libc10.so
    libc10_path = os.path.join(torch_lib_path, 'libc10.so')
    if os.path.exists(libc10_path):
        ctypes.CDLL(libc10_path, mode=ctypes.RTLD_GLOBAL)
        print(f"✅ 預載入 libc10.so")
    
    # 載入 libtorch_cpu.so
    libtorch_cpu_path = os.path.join(torch_lib_path, 'libtorch_cpu.so')
    if os.path.exists(libtorch_cpu_path):
        ctypes.CDLL(libtorch_cpu_path, mode=ctypes.RTLD_GLOBAL)
        print(f"✅ 預載入 libtorch_cpu.so")
    
    # 僅載入 libtorch_cuda.so (避免載入不相容 CUDA 更新版 libtorch_cuda_linalg.so 引發符號未定義錯誤)
    libtorch_cuda_path = os.path.join(torch_lib_path, 'libtorch_cuda.so')
    if os.path.exists(libtorch_cuda_path):
        ctypes.CDLL(libtorch_cuda_path, mode=ctypes.RTLD_GLOBAL)
        print(f"✅ 預載入 libtorch_cuda.so")
        
except Exception as e:
    print(f"⚠️  預載入庫時發生錯誤: {e}")

# 設置 LD_LIBRARY_PATH (對子進程有效)
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
new_ld_path = f"{torch_lib_path}:{conda_lib_path}:{current_ld_path}" if current_ld_path else f"{torch_lib_path}:{conda_lib_path}"
os.environ['LD_LIBRARY_PATH'] = new_ld_path

import json
import time
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, Response, send_from_directory
from werkzeug.utils import secure_filename
import uuid
import cv2
from fuzzywuzzy import process
from dotenv import load_dotenv

# 載入 .env 檔案
load_dotenv()

# 導入 MongoDB 管理器
from mongo_manager import start_local_mongodb, get_mongo_manager

# 在應用啟動前啟動 MongoDB
print("🔧 正在初始化 MongoDB...")
if start_local_mongodb():
    print("✅ MongoDB 初始化成功")
    mongo_manager = get_mongo_manager()
else:
    print("❌ MongoDB 初始化失敗，將使用檔案系統儲存")
    mongo_manager = None

# 設定 GroundingDINO 環境
import torch
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
if torch_lib_path not in current_ld_path:
    if current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{current_ld_path}"
    else:
        os.environ['LD_LIBRARY_PATH'] = torch_lib_path
    print(f"已設定 LD_LIBRARY_PATH 包含 PyTorch 庫: {torch_lib_path}")

# 測試 C++ 擴展載入
try:
    from groundingdino import _C
    print("✅ GroundingDINO C++ 擴展載入成功，將使用 GPU 加速")
except ImportError as e:
    print(f"⚠️  GroundingDINO C++ 擴展載入失敗，將使用 CPU 模式: {e}")

# 導入我們的影片處理類別
from video_screen_digit_extractor import VideoScreenDigitExtractor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 2000 * 1024 * 1024  # 2GB 最大檔案大小

# 確保上傳目錄存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# 全域變數來追蹤處理狀態
processing_status = {
    'is_processing': False,
    'current_video': None,
    'current_frame': 0,
    'total_frames': 0,
    'current_minute': 0,
    'progress_percentage': 0,
    'results': [],
    'error': None,
    'warning': None,
    'llm_failed_count': 0,
    'llm_success_count': 0,
    'llm_last_error': None,
    'session_id': None
}


def summarize_results_llm_stats(results):
    """從影片處理結果統計 LLM 成功/失敗筆數與最後一則錯誤"""
    failed = 0
    success = 0
    last_error = None
    for frame in results or []:
        for sa in frame.get('screen_analyses', []):
            analysis = sa.get('analysis') or {}
            if analysis.get('success'):
                success += 1
            elif analysis:
                failed += 1
                if analysis.get('error'):
                    last_error = analysis['error']
    return failed, success, last_error

# 儲存處理結果
video_results = {}

# 儲存活躍的串流處理器實例 (session_id -> extractor instance)
active_stream_processors = {}
stream_status_lock = threading.Lock()

LOCAL_LLM_MODELS = frozenset({'qwen3-vl-235b-a22b-instruct-1m_moe', 'glm-4.6v'})
LOCAL_LLM_BASE_URL = "http://192.168.1.237:1234/v1"


def resolve_llm_credentials(provider, model, api_key):
    """
    從前端參數解析 LLM 連線設定。API Key 僅接受前端傳入（local 除外）。
    回傳 (provider, api_key, base_url, error_message)。
    """
    if not provider or not str(provider).strip():
        return None, None, None, '缺少 provider 參數'
    if not model or not str(model).strip():
        return None, None, None, '缺少 model 參數'

    provider = str(provider).strip().lower()
    model = str(model).strip()

    if provider == 'local':
        if model not in LOCAL_LLM_MODELS:
            return None, None, None, f'不支援的地端模型: {model}'
        return provider, 'lm-studio', LOCAL_LLM_BASE_URL, None

    if provider != 'openrouter':
        return None, None, None, f'不支援的 provider: {provider}（僅支援 openrouter、local）'

    key = (api_key or '').strip()
    if not key:
        return None, None, None, '請在前端輸入 OpenRouter API Key'

    return provider, key, None, None


class WebVideoProcessor:
    """Web 介面的影片處理器"""
    
    def __init__(self, api_key, config_file, checkpoint_path, provider="openrouter", base_url=None, model=None, cpu_only=True, frame_interval_seconds=60, rtsp_url=None, camera_name=None):
        self.extractor = VideoScreenDigitExtractor(
            api_key=api_key,
            config_file=config_file,
            checkpoint_path=checkpoint_path,
            provider=provider,
            base_url=base_url,
            model=model,
            cpu_only=cpu_only,
            target_data="medical_values",
            frame_interval_seconds=frame_interval_seconds,
            rtsp_url=rtsp_url, # 新增
            camera_name=camera_name # 新增
        )
        self.session_id = None
        self.frame_interval_seconds = frame_interval_seconds
        
    def process_video_with_updates(self, video_path, session_id):
        """處理影片並更新全域狀態"""
        global processing_status, video_results
        
        self.session_id = session_id
        processing_status['session_id'] = session_id
        processing_status['is_processing'] = True
        processing_status['current_video'] = os.path.basename(video_path)
        processing_status['error'] = None
        processing_status['warning'] = None
        processing_status['llm_failed_count'] = 0
        processing_status['llm_success_count'] = 0
        processing_status['llm_last_error'] = None
        processing_status['results'] = []
        
        try:
            # 載入模型
            self.extractor.load_grounding_model()
            
            # 建立輸出目錄
            video_name = Path(video_path).stem
            analysis_dir = os.path.join("video_screen_analysis", video_name)
            #frames_dir = os.extractorpath.join(analysis_dir, "frames")
            frames_dir = os.path.join(analysis_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            # 開啟影片並取得資訊
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"無法開啟影片: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_seconds = total_frames / fps
            duration_minutes = duration_seconds / 60
            total_intervals = int(duration_seconds / self.frame_interval_seconds) + (1 if duration_seconds % self.frame_interval_seconds > 0 else 0)
            
            processing_status['total_frames'] = total_intervals
            
            results = []
            
            for interval in range(total_intervals):
                if not processing_status['is_processing']:
                    break
                    
                time_seconds = interval * self.frame_interval_seconds
                processing_status['current_minute'] = time_seconds  # 保持舊的欄位名稱以向後相容
                processing_status['progress_percentage'] = int((interval / total_intervals) * 100)
                
                # 計算目標時間（毫秒）
                target_time_ms = time_seconds * 1000
                
                # 使用 seek 直接跳到指定時間
                cap.set(cv2.CAP_PROP_POS_MSEC, target_time_ms)
                
                # 讀取該時間點的畫面
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 保存當前畫面
                filename = f"frame_{time_seconds:04d}s.jpg"
                filepath = os.path.join(frames_dir, filename)
                cv2.imwrite(filepath, frame)

                # 處理畫面
                frame_result = self.extractor.process_single_video_frame(filepath, time_seconds, analysis_dir)
                
                # 添加相對路徑供 Web 顯示
                frame_result['frame_image_url'] = f"./analysis/{video_name}/frames/{filename}"
                
                # 處理螢幕圖片路徑
                for screen_analysis in frame_result['screen_analyses']:
                    if screen_analysis['screen_path']:
                        # 轉換為相對路徑
                        rel_path = os.path.relpath(screen_analysis['screen_path'], "video_screen_analysis")
                        screen_analysis['screen_image_url'] = f"./analysis/{rel_path}"
                
                results.append(frame_result)
                processing_status['results'] = results
                
                # 短暫延遲以避免 API 限制
                time.sleep(0.5)
            
            cap.release()
            
            # 保存最終結果到檔案系統
            self.extractor.save_video_results(video_path, results, analysis_dir)
            
            # 新增：儲存到 MongoDB (標記為手動上傳)
            if mongo_manager:
                try:
                    video_analysis_id = mongo_manager.save_analysis_result(
                        video_path, session_id, results, source="manual",
                        llm_model=self.extractor.model
                    )
                    print(f"📊 結果已同步到 MongoDB: {video_analysis_id}")
                except Exception as e:
                    print(f"⚠️  MongoDB 儲存失敗: {e}")
            
            video_results[session_id] = {
                'video_name': video_name,
                'results': results,
                'analysis_dir': analysis_dir,
                'completed_at': datetime.now().isoformat()
            }
            
            processing_status['progress_percentage'] = 100

            failed, success, last_error = summarize_results_llm_stats(results)
            processing_status['llm_failed_count'] = failed
            processing_status['llm_success_count'] = success
            processing_status['llm_last_error'] = last_error
            if failed > 0 and success == 0:
                processing_status['error'] = last_error or f'LLM 分析全部失敗（共 {failed} 個螢幕）'
            elif failed > 0:
                processing_status['warning'] = (
                    f'部分螢幕 LLM 分析失敗：成功 {success}、失敗 {failed}。'
                    f' {last_error or ""}'
                ).strip()
            
        except Exception as e:
            processing_status['error'] = str(e)
            print(f"處理影片時發生錯誤: {e}")
        finally:
            processing_status['is_processing'] = False

# 全域處理器實例
processor = None

def allowed_file(filename):
    """檢查檔案類型是否允許"""
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """主頁面 - 影片上傳"""
    return render_template('upload.html')

@app.route('/database')
def database():
    """資料庫檢視頁面"""
    return render_template('database.html')

@app.route('/templates_editor')
def templates_editor():
    """設備模板編輯頁面"""
    return render_template('templates.html')

@app.route('/details')
def details():
    """醫療數值詳情頁面"""
    return render_template('details.html')

@app.route('/stream_monitor')
def stream_monitor():
    """即時串流監測頁面"""
    return render_template('stream_monitor.html')

@app.route('/upload/', methods=['GET','POST'])
def upload_video():
    """處理影片上傳"""
    global processor
    
    if 'video' not in request.files:
        return jsonify({'error': '沒有選擇檔案'}), 400
    
    file = request.files['video']
    
    provider = request.form.get('provider')
    model = request.form.get('model')
    user_api_key = request.form.get('api_key')

    provider, api_key, base_url, cred_error = resolve_llm_credentials(
        provider, model, user_api_key
    )
    if cred_error:
        return jsonify({'error': cred_error}), 400
    
    if file.filename == '':
        return jsonify({'error': '沒有選擇檔案'}), 400
    
    if file and allowed_file(file.filename):
        # 檢查是否正在處理其他影片
        if processing_status['is_processing']:
            return jsonify({'error': '目前正在處理其他影片，請稍後再試'}), 409
        
        filename = secure_filename(file.filename)
        session_id = str(uuid.uuid4())
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(filepath)
        
        # 初始化處理器（可配置擷取間隔）
        frame_interval = 60
        processor = WebVideoProcessor(
            provider=provider,
            base_url=base_url,
            model=model,
            api_key=api_key,
            config_file="groundingdino_source/config/GroundingDINO_SwinT_OGC.py",
            checkpoint_path="groundingdino_swint_ogc.pth",
            cpu_only=False,
            frame_interval_seconds=frame_interval,
            rtsp_url=None,
            camera_name=None
        )
        
        # 在背景執行緒中開始處理
        thread = threading.Thread(
            target=processor.process_video_with_updates,
            args=(filepath, session_id)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': '影片上傳成功，開始處理...'
        })
    
    return jsonify({'error': '不支援的檔案格式'}), 400

def normalize_device_name(llm_returned_name):
    """使用模糊比對將 LLM 回傳的設備名稱正規化"""
    if not llm_returned_name or llm_returned_name.lower() == 'unknown':
        return 'unknown'
    
    template_dir = 'device_templates'
    try:
        # 獲取所有標準設備名稱（不含副檔名）
        standard_names = [f.replace('.txt', '') for f in os.listdir(template_dir) if f.endswith('.txt')]
        if not standard_names:
            return llm_returned_name
            
        # 使用模糊比對找到最接近的標準名稱
        best_match, score = process.extractOne(llm_returned_name, standard_names)
        # 如果分數大於 70，則認為是該設備，否則保留原名或標記為 unknown
        return best_match if score > 70 else llm_returned_name
    except Exception as e:
        print(f"⚠️ 正規化設備名稱時發生錯誤: {e}")
        return llm_returned_name

@app.route('/api/stream/start', methods=['POST'])
def start_stream_monitor():
    global active_stream_processors
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用，無法啟動串流監測'}), 503

    data = request.get_json()
    rtsp_url = data.get('rtsp_url')
    camera_name = data.get('camera_name')
    api_key = data.get('api_key')
    provider = data.get('provider')
    model = data.get('model')
    task_name = data.get('task_name')

    # 讀取擷取間隔，預設為 60 秒
    capture_interval = data.get('interval') or data.get('capture_interval_seconds') or 60
    try:
        capture_interval = int(capture_interval)
    except (ValueError, TypeError):
        capture_interval = 60

    provider, api_key, base_url, cred_error = resolve_llm_credentials(
        provider, model, api_key
    )
    if cred_error:
        return jsonify({'error': cred_error}), 400

    if not rtsp_url or not camera_name:
        return jsonify({'error': '缺少 RTSP URL 或攝影機名稱'}), 400

    # 驗證 RTSP URL 格式
    if not (rtsp_url.startswith('rtsp://') or rtsp_url.startswith('rtsps://')):
        return jsonify({'error': '無效的 RTSP URL 格式，必須以 rtsp:// 或 rtsps:// 開頭'}), 400

    # 如果前端有傳入現有的 session_id，則重複使用它；否則生成一個新的
    session_id = data.get('session_id') or str(uuid.uuid4())

    with stream_status_lock:
        if any(p.rtsp_url == rtsp_url for p in active_stream_processors.values()):
            return jsonify({'error': '該 RTSP URL 已在監測中'}), 409

        try:
            extractor = VideoScreenDigitExtractor(
                api_key=api_key,
                config_file="groundingdino_source/config/GroundingDINO_SwinT_OGC.py",
                checkpoint_path="groundingdino_swint_ogc.pth",
                provider=provider,
                base_url=base_url,
                model=model,
                cpu_only=False, # 串流監測通常需要 GPU 加速
                target_data="medical_values",
                rtsp_url=rtsp_url, # 傳遞給 extractor 實例
                camera_name=camera_name # 傳遞攝影機名稱
            )
            
            # 儲存串流會話到 MongoDB
            mongo_manager.save_stream_session(session_id, camera_name, rtsp_url, llm_model=model, task_name=task_name)
            mongo_manager.update_stream_session_status(session_id, "active")

            # 啟動背景串流捕獲線程
            extractor.start_rtsp_capture(
                session_id=session_id,
                rtsp_url=rtsp_url,
                camera_name=camera_name,
                capture_interval_seconds=capture_interval,
                save_to_mongodb=True
            )
            active_stream_processors[session_id] = extractor

            return jsonify({'success': True, 'session_id': session_id, 'message': '串流監測已啟動'}), 200
        except Exception as e:
            print(f"啟動串流監測失敗: {e}")
            return jsonify({'error': str(e)}), 500

@app.route('/api/stream/stop/<session_id>', methods=['POST'])
def stop_stream_monitor(session_id):
    global active_stream_processors
    with stream_status_lock:
        if session_id not in active_stream_processors:
            return jsonify({'error': '找不到該串流會話'}), 404

        extractor = active_stream_processors[session_id]
        if extractor.stop_capture():
            del active_stream_processors[session_id]
            if mongo_manager:
                mongo_manager.update_stream_session_status(session_id, "inactive")
            return jsonify({'success': True, 'message': '串流監測已停止'}), 200
        else:
            return jsonify({'error': '停止串流監測失敗或未運行'}), 500

@app.route('/api/stream/delete/<session_id>', methods=['POST'])
def delete_stream_monitor(session_id):
    global active_stream_processors
    with stream_status_lock:
        # 如果該串流還在運行中，不允許直接刪除，必須先停止
        if session_id in active_stream_processors:
            return jsonify({'error': '該串流正在運行中，請先停止監測再刪除卡片'}), 400

        try:
            if mongo_manager:
                if mongo_manager.delete_stream_session(session_id):
                    return jsonify({'success': True, 'message': '串流會話已成功刪除'}), 200
                else:
                    return jsonify({'error': '刪除串流會話失敗'}), 500
            else:
                return jsonify({'error': 'MongoDB 未啟用'}), 503
        except Exception as e:
            print(f"刪除串流監測失敗: {e}")
            return jsonify({'error': str(e)}), 500

@app.route('/api/stream/list')
def list_stream_monitors():
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503
    try:
        sessions = mongo_manager.get_stream_sessions()
        # 補充電腦中的即時狀態
        with stream_status_lock:
            for session in sessions:
                session_id = session['session_id']
                if session_id in active_stream_processors:
                    session['current_status'] = active_stream_processors[session_id]._is_running
                    # 如果 MongoDB 狀態是 inactive 但記憶體中是 running，則修正狀態
                    if session['status'] == "inactive" and session['current_status']:
                        session['status'] = "active"
                else:
                    session['current_status'] = False
                    # 如果 MongoDB 狀態是 active 但記憶體中已停止，則修正狀態
                    if session['status'] == "active" and not session['current_status']:
                        session['status'] = "inactive"

        # 回傳所有會話（包含已停止的），讓前端可以保留卡片並提供重啟選項
        return jsonify({'success': True, 'streams': sessions}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream/status/<session_id>')
def get_stream_monitor_status(session_id):
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503
    try:
        # 從 MongoDB 獲取最新的數據
        medical_values = mongo_manager.get_medical_values_by_session(session_id)
        
        # 找到最新的醫療數值和對應的圖片 URL (支援多個螢幕)
        latest_analyses = []
        if medical_values:
            # 取得最新的一筆時間戳記
            latest_timestamp = medical_values[0].get('analyzed_at')
            
            # 找出所有具有相同最新時間戳記的記錄 (代表同一個畫面中的多個螢幕)
            for item in medical_values:
                if item.get('analyzed_at') == latest_timestamp:
                    # 轉換圖片路徑為 Web URL
                    if item.get('screen_image_path'):
                        path = item['screen_image_path']
                        if path.startswith('video_screen_analysis/'):
                            path = path.replace('video_screen_analysis/', '', 1)
                        item['screen_image_url'] = f"./analysis/{path}"
                    
                    # 轉換 analyzed_at 時間格式
                    if 'analyzed_at' in item and item['analyzed_at']:
                        if isinstance(item['analyzed_at'], datetime):
                            item['analyzed_at'] = item['analyzed_at'].isoformat()
                    
                    # 處理 ObjectId 序列化問題
                    if '_id' in item:
                        item['_id'] = str(item['_id'])
                    if 'frame_result_id' in item:
                        item['frame_result_id'] = str(item['frame_result_id'])
                    
                    latest_analyses.append(item)
                else:
                    # 因為已經排序過，遇到不同時間戳記就可以停止了
                    break
            
        status = {
            'is_running_in_memory': session_id in active_stream_processors, # 檢查是否在記憶體中運行
            'latest_analysis': latest_analyses[0] if latest_analyses else None, # 保持向後相容
            'latest_analyses': latest_analyses, # 新增：回傳所有最新的分析
            'total_analysis_count': len(medical_values),
            'last_error': active_stream_processors[session_id].last_error if session_id in active_stream_processors else None
        }

        # 補充電腦中的即時狀態
        with stream_status_lock:
            if session_id in active_stream_processors:
                extractor = active_stream_processors[session_id]
                status['extractor_is_running'] = extractor._is_running
                status['extractor_stop_event_is_set'] = extractor._stop_event.is_set()
                # 可以根據需要添加更多來自 extractor 的狀態

        return jsonify({'success': True, 'status': status}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 相機監測範本 API 端點
@app.route('/api/stream/templates', methods=['GET'])
def list_stream_templates():
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503
    try:
        templates = mongo_manager.get_camera_templates()
        return jsonify({'success': True, 'templates': templates}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream/templates', methods=['POST'])
def save_stream_template():
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503
    try:
        data = request.get_json()
        template_id = data.get('template_id')
        camera_name = data.get('camera_name')
        rtsp_url = data.get('rtsp_url')
        provider = data.get('provider')
        model = data.get('model')
        api_key = data.get('api_key')

        if not template_id or not camera_name or not rtsp_url:
            return jsonify({'error': '缺少必要欄位 (範本名稱、攝影機名稱或 RTSP URL)'}), 400

        success = mongo_manager.save_camera_template(
            template_id=template_id,
            camera_name=camera_name,
            rtsp_url=rtsp_url,
            provider=provider,
            model=model,
            api_key=api_key
        )
        if success:
            return jsonify({'success': True, 'message': '範本儲存成功'}), 200
        else:
            return jsonify({'error': '儲存範本失敗'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream/templates/<template_id>', methods=['DELETE'])
def delete_stream_template(template_id):
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503
    try:
        success = mongo_manager.delete_camera_template(template_id)
        if success:
            return jsonify({'success': True, 'message': '範本刪除成功'}), 200
        else:
            return jsonify({'error': '刪除範本失敗，可能範本不存在'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status/')
def get_status():
    """取得處理狀態"""
    return jsonify(processing_status)

@app.route('/stop/')
def stop_processing():
    """停止處理"""
    global processing_status
    processing_status['is_processing'] = False
    return jsonify({'success': True, 'message': '已停止處理'})

@app.route('/results/<session_id>')
def get_results(session_id):
    """取得特定 session 的結果"""
    if session_id in video_results:
        return jsonify(video_results[session_id])
    return jsonify({'error': '找不到結果'}), 404

@app.route('/analysis/<path:filename>')
def serve_analysis_file(filename):
    """提供分析結果檔案"""
    return send_from_directory('video_screen_analysis', filename)

# 移除 events 端點，改用 polling

@app.route('/medical_values/<session_id>/')
def get_medical_values(session_id):
    """取得醫療數值列表"""
    if session_id not in video_results:
        return jsonify({'error': '找不到結果'}), 404
    
    results = video_results[session_id]['results']
    medical_values_list = []
    
    for frame_result in results:
        time_seconds = frame_result['time_seconds']
        for screen_analysis in frame_result['screen_analyses']:
            if screen_analysis['analysis']['success']:
                result_data = screen_analysis['analysis']['result']
                medical_values = result_data.get('medical_values', {})
                
                if medical_values:
                    medical_values_list.append({
                        'time_seconds': time_seconds,
                        'screen_number': screen_analysis['screen_number'],
                        'medical_values': medical_values,
                        'frame_image_url': frame_result['frame_image_url'],
                        'screen_image_url': screen_analysis.get('screen_image_url', ''),
                        'model': result_data.get('model', 'unknown')
                    })
    
    return jsonify(medical_values_list)

# MongoDB 查詢 API
@app.route('/api/mongodb/status')
def mongodb_status():
    """檢查 MongoDB 連接狀態"""
    if mongo_manager and mongo_manager.is_mongodb_running():
        return jsonify({
            'mongodb_connected': True,
            'database': mongo_manager.database_name,
            'port': mongo_manager.port
        })
    else:
        return jsonify({'mongodb_connected': False})

@app.route('/api/mongodb/medical_values/<session_id>')
def get_mongodb_medical_values(session_id):
    """從 MongoDB 取得醫療數值 (含圖片路徑)"""
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503
    
    try:
        model = request.args.get('model')
        medical_values = mongo_manager.get_medical_values_by_session(session_id, model)
        
        # 轉換相對路徑為 Web 可存取的 URL，並處理 ObjectId
        for item in medical_values:
            # 轉換 ObjectId 為字串
            if '_id' in item:
                item['_id'] = str(item['_id'])
            
            # 如果聚合查詢中沒有包含 id，從 _id 轉換 (雖然聚合查詢 project 已經做了)
            if 'id' not in item and '_id' in item:
                item['id'] = item['_id']
            
            if item.get('original_image_path'):
                # 移除 video_screen_analysis/ 前綴並使用 /analysis/ 路由
                path = item['original_image_path']
                if path.startswith('video_screen_analysis/'):
                    path = path.replace('video_screen_analysis/', '', 1)
                item['original_image_url'] = f"./analysis/{path}"
            
            if item.get('screen_image_path'):
                # 移除 video_screen_analysis/ 前綴並使用 /analysis/ 路由
                path = item['screen_image_path']
                if path.startswith('video_screen_analysis/'):
                    path = path.replace('video_screen_analysis/', '', 1)
                item['screen_image_url'] = f"./analysis/{path}"
            
            # 轉換 analyzed_at 時間格式
            if 'analyzed_at' in item and item['analyzed_at']:
                item['analyzed_at'] = item['analyzed_at'].isoformat()
        
        failure_summary = None
        camera_name = None
        task_name = None
        if mongo_manager:
            failure_summary = mongo_manager.get_analysis_failure_summary(session_id, model)
            if mongo_manager.db is not None:
                try:
                    video_analysis = mongo_manager.db.video_analysis.find_one({"session_id": session_id})
                    if video_analysis:
                        camera_name = video_analysis.get("camera_name")
                        task_name = video_analysis.get("task_name")
                except Exception as e:
                    print(f"⚠️ 獲取 video_analysis 資訊失敗: {e}")

        return jsonify({
            'success': True,
            'data': medical_values,
            'count': len(medical_values),
            'failure_summary': failure_summary,
            'camera_name': camera_name,
            'task_name': task_name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mongodb/session_models/<session_id>')
def get_session_models(session_id):
    """取得特定影片使用過的所有模型"""
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503
    
    try:
        models = mongo_manager.get_session_models(session_id)
        return jsonify({
            'success': True,
            'models': models
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mongodb/query', methods=['POST'])
def query_mongodb():
    """MongoDB 進階查詢 (檔案路徑版本)"""
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503
    
    try:
        data = request.get_json()
        query_filters = {}
        
        # 設備類型過濾
        if data.get('device_model'):
            query_filters['detected_model'] = data['device_model']
        
        # 數值範圍過濾
        if data.get('heart_rate_range'):
            min_hr, max_hr = data['heart_rate_range']
            query_filters['medical_values.heart_rate'] = {'$gte': min_hr, '$lte': max_hr}
        
        # 時間範圍過濾
        if data.get('date_range'):
            start_date, end_date = data['date_range']
            query_filters['analyzed_at'] = {
                '$gte': datetime.fromisoformat(start_date),
                '$lte': datetime.fromisoformat(end_date)
            }
        
        # 只查詢成功的結果
        query_filters['success'] = True
        
        results = mongo_manager.query_screen_analysis(query_filters)
        
        # 轉換 ObjectId 為字串並處理圖片路徑
        for result in results:
            result['_id'] = str(result['_id'])
            result['frame_result_id'] = str(result['frame_result_id'])
            
            # 轉換圖片路徑為 Web URL
            if result.get('screen_image_path'):
                rel_path = os.path.relpath(result['screen_image_path'], '.')
                result['screen_image_url'] = f"./{rel_path}"
        
        return jsonify({
            'success': True,
            'data': results,
            'count': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mongodb/stats')
def get_mongodb_stats():
    """取得設備統計資訊"""
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503
    
    try:
        session_id = request.args.get('session_id')
        model = request.args.get('model')  # 新增：取得模型參數
        if not session_id:
            return jsonify({'error': '缺少 session_id'}), 400
            
        # 1. 取得該 session 的所有醫療數值 (支援模型過濾)
        medical_values = mongo_manager.get_medical_values_by_session(session_id, model)
        
        # 2. 統計各設備的錯誤率 (TP/FP 邏輯：以 AI 辨識結果為基準)
        device_stats = {}
        
        for item in medical_values:
            # AI 原始辨識的設備 (這是我們的 "Prediction")
            raw_device_name = item.get('model', 'unknown')
            # 使用者訂正後的設備 (這是我們的 "Ground Truth")
            corrected_device_name = item.get('corrected_model')
            
            # 以 AI 辨識結果作為統計分組 (TP+FP 的基礎)
            device_name = normalize_device_name(raw_device_name)
            
            if device_name not in device_stats:
                device_stats[device_name] = {
                    'total_count': 0,      # TP + FP (AI 辨識為此設備的總次數)
                    'corrected_count': 0,  # FP (AI 辨識為此設備但被訂正為其他的次數)
                    'accuracy': 0
                }
            
            device_stats[device_name]['total_count'] += 1
            
            # 如果有訂正，且訂正後的名稱與原始辨識不同，則視為 FP (False Positive)
            # 注意：如果 corrected_device_name 為空，代表 AI 辨識正確 (TP)
            is_fp = corrected_device_name is not None and corrected_device_name != "" and corrected_device_name != raw_device_name
            if is_fp:
                device_stats[device_name]['corrected_count'] += 1
        
        # 3. 計算正確率 (Precision = TP / (TP + FP))
        for device_name in device_stats:
            stats = device_stats[device_name]
            if stats['total_count'] > 0:
                # 正確率 = (總辨識數 - 錯誤辨識數) / 總辨識數 = TP / (TP + FP)
                stats['accuracy'] = round(((stats['total_count'] - stats['corrected_count']) / stats['total_count']) * 100, 1)
        
        return jsonify({
            'success': True,
            'device_stats': device_stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mongodb/videos')
def get_all_videos():
    """取得所有影片分析記錄 (支援過濾 source 和搜尋)"""
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503
    
    try:
        source_type = request.args.get('type', 'manual') # manual or stream
        search_query = request.args.get('search') # 新增：取得搜尋關鍵字
        
        # 將搜尋關鍵字傳遞給 mongo_manager
        videos = mongo_manager.get_all_videos(source_filter=source_type, search_query=search_query)
        
        # 轉換 ObjectId 為字串並格式化時間
        for video in videos:
            video['_id'] = str(video['_id'])
            video['timestamp'] = video['timestamp'].isoformat() if video.get('timestamp') else None
        
        return jsonify({
            'success': True,
            'data': videos,
            'count': len(videos)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mongodb/videos/<session_id>/rename', methods=['POST'])
def rename_video(session_id):
    """重新命名資料"""
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503
    
    try:
        data = request.get_json()
        new_name = data.get('new_name')
        
        if not new_name:
            return jsonify({'error': '缺少新名稱'}), 400
            
        success = mongo_manager.update_video_name(session_id, new_name)
        
        if success:
            return jsonify({'success': True, 'message': '名稱更新成功'})
        else:
            return jsonify({'error': '找不到該資料或名稱未變更'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mongodb/videos/<session_id>', methods=['DELETE'])
def delete_video(session_id):
    """刪除資料"""
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503
    
    try:
        success = mongo_manager.delete_video_record(session_id)
        
        if success:
            return jsonify({'success': True, 'message': '資料刪除成功'})
        else:
            return jsonify({'error': '找不到該資料或刪除失敗'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mongodb/correct_value/', methods=['POST'])
def correct_value():
    """處理數據訂正請求"""
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503
    
    try:
        data = request.get_json()
        analysis_id = data.get('analysis_id')
        key = data.get('key')
        value = data.get('value')
        
        if not analysis_id or not key:
            return jsonify({'error': '缺少必要參數'}), 400
            
        success = mongo_manager.update_corrected_value(analysis_id, key, value)
        
        if success:
            return jsonify({'success': True, 'message': '數據訂正成功'})
        else:
            return jsonify({'error': '更新資料庫失敗'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mongodb/correct_model/', methods=['POST'])
def correct_model():
    """處理設備型號訂正請求"""
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503
    
    try:
        data = request.get_json()
        analysis_id = data.get('analysis_id')
        model = data.get('model')
        
        if not analysis_id or not model:
            return jsonify({'error': '缺少必要參數'}), 400
            
        success = mongo_manager.update_corrected_model(analysis_id, model)
        
        if success:
            return jsonify({'success': True, 'message': '型號訂正成功'})
        else:
            return jsonify({'error': '更新資料庫失敗'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mongodb/mark_quality/', methods=['POST'])
def mark_quality():
    """更新反光/遮擋人工標註"""
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503

    try:
        data = request.get_json()
        analysis_id = data.get('analysis_id')

        if not analysis_id:
            return jsonify({'error': '缺少必要參數 analysis_id'}), 400

        has_glare = data.get('has_glare')
        has_occlusion = data.get('has_occlusion')

        if has_glare is None and has_occlusion is None:
            return jsonify({'error': '至少要提供 has_glare 或 has_occlusion'}), 400

        success = mongo_manager.update_quality_flags(
            analysis_id=analysis_id,
            has_glare=has_glare,
            has_occlusion=has_occlusion
        )

        if success:
            return jsonify({'success': True, 'message': '畫面品質標註已更新'})
        return jsonify({'error': '更新資料庫失敗'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mongodb/undo_value_correction/', methods=['POST'])
def undo_value_correction():
    """撤銷數值訂正"""
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503
    
    try:
        data = request.get_json()
        analysis_id = data.get('analysis_id')
        key = data.get('key')
        
        if not analysis_id or not key:
            return jsonify({'error': '缺少必要參數'}), 400
            
        # 使用 $unset 移除特定欄位的訂正值
        from bson.objectid import ObjectId
        result = mongo_manager.db.screen_analysis.update_one(
            {"_id": ObjectId(analysis_id)},
            {"$unset": {f"corrected_medical_values.{key}": ""}}
        )
        
        # 如果該字典空了，可以考慮把 is_corrected 也設為 false
        doc = mongo_manager.db.screen_analysis.find_one({"_id": ObjectId(analysis_id)})
        if doc and (not doc.get('corrected_medical_values') or len(doc.get('corrected_medical_values')) == 0):
            mongo_manager.db.screen_analysis.update_one(
                {"_id": ObjectId(analysis_id)},
                {"$set": {"is_corrected": False}}
            )
            
        return jsonify({'success': True, 'message': '已撤銷數值訂正'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mongodb/undo_model_correction/', methods=['POST'])
def undo_model_correction():
    """撤銷型號訂正"""
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503
    
    try:
        data = request.get_json()
        analysis_id = data.get('analysis_id')
        
        if not analysis_id:
            return jsonify({'error': '缺少必要參數'}), 400
            
        from bson.objectid import ObjectId
        result = mongo_manager.db.screen_analysis.update_one(
            {"_id": ObjectId(analysis_id)},
            {"$unset": {"corrected_model": "", "model_corrected_at": ""},
             "$set": {"is_model_corrected": False}}
        )
        
        return jsonify({'success': True, 'message': '已撤銷型號訂正'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reanalyze_image', methods=['POST'])
def reanalyze_image():
    """重新分析單張圖片並存入資料庫"""
    try:
        data = request.get_json()
        if not data or 'image_path' not in data:
            return jsonify({'error': '缺少圖片路徑參數 (image_path)'}), 400
        
        image_path = data['image_path']
        model = data.get('model')
        provider = data.get('provider')
        api_key_input = data.get('api_key')
        frame_result_id = data.get('frame_result_id')
        original_screen_number = data.get('original_screen_number')
        screen_number = data.get('screen_number')
        save_to_db = data.get('save_to_db', False)

        provider, api_key, base_url, cred_error = resolve_llm_credentials(
            provider, model, api_key_input
        )
        if cred_error:
            return jsonify({'error': cred_error}), 400
        
        # 路徑處理：將前端 URL 路徑 (./analysis/...) 轉換為後端真實檔案路徑
        if image_path.startswith('./analysis/'):
            image_path = image_path.replace('./analysis/', 'video_screen_analysis/', 1)
        elif image_path.startswith('/analysis/'):
            image_path = image_path.replace('/analysis/', 'video_screen_analysis/', 1)
        elif image_path.startswith('./'):
             image_path = image_path.replace('./', '', 1)

        # 確保檔案存在
        if not os.path.exists(image_path):
            return jsonify({'error': f'找不到檔案: {image_path}'}), 404
        
        # 初始化 Extractor
        extractor = VideoScreenDigitExtractor(
            api_key=api_key,
            config_file="groundingdino_source/config/GroundingDINO_SwinT_OGC.py",
            checkpoint_path="groundingdino_swint_ogc.pth",
            model=model,
            provider=provider,
            base_url=base_url,
            cpu_only=True  # 這裡只用 API，不需要 GPU
        )
        
        print(f"🔄 正在重新分析圖片: {image_path}, 使用模型: {model}")
        
        # 直接呼叫識別函數
        result = extractor.extract_digits_from_screen(image_path)
        
        # 儲存到 MongoDB
        if save_to_db and mongo_manager and frame_result_id and original_screen_number is not None:
            save_success = mongo_manager.save_single_screen_analysis(
                frame_result_id=frame_result_id,
                original_screen_number=original_screen_number,
                screen_number=screen_number or original_screen_number,
                screen_image_path=image_path,
                analysis_result=result,
                llm_model=model
            )
            if not save_success:
                print("⚠️ 儲存重新分析結果到 MongoDB 失敗")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ 重新分析失敗: {e}")
        return jsonify({'error': str(e)}), 500

# 設備模板管理 API
@app.route('/api/templates')
def list_templates():
    """列出所有設備模板"""
    template_dir = 'device_templates'
    try:
        files = [f for f in os.listdir(template_dir) if f.endswith('.txt')]
        return jsonify({'success': True, 'templates': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/templates/<filename>')
def get_template(filename):
    """取得特定模板內容"""
    template_dir = 'device_templates'
    # 安全檢查：確保檔名是 .txt 且不包含路徑遍歷
    if not filename.endswith('.txt') or '..' in filename or '/' in filename:
        return jsonify({'error': '不合法的檔案名稱'}), 400
    
    filepath = os.path.join(template_dir, filename)
    try:
        if not os.path.exists(filepath):
            return jsonify({'error': '檔案不存在'}), 404
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({'success': True, 'content': content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/templates/create', methods=['POST'])
def create_template():
    """建立新的設備模板"""
    template_dir = 'device_templates'
    data = request.get_json()
    if not data or 'filename' not in data:
        return jsonify({'error': '缺少檔案名稱'}), 400
    
    filename = data['filename'].strip()
    if not filename:
        return jsonify({'error': '檔案名稱不能為空'}), 400
        
    if not filename.endswith('.txt'):
        filename += '.txt'
    
    # 安全檢查
    if '..' in filename or '/' in filename or '\\' in filename:
        return jsonify({'error': '不合法的檔案名稱'}), 400
        
    filepath = os.path.join(template_dir, filename)
    if os.path.exists(filepath):
        return jsonify({'error': '檔案已存在'}), 400
        
    try:
        # 建立空檔案
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('')
        return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/templates/<filename>', methods=['POST'])
def save_template(filename):
    """儲存模板內容"""
    template_dir = 'device_templates'
    # 安全檢查
    if not filename.endswith('.txt') or '..' in filename or '/' in filename:
        return jsonify({'error': '不合法的檔案名稱'}), 400
    
    data = request.get_json()
    if not data or 'content' not in data:
        return jsonify({'error': '缺少內容'}), 400
    
    filepath = os.path.join(template_dir, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(data['content'])
        return jsonify({'success': True, 'message': '模板儲存成功'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/templates/<filename>', methods=['DELETE'])
def delete_template(filename):
    """刪除設備模板"""
    template_dir = 'device_templates'
    # 安全檢查
    if not filename.endswith('.txt') or '..' in filename or '/' in filename:
        return jsonify({'error': '不合法的檔案名稱'}), 400
    
    filepath = os.path.join(template_dir, filename)
    try:
        if not os.path.exists(filepath):
            return jsonify({'error': '檔案不存在'}), 404
        os.remove(filepath)
        return jsonify({'success': True, 'message': '模板刪除成功'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 使用 watchdog reloader 或禁用 reloader 以保留環境變數
    try:
        app.run(host='0.0.0.0', port=3001, debug=True, use_reloader=False)
    finally:
        get_mongo_manager().cleanup()