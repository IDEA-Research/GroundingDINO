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
    
    # 載入 libtorch_cuda.so (如果存在)
    for cuda_lib in glob.glob(os.path.join(torch_lib_path, 'libtorch_cuda*.so')):
        ctypes.CDLL(cuda_lib, mode=ctypes.RTLD_GLOBAL)
        print(f"✅ 預載入 {os.path.basename(cuda_lib)}")
        
except Exception as e:
    print(f"⚠️  預載入庫時發生錯誤: {e}")

# 設置 LD_LIBRARY_PATH (對子進程有效)
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
new_ld_path = f"{torch_lib_path}:{conda_lib_path}:{current_ld_path}" if current_ld_path else f"{torch_lib_path}:{conda_lib_path}"
os.environ['LD_LIBRARY_PATH'] = new_ld_path

import json
import csv
import io
import time
import threading
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
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
app.config['DATASET_FOLDER'] = 'datasets'
app.config['MAX_CONTENT_LENGTH'] = 2000 * 1024 * 1024  # 2GB 最大檔案大小

# 確保上傳目錄存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATASET_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)


@app.errorhandler(413)
def handle_request_entity_too_large(error):
    """讓過大的上傳請求固定回傳 JSON，避免前端收到 HTML 錯誤頁。"""
    return jsonify({
        'error': '單一影片超過 2GB 上傳上限，請壓縮或分割影片後重試'
    }), 413


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

dataset_batch_status = {
    'is_running': False,
    'stop_requested': False,
    'run_id': None,
    'dataset_id': None,
    'dataset_name': None,
    'current_video': None,
    'current_index': 0,
    'total_videos': 0,
    'completed_videos': 0,
    'failed_videos': 0,
    'error': None,
}
dataset_batch_lock = threading.Lock()
dataset_storage_lock = threading.Lock()


def parse_env_bool(name, default=False):
    """解析環境變數布林值。"""
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ('1', 'true', 'yes', 'on')


AUTO_DELETE_UPLOADED_VIDEO = parse_env_bool('AUTO_DELETE_UPLOADED_VIDEO', default=True)


def sanitize_export_filename_component(name, fallback='unknown'):
    """清理可作為檔名片段的字串，避免非法字元造成下載失敗。"""
    text = str(name or '').strip()
    if not text:
        text = fallback
    text = re.sub(r'[\\/:*?"<>|]+', '_', text)
    text = re.sub(r'\s+', ' ', text).strip(' .')
    return text or fallback


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


def safe_delete_uploaded_video(video_path):
    """只允許刪除 uploads 目錄下的原始影片，避免誤刪。"""
    if not video_path:
        return {'deleted': False, 'reason': 'empty_path'}

    uploads_root = Path(app.config['UPLOAD_FOLDER']).resolve()
    target = Path(video_path).resolve()

    if uploads_root not in target.parents:
        return {'deleted': False, 'reason': 'outside_uploads'}

    if not target.exists():
        return {'deleted': True, 'reason': 'already_missing'}

    if not target.is_file():
        return {'deleted': False, 'reason': 'not_a_file'}

    try:
        target.unlink()
        return {'deleted': True, 'reason': 'deleted'}
    except Exception as e:
        return {'deleted': False, 'reason': f'failed: {e}'}

# 儲存處理結果
video_results = {}

# 儲存活躍的串流處理器實例 (session_id -> extractor instance)
active_stream_processors = {}
stream_status_lock = threading.Lock()

LOCAL_LLM_MODELS = frozenset({
    'qwen3-vl-235b-a22b-instruct-1m_moe',
    'glm-4.6v',
    'internvl3-78b',
})
LOCAL_LLM_BASE_URL = "http://192.168.1.223:1234/v1"


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
    
    def __init__(self, api_key, config_file, checkpoint_path, provider="openrouter", base_url=None, model=None, cpu_only=True, frame_interval_seconds=60, rtsp_url=None, camera_name=None, remove_screen_content=False):
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
            camera_name=camera_name, # 新增
            remove_screen_content=remove_screen_content
        )
        self.session_id = None
        self.frame_interval_seconds = frame_interval_seconds
        
    def process_video_with_updates(self, video_path, session_id, display_name=None, dataset_context=None):
        """處理影片並更新全域狀態"""
        global processing_status, video_results
        
        self.session_id = session_id
        processing_status['session_id'] = session_id
        if dataset_context:
            with dataset_batch_lock:
                if dataset_batch_status['stop_requested']:
                    return
                processing_status['is_processing'] = True
        else:
            processing_status['is_processing'] = True
        processing_status['current_video'] = display_name or os.path.basename(video_path)
        processing_status['error'] = None
        processing_status['warning'] = None
        processing_status['llm_failed_count'] = 0
        processing_status['llm_success_count'] = 0
        processing_status['llm_last_error'] = None
        processing_status['current_minute'] = 0
        processing_status['progress_percentage'] = 0
        processing_status['results'] = []
        stopped_by_user = False
        
        try:
            # 載入模型
            self.extractor.load_grounding_model()
            
            # 建立輸出目錄
            video_name = display_name or Path(video_path).stem
            analysis_key = session_id
            analysis_dir = os.path.join("video_screen_analysis", analysis_key)
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
                    stopped_by_user = True
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
                frame_result['frame_image_url'] = f"./analysis/{analysis_key}/frames/{filename}"
                
                # 處理螢幕圖片路徑
                for screen_analysis in frame_result['screen_analyses']:
                    if screen_analysis['screen_path']:
                        # 轉換為相對路徑
                        rel_path = os.path.relpath(screen_analysis['screen_path'], "video_screen_analysis")
                        screen_analysis['screen_image_url'] = f"./analysis/{rel_path}"
                    if screen_analysis.get('screen_path_for_vlm'):
                        rel_path_for_vlm = os.path.relpath(screen_analysis['screen_path_for_vlm'], "video_screen_analysis")
                        screen_analysis['screen_image_url_for_vlm'] = f"./analysis/{rel_path_for_vlm}"
                
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
                    context = dataset_context or {}
                    video_analysis_id = mongo_manager.save_analysis_result(
                        video_path, session_id, results,
                        source="dataset" if dataset_context else "manual",
                        llm_model=self.extractor.model,
                        dataset_id=context.get("dataset_id"),
                        dataset_video_id=context.get("video_id"),
                        dataset_run_id=context.get("run_id"),
                        original_video_name=display_name,
                    )
                    print(f"📊 結果已同步到 MongoDB: {video_analysis_id}")
                except Exception as e:
                    processing_status['error'] = f'MongoDB 儲存分析結果失敗: {e}'
                    print(f"⚠️  MongoDB 儲存失敗: {e}")

            # 分析完成後可自動刪除 uploads 原始影片（僅手動上傳來源）
            if AUTO_DELETE_UPLOADED_VIDEO and not stopped_by_user and not dataset_context:
                cleanup_result = safe_delete_uploaded_video(video_path)
                if cleanup_result['deleted']:
                    print(f"🗑️ 已自動清理上傳原始影片: {video_path} ({cleanup_result['reason']})")
                else:
                    print(f"⚠️ 自動清理上傳原始影片失敗/跳過: {video_path} ({cleanup_result['reason']})")
            
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
@app.route('/details/<session_id>')
def details(session_id=None):
    """醫療數值詳情頁面"""
    return render_template('details.html', initial_session_id=session_id)

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
    remove_screen_content = str(request.form.get('remove_screen_content', '0')).lower() in ('1', 'true', 'yes', 'on')

    provider, api_key, base_url, cred_error = resolve_llm_credentials(
        provider, model, user_api_key
    )
    if cred_error:
        return jsonify({'error': cred_error}), 400
    
    if file.filename == '':
        return jsonify({'error': '沒有選擇檔案'}), 400
    
    if file and allowed_file(file.filename):
        # 檢查是否正在處理其他影片
        with dataset_batch_lock:
            if processing_status['is_processing'] or dataset_batch_status['is_running']:
                return jsonify({'error': '目前正在處理其他影片，請稍後再試'}), 409
            # 在背景執行緒啟動前先保留單工處理權，避免批次請求插入。
            processing_status['is_processing'] = True
        
        filename = secure_filename(file.filename)
        session_id = str(uuid.uuid4())
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        try:
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
                camera_name=None,
                remove_screen_content=remove_screen_content
            )

            # 在背景執行緒中開始處理
            thread = threading.Thread(
                target=processor.process_video_with_updates,
                args=(filepath, session_id)
            )
            thread.daemon = True
            thread.start()
        except Exception as e:
            processing_status['is_processing'] = False
            safe_delete_uploaded_video(filepath)
            return jsonify({'error': f'啟動影片處理失敗: {e}'}), 500
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': '影片上傳成功，開始處理...'
        })
    
    return jsonify({'error': '不支援的檔案格式'}), 400

def serialize_dataset_document(document):
    """將資料集 MongoDB 文件轉成可 JSON 化的資料。"""
    if not document:
        return document
    serialized = dict(document)
    serialized.pop('_id', None)
    for key, value in list(serialized.items()):
        if isinstance(value, datetime):
            serialized[key] = value.isoformat()
        elif isinstance(value, list):
            serialized[key] = [
                serialize_dataset_document(item) if isinstance(item, dict) else item
                for item in value
            ]
    return serialized


def normalize_dataset_video_path(stored_path):
    """將資料集影片路徑限制在 datasets 目錄內。"""
    if not stored_path:
        return None
    datasets_root = Path(app.config['DATASET_FOLDER']).resolve()
    candidate = Path(stored_path)
    if not candidate.is_absolute():
        candidate = (Path('.') / candidate).resolve()
    else:
        candidate = candidate.resolve()
    try:
        candidate.relative_to(datasets_root)
    except ValueError:
        return None
    return candidate


def resolve_dataset_video_path(stored_path):
    """只允許批次工作讀取 datasets 目錄內的既有影片。"""
    candidate = normalize_dataset_video_path(stored_path)
    if candidate is None:
        return None
    return candidate if candidate.is_file() else None


def run_dataset_batch(run_id, dataset, videos, processor_options):
    """在單一背景執行緒中依序分析資料集的所有影片。"""
    global processor, dataset_batch_status
    completed = 0
    failed = 0
    final_status = 'completed'

    try:
        mongo_manager.update_dataset_run(run_id, status='running', started_at=datetime.now())
        for index, video in enumerate(videos, start=1):
            with dataset_batch_lock:
                if dataset_batch_status['stop_requested']:
                    final_status = 'stopped'
                    break
                dataset_batch_status.update({
                    'current_video': video['original_name'],
                    'current_index': index,
                    'completed_videos': completed,
                    'failed_videos': failed,
                })

            video_path = resolve_dataset_video_path(video.get('stored_path'))
            if video_path is None:
                failed += 1
                error = '資料集影片不存在或路徑無效'
                mongo_manager.update_dataset_run_item(
                    run_id, video['video_id'], status='failed', error=error
                )
                mongo_manager.update_dataset_run(
                    run_id, completed_videos=completed, failed_videos=failed
                )
                continue

            session_id = str(uuid.uuid4())
            mongo_manager.update_dataset_run_item(
                run_id, video['video_id'], status='running',
                session_id=session_id, started_at=datetime.now()
            )

            processor = WebVideoProcessor(
                provider=processor_options['provider'],
                base_url=processor_options['base_url'],
                model=processor_options['model'],
                api_key=processor_options['api_key'],
                config_file="groundingdino_source/config/GroundingDINO_SwinT_OGC.py",
                checkpoint_path="groundingdino_swint_ogc.pth",
                cpu_only=False,
                frame_interval_seconds=60,
                rtsp_url=None,
                camera_name=None,
                remove_screen_content=processor_options['remove_screen_content'],
            )
            processor.process_video_with_updates(
                str(video_path),
                session_id,
                display_name=video['original_name'],
                dataset_context={
                    'dataset_id': dataset['dataset_id'],
                    'video_id': video['video_id'],
                    'run_id': run_id,
                },
            )

            with dataset_batch_lock:
                stop_requested = dataset_batch_status['stop_requested']
            if stop_requested:
                mongo_manager.update_dataset_run_item(
                    run_id, video['video_id'], status='stopped',
                    error='使用者停止批次工作', finished_at=datetime.now()
                )
                final_status = 'stopped'
                break

            if processing_status.get('error'):
                failed += 1
                mongo_manager.update_dataset_run_item(
                    run_id, video['video_id'], status='failed',
                    error=processing_status['error'], finished_at=datetime.now()
                )
            else:
                completed += 1
                mongo_manager.update_dataset_run_item(
                    run_id, video['video_id'], status='completed',
                    session_id=session_id, finished_at=datetime.now()
                )
            mongo_manager.update_dataset_run(
                run_id, completed_videos=completed, failed_videos=failed
            )
    except Exception as e:
        final_status = 'failed'
        with dataset_batch_lock:
            dataset_batch_status['error'] = str(e)
        print(f"❌ 資料集批次分析失敗: {e}")
    finally:
        try:
            mongo_manager.update_dataset_run(
                run_id,
                status=final_status,
                completed_videos=completed,
                failed_videos=failed,
                finished_at=datetime.now(),
            )
        finally:
            with dataset_batch_lock:
                dataset_batch_status.update({
                    'is_running': False,
                    'current_video': None,
                    'completed_videos': completed,
                    'failed_videos': failed,
                })


@app.route('/api/datasets', methods=['GET', 'POST'])
def datasets_api():
    """列出或建立影片資料集。"""
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503
    if request.method == 'GET':
        datasets = [
            serialize_dataset_document(item)
            for item in mongo_manager.get_datasets()
        ]
        return jsonify({'success': True, 'data': datasets})

    data = request.get_json(silent=True) or {}
    name = str(data.get('name') or '').strip()
    description = str(data.get('description') or '').strip()
    if not name:
        return jsonify({'error': '資料集名稱不可為空'}), 400
    if len(name) > 100:
        return jsonify({'error': '資料集名稱不可超過 100 個字元'}), 400
    dataset_id = mongo_manager.create_dataset(name, description)
    if not dataset_id:
        return jsonify({'error': '建立資料集失敗'}), 500
    return jsonify({'success': True, 'dataset_id': dataset_id}), 201


@app.route('/api/datasets/<dataset_id>', methods=['PATCH', 'DELETE'])
def dataset_detail_api(dataset_id):
    """重新命名或刪除資料集；刪除時保留歷史分析結果。"""
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503

    with dataset_storage_lock:
        dataset = mongo_manager.get_dataset(dataset_id)
        if not dataset:
            return jsonify({'error': '找不到資料集'}), 404

        if request.method == 'PATCH':
            data = request.get_json(silent=True) or {}
            new_name = str(data.get('name') or '').strip()
            if not new_name:
                return jsonify({'error': '資料集名稱不可為空'}), 400
            if len(new_name) > 100:
                return jsonify({'error': '資料集名稱不可超過 100 個字元'}), 400
            if not mongo_manager.rename_dataset(dataset_id, new_name):
                return jsonify({'error': '重新命名資料集失敗'}), 500
            return jsonify({'success': True, 'name': new_name})

        with dataset_batch_lock:
            if (
                dataset_batch_status['is_running']
                and dataset_batch_status['dataset_id'] == dataset_id
            ):
                return jsonify({'error': '資料集正在分析中，請先停止批次工作'}), 409

        videos = mongo_manager.get_dataset_videos(dataset_id)
        deleted_files = 0
        cleanup_errors = 0
        for video in videos:
            video_path = normalize_dataset_video_path(video.get('stored_path'))
            if video_path is None:
                cleanup_errors += 1
                continue
            if not video_path.exists():
                continue
            if not video_path.is_file():
                cleanup_errors += 1
                continue
            try:
                video_path.unlink()
                deleted_files += 1
            except OSError as e:
                cleanup_errors += 1
                print(f"⚠️ 刪除資料集影片失敗: {video_path} ({e})")

        if cleanup_errors:
            return jsonify({
                'error': f'有 {cleanup_errors} 個來源影片無法清除，資料集尚未刪除',
                'deleted_files': deleted_files,
            }), 500

        if not mongo_manager.delete_dataset(dataset_id):
            return jsonify({'error': '刪除資料集中繼資料失敗，請重試'}), 500

        dataset_dir = (Path(app.config['DATASET_FOLDER']) / dataset_id).resolve()
        datasets_root = Path(app.config['DATASET_FOLDER']).resolve()
        try:
            dataset_dir.relative_to(datasets_root)
            dataset_dir.rmdir()
        except (ValueError, OSError):
            pass

        return jsonify({
            'success': True,
            'deleted_files': deleted_files,
            'message': '資料集已刪除；歷史分析結果已保留',
        })


@app.route('/api/datasets/<dataset_id>/videos', methods=['GET', 'POST'])
def dataset_videos_api(dataset_id):
    """列出影片，或將多部影片永久加入資料集。"""
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503
    dataset = mongo_manager.get_dataset(dataset_id)
    if not dataset:
        return jsonify({'error': '找不到資料集'}), 404

    if request.method == 'GET':
        videos = [
            serialize_dataset_document(item)
            for item in mongo_manager.get_dataset_videos(dataset_id)
        ]
        return jsonify({'success': True, 'data': videos})

    with dataset_storage_lock:
        if not mongo_manager.get_dataset(dataset_id):
            return jsonify({'error': '找不到資料集'}), 404
        files = request.files.getlist('videos')
        if not files:
            return jsonify({'error': '沒有選擇影片'}), 400
        invalid_names = [
            file.filename for file in files
            if not file.filename or not allowed_file(file.filename)
        ]
        if invalid_names:
            return jsonify({
                'error': '包含不支援的影片格式',
                'files': invalid_names,
            }), 400

        dataset_dir = Path(app.config['DATASET_FOLDER']) / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        uploaded = []
        for file in files:
            original_name = file.filename
            safe_name = secure_filename(original_name) or 'video'
            stored_path = dataset_dir / f"{uuid.uuid4()}_{safe_name}"
            file.save(str(stored_path))
            try:
                video_id = mongo_manager.add_dataset_video(
                    dataset_id, original_name, str(stored_path), stored_path.stat().st_size
                )
            except Exception:
                stored_path.unlink(missing_ok=True)
                raise
            uploaded.append({'video_id': video_id, 'original_name': original_name})
        return jsonify({'success': True, 'data': uploaded, 'count': len(uploaded)}), 201


@app.route('/api/datasets/<dataset_id>/run', methods=['POST'])
def start_dataset_run(dataset_id):
    """依序分析資料集內的全部影片。"""
    global dataset_batch_status
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503

    data = request.get_json(silent=True) or {}
    provider = data.get('provider')
    model = data.get('model')
    user_api_key = data.get('api_key')
    provider, api_key, base_url, cred_error = resolve_llm_credentials(
        provider, model, user_api_key
    )
    if cred_error:
        return jsonify({'error': cred_error}), 400

    with dataset_storage_lock:
        dataset = mongo_manager.get_dataset(dataset_id)
        if not dataset:
            return jsonify({'error': '找不到資料集'}), 404
        videos = mongo_manager.get_dataset_videos(dataset_id)
        if not videos:
            return jsonify({'error': '資料集內沒有影片'}), 400

        with dataset_batch_lock:
            if dataset_batch_status['is_running'] or processing_status['is_processing']:
                return jsonify({'error': '目前已有影片或資料集正在處理'}), 409
            run_id = str(uuid.uuid4())
            dataset_batch_status = {
                'is_running': True,
                'stop_requested': False,
                'run_id': run_id,
                'dataset_id': dataset_id,
                'dataset_name': dataset['name'],
                'current_video': None,
                'current_index': 0,
                'total_videos': len(videos),
                'completed_videos': 0,
                'failed_videos': 0,
                'error': None,
            }

    try:
        mongo_manager.create_dataset_run(
            run_id, dataset_id, dataset['name'], model, provider, videos
        )
    except Exception as e:
        with dataset_batch_lock:
            dataset_batch_status['is_running'] = False
        return jsonify({'error': f'建立批次工作失敗: {e}'}), 500
    options = {
        'provider': provider,
        'model': model,
        'api_key': api_key,
        'base_url': base_url,
        'remove_screen_content': bool(data.get('remove_screen_content', False)),
    }
    try:
        thread = threading.Thread(
            target=run_dataset_batch,
            args=(run_id, dataset, videos, options),
            daemon=True,
        )
        thread.start()
    except Exception as e:
        try:
            try:
                mongo_manager.update_dataset_run(
                    run_id,
                    status='failed',
                    error=f'無法啟動背景工作: {e}',
                    finished_at=datetime.now(),
                )
            except Exception as update_error:
                print(f"⚠️ 更新批次啟動失敗狀態時發生錯誤: {update_error}")
        finally:
            with dataset_batch_lock:
                dataset_batch_status['is_running'] = False
        return jsonify({'error': f'無法啟動批次工作: {e}'}), 500
    return jsonify({
        'success': True,
        'run_id': run_id,
        'message': f"已開始分析資料集，共 {len(videos)} 部影片",
    }), 202


@app.route('/api/dataset-runs/<run_id>/status')
def dataset_run_status(run_id):
    """取得資料集批次工作狀態。"""
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503
    run = mongo_manager.get_dataset_run(run_id)
    if not run:
        return jsonify({'error': '找不到批次工作'}), 404
    return jsonify({'success': True, 'data': serialize_dataset_document(run)})


@app.route('/api/dataset-runs/<run_id>/stop', methods=['POST'])
def stop_dataset_run(run_id):
    """要求目前的資料集批次工作停止。"""
    with dataset_batch_lock:
        if not dataset_batch_status['is_running'] or dataset_batch_status['run_id'] != run_id:
            return jsonify({'error': '批次工作目前未執行'}), 409
        dataset_batch_status['stop_requested'] = True
        processing_status['is_processing'] = False
    return jsonify({'success': True, 'message': '正在停止批次工作'})


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


MIN_STREAM_INTERVAL_MINUTES = 1
MAX_STREAM_INTERVAL_MINUTES = 60
DEFAULT_STREAM_INTERVAL_MINUTES = 1


def _validate_interval_minutes(minutes):
    if minutes < MIN_STREAM_INTERVAL_MINUTES or minutes > MAX_STREAM_INTERVAL_MINUTES:
        raise ValueError('分析間隔必須介於 1 到 60 分鐘')
    return minutes


def _resolve_global_interval_minutes():
    if not mongo_manager:
        return DEFAULT_STREAM_INTERVAL_MINUTES
    minutes = mongo_manager.get_global_capture_interval_minutes(DEFAULT_STREAM_INTERVAL_MINUTES)
    try:
        minutes = int(minutes)
        return _validate_interval_minutes(minutes)
    except (ValueError, TypeError):
        return DEFAULT_STREAM_INTERVAL_MINUTES


def _resolve_capture_interval_minutes_from_payload(data):
    raw_interval_minutes = data.get('capture_interval_minutes')
    raw_interval_seconds = data.get('interval') or data.get('capture_interval_seconds')

    if raw_interval_minutes is not None:
        try:
            return _validate_interval_minutes(int(raw_interval_minutes))
        except (ValueError, TypeError):
            raise ValueError('分析間隔格式錯誤，請輸入 1~60 的整數分鐘')

    if raw_interval_seconds is not None:
        try:
            capture_interval_seconds = int(raw_interval_seconds)
        except (ValueError, TypeError):
            raise ValueError('分析間隔格式錯誤，請輸入 1~60 的整數分鐘')
        if capture_interval_seconds <= 0 or capture_interval_seconds % 60 != 0:
            raise ValueError('分析間隔必須是整數分鐘（1~60）')
        return _validate_interval_minutes(capture_interval_seconds // 60)

    return _resolve_global_interval_minutes()


@app.route('/api/stream/global_interval', methods=['GET'])
def get_stream_global_interval():
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503
    minutes = _resolve_global_interval_minutes()
    return jsonify({'success': True, 'capture_interval_minutes': minutes}), 200


@app.route('/api/stream/global_interval', methods=['POST'])
def set_stream_global_interval():
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503
    try:
        data = request.get_json() or {}
        minutes = _validate_interval_minutes(int(data.get('capture_interval_minutes')))
    except (ValueError, TypeError):
        return jsonify({'error': '分析間隔格式錯誤，請輸入 1~60 的整數分鐘'}), 400

    if not mongo_manager.set_global_capture_interval_minutes(minutes):
        return jsonify({'error': '更新全域分析間隔失敗'}), 500
    return jsonify({'success': True, 'capture_interval_minutes': minutes}), 200

@app.route('/api/patient-ids')
def list_patient_ids():
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503
    try:
        return jsonify({
            'success': True,
            'patient_ids': mongo_manager.get_patient_ids()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream/start', methods=['POST'])
def start_stream_monitor():
    global active_stream_processors
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用，無法啟動串流監測'}), 503

    data = request.get_json() or {}
    rtsp_url = data.get('rtsp_url')
    camera_name = data.get('camera_name')
    api_key = data.get('api_key')
    provider = data.get('provider')
    model = data.get('model')
    # patient_id 可跨多個 session 延續；task_name 是舊版前端的相容欄位。
    patient_id = str(data.get('patient_id') or data.get('task_name') or '').strip()
    
    try:
        capture_interval_minutes = _resolve_capture_interval_minutes_from_payload(data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    capture_interval = capture_interval_minutes * 60
    mongo_manager.set_global_capture_interval_minutes(capture_interval_minutes)

    provider, api_key, base_url, cred_error = resolve_llm_credentials(
        provider, model, api_key
    )
    if cred_error:
        return jsonify({'error': cred_error}), 400

    if not rtsp_url or not camera_name or not patient_id:
        return jsonify({'error': '缺少病患 ID、RTSP URL 或攝影機名稱'}), 400

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
            mongo_manager.save_stream_session(
                session_id,
                camera_name,
                rtsp_url,
                llm_model=model,
                patient_id=patient_id,
                capture_interval_seconds=capture_interval,
                capture_interval_minutes=capture_interval_minutes
            )
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
        # 只取得最新一小頁；總筆數由分頁查詢的 count_documents 提供。
        page_data = mongo_manager.get_medical_values_page(
            session_id, page=1, page_size=100
        )
        medical_values = page_data["items"]
        
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
            'total_analysis_count': page_data["total_items"],
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
        data = request.get_json() or {}
        template_id = data.get('template_id')
        camera_name = data.get('camera_name')
        rtsp_url = data.get('rtsp_url')
        provider = data.get('provider')
        model = data.get('model')
        api_key = data.get('api_key')
        capture_interval_minutes = data.get('capture_interval_minutes', _resolve_global_interval_minutes())

        if not template_id or not camera_name or not rtsp_url:
            return jsonify({'error': '缺少必要欄位 (範本名稱、攝影機名稱或 RTSP URL)'}), 400
        try:
            capture_interval_minutes = int(capture_interval_minutes)
        except (ValueError, TypeError):
            return jsonify({'error': '分析間隔格式錯誤，請輸入 1~60 的整數分鐘'}), 400
        try:
            _validate_interval_minutes(capture_interval_minutes)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400

        success = mongo_manager.save_camera_template(
            template_id=template_id,
            camera_name=camera_name,
            rtsp_url=rtsp_url,
            provider=provider,
            model=model,
            api_key=api_key,
            capture_interval_minutes=capture_interval_minutes
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
    with dataset_batch_lock:
        if dataset_batch_status['is_running']:
            dataset_batch_status['stop_requested'] = True
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
    """從 MongoDB 分頁取得醫療數值（含圖片路徑）。"""
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503
    
    try:
        model = request.args.get('model')
        try:
            page = max(int(request.args.get('page', 1)), 1)
        except (TypeError, ValueError):
            page = 1
        try:
            page_size = int(request.args.get('page_size', 12))
        except (TypeError, ValueError):
            page_size = 12
        page_size = min(max(page_size, 1), 100)

        page_data = mongo_manager.get_medical_values_page(
            session_id, model, page, page_size
        )
        medical_values = page_data["items"]
        
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
                item['original_image_url'] = f"/analysis/{path}"
            
            if item.get('screen_image_path'):
                # 移除 video_screen_analysis/ 前綴並使用 /analysis/ 路由
                path = item['screen_image_path']
                if path.startswith('video_screen_analysis/'):
                    path = path.replace('video_screen_analysis/', '', 1)
                item['screen_image_url'] = f"/analysis/{path}"
            if item.get('screen_image_path_for_vlm'):
                path_for_vlm = item['screen_image_path_for_vlm']
                if path_for_vlm.startswith('video_screen_analysis/'):
                    path_for_vlm = path_for_vlm.replace('video_screen_analysis/', '', 1)
                item['screen_image_url_for_vlm'] = f"/analysis/{path_for_vlm}"
            
            # 轉換 analyzed_at 時間格式
            if 'analyzed_at' in item and item['analyzed_at']:
                item['analyzed_at'] = item['analyzed_at'].isoformat()
        
        failure_summary = None
        camera_name = None
        patient_id = None
        if mongo_manager:
            failure_summary = mongo_manager.get_analysis_failure_summary(session_id, model)
            if mongo_manager.db is not None:
                try:
                    video_analysis = mongo_manager.db.video_analysis.find_one({"session_id": session_id})
                    if video_analysis:
                        camera_name = video_analysis.get("camera_name")
                        patient_id = video_analysis.get("patient_id") or video_analysis.get("task_name")
                except Exception as e:
                    print(f"⚠️ 獲取 video_analysis 資訊失敗: {e}")

        return jsonify({
            'success': True,
            'items': medical_values,
            'pagination': {
                'page': page_data["page"],
                'page_size': page_data["page_size"],
                'total_items': page_data["total_items"],
                'total_pages': page_data["total_pages"],
            },
            'failure_summary': failure_summary,
            'camera_name': camera_name,
            'patient_id': patient_id,
            'task_name': patient_id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mongodb/histogram/<session_id>')
def get_mongodb_histogram(session_id):
    """取得整個 Session 的輕量 Histogram 數值資料。"""
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503

    try:
        model = request.args.get('model')
        items = mongo_manager.get_histogram_values_by_session(session_id, model)
        for item in items:
            analyzed_at = item.get('analyzed_at')
            if isinstance(analyzed_at, datetime):
                item['analyzed_at'] = analyzed_at.isoformat()
            elif analyzed_at is not None:
                item['analyzed_at'] = str(analyzed_at)
            item['medical_values'] = item.get('medical_values') or {}
            item['corrected_medical_values'] = (
                item.get('corrected_medical_values') or {}
            )

        return jsonify({
            'success': True,
            'items': items,
            'count': len(items),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mongodb/session_screen_sources/<session_id>')
def get_session_screen_sources(session_id):
    """取得批次重新分析所需的最小螢幕來源資料。"""
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503
    try:
        try:
            page = max(int(request.args.get('page', 1)), 1)
            page_size = min(
                max(int(request.args.get('page_size', 50)), 1),
                100
            )
        except (TypeError, ValueError):
            page, page_size = 1, 50
        page_data = mongo_manager.get_session_screen_sources_page(
            session_id, page, page_size
        )
        return jsonify({
            'success': True,
            'items': page_data["items"],
            'pagination': {
                'page': page_data["page"],
                'page_size': page_data["page_size"],
                'total_items': page_data["total_items"],
                'total_pages': page_data["total_pages"],
            },
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mongodb/export/medical_values/<session_id>')
def export_mongodb_medical_values(session_id):
    """匯出單一 Session 的醫療數值 (CSV)"""
    if not mongo_manager:
        return jsonify({'error': 'MongoDB 未啟用'}), 503

    try:
        model = request.args.get('model')
        mode = (request.args.get('mode') or 'raw').lower()
        if mode not in {'raw', 'corrected', 'both'}:
            return jsonify({'error': '無效的匯出模式，僅支援 raw/corrected/both'}), 400

        # corrected 模式要優先確保人工訂正能被帶出，不受目前下拉模型限制
        query_model = model if mode in {'raw', 'both'} else None
        medical_values = mongo_manager.get_medical_values_by_session(session_id, query_model)

        # 依時間與螢幕排序，確保匯出內容可讀
        medical_values = sorted(
            medical_values,
            key=lambda item: (
                item.get('minute', 0),
                item.get('screen_number', 0),
                item.get('original_screen_number', 0)
            )
        )

        medical_keys = sorted({
            key
            for item in medical_values
            for key in (item.get('medical_values') or {}).keys()
        })
        corrected_keys = sorted({
            key
            for item in medical_values
            for key in (item.get('corrected_medical_values') or {}).keys()
        })
        final_keys = sorted(set(medical_keys) | set(corrected_keys))

        output = io.StringIO()
        writer = csv.writer(output)

        header = [
            'session_id',
            'time_seconds',
            'analyzed_at',
            'screen_number',
            'original_screen_number',
            'model',
            'corrected_model',
            'llm_model',
            'has_glare',
            'has_occlusion',
        ]
        if mode == 'raw':
            header.extend([f'medical_{key}' for key in medical_keys])
        elif mode == 'corrected':
            header.extend([f'final_{key}' for key in final_keys])
        else:
            header.extend([f'medical_{key}' for key in medical_keys])
            header.extend([f'corrected_{key}' for key in corrected_keys])
        writer.writerow(header)

        for item in medical_values:
            analyzed_at = item.get('analyzed_at')
            row = [
                item.get('session_id', session_id),
                item.get('minute', ''),
                analyzed_at.isoformat() if analyzed_at else '',
                item.get('screen_number', ''),
                item.get('original_screen_number', ''),
                item.get('model', ''),
                item.get('corrected_model', ''),
                item.get('llm_model', ''),
                item.get('has_glare', False),
                item.get('has_occlusion', False),
            ]

            item_medical_values = item.get('medical_values') or {}
            item_corrected_values = item.get('corrected_medical_values') or {}

            if mode == 'raw':
                row.extend([item_medical_values.get(key, '') for key in medical_keys])
            elif mode == 'corrected':
                row.extend([
                    item_corrected_values.get(key)
                    if item_corrected_values.get(key) not in (None, '')
                    else item_medical_values.get(key, '')
                    for key in final_keys
                ])
            else:
                row.extend([item_medical_values.get(key, '') for key in medical_keys])
                row.extend([item_corrected_values.get(key, '') for key in corrected_keys])
            writer.writerow(row)

        # 檔名優先使用病患 ID；舊資料回退 task_name，再回退 session_id。
        patient_id = None
        if mongo_manager and mongo_manager.db is not None:
            try:
                video_analysis = mongo_manager.db.video_analysis.find_one(
                    {"session_id": session_id},
                    {"patient_id": 1, "task_name": 1, "_id": 0}
                )
                if video_analysis:
                    patient_id = video_analysis.get('patient_id') or video_analysis.get('task_name')
            except Exception as e:
                print(f"⚠️ 取得匯出檔名 patient_id 失敗，改用 session_id: {e}")

        session_label = sanitize_export_filename_component(session_id, fallback='session')
        if patient_id and str(patient_id).strip():
            patient_label = sanitize_export_filename_component(patient_id, fallback=session_label)
            base_name = f'{patient_label}_{session_label}'
        else:
            base_name = session_label
        model_label = sanitize_export_filename_component(
            (query_model or 'all_models').replace('/', '_'),
            fallback='all_models'
        )
        utf8_filename = f'{base_name}_{model_label}_{mode}.csv'
        ascii_filename = secure_filename(utf8_filename) or f'{session_id}_{mode}.csv'
        content_disposition = (
            f'attachment; filename="{ascii_filename}"; '
            f"filename*=UTF-8''{quote(utf8_filename)}"
        )

        return Response(
            output.getvalue().encode('utf-8-sig'),
            mimetype='text/csv; charset=utf-8',
            headers={'Content-Disposition': content_disposition}
        )
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
            
        # 在 MongoDB 端先分組，避免把整個 session 載入 Flask。
        grouped_stats = mongo_manager.get_session_device_stats(session_id, model)
        device_stats = {}
        
        for row in grouped_stats:
            group = row.get('_id') or {}
            raw_device_name = group.get('model') or 'unknown'
            corrected_device_name = group.get('corrected_model')
            count = row.get('count', 0)
            
            # 以 AI 辨識結果統計 TP/FP，並把訂正後的設備計入 FN。
            device_name = normalize_device_name(raw_device_name)
            corrected_name = (
                normalize_device_name(corrected_device_name)
                if corrected_device_name is not None and corrected_device_name != ""
                else device_name
            )
            
            if device_name not in device_stats:
                device_stats[device_name] = {
                    'total_count': 0,      # TP + FP (AI 辨識為此設備的總次數)
                    'corrected_count': 0,  # FP (AI 辨識為此設備但被訂正為其他的次數)
                    'true_positive_count': 0,
                    'false_positive_count': 0,
                    'false_negative_count': 0,
                    'true_count': 0,       # TP + FN (人工訂正後屬於此設備的總次數)
                    'accuracy': 0,         # Precision，保留既有 API 欄位
                    'recall': 0
                }
            
            device_stats[device_name]['total_count'] += count
            
            # 如果有訂正，且訂正後的名稱與原始辨識不同，則視為 FP (False Positive)
            # 注意：如果 corrected_device_name 為空，代表 AI 辨識正確 (TP)
            is_fp = corrected_name != device_name
            if is_fp:
                device_stats[device_name]['corrected_count'] += count
                device_stats[device_name]['false_positive_count'] += count

                if corrected_name not in device_stats:
                    device_stats[corrected_name] = {
                        'total_count': 0,
                        'corrected_count': 0,
                        'true_positive_count': 0,
                        'false_positive_count': 0,
                        'false_negative_count': 0,
                        'true_count': 0,
                        'accuracy': 0,
                        'recall': 0
                    }
                device_stats[corrected_name]['false_negative_count'] += count
            else:
                device_stats[device_name]['true_positive_count'] += count
        
        # 3. 同時計算 Precision 與 Recall。
        for device_name in device_stats:
            stats = device_stats[device_name]
            if stats['total_count'] > 0:
                stats['accuracy'] = round(
                    (stats['true_positive_count'] / stats['total_count']) * 100,
                    1
                )
            stats['true_count'] = (
                stats['true_positive_count'] + stats['false_negative_count']
            )
            if stats['true_count'] > 0:
                stats['recall'] = round(
                    (stats['true_positive_count'] / stats['true_count']) * 100,
                    1
                )
        
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