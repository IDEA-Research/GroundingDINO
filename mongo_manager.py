#!/usr/bin/env python3
"""
MongoDB 管理模組 (檔案系統儲存版本)
自動啟動、停止和管理本地 MongoDB 實例，圖片存放在檔案系統
"""

import os
import re
import sys
import time
import signal
import shutil
import subprocess
import atexit
import uuid
from pathlib import Path
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from datetime import datetime
import traceback

class LocalMongoDBManager:
    """本地 MongoDB 管理器 (檔案系統儲存)"""
    
    def __init__(self, 
                 data_dir="mongodb_data",
                 config_file="mongodb_config/mongod.conf",
                 port=27017,
                 database_name="medical_monitor_db"):
        
        self.data_dir = Path(data_dir)
        self.config_file = Path(config_file)
        self.port = port
        self.database_name = database_name
        self.mongod_process = None
        self.client = None
        self.db = None
        self.medical_exporter_process = None  # 醫療監控服務進程
        self._we_started_mongodb = False
        self._cleanup_done = False
        self._signals_registered = False
        
        # 確保目錄存在
        self.data_dir.mkdir(exist_ok=True)
        (self.data_dir / "db").mkdir(exist_ok=True)
        (self.data_dir / "logs").mkdir(exist_ok=True)
        self.config_file.parent.mkdir(exist_ok=True)
        
        # 註冊清理函數（Ctrl+C / SIGTERM 與正常退出）
        atexit.register(self.cleanup)
        self._register_exit_handlers()
    
    def _mongod_executable(self):
        """解析 mongod 路徑（需在 webapp_mongo_gpu 等已安裝 Mongo 的環境中執行）"""
        exe = shutil.which('mongod')
        if exe:
            return exe
        raise FileNotFoundError('mongod')
    
    def _register_exit_handlers(self):
        if self._signals_registered:
            return
        signal.signal(signal.SIGINT, self._handle_exit_signal)
        signal.signal(signal.SIGTERM, self._handle_exit_signal)
        self._signals_registered = True
    
    def _handle_exit_signal(self, signum, frame):
        print("\n收到結束信號，正在清理 MongoDB 與相關服務...")
        self.cleanup()
        sys.exit(0)
    
    def create_config_file(self):
        """創建 MongoDB 配置檔案"""
        config_content = f"""storage:
  dbPath: {self.data_dir}/db

systemLog:
  destination: file
  path: {self.data_dir}/logs/mongod.log
  logAppend: true

net:
  port: {self.port}
  bindIp: 127.0.0.1

processManagement:
  fork: false

security:
  authorization: disabled
"""
        
        with open(self.config_file, 'w') as f:
            f.write(config_content)
        
        print(f"✅ MongoDB 配置檔案已創建: {self.config_file}")
    
    def start_mongodb(self):
        """啟動 MongoDB"""
        if self.is_mongodb_running():
            print("📄 MongoDB 已經在運行中")
            self._we_started_mongodb = False
            # 外部啟動的 MongoDB（例如 docker medical-mongodb）也要確保索引存在；
            # create_index 為冪等操作，已存在的索引不會重建
            if self.connect_to_mongodb():
                self.setup_database()
            return True
        
        # mongodb_data/db 由 docker medical-mongodb (mongo:8.0) 管理且 FCV 為 8.0，
        # 不可再用 conda 環境中較舊版本的 mongod 開啟同一份資料
        print("❌ MongoDB (docker medical-mongodb) 未回應，不會在本機另外啟動 mongod")
        print("💡 請執行: docker compose -f docker-compose.monitoring.yml up -d mongodb")
        return False
    
    def is_mongodb_running(self):
        """檢查 MongoDB 是否正在運行"""
        try:
            client = MongoClient(f'mongodb://localhost:{self.port}', 
                               serverSelectionTimeoutMS=1000)
            client.admin.command('ping')
            client.close()
            return True
        except:
            # print(traceback.format_exc())
            return False
    
    def _graceful_shutdown_mongod(self, timeout=30):
        """使用 mongod --shutdown 優雅關閉本專案 dbpath 的實例（策略 2）"""
        db_path = (self.data_dir / "db").resolve()
        if not self.is_mongodb_running():
            return True
        try:
            mongod = self._mongod_executable()
        except FileNotFoundError:
            print("❌ 找不到 mongod 命令，無法優雅關閉 MongoDB")
            print("💡 請先 conda activate webapp_mongo_gpu")
            return False
        
        print(f"🛑 正在優雅關閉 MongoDB (dbpath: {db_path})...")
        result = subprocess.run(
            [mongod, '--shutdown', '--dbpath', str(db_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            msg = (result.stderr or result.stdout or "").strip()
            if msg:
                print(f"⚠️  mongod --shutdown: {msg}")
        
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.is_mongodb_running():
                print("✅ MongoDB 已正常關閉")
                return True
            time.sleep(0.5)
        
        print("⚠️  MongoDB 優雅關閉等待超時")
        return False
    
    def _force_stop_mongod_process(self):
        """shutdown 失敗時，僅對本程式啟動的 mongod 子進程強制終止"""
        if not self.mongod_process:
            return
        if self.mongod_process.poll() is not None:
            return
        print("🔨 強制終止 MongoDB 子進程...")
        self.mongod_process.terminate()
        try:
            self.mongod_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.mongod_process.kill()
            self.mongod_process.wait()
    
    def connect_to_mongodb(self):
        """連接到 MongoDB"""
        try:
            self.client = MongoClient(f'mongodb://localhost:{self.port}')
            self.db = self.client[self.database_name]
            
            # 測試連接
            self.client.admin.command('ping')
            print(f"🔗 已連接到 MongoDB 數據庫: {self.database_name}")
            return True
            
        except Exception as e:
            print(f"❌ 連接 MongoDB 失敗: {e}")
            return False
    
    def setup_database(self):
        """設置數據庫和索引"""
        try:
            # 創建 collections
            collections = [
                'video_analysis', 'frame_results', 'screen_analysis', 'stream_sessions',
                'datasets', 'dataset_videos', 'dataset_runs', 'dataset_run_items'
            ]
            for collection_name in collections:
                if collection_name not in self.db.list_collection_names():
                    self.db.create_collection(collection_name)
            
            # 創建索引
            self.create_indexes()
            print("🗂️  數據庫設置完成")
            
        except Exception as e:
            print(f"⚠️  設置數據庫時發生錯誤: {e}")
    
    def create_indexes(self):
        """創建必要的索引（逐一建立，單一索引失敗不影響其他索引）"""
        index_specs = [
            # VideoAnalysis 索引
            ("video_analysis", "session_id", {}),
            ("video_analysis", [("timestamp", -1)], {}),
            ("video_analysis", "video_name", {}),
            ("video_analysis", "patient_id", {}),
            # camera 查詢第 1 步：source + camera_name 找出 session
            ("video_analysis", [("source", 1), ("camera_name", 1)], {}),

            # FrameResult 索引
            ("frame_results", [("video_analysis_id", 1), ("minute", 1)], {}),
            ("frame_results", [("processed_at", -1)], {}),
            # camera 查詢第 2 步：session + 時間範圍找出 frame
            ("frame_results", [("video_analysis_id", 1), ("processed_at", 1)], {}),

            # ScreenAnalysis 索引
            ("screen_analysis", "frame_result_id", {}),
            ("screen_analysis", "detected_model", {}),
            ("screen_analysis", "medical_values.heart_rate", {}),
            ("screen_analysis", [("analyzed_at", -1)], {}),
            ("screen_analysis", [
                ("frame_result_id", 1),
                ("success", 1),
                ("llm_model", 1),
                ("analyzed_at", -1),
            ], {}),
            # camera 查詢第 3 步：frame + 時間範圍取讀值
            ("screen_analysis", [("frame_result_id", 1), ("analyzed_at", 1)], {}),

            # StreamSessions 索引
            ("stream_sessions", "session_id", {"unique": True}),
            ("stream_sessions", "camera_name", {}),
            ("stream_sessions", "rtsp_url", {}),

            # Dataset / batch run indexes
            ("datasets", "dataset_id", {"unique": True}),
            ("datasets", [("created_at", -1)], {}),
            ("dataset_videos", "video_id", {"unique": True}),
            ("dataset_videos", [("dataset_id", 1), ("created_at", 1)], {}),
            ("dataset_runs", "run_id", {"unique": True}),
            ("dataset_runs", [("dataset_id", 1), ("created_at", -1)], {}),
            ("dataset_run_items", [("run_id", 1), ("position", 1)], {"unique": True}),
        ]

        failed = 0
        for collection_name, keys, options in index_specs:
            try:
                self.db[collection_name].create_index(keys, **options)
            except Exception as e:
                failed += 1
                print(f"⚠️  創建索引失敗 {collection_name} {keys}: {e}")

        if failed:
            print(f"⚠️  索引創建完成，{failed} 個失敗")
        else:
            print("📊 索引創建完成")
    
    def stop_mongodb(self):
        """停止 MongoDB（策略 2：本機在跑則對 mongodb_data/db 執行 --shutdown）"""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
        
        if self._we_started_mongodb and self.is_mongodb_running():
            if not self._graceful_shutdown_mongod():
                self._force_stop_mongod_process()
        else:
            if self.is_mongodb_running():
                print("MongoDB 是由其他進程啟動或本已在運行，當前進程退出時將保持其運行狀態。")
        
        self.mongod_process = None
    
    def cleanup(self):
        """清理資源"""
        if self._cleanup_done:
            return
        self._cleanup_done = True
        self.stop_medical_monitoring()
        self.stop_mongodb()
    
    def save_analysis_result(self, video_path, session_id, results, source="manual", llm_model="unknown", camera_name=None, dataset_id=None, dataset_video_id=None, dataset_run_id=None, original_video_name=None):
        """儲存分析結果到 MongoDB (檔案路徑版本)"""
        if self.db is None:
            raise Exception("MongoDB 連接未建立")
        
        try:
            # 串流的 patient_id 是跨會話的病患識別；task_name 僅保留給舊資料相容。
            patient_id = None
            if source == "stream":
                try:
                    stream_sess = self.db.stream_sessions.find_one({"session_id": session_id})
                    if stream_sess:
                        patient_id = stream_sess.get("patient_id") or stream_sess.get("task_name")
                except Exception as e:
                    print(f"⚠️ 獲取串流會話病患 ID 失敗: {e}")

            # 1. 更新或建立 VideoAnalysis 記錄 (使用 upsert 避免重複)
            video_analysis_doc = {
                "video_path": video_path,
                "video_name": (
                    Path(original_video_name).stem
                    if original_video_name else Path(video_path).stem
                ),
                "session_id": session_id,
                "timestamp": datetime.now(),
                "total_frames": len(results),
                "status": "processing",
                "source": source,  # manual, stream 等
                "llm_model": llm_model,  # 記錄使用的 LLM 模型
                "camera_name": camera_name, # 記錄攝影機名稱 (用於串流)
                "patient_id": patient_id,
                "task_name": patient_id, # 舊版欄位相容，後續讀取應優先使用 patient_id
                "dataset_id": dataset_id,
                "dataset_video_id": dataset_video_id,
                "dataset_run_id": dataset_run_id,
                "original_video_name": original_video_name,
            }
            
            # 使用 session_id 作為唯一標識進行 upsert
            self.db.video_analysis.update_one(
                {"session_id": session_id},
                {"$set": video_analysis_doc},
                upsert=True
            )
            
            # 取得該 session 的文檔 ID
            video_analysis_id = self.db.video_analysis.find_one({"session_id": session_id})["_id"]
            
            for result in results:
                # 2. 建立 FrameResult 記錄 (只儲存圖片路徑)
                # 對於 frame，我們可以使用 (video_analysis_id, minute) 作為唯一標識
                frame_result_doc = {
                    "video_analysis_id": video_analysis_id,
                    "minute": result['time_seconds'],
                    "original_image_path": result['image_path'],
                    "screens_detected": result['screens_detected'],
                    "processed_at": datetime.now()
                }
                
                self.db.frame_results.update_one(
                    {
                        "video_analysis_id": video_analysis_id, 
                        "minute": result['time_seconds']
                    },
                    {"$set": frame_result_doc},
                    upsert=True
                )
                
                frame_result = self.db.frame_results.find_one({
                    "video_analysis_id": video_analysis_id, 
                    "minute": result['time_seconds']
                })
                frame_result_id = frame_result["_id"]
                
                # 3. 處理每個螢幕分析結果
                original_screen_count = result.get('original_screens_detected', len(result['screen_analyses']))
                
                for idx, screen_analysis in enumerate(result['screen_analyses'], 1):
                    if not screen_analysis['screen_path']:
                        continue
                    
                    analysis = screen_analysis['analysis']
                    medical_values = {}
                    detected_model = 'unknown'
                    raw_analysis = {}
                    
                    if analysis['success']:
                        result_data = analysis['result']
                        medical_values = result_data.get('medical_values', {})
                        detected_model = result_data.get('model', 'unknown')
                        raw_analysis = result_data
                    
                    screen_analysis_doc = {
                        "frame_result_id": frame_result_id,
                        "screen_number": idx,
                        "original_screen_number": screen_analysis['screen_number'],
                        "screen_image_path": screen_analysis['screen_path'],
                        "screen_image_path_for_vlm": (
                            screen_analysis.get('screen_path_for_vlm')
                            or screen_analysis['screen_path']
                        ),
                        "detected_model": detected_model,
                        "bbox": screen_analysis.get('bbox'),
                        "medical_values": medical_values,
                        "raw_analysis": raw_analysis,
                        "success": analysis['success'],
                        "error_message": analysis.get('error') if not analysis['success'] else None,
                        "is_merged_result": len(result['screen_analyses']) < original_screen_count,
                        # 修改：優先使用擷取時間，若無則使用當下時間
                        "analyzed_at": result.get('capture_timestamp', datetime.now()),
                        "llm_model": llm_model,  # 記錄使用的模型
                        "has_glare": False,
                        "has_occlusion": False
                    }
                    
                    # 對於 screen，使用 (frame_result_id, original_screen_number, llm_model) 作為唯一標識
                    self.db.screen_analysis.update_one(
                        {
                            "frame_result_id": frame_result_id,
                            "original_screen_number": screen_analysis['screen_number'],
                            "llm_model": llm_model
                        },
                        {"$set": screen_analysis_doc},
                        upsert=True
                    )
            
            # 4. 更新 VideoAnalysis 狀態
            self.db.video_analysis.update_one(
                {"_id": video_analysis_id},
                {"$set": {"status": "completed"}}
            )
            
            print(f"💾 分析結果已更新到 MongoDB (Source: {source}), Session: {session_id}")
            return video_analysis_id
            
        except Exception as e:
            print(f"❌ 儲存分析結果時發生錯誤: {e}")
            raise
    
    @staticmethod
    def _llm_model_match_filter(model):
        """建立 llm_model 查詢條件（相容舊版 gpt-4o 與 OpenRouter 的 openai/gpt-4o）"""
        if model == 'gpt-4o':
            return {
                "$or": [
                    {"screens.llm_model": model},
                    {"screens.llm_model": {"$exists": False}},
                    {"screens.llm_model": "openai/gpt-4o"},
                ]
            }
        if model == 'openai/gpt-4o':
            return {
                "$or": [
                    {"screens.llm_model": model},
                    {"screens.llm_model": "gpt-4o"},
                    {"screens.llm_model": {"$exists": False}},
                ]
            }
        return {"screens.llm_model": model}

    @staticmethod
    def _direct_llm_model_filter(model):
        """建立直接查詢 screen_analysis 時使用的模型條件。"""
        if model == 'gpt-4o':
            return {
                "$or": [
                    {"llm_model": model},
                    {"llm_model": {"$exists": False}},
                    {"llm_model": "openai/gpt-4o"},
                ]
            }
        if model == 'openai/gpt-4o':
            return {
                "$or": [
                    {"llm_model": model},
                    {"llm_model": "gpt-4o"},
                    {"llm_model": {"$exists": False}},
                ]
            }
        return {"llm_model": model}

    @staticmethod
    def _is_stream_video(video):
        return (
            video.get("source") == "stream"
            or str(video.get("video_path", "")).startswith(("RTSP:", "Camera:"))
        )

    def _get_related_video_analysis_ids(self, session_id):
        """取得同一病患所有串流 session 的分析記錄 ID。"""
        if self.db is None:
            return []

        video = self.db.video_analysis.find_one(
            {"session_id": session_id},
            {
                "_id": 1,
                "source": 1,
                "video_path": 1,
                "patient_id": 1,
                "task_name": 1,
            }
        )
        if not video:
            return []

        patient_id = str(
            video.get("patient_id") or video.get("task_name") or ""
        ).strip()
        if not patient_id or not self._is_stream_video(video):
            return [video["_id"]]

        related = self.db.video_analysis.find(
            {
                "$and": [
                    {
                        "$or": [
                            {"source": "stream"},
                            {"video_path": {"$regex": "^(RTSP|Camera):"}},
                        ]
                    },
                    {
                        "$or": [
                            {"patient_id": patient_id},
                            {"task_name": patient_id},
                        ]
                    },
                ]
            },
            {"_id": 1},
        )
        ids = [item["_id"] for item in related]
        return ids or [video["_id"]]

    def _get_session_frame_ids(self, session_id):
        """取得 session 所屬病患的所有串流 frame ID。"""
        video_ids = self._get_related_video_analysis_ids(session_id)
        if not video_ids:
            return []

        return self.db.frame_results.distinct(
            "_id", {"video_analysis_id": {"$in": video_ids}}
        )

    def _session_screens_lookup_pipeline(self, session_id):
        """病患相關 sessions → frame → screen 的共用聚合前段"""
        video_ids = self._get_related_video_analysis_ids(session_id)
        return [
            {"$match": {"_id": {"$in": video_ids}}},
            {"$lookup": {
                "from": "frame_results",
                "localField": "_id",
                "foreignField": "video_analysis_id",
                "as": "frames"
            }},
            {"$unwind": "$frames"},
            {"$lookup": {
                "from": "screen_analysis",
                "localField": "frames._id",
                "foreignField": "frame_result_id",
                "as": "screens"
            }},
            {"$unwind": "$screens"},
        ]

    def get_analysis_failure_summary(self, session_id, model=None):
        """取得指定 session（與可選模型）的 LLM 分析失敗摘要"""
        empty = {
            "failed_count": 0,
            "success_count": 0,
            "last_error": None,
            "sample_errors": [],
        }
        if self.db is None:
            return empty

        pipeline = self._session_screens_lookup_pipeline(session_id)
        if model:
            pipeline.append({"$match": self._llm_model_match_filter(model)})

        pipeline.extend([
            {"$facet": {
                "failed": [
                    {"$match": {"screens.success": False}},
                    {"$count": "count"},
                ],
                "success": [
                    {"$match": {"screens.success": True}},
                    {"$count": "count"},
                ],
                "recent_errors": [
                    {"$match": {
                        "screens.success": False,
                        "screens.error_message": {"$nin": [None, ""]},
                    }},
                    {"$sort": {"screens.analyzed_at": -1}},
                    {"$limit": 10},
                    {"$project": {
                        "_id": 0,
                        "message": "$screens.error_message",
                    }},
                ],
            }},
        ])

        try:
            rows = list(self.db.video_analysis.aggregate(pipeline))
            if not rows:
                return empty
            facet = rows[0]
            failed_block = (facet.get("failed") or [{}])[0]
            success_block = facet.get("success") or []
            failed_count = failed_block.get("count", 0)
            success_count = success_block[0]["count"] if success_block else 0
            raw_errors = [
                row.get("message")
                for row in (facet.get("recent_errors") or [])
                if row.get("message")
            ]
            sample_errors = list(dict.fromkeys(raw_errors))[:3]
            return {
                "failed_count": failed_count,
                "success_count": success_count,
                "last_error": raw_errors[0] if raw_errors else None,
                "sample_errors": sample_errors,
            }
        except Exception as e:
            print(f"❌ 取得分析失敗摘要時發生錯誤: {e}")
            return empty

    def get_medical_values_by_session(self, session_id, model=None):
        """取得特定 session 的醫療數值"""
        if self.db is None:
            return []
        
        # 使用聚合查詢優化效能
        pipeline = self._session_screens_lookup_pipeline(session_id)
        pipeline.append({"$match": {"screens.success": True}})
        
        if model:
            pipeline.append({"$match": self._llm_model_match_filter(model)})
            
        pipeline.append(
            {"$sort": {"screens.analyzed_at": -1}}
        )
        
        pipeline.append(
            {"$project": {
                "id": {"$toString": "$screens._id"},
                "session_id": "$session_id",
                "minute": "$frames.minute",
                "screen_number": "$screens.screen_number",
                "original_screen_number": "$screens.original_screen_number",
                "frame_result_id": {"$toString": "$frames._id"},
                "medical_values": "$screens.medical_values", 
                "corrected_medical_values": "$screens.corrected_medical_values",
                "model": "$screens.detected_model",
                "corrected_model": "$screens.corrected_model",
                "has_glare": "$screens.has_glare",
                "has_occlusion": "$screens.has_occlusion",
                "analyzed_at": "$screens.analyzed_at",
                "original_image_path": "$frames.original_image_path",
                "screen_image_path": "$screens.screen_image_path",
                "screen_image_path_for_vlm": "$screens.screen_image_path_for_vlm",
                "llm_model": "$screens.llm_model"
            }}
        )
        
        return list(self.db.video_analysis.aggregate(pipeline))

    def get_medical_values_page(self, session_id, model=None, page=1, page_size=12):
        """只取得指定 session 當頁的醫療數值。"""
        if self.db is None:
            return {
                "items": [],
                "page": 1,
                "page_size": page_size,
                "total_items": 0,
                "total_pages": 0,
            }

        frame_ids = self._get_session_frame_ids(session_id)
        query = {
            "frame_result_id": {"$in": frame_ids},
            "success": True,
        }
        if model:
            query = {"$and": [query, self._direct_llm_model_filter(model)]}

        total_items = self.db.screen_analysis.count_documents(query)
        total_pages = (
            (total_items + page_size - 1) // page_size
            if total_items else 0
        )
        page = min(max(page, 1), total_pages or 1)

        screens = list(
            self.db.screen_analysis.find(query)
            .sort([("analyzed_at", -1), ("_id", -1)])
            .skip((page - 1) * page_size)
            .limit(page_size)
        )

        page_frame_ids = list({
            screen.get("frame_result_id")
            for screen in screens
            if screen.get("frame_result_id") is not None
        })
        page_frames = self.db.frame_results.find(
            {"_id": {"$in": page_frame_ids}},
            {"minute": 1, "original_image_path": 1}
        )
        frame_map = {frame["_id"]: frame for frame in page_frames}

        items = []
        for screen in screens:
            frame = frame_map.get(screen.get("frame_result_id"), {})
            items.append({
                "id": str(screen.get("_id", "")),
                "minute": frame.get("minute", 0),
                "screen_number": screen.get("screen_number"),
                "original_screen_number": screen.get("original_screen_number"),
                "frame_result_id": str(screen.get("frame_result_id", "")),
                "medical_values": screen.get("medical_values") or {},
                "corrected_medical_values": screen.get("corrected_medical_values") or {},
                "model": screen.get("detected_model") or "unknown",
                "corrected_model": screen.get("corrected_model"),
                "has_glare": bool(screen.get("has_glare", False)),
                "has_occlusion": bool(screen.get("has_occlusion", False)),
                "analyzed_at": screen.get("analyzed_at"),
                "original_image_path": frame.get("original_image_path", ""),
                "screen_image_path": screen.get("screen_image_path", ""),
                "screen_image_path_for_vlm": (
                    screen.get("screen_image_path_for_vlm")
                    or screen.get("screen_image_path", "")
                ),
                "llm_model": screen.get("llm_model"),
            })

        return {
            "items": items,
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "total_pages": total_pages,
        }

    def get_histogram_values_by_session(self, session_id, model=None):
        """取得整個 session 繪製 Histogram 所需的輕量資料。"""
        if self.db is None:
            return []

        frame_ids = self._get_session_frame_ids(session_id)
        query = {
            "frame_result_id": {"$in": frame_ids},
            "success": True,
        }
        if model:
            query = {"$and": [query, self._direct_llm_model_filter(model)]}

        return list(
            self.db.screen_analysis.find(
                query,
                {
                    "_id": 0,
                    "analyzed_at": 1,
                    "medical_values": 1,
                    "corrected_medical_values": 1,
                }
            ).sort([("analyzed_at", 1), ("_id", 1)])
        )

    def get_session_device_stats(self, session_id, model=None):
        """在 MongoDB 端彙總設備訂正統計，避免載入整個 session。"""
        if self.db is None:
            return []
        frame_ids = self._get_session_frame_ids(session_id)
        match = {
            "frame_result_id": {"$in": frame_ids},
            "success": True,
        }
        if model:
            match = {"$and": [match, self._direct_llm_model_filter(model)]}

        return list(self.db.screen_analysis.aggregate([
            {"$match": match},
            {"$group": {
                "_id": {
                    "model": {"$ifNull": ["$detected_model", "unknown"]},
                    "corrected_model": "$corrected_model",
                },
                "count": {"$sum": 1},
            }},
        ]))

    def get_session_screen_sources_page(self, session_id, page=1, page_size=50):
        """分頁取得批次重新分析需要的最小螢幕來源資料。"""
        if self.db is None:
            return {"items": [], "page": 1, "total_items": 0, "total_pages": 0}
        frame_ids = self._get_session_frame_ids(session_id)
        pipeline = [
            {"$match": {
                "frame_result_id": {"$in": frame_ids},
                "success": True,
            }},
            {"$sort": {"analyzed_at": -1, "_id": -1}},
            {"$group": {
                "_id": {
                    "frame_result_id": "$frame_result_id",
                    "original_screen_number": "$original_screen_number",
                },
                "source": {"$first": {
                    "frame_result_id": {"$toString": "$frame_result_id"},
                    "original_screen_number": "$original_screen_number",
                    "screen_number": "$screen_number",
                    "screen_image_path": {
                        "$ifNull": [
                            "$screen_image_path_for_vlm",
                            "$screen_image_path",
                        ]
                    },
                }},
            }},
            {"$sort": {
                "_id.frame_result_id": 1,
                "_id.original_screen_number": 1,
            }},
            {"$facet": {
                "metadata": [{"$count": "total"}],
                "items": [
                    {"$skip": (page - 1) * page_size},
                    {"$limit": page_size},
                    {"$replaceWith": "$source"},
                ],
            }},
        ]
        rows = list(self.db.screen_analysis.aggregate(pipeline))
        facet = rows[0] if rows else {}
        metadata = facet.get("metadata") or []
        total_items = metadata[0]["total"] if metadata else 0
        total_pages = (
            (total_items + page_size - 1) // page_size
            if total_items else 0
        )
        page = min(max(page, 1), total_pages or 1)
        if page > 1 and not facet.get("items") and total_pages:
            return self.get_session_screen_sources_page(
                session_id, total_pages, page_size
            )
        return {
            "items": facet.get("items") or [],
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "total_pages": total_pages,
        }
    
    def get_session_models(self, session_id):
        """取得特定 session 中使用過的所有模型"""
        if self.db is None:
            return []

        video_ids = self._get_related_video_analysis_ids(session_id)
        pipeline = [
            {"$match": {"_id": {"$in": video_ids}}},
            {"$lookup": {
                "from": "frame_results",
                "localField": "_id",
                "foreignField": "video_analysis_id",
                "as": "frames"
            }},
            {"$unwind": "$frames"},
            {"$lookup": {
                "from": "screen_analysis", 
                "localField": "frames._id",
                "foreignField": "frame_result_id",
                "as": "screens"
            }},
            {"$unwind": "$screens"},
            {"$match": {"screens.success": True}},
            {"$group": {"_id": "$screens.llm_model"}},
            {"$project": {"_id": 0, "model": "$_id"}}
        ]
        
        results = list(self.db.video_analysis.aggregate(pipeline))
        models = [r['model'] for r in results if r.get('model')]
        
        # 為了相容舊資料 (沒有 llm_model 欄位)，如果 models 為空，預設加入 gpt-4o
        if not models:
            # 檢查是否有任何 screen_analysis
            count_pipeline = [
                {"$match": {"_id": {"$in": video_ids}}},
                {"$lookup": {
                    "from": "frame_results",
                    "localField": "_id",
                    "foreignField": "video_analysis_id",
                    "as": "frames"
                }},
                {"$unwind": "$frames"},
                {"$lookup": {
                    "from": "screen_analysis", 
                    "localField": "frames._id",
                    "foreignField": "frame_result_id",
                    "as": "screens"
                }},
                {"$unwind": "$screens"},
                {"$count": "total"}
            ]
            count_result = list(self.db.video_analysis.aggregate(count_pipeline))
            if count_result and count_result[0]['total'] > 0:
                models = ['gpt-4o']
                
        return models

    def save_stream_session(self, session_id, camera_name, rtsp_url, llm_model="unknown", patient_id=None, task_name=None, capture_interval_seconds=60, capture_interval_minutes=1):
        """儲存一個新的串流監測會話"""
        if self.db is None:
            raise Exception("MongoDB 連接未建立")

        try:
            patient_id = str(patient_id or task_name or "").strip() or None
            session_doc = {
                "session_id": session_id,
                "camera_name": camera_name,
                "rtsp_url": rtsp_url,
                "llm_model": llm_model,
                "patient_id": patient_id,
                "task_name": patient_id, # 舊版欄位相容
                "capture_interval_seconds": capture_interval_seconds,
                "capture_interval_minutes": capture_interval_minutes,
                "status": "inactive", # inactive, active, recording
                "created_at": datetime.now(),
                "last_updated_at": datetime.now()
            }
            self.db.stream_sessions.update_one(
                {"session_id": session_id},
                {"$set": session_doc},
                upsert=True
            )
            print(f"💾 串流會話已儲存或更新: {camera_name} ({rtsp_url})")
            return True
        except Exception as e:
            print(f"❌ 儲存串流會話時發生錯誤: {e}")
            return False

    def update_stream_session_status(self, session_id, status):
        """更新串流會話的狀態 (inactive, active, recording)"""
        if self.db is None:
            return False
        try:
            result = self.db.stream_sessions.update_one(
                {"session_id": session_id},
                {"$set": {"status": status, "last_updated_at": datetime.now()}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"❌ 更新串流會話狀態失敗: {e}")
            return False

    def get_stream_sessions(self):
        """獲取所有串流會話的列表"""
        if self.db is None:
            return []
        try:
            sessions = list(self.db.stream_sessions.find({}).sort("created_at", -1))
            # 轉換 ObjectId 為字串
            for session in sessions:
                session['_id'] = str(session['_id'])
            return sessions
        except Exception as e:
            print(f"❌ 獲取串流會話失敗: {e}")
            return []

    def get_patient_ids(self):
        """取得歷史分析與串流卡面中曾使用過的病患 ID。"""
        if self.db is None:
            return []

        patient_ids = set()
        try:
            for collection_name in ("video_analysis", "stream_sessions"):
                collection = self.db[collection_name]
                for field_name in ("patient_id", "task_name"):
                    for value in collection.distinct(field_name):
                        normalized = str(value or "").strip()
                        if normalized:
                            patient_ids.add(normalized)
            return sorted(patient_ids, key=str.casefold)
        except Exception as e:
            print(f"❌ 獲取病患 ID 清單失敗: {e}")
            return []

    def save_camera_template(self, template_id, camera_name, rtsp_url, provider, model, api_key=None, capture_interval_minutes=1):
        """儲存或更新一個相機監測範本"""
        if self.db is None:
            raise Exception("MongoDB 連接未建立")

        try:
            template_doc = {
                "template_id": template_id,
                "camera_name": camera_name,
                "rtsp_url": rtsp_url,
                "provider": provider,
                "model": model,
                "api_key": api_key,
                "capture_interval_minutes": capture_interval_minutes,
                "updated_at": datetime.now()
            }
            self.db.camera_templates.update_one(
                {"template_id": template_id},
                {"$set": template_doc},
                upsert=True
            )
            print(f"💾 相機範本已儲存或更新: {template_id} ({camera_name})")
            return True
        except Exception as e:
            print(f"❌ 儲存相機範本時發生錯誤: {e}")
            return False

    def get_camera_templates(self):
        """獲取所有相機範本的列表"""
        if self.db is None:
            return []
        try:
            templates = list(self.db.camera_templates.find({}).sort("updated_at", -1))
            for t in templates:
                t['_id'] = str(t['_id'])
            return templates
        except Exception as e:
            print(f"❌ 獲取相機範本失敗: {e}")
            return []

    def delete_camera_template(self, template_id):
        """刪除一個相機範本"""
        if self.db is None:
            return False
        try:
            result = self.db.camera_templates.delete_one({"template_id": template_id})
            return result.deleted_count > 0
        except Exception as e:
            print(f"❌ 刪除相機範本失敗: {e}")
            return False

    def get_global_capture_interval_minutes(self, default_value=1):
        """取得全域預設分析間隔（分鐘）"""
        if self.db is None:
            return default_value
        try:
            doc = self.db.system_settings.find_one({"key": "stream_capture_interval_minutes"})
            value = int(doc.get("value")) if doc and doc.get("value") is not None else default_value
            return value
        except Exception as e:
            print(f"❌ 讀取全域分析間隔失敗: {e}")
            return default_value

    def set_global_capture_interval_minutes(self, minutes):
        """設定全域預設分析間隔（分鐘）"""
        if self.db is None:
            return False
        try:
            self.db.system_settings.update_one(
                {"key": "stream_capture_interval_minutes"},
                {
                    "$set": {
                        "value": int(minutes),
                        "updated_at": datetime.now()
                    }
                },
                upsert=True
            )
            return True
        except Exception as e:
            print(f"❌ 更新全域分析間隔失敗: {e}")
            return False

    def save_single_screen_analysis(self, frame_result_id, original_screen_number, screen_number, screen_image_path, analysis_result, llm_model):
        """儲存單張螢幕的重新分析結果"""
        from bson.objectid import ObjectId
        if self.db is None:
            return False
            
        try:
            medical_values = {}
            detected_model = 'unknown'
            raw_analysis = {}
            
            if analysis_result.get('success'):
                result_data = analysis_result.get('result', {})
                medical_values = result_data.get('medical_values', {})
                detected_model = result_data.get('model', 'unknown')
                raw_analysis = result_data
                
            screen_analysis_doc = {
                "frame_result_id": ObjectId(frame_result_id),
                "screen_number": screen_number,
                "original_screen_number": original_screen_number,
                "screen_image_path": screen_image_path,
                "screen_image_path_for_vlm": screen_image_path,
                "detected_model": detected_model,
                "medical_values": medical_values,
                "raw_analysis": raw_analysis,
                "success": analysis_result.get('success', False),
                "error_message": analysis_result.get('error') if not analysis_result.get('success') else None,
                "analyzed_at": datetime.now(),
                "llm_model": llm_model,
                "has_glare": False,
                "has_occlusion": False
            }
            
            self.db.screen_analysis.update_one(
                {
                    "frame_result_id": ObjectId(frame_result_id),
                    "original_screen_number": original_screen_number,
                    "llm_model": llm_model
                },
                {"$set": screen_analysis_doc},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"❌ 儲存單張分析結果時發生錯誤: {e}")
            return False
    
    def query_screen_analysis(self, filters=None):
        """查詢螢幕分析結果"""
        if self.db is None:
            return []
        
        query_filters = filters or {}
        return list(self.db.screen_analysis.find(query_filters))
    
    def update_corrected_value(self, analysis_id, key, corrected_value):
        """更新特定欄位的訂正值"""
        from bson.objectid import ObjectId
        if self.db is None:
            return False
        
        try:
            # 使用 $set 更新 corrected_medical_values 字典中的特定 key
            self.db.screen_analysis.update_one(
                {"_id": ObjectId(analysis_id)},
                {
                    "$set": {
                        f"corrected_medical_values.{key}": corrected_value,
                        "is_corrected": True,
                        "last_corrected_at": datetime.now()
                    }
                }
            )
            return True
        except Exception as e:
            print(f"❌ 訂正失敗: {e}")
            return False

    def update_corrected_model(self, analysis_id, corrected_model):
        """更新訂正後的設備型號"""
        from bson.objectid import ObjectId
        if self.db is None:
            return False
        
        try:
            self.db.screen_analysis.update_one(
                {"_id": ObjectId(analysis_id)},
                {
                    "$set": {
                        "corrected_model": corrected_model,
                        "is_model_corrected": True,
                        "model_corrected_at": datetime.now()
                    }
                }
            )
            return True
        except Exception as e:
            print(f"❌ 型號訂正失敗: {e}")
            return False

    def update_quality_flags(self, analysis_id, has_glare=None, has_occlusion=None):
        """更新反光/遮擋人工標註欄位"""
        from bson.objectid import ObjectId
        if self.db is None:
            return False

        try:
            set_fields = {"quality_marked_at": datetime.now()}
            if has_glare is not None:
                set_fields["has_glare"] = bool(has_glare)
            if has_occlusion is not None:
                set_fields["has_occlusion"] = bool(has_occlusion)

            if len(set_fields) == 1:
                return False

            self.db.screen_analysis.update_one(
                {"_id": ObjectId(analysis_id)},
                {"$set": set_fields}
            )
            return True
        except Exception as e:
            print(f"❌ 更新畫面品質標註失敗: {e}")
            return False

    def create_dataset(self, name, description=""):
        """建立可重複使用的影片資料集。"""
        if self.db is None:
            return None
        now = datetime.now()
        dataset_id = str(uuid.uuid4())
        self.db.datasets.insert_one({
            "dataset_id": dataset_id,
            "name": name,
            "description": description,
            "created_at": now,
            "updated_at": now,
        })
        return dataset_id

    def get_dataset(self, dataset_id):
        """取得單一資料集。"""
        if self.db is None:
            return None
        return self.db.datasets.find_one({"dataset_id": dataset_id})

    def get_datasets(self):
        """取得資料集清單並附上影片數量。"""
        if self.db is None:
            return []
        datasets = list(self.db.datasets.find().sort("created_at", -1))
        for dataset in datasets:
            dataset["video_count"] = self.db.dataset_videos.count_documents({
                "dataset_id": dataset["dataset_id"]
            })
        return datasets

    def rename_dataset(self, dataset_id, new_name):
        """重新命名資料集。"""
        if self.db is None:
            return False
        result = self.db.datasets.update_one(
            {"dataset_id": dataset_id},
            {"$set": {
                "name": new_name,
                "updated_at": datetime.now(),
            }}
        )
        return result.matched_count > 0

    def delete_dataset(self, dataset_id):
        """刪除資料集與來源影片中繼資料，保留歷史分析結果。"""
        if self.db is None:
            return False
        dataset = self.db.datasets.find_one({"dataset_id": dataset_id})
        if not dataset:
            return False
        self.db.dataset_videos.delete_many({"dataset_id": dataset_id})
        self.db.dataset_ground_truths.delete_many({"dataset_id": dataset_id})
        result = self.db.datasets.delete_one({"dataset_id": dataset_id})
        return result.deleted_count > 0

    def add_dataset_video(self, dataset_id, original_name, stored_path, size_bytes):
        """新增資料集影片的中繼資料。"""
        if self.db is None:
            return None
        video_id = str(uuid.uuid4())
        self.db.dataset_videos.insert_one({
            "video_id": video_id,
            "dataset_id": dataset_id,
            "original_name": original_name,
            "stored_path": stored_path,
            "size_bytes": size_bytes,
            "created_at": datetime.now(),
        })
        self.db.datasets.update_one(
            {"dataset_id": dataset_id},
            {"$set": {"updated_at": datetime.now()}}
        )
        return video_id

    def get_dataset_videos(self, dataset_id):
        """依上傳順序取得資料集中的所有影片。"""
        if self.db is None:
            return []
        return list(
            self.db.dataset_videos.find({"dataset_id": dataset_id})
            .sort("created_at", 1)
        )

    def create_dataset_run(self, run_id, dataset_id, dataset_name, model, provider, videos):
        """建立資料集批次工作及每部影片的工作項目。"""
        if self.db is None:
            return False
        now = datetime.now()
        self.db.dataset_runs.insert_one({
            "run_id": run_id,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "provider": provider,
            "model": model,
            "status": "queued",
            "total_videos": len(videos),
            "completed_videos": 0,
            "failed_videos": 0,
            "created_at": now,
            "updated_at": now,
        })
        if videos:
            self.db.dataset_run_items.insert_many([
                {
                    "run_id": run_id,
                    "position": position,
                    "video_id": video["video_id"],
                    "video_name": video["original_name"],
                    "status": "queued",
                    "session_id": None,
                    "error": None,
                    "updated_at": now,
                }
                for position, video in enumerate(videos, start=1)
            ])
        return True

    def update_dataset_run(self, run_id, **fields):
        """更新批次工作的彙總狀態。"""
        if self.db is None:
            return False
        fields["updated_at"] = datetime.now()
        result = self.db.dataset_runs.update_one(
            {"run_id": run_id}, {"$set": fields}
        )
        return result.matched_count > 0

    def update_dataset_run_item(self, run_id, video_id, **fields):
        """更新批次工作中的單一影片狀態。"""
        if self.db is None:
            return False
        fields["updated_at"] = datetime.now()
        result = self.db.dataset_run_items.update_one(
            {"run_id": run_id, "video_id": video_id},
            {"$set": fields}
        )
        return result.matched_count > 0

    def get_dataset_run(self, run_id):
        """取得批次工作與各影片結果。"""
        if self.db is None:
            return None
        run = self.db.dataset_runs.find_one({"run_id": run_id})
        if not run:
            return None
        run["items"] = list(
            self.db.dataset_run_items.find({"run_id": run_id})
            .sort("position", 1)
        )
        return run

    def get_latest_dataset_run(self, dataset_id):
        """取得資料集最近一次的批次工作。"""
        if self.db is None:
            return None
        latest = self.db.dataset_runs.find_one(
            {"dataset_id": dataset_id}, {"run_id": 1}, sort=[("created_at", -1)]
        )
        return self.get_dataset_run(latest["run_id"]) if latest else None

    def _get_session_screens(self, session_id, llm_model=None):
        """取得 session 的所有成功螢幕分析，並附上擷取秒數。"""
        video = self.db.video_analysis.find_one({"session_id": session_id}, {"_id": 1})
        if not video:
            return []
        frames = list(self.db.frame_results.find(
            {"video_analysis_id": video["_id"]}, {"minute": 1}
        ))
        minute_by_frame = {frame["_id"]: frame.get("minute") for frame in frames}
        query = {
            "frame_result_id": {"$in": list(minute_by_frame)},
            "success": True,
        }
        if llm_model:
            query["llm_model"] = llm_model
        screens = list(self.db.screen_analysis.find(query, {
            "frame_result_id": 1,
            "detected_model": 1,
            "corrected_model": 1,
            "is_model_corrected": 1,
            "bbox": 1,
            "medical_values": 1,
            "corrected_medical_values": 1,
            "is_corrected": 1,
            "analyzed_at": 1,
            "llm_model": 1,
            "success": 1,
        }))
        for screen in screens:
            screen["minute"] = minute_by_frame.get(screen.get("frame_result_id"))
        return screens

    @staticmethod
    def _video_name_key(name):
        """去掉上傳時加的 uuid 前綴與副檔名，空白/底線視為相同，用來比對同一支影片。"""
        name = re.sub(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_', '', name or '')
        name = re.sub(r'\.(mp4|avi|mov|mkv|m4v|webm)$', '', name, flags=re.IGNORECASE)
        return re.sub(r'[\s_]+', '_', name).strip('_').lower()

    def get_ground_truth_candidates(self, session_id):
        """列出可指定的資料集影片（檔名相同的排前面），並附上 session 的涵蓋程度。"""
        from dataset_evaluation import build_entries

        if self.db is None:
            return None, "MongoDB 未連接"
        video_analysis = self.db.video_analysis.find_one(
            {"session_id": session_id},
            {"video_name": 1, "original_video_name": 1, "dataset_video_id": 1},
        )
        if not video_analysis:
            return None, "找不到該 Session"
        session_key = self._video_name_key(
            video_analysis.get("original_video_name") or video_analysis.get("video_name")
        )
        dataset_names = {
            item["dataset_id"]: item.get("name")
            for item in self.db.datasets.find({}, {"dataset_id": 1, "name": 1})
        }
        candidates = []
        for video in self.db.dataset_videos.find({}, {
            "video_id": 1, "dataset_id": 1, "original_name": 1, "ground_truth_session_id": 1,
        }):
            candidates.append({
                "video_id": video["video_id"],
                "video_name": video.get("original_name"),
                "dataset_id": video.get("dataset_id"),
                "dataset_name": dataset_names.get(video.get("dataset_id")),
                "name_match": bool(session_key) and self._video_name_key(video.get("original_name")) == session_key,
                "is_source_video": video["video_id"] == video_analysis.get("dataset_video_id"),
                "ground_truth_session_id": video.get("ground_truth_session_id"),
            })
        candidates.sort(key=lambda c: (
            not (c["is_source_video"] or c["name_match"]),
            c["dataset_name"] or "",
            c["video_name"] or "",
        ))
        screens = self._get_session_screens(session_id)
        entries, _ = build_entries(screens, use_corrections=True)
        summary = {
            "entry_count": len(entries),
            "capture_count": len({key[0] for key in entries}),
            "corrected_count": sum(
                1 for screen in screens
                if screen.get("is_corrected") or screen.get("is_model_corrected")
            ),
        }
        return {"candidates": candidates, "session": summary}, None

    def set_dataset_ground_truth(self, session_id, video_id=None):
        """將 session 設為資料集影片的標準答案。

        未指定 video_id 時使用資料集批次 session 的來源影片；
        指定時可把舊的（非資料集批次產生的）已訂正 session 指定給任一資料集影片。
        """
        from dataset_evaluation import build_entries, count_entries, serialize_entries

        if self.db is None:
            return None, "MongoDB 未連接"
        video_analysis = self.db.video_analysis.find_one({"session_id": session_id})
        if not video_analysis:
            return None, "找不到該 Session"
        video_id = video_id or video_analysis.get("dataset_video_id")
        if not video_id:
            return None, "請指定要作為哪一部資料集影片的標準答案"
        dataset_video = self.db.dataset_videos.find_one({"video_id": video_id})
        if not dataset_video:
            return None, "資料集影片不存在"
        current = self.get_ground_truth_video_for_session(session_id)
        if current and current["video_id"] != video_id:
            return None, (
                f"這個 Session 已是「{current.get('original_name')}」的標準答案，"
                "請先取消再指定給其他影片"
            )

        entries, duplicates = build_entries(
            self._get_session_screens(session_id), use_corrections=True
        )
        if not entries:
            return None, "這個 Session 沒有可辨識設備的分析結果，無法作為標準答案"

        now = datetime.now()
        self.db.dataset_ground_truths.update_one(
            {"video_id": video_id},
            {"$set": {
                "video_id": video_id,
                "dataset_id": dataset_video.get("dataset_id"),
                "session_id": session_id,
                "entries": serialize_entries(entries),
                "duplicate_count": duplicates,
                "updated_at": now,
            }, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        self.db.dataset_videos.update_one(
            {"video_id": video_id},
            {"$set": {
                "ground_truth_session_id": session_id,
                "ground_truth_set_at": now,
            }}
        )
        return {
            "video_id": video_id,
            "dataset_id": dataset_video.get("dataset_id"),
            "video_name": dataset_video.get("original_name"),
            "entry_count": count_entries(entries),
            "duplicate_count": duplicates,
        }, None

    def get_ground_truth_video_for_session(self, session_id):
        """取得以此 session 作為標準答案的資料集影片。"""
        if self.db is None:
            return None
        return self.db.dataset_videos.find_one({"ground_truth_session_id": session_id})

    def clear_dataset_ground_truth(self, video_id):
        """移除資料集影片的標準答案。"""
        if self.db is None:
            return False
        self.db.dataset_ground_truths.delete_one({"video_id": video_id})
        result = self.db.dataset_videos.update_one(
            {"video_id": video_id},
            {"$unset": {"ground_truth_session_id": "", "ground_truth_set_at": ""}}
        )
        return result.matched_count > 0

    def get_dataset_ground_truth_entries(self, video_id):
        """取得影片標準答案；標準答案 Session 仍存在時重新讀取以納入最新訂正。"""
        from dataset_evaluation import build_entries, deserialize_entries, serialize_entries

        record = self.db.dataset_ground_truths.find_one({"video_id": video_id})
        if not record:
            return None, None
        session_id = record.get("session_id")
        if self.db.video_analysis.count_documents({"session_id": session_id}, limit=1):
            entries, duplicates = build_entries(
                self._get_session_screens(session_id), use_corrections=True
            )
            if entries:
                self.db.dataset_ground_truths.update_one(
                    {"video_id": video_id},
                    {"$set": {
                        "entries": serialize_entries(entries),
                        "duplicate_count": duplicates,
                        "updated_at": datetime.now(),
                    }}
                )
                return entries, {"session_id": session_id, "snapshot_only": False}
        return (
            deserialize_entries(record.get("entries")),
            {"session_id": session_id, "snapshot_only": True},
        )

    def evaluate_session(self, session_id, llm_model=None, include_screen_annotations=False):
        """以來源影片（或此 session 被指定的影片）的標準答案比對 session 的模型原始輸出。"""
        from dataset_evaluation import (
            annotate_screens,
            build_entries,
            compare_entries,
            pair_entries,
        )

        if self.db is None:
            return None, "MongoDB 未連接"
        video_analysis = self.db.video_analysis.find_one({"session_id": session_id})
        if not video_analysis:
            return None, "找不到該 Session"
        video_id = video_analysis.get("dataset_video_id")
        if not video_id:
            # 被指定為標準答案的舊 session：比對它訂正前後的差異
            assigned_video = self.get_ground_truth_video_for_session(session_id)
            video_id = assigned_video and assigned_video.get("video_id")
        if not video_id:
            return None, "只有資料集批次分析產生、或已被指定為標準答案的 Session 可以驗證正確率"
        dataset_video = self.db.dataset_videos.find_one(
            {"video_id": video_id}, {"original_name": 1}
        )

        reference, gt_info = self.get_dataset_ground_truth_entries(video_id)
        if reference is None:
            return None, "這部資料集影片尚未設定標準答案"

        available_models = sorted(
            model for model in self.db.screen_analysis.distinct(
                "llm_model",
                {"frame_result_id": {"$in": self.db.frame_results.distinct(
                    "_id", {"video_analysis_id": video_analysis["_id"]}
                )}}
            ) if model
        )
        if llm_model not in available_models:
            session_model = video_analysis.get("llm_model")
            llm_model = session_model if session_model in available_models else None

        screens = self._get_session_screens(session_id, llm_model)
        candidate_groups, _ = build_entries(screens, use_corrections=False)
        reference, candidate = pair_entries(reference, candidate_groups)
        result = compare_entries(reference, candidate)
        if include_screen_annotations:
            result["screen_annotations"] = annotate_screens(screens, reference, candidate)
        result.update({
            "session_id": session_id,
            "video_id": video_id,
            "video_name": (dataset_video or {}).get("original_name")
            or video_analysis.get("original_video_name")
            or video_analysis.get("video_name"),
            "llm_model": llm_model,
            "available_llm_models": available_models,
            "ground_truth_session_id": gt_info["session_id"],
            "ground_truth_snapshot_only": gt_info["snapshot_only"],
            "is_ground_truth_session": gt_info["session_id"] == session_id,
            "evaluated_at": datetime.now(),
        })
        return result, None

    def evaluate_dataset_run(self, run_id):
        """重新計算批次工作中每部已完成影片的正確率與整體正確率。"""
        from dataset_evaluation import aggregate_evaluations

        if self.db is None:
            return None
        run = self.db.dataset_runs.find_one({"run_id": run_id})
        if not run:
            return None
        evaluations = []
        items = self.db.dataset_run_items.find({"run_id": run_id, "status": "completed"})
        for item in items:
            session_id = item.get("session_id")
            evaluation, error = (
                self.evaluate_session(session_id, run.get("model"))
                if session_id else (None, "缺少 session")
            )
            if evaluation:
                summary = {
                    key: value for key, value in evaluation.items()
                    if key not in ("mismatches", "extra_devices", "available_llm_models")
                }
                self.update_dataset_run_item(
                    run_id, item["video_id"], evaluation=summary, evaluation_error=None
                )
                evaluations.append(summary)
            else:
                self.db.dataset_run_items.update_one(
                    {"run_id": run_id, "video_id": item["video_id"]},
                    {"$set": {"evaluation_error": error, "updated_at": datetime.now()},
                     "$unset": {"evaluation": ""}}
                )
        aggregate = aggregate_evaluations(evaluations) if evaluations else None
        if aggregate:
            aggregate["evaluated_at"] = datetime.now()
        self.update_dataset_run(run_id, evaluation=aggregate)
        return aggregate

    def get_all_videos(self, source_filter=None, search_query=None):
        """取得所有影片分析記錄 (支援搜尋)"""
        if self.db is None:
            return []
        
        query = {}
        if source_filter:
            if source_filter == 'stream':
                # 篩選串流來源 (包含 stream 標記或 video_path 包含 RTSP/Camera)
                query = {
                    "$or": [
                        {"source": "stream"},
                        {"video_path": {"$regex": "^(RTSP|Camera):"}}
                    ]
                }
            elif source_filter == 'manual':
                # 篩選非串流來源
                query = {
                    "$and": [
                        {"source": {"$ne": "stream"}},
                        {"video_path": {"$not": {"$regex": "^(RTSP|Camera):"}}}
                    ]
                }
        
        # 新增搜尋邏輯：針對名稱、路徑、ID 或攝影機名稱進行不分大小寫的模糊搜尋
        if search_query:
            search_condition = {
                "$or": [
                    {"video_name": {"$regex": search_query, "$options": "i"}},
                    {"video_path": {"$regex": search_query, "$options": "i"}},
                    {"session_id": {"$regex": search_query, "$options": "i"}},
                    {"camera_name": {"$regex": search_query, "$options": "i"}},
                    {"patient_id": {"$regex": search_query, "$options": "i"}},
                    {"task_name": {"$regex": search_query, "$options": "i"}}
                ]
            }
            if query:
                query = {"$and": [query, search_condition]}
            else:
                query = search_condition
        
        videos = list(self.db.video_analysis.find(query).sort('timestamp', -1))
        if source_filter != "stream":
            dataset_video_ids = list({
                video["dataset_video_id"] for video in videos
                if video.get("dataset_video_id")
            })
            session_ids = [video.get("session_id") for video in videos]
            projection = {
                "video_id": 1, "dataset_id": 1, "original_name": 1,
                "ground_truth_session_id": 1,
            }
            dataset_videos = {
                item["video_id"]: item
                for item in self.db.dataset_videos.find(
                    {"$or": [
                        {"video_id": {"$in": dataset_video_ids}},
                        {"ground_truth_session_id": {"$in": session_ids}},
                    ]},
                    projection,
                )
            }
            ground_truth_video_by_session = {
                item["ground_truth_session_id"]: item
                for item in dataset_videos.values()
                if item.get("ground_truth_session_id")
            }
            dataset_names = {
                item["dataset_id"]: item.get("name")
                for item in self.db.datasets.find(
                    {"dataset_id": {"$in": list({
                        item.get("dataset_id") for item in dataset_videos.values()
                    })}},
                    {"dataset_id": 1, "name": 1},
                )
            }
            for video in videos:
                source_video = dataset_videos.get(video.get("dataset_video_id"))
                video["dataset_video_exists"] = source_video is not None
                video["dataset_video_name"] = (source_video or {}).get("original_name")
                video["dataset_name"] = dataset_names.get((source_video or {}).get("dataset_id"))
                video["is_ground_truth"] = video.get("session_id") in ground_truth_video_by_session
                ground_truth_video = ground_truth_video_by_session.get(video.get("session_id"))
                video["ground_truth_video_name"] = (ground_truth_video or {}).get("original_name")
                video["ground_truth_dataset_name"] = dataset_names.get(
                    (ground_truth_video or {}).get("dataset_id")
                )
                video["can_evaluate"] = bool(
                    source_video and source_video.get("ground_truth_session_id")
                )
            return videos

        # 資料庫卡片以病患為單位顯示；各次 session 仍保留供追溯，
        # 詳情查詢會透過代表 session 聚合該病患的全部歷史資料。
        grouped = {}
        for video in videos:
            patient_id = str(
                video.get("patient_id") or video.get("task_name") or ""
            ).strip()
            group_key = f"patient:{patient_id}" if patient_id else f"session:{video.get('session_id')}"

            if group_key not in grouped:
                grouped[group_key] = video
                grouped[group_key]["patient_id"] = patient_id or None
                grouped[group_key]["session_ids"] = [video.get("session_id")]
                grouped[group_key]["session_count"] = 1
                continue

            group = grouped[group_key]
            group["session_ids"].append(video.get("session_id"))
            group["session_count"] += 1
            group["total_frames"] = (
                int(group.get("total_frames") or 0)
                + int(video.get("total_frames") or 0)
            )

        return list(grouped.values())

    def update_video_name(self, session_id, new_name):
        """更新影片/資料名稱"""
        if self.db is None:
            return False
        try:
            result = self.db.video_analysis.update_one(
                {"session_id": session_id},
                {"$set": {"video_name": new_name}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"❌ 更新名稱失敗: {e}")
            return False

    def _normalize_analysis_file_path(self, path_value):
        """正規化並限制只允許刪除 video_screen_analysis 目錄內檔案"""
        if not isinstance(path_value, str) or not path_value.strip():
            return None

        analysis_root = Path("video_screen_analysis").resolve()
        candidate = Path(path_value.strip())
        if not candidate.is_absolute():
            candidate = (Path(".") / candidate).resolve()
        else:
            candidate = candidate.resolve()

        try:
            candidate.relative_to(analysis_root)
        except ValueError:
            return None
        return candidate

    def _remove_empty_analysis_dirs(self, start_dir):
        """從葉節點往上刪除空資料夾，最多到 video_screen_analysis"""
        analysis_root = Path("video_screen_analysis").resolve()
        current = start_dir

        while True:
            try:
                current.relative_to(analysis_root)
            except ValueError:
                break

            if current == analysis_root:
                break

            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent

    def _delete_analysis_files(self, file_paths):
        """刪除分析圖像檔，忽略不存在檔案並清理空資料夾"""
        deleted_files = 0
        skipped_paths = 0
        failed_files = 0
        touched_dirs = set()

        for raw_path in file_paths:
            normalized = self._normalize_analysis_file_path(raw_path)
            if normalized is None:
                skipped_paths += 1
                continue

            touched_dirs.add(normalized.parent)
            if not normalized.exists():
                continue
            if not normalized.is_file():
                skipped_paths += 1
                continue

            try:
                normalized.unlink()
                deleted_files += 1
            except Exception as e:
                failed_files += 1
                print(f"⚠️ 刪除圖像檔失敗: {normalized} ({e})")

        for folder in sorted(touched_dirs, key=lambda p: len(p.parts), reverse=True):
            self._remove_empty_analysis_dirs(folder)

        return {
            "deleted_files": deleted_files,
            "skipped_paths": skipped_paths,
            "failed_files": failed_files
        }

    def _delete_stream_analysis_dir(self, session_id):
        """刪除指定 session 的串流分析目錄，清除可能殘留的孤兒檔"""
        if not isinstance(session_id, str) or len(session_id) < 8:
            return {"removed": False, "reason": "invalid_session_id"}

        analysis_root = Path("video_screen_analysis").resolve()
        stream_dir_name = f"stream_{session_id[:8]}"
        target_dir = (analysis_root / stream_dir_name).resolve()

        # 僅允許刪除 video_screen_analysis 下、且名稱符合 stream_<id8> 的目錄
        try:
            target_dir.relative_to(analysis_root)
        except ValueError:
            return {"removed": False, "reason": "out_of_scope"}

        if target_dir.name != stream_dir_name:
            return {"removed": False, "reason": "name_mismatch"}

        if not target_dir.exists():
            return {"removed": False, "reason": "not_found"}
        if not target_dir.is_dir():
            return {"removed": False, "reason": "not_a_directory"}

        try:
            shutil.rmtree(target_dir)
            return {"removed": True, "reason": "removed"}
        except Exception as e:
            print(f"⚠️ 刪除串流分析目錄失敗: {target_dir} ({e})")
            return {"removed": False, "reason": "failed"}

    def delete_video_record(self, session_id):
        """刪除影片記錄及其相關資料"""
        if self.db is None:
            return False
        try:
            # 1. 找到對應的 video_analysis_id
            video = self.db.video_analysis.find_one({"session_id": session_id})
            if not video:
                return False
            
            video_id = video["_id"]
            is_stream_source = (
                video.get("source") == "stream" or
                str(video.get("video_path", "")).startswith(("RTSP:", "Camera:"))
            )
            
            # 2. 找到所有相關的 frame_results
            frames = list(self.db.frame_results.find({"video_analysis_id": video_id}))
            frame_ids = [f["_id"] for f in frames]

            # 2.1 收集待刪除圖像路徑（在刪 DB 前先收集）
            file_paths = set()
            for frame in frames:
                frame_path = frame.get("original_image_path")
                if frame_path:
                    file_paths.add(frame_path)
            
            # 3. 刪除相關的 screen_analysis (如果有)
            if frame_ids:
                screens = list(self.db.screen_analysis.find(
                    {"frame_result_id": {"$in": frame_ids}},
                    {"screen_image_path": 1, "screen_image_path_for_vlm": 1}
                ))
                for screen in screens:
                    screen_path = screen.get("screen_image_path")
                    screen_path_for_vlm = screen.get("screen_image_path_for_vlm")
                    if screen_path:
                        file_paths.add(screen_path)
                    if screen_path_for_vlm:
                        file_paths.add(screen_path_for_vlm)
                self.db.screen_analysis.delete_many({"frame_result_id": {"$in": frame_ids}})
            
            # 4. 刪除 frame_results
            self.db.frame_results.delete_many({"video_analysis_id": video_id})
            
            # 5. 刪除 video_analysis 主記錄
            self.db.video_analysis.delete_one({"_id": video_id})

            # 6. 刪除檔案系統中的相關圖像檔
            file_cleanup = self._delete_analysis_files(file_paths)

            # 7. 若為串流來源，額外刪除 stream_<session前8碼> 目錄以清理孤兒檔
            stream_dir_cleanup = {"removed": False, "reason": "not_stream_source"}
            if is_stream_source:
                stream_dir_cleanup = self._delete_stream_analysis_dir(session_id)
            
            print(
                f"🗑️ 已刪除 Session ID: {session_id} 的所有相關資料 "
                f"(圖像檔刪除 {file_cleanup['deleted_files']}、"
                f"跳過 {file_cleanup['skipped_paths']}、失敗 {file_cleanup['failed_files']}；"
                f"串流目錄清理: {stream_dir_cleanup['reason']})"
            )
            return True
        except Exception as e:
            print(f"❌ 刪除資料失敗: {e}")
            return False

    def delete_stream_session(self, session_id):
        """僅刪除串流會話的卡面配置，保留所有歷史分析數據"""
        if self.db is None:
            return False
        try:
            # 僅刪除 stream_sessions 記錄，不呼叫 delete_video_record，
            # 這樣該鏡頭過去產生的所有分析結果與歷史資料都會完整保留在資料庫中。
            result = self.db.stream_sessions.delete_one({"session_id": session_id})
            print(f"🗑️ 已從面板移除串流會話卡面: {session_id} (歷史數據已保留)")
            return True
        except Exception as e:
            print(f"❌ 移除串流會話卡面失敗: {e}")
            return False

    def start_medical_monitoring(self):
        """啟動醫療監控服務"""
        try:
            print("🔄 正在啟動醫療監控服務...")
            
            # 檢查是否已有醫療監控進程在運行
            import subprocess
            try:
                result = subprocess.run(['pgrep', '-f', 'medical_mongodb_exporter.py'],
                                      capture_output=True, text=True)
                if result.stdout.strip():
                    print("📊 醫療監控服務已在運行")
                    return True
            except:
                pass
            
            # 啟動醫療監控服務
            self.medical_exporter_process = subprocess.Popen([
                'python3', 'medical_mongodb_exporter.py',
                '--mongodb-uri', 'mongodb://localhost:27017',
                '--database', self.database_name,
                '--port', '8000',
                '--interval', '15',
                '--log-level', 'INFO'
            ], stdout=open('medical_exporter_auto.log', 'w'),
               stderr=subprocess.STDOUT)
            
            # 等待服務啟動
            import time
            time.sleep(5)
            
            # 檢查服務是否正常運行
            try:
                import urllib.request
                urllib.request.urlopen('http://localhost:8000/metrics', timeout=5)
                print("✅ 醫療監控服務在 http://localhost:8000/metrics 啟動成功")
                
                # 保存 PID
                with open('medical_exporter_auto.pid', 'w') as f:
                    f.write(str(self.medical_exporter_process.pid))
                
                return True
            except:
                print("⚠️  醫療監控服務啟動失敗，但 MongoDB 正常運行")
                return False
                
        except Exception as e:
            print(f"⚠️  啟動醫療監控服務時發生錯誤: {e}")
            return False
    
    def stop_medical_monitoring(self):
        """停止醫療監控服務"""
        try:
            # 通過 PID 檔案停止
            if os.path.exists('medical_exporter_auto.pid'):
                with open('medical_exporter_auto.pid', 'r') as f:
                    pid = int(f.read().strip())
                try:
                    os.kill(pid, 15)  # SIGTERM
                    import time
                    time.sleep(2)
                    os.kill(pid, 0)  # 檢查進程是否還存在
                except (ProcessLookupError, OSError):
                    pass  # 進程已經結束
                os.remove('medical_exporter_auto.pid')
            
            # 停止進程對象
            if self.medical_exporter_process:
                try:
                    self.medical_exporter_process.terminate()
                    self.medical_exporter_process.wait(timeout=5)
                except:
                    try:
                        self.medical_exporter_process.kill()
                    except:
                        pass
                self.medical_exporter_process = None
            
            # 通過進程名稱停止（備用方法）
            try:
                subprocess.run(['pkill', '-f', 'medical_mongodb_exporter.py'],
                             capture_output=True)
            except:
                pass
                
            print("🛑 醫療監控服務已停止")
            
        except Exception as e:
            print(f"⚠️  停止醫療監控服務時發生錯誤: {e}")


# 全域 MongoDB 管理器實例
mongo_manager = LocalMongoDBManager()


def start_local_mongodb():
    """啟動本地 MongoDB 的便捷函數"""
    return mongo_manager.start_mongodb()


def get_mongo_manager():
    """取得 MongoDB 管理器實例"""
    return mongo_manager
