#!/usr/bin/env python3
"""
MongoDB 管理模組 (檔案系統儲存版本)
自動啟動、停止和管理本地 MongoDB 實例，圖片存放在檔案系統
"""

import os
import sys
import time
import signal
import shutil
import subprocess
import atexit
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
            self.connect_to_mongodb()
            return True
        
        # 創建配置檔案（如果不存在）
        if not self.config_file.exists():
            self.create_config_file()
        
        try:
            print("🚀 正在啟動 MongoDB...")
            
            # 啟動 MongoDB 進程
            self.mongod_process = subprocess.Popen([
                self._mongod_executable(), '--config', str(self.config_file)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self._we_started_mongodb = True
            
            # 等待 MongoDB 啟動
            max_wait = 30
            for i in range(max_wait):
                if self.is_mongodb_running():
                    print("✅ MongoDB 啟動成功！")
                    self.connect_to_mongodb()
                    self.setup_database()
                    # 自動啟動醫療監控服務
                    self.start_medical_monitoring()
                    return True
                time.sleep(1)
                print(f"⏳ 等待 MongoDB 啟動... ({i+1}/{max_wait})")
            
            print("❌ MongoDB 啟動超時")
            return False
            
        except FileNotFoundError:
            print("❌ 找不到 mongod 命令，請確保 MongoDB 已安裝到 conda 環境")
            print("💡 安裝命令: conda install -c conda-forge mongodb")
            return False
        except Exception as e:
            print(f"❌ 啟動 MongoDB 時發生錯誤: {e}")
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
            collections = ['video_analysis', 'frame_results', 'screen_analysis', 'stream_sessions']
            for collection_name in collections:
                if collection_name not in self.db.list_collection_names():
                    self.db.create_collection(collection_name)
            
            # 創建索引
            self.create_indexes()
            print("🗂️  數據庫設置完成")
            
        except Exception as e:
            print(f"⚠️  設置數據庫時發生錯誤: {e}")
    
    def create_indexes(self):
        """創建必要的索引"""
        try:
            # VideoAnalysis 索引
            self.db.video_analysis.create_index("session_id")
            self.db.video_analysis.create_index([("timestamp", -1)])
            self.db.video_analysis.create_index("video_name")
            
            # FrameResult 索引
            self.db.frame_results.create_index([("video_analysis_id", 1), ("minute", 1)])
            self.db.frame_results.create_index([("processed_at", -1)])
            
            # ScreenAnalysis 索引
            self.db.screen_analysis.create_index("frame_result_id")
            self.db.screen_analysis.create_index("detected_model")
            self.db.screen_analysis.create_index("medical_values.heart_rate")
            self.db.screen_analysis.create_index([("analyzed_at", -1)])

            # StreamSessions 索引
            self.db.stream_sessions.create_index("session_id", unique=True)
            self.db.stream_sessions.create_index("camera_name")
            self.db.stream_sessions.create_index("rtsp_url")
            
            print("📊 索引創建完成")
            
        except Exception as e:
            print(f"⚠️  創建索引時發生錯誤: {e}")
    
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
    
    def save_analysis_result(self, video_path, session_id, results, source="manual", llm_model="unknown", camera_name=None):
        """儲存分析結果到 MongoDB (檔案路徑版本)"""
        if self.db is None:
            raise Exception("MongoDB 連接未建立")
        
        try:
            # 自動獲取串流會話的任務名稱
            task_name = None
            if source == "stream":
                try:
                    stream_sess = self.db.stream_sessions.find_one({"session_id": session_id})
                    if stream_sess:
                        task_name = stream_sess.get("task_name")
                except Exception as e:
                    print(f"⚠️ 獲取串流會話任務名稱失敗: {e}")

            # 1. 更新或建立 VideoAnalysis 記錄 (使用 upsert 避免重複)
            video_analysis_doc = {
                "video_path": video_path,
                "video_name": Path(video_path).stem,
                "session_id": session_id,
                "timestamp": datetime.now(),
                "total_frames": len(results),
                "status": "processing",
                "source": source,  # manual, stream 等
                "llm_model": llm_model,  # 記錄使用的 LLM 模型
                "camera_name": camera_name, # 記錄攝影機名稱 (用於串流)
                "task_name": task_name # 記錄任務名稱 (用於串流)
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
                        "detected_model": detected_model,
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

    def _session_screens_lookup_pipeline(self, session_id):
        """session → frame → screen 的共用聚合前段"""
        return [
            {"$match": {"session_id": session_id}},
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
                    {"$sort": {"screens.analyzed_at": -1}},
                    {"$group": {
                        "_id": None,
                        "count": {"$sum": 1},
                        "errors": {"$push": "$screens.error_message"},
                        "last_error": {"$first": "$screens.error_message"},
                    }},
                ],
                "success": [
                    {"$match": {"screens.success": True}},
                    {"$count": "count"},
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
            raw_errors = [e for e in (failed_block.get("errors") or []) if e]
            sample_errors = list(dict.fromkeys(raw_errors))[:3]
            return {
                "failed_count": failed_count,
                "success_count": success_count,
                "last_error": failed_block.get("last_error"),
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
                "llm_model": "$screens.llm_model"
            }}
        )
        
        return list(self.db.video_analysis.aggregate(pipeline))
    
    def get_session_models(self, session_id):
        """取得特定 session 中使用過的所有模型"""
        if self.db is None:
            return []
        
        pipeline = [
            {"$match": {"session_id": session_id}},
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
                {"$match": {"session_id": session_id}},
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

    def save_stream_session(self, session_id, camera_name, rtsp_url, llm_model="unknown", task_name=None):
        """儲存一個新的串流監測會話"""
        if self.db is None:
            raise Exception("MongoDB 連接未建立")

        try:
            session_doc = {
                "session_id": session_id,
                "camera_name": camera_name,
                "rtsp_url": rtsp_url,
                "llm_model": llm_model,
                "task_name": task_name,
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

    def save_camera_template(self, template_id, camera_name, rtsp_url, provider, model, api_key=None):
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
                    {"camera_name": {"$regex": search_query, "$options": "i"}}
                ]
            }
            if query:
                query = {"$and": [query, search_condition]}
            else:
                query = search_condition
        
        return list(self.db.video_analysis.find(query).sort('timestamp', -1))

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
            
            # 2. 找到所有相關的 frame_results
            frames = list(self.db.frame_results.find({"video_analysis_id": video_id}))
            frame_ids = [f["_id"] for f in frames]
            
            # 3. 刪除相關的 screen_analysis (如果有)
            if frame_ids:
                self.db.screen_analysis.delete_many({"frame_result_id": {"$in": frame_ids}})
            
            # 4. 刪除 frame_results
            self.db.frame_results.delete_many({"video_analysis_id": video_id})
            
            # 5. 刪除 video_analysis 主記錄
            self.db.video_analysis.delete_one({"_id": video_id})
            
            print(f"🗑️ 已刪除 Session ID: {session_id} 的所有相關資料")
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
