#!/usr/bin/env python3
"""
MongoDB 醫療數據 Prometheus 導出器 (Custom Collector Version)

為醫師提供即時的患者數據監控儀表板，將 MongoDB 中的醫療數據轉換為 Prometheus 時間序列格式。
使用 Custom Collector 模式以支援明確的時間戳記 (Explicit Timestamps)，解決數據延遲問題。
"""

import os
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import traceback

import pymongo
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, AutoReconnect
from prometheus_client import start_http_server
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily, REGISTRY

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medical_exporter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MedicalCustomCollector:
    """
    Prometheus 自定義收集器
    
    負責從 MongoDB 抓取數據並轉換為帶有時間戳的 Prometheus 指標。
    """
    
    def __init__(self, db, logger_instance=None):
        self.db = db
        self.logger = logger_instance or logging.getLogger(__name__)
        
        # [新增] 執行緒鎖與快取
        self._lock = threading.Lock()
        self._cache = []
        
        # 醫療設備正常值範圍（用於異常檢測邏輯，雖在此不一定報警，但保留參考）
        self.normal_ranges = {
            'heart_rate': (50, 150),
            'spo2': (85, 100),
            'respiration_rate': (10, 35),
            'blood_pressure_systolic': (80, 180),
            'blood_pressure_diastolic': (50, 110),
            'FiO2': (21, 100),
            'PEEP': (0, 25),
            'PIP': (10, 50),
            'rSO2_left': (50, 85),
            'rSO2_right': (50, 85),
        }

    def _safe_float_conversion(self, value: Any) -> Optional[float]:
        """安全的數值轉換"""
        if value is None:
            return None
        
        try:
            # 處理字串格式的血壓值 (例如: "120/80")
            if isinstance(value, str):
                # 血壓格式
                if '/' in value:
                    systolic, diastolic = value.split('/')
                    return float(systolic.strip()), float(diastolic.strip())
                
                # 移除非數字字符
                value = ''.join(c for c in value if c.isdigit() or c in '.-')
                if not value:
                    return None
            
            return float(value)
        
        except (ValueError, TypeError):
            # self.logger.warning(f"無法轉換數值: {value}")
            return None

    def _get_latest_medical_data(self) -> List[Dict]:
        """從 MongoDB 獲取最新的醫療數據"""
        try:
            # 查詢最近 10 分鐘的數據
            cutoff_time = datetime.now() - timedelta(minutes=10)
            
            # 聚合查詢：獲取每個會話的最新醫療數據
            pipeline = [
                # 過濾最近的分析數據，且 source 必須為 "stream"
                {
                    "$match": {
                        "analyzed_at": {"$gte": cutoff_time},
                        "success": True,
                        "medical_values": {"$ne": {}, "$exists": True}
                    }
                },
                
                # 按會話和設備類型排序，取最新的
                {"$sort": {"analyzed_at": -1}},
                
                # 按會話ID和設備模型分組，取最新數據
                {
                    "$group": {
                        "_id": {
                            "session_id": "$session_id",
                            "device_model": "$detected_model"
                        },
                        "latest_data": {"$first": "$$ROOT"}
                    }
                },
                
                # 關聯查詢 frame_results 獲取會話資訊
                {
                    "$lookup": {
                        "from": "frame_results",
                        "localField": "latest_data.frame_result_id",
                        "foreignField": "_id",
                        "as": "frame_info"
                    }
                },
                
                # 關聯查詢 video_analysis 獲取會話詳情
                {
                    "$lookup": {
                        "from": "video_analysis",
                        "localField": "frame_info.video_analysis_id",
                        "foreignField": "_id",
                        "as": "session_info"
                    }
                },
                
                # 投影需要的欄位
                {
                    "$project": {
                        "session_id": {"$arrayElemAt": ["$session_info.session_id", 0]},
                        "device_model": "$latest_data.detected_model",
                        "medical_values": "$latest_data.medical_values",
                        "analyzed_at": "$latest_data.analyzed_at",
                        "video_name": {"$arrayElemAt": ["$session_info.video_name", 0]},
                        # 若無 video_name，使用 session_id
                        "fallback_video_name": {"$ifNull": [{"$arrayElemAt": ["$session_info.video_name", 0]}, "$latest_data.session_id"]}
                    }
                }
            ]
            
            results = list(self.db.screen_analysis.aggregate(pipeline))
            self.logger.debug(f"查詢到 {len(results)} 筆即時醫療數據")
            return results
            
        except Exception as e:
            self.logger.error(f"查詢 MongoDB 時發生錯誤: {e}")
            self.logger.error(traceback.format_exc())
            return []

    def start_background_loop(self, interval):
        """背景執行緒：定期更新快取數據"""
        self.logger.info(f"啟動背景數據更新，間隔: {interval} 秒")
        while True:
            try:
                # 1. 查 DB (耗時操作，不用鎖)
                new_data = self._get_latest_medical_data()
                
                # 2. 更新快取 (快速操作，要鎖)
                with self._lock:
                    self._cache = new_data
                    
                self.logger.debug(f"快取已更新，共 {len(new_data)} 筆數據")
            except Exception as e:
                self.logger.error(f"背景更新失敗: {e}")
            
            # 3. 等待
            time.sleep(interval)

    def collect(self):
        """
        Prometheus 抓取指標時會呼叫此方法
        在這裡我們從快取讀取數據並回傳帶有時間戳的指標
        """
        # 1. 從快取讀取數據
        data_list = []
        with self._lock:
            data_list = self._cache[:]  # 淺拷貝，避免迭代時被修改
        
        # 2. 定義指標家族 (Metric Families)
        
        # === 生理指標 ===
        metric_hr = GaugeMetricFamily(
            'patient_heart_rate_bpm', 
            '患者心率 (每分鐘心跳數)', 
            labels=['session_id', 'device_model', 'patient_id']
        )
        metric_spo2 = GaugeMetricFamily(
            'patient_spo2_percentage', 
            '患者血氧飽和度 (%)', 
            labels=['session_id', 'device_model', 'patient_id']
        )
        metric_rr = GaugeMetricFamily(
            'patient_respiration_rate_per_minute', 
            '患者呼吸頻率 (每分鐘)', 
            labels=['session_id', 'device_model', 'patient_id']
        )
        metric_bp_sys = GaugeMetricFamily(
            'patient_blood_pressure_systolic_mmhg', 
            '患者收縮壓 (mmHg)', 
            labels=['session_id', 'device_model', 'patient_id']
        )
        metric_bp_dia = GaugeMetricFamily(
            'patient_blood_pressure_diastolic_mmhg', 
            '患者舒張壓 (mmHg)', 
            labels=['session_id', 'device_model', 'patient_id']
        )
        
        # === 呼吸機參數 ===
        metric_fio2 = GaugeMetricFamily(
            'ventilator_fio2_percentage', 
            '吸氧濃度 (%)', 
            labels=['session_id', 'device_model', 'patient_id']
        )
        metric_peep = GaugeMetricFamily(
            'ventilator_peep_cmh2o', 
            '呼氣末正壓 (cmH2O)', 
            labels=['session_id', 'device_model', 'patient_id']
        )
        metric_pip = GaugeMetricFamily(
            'ventilator_pip_cmh2o', 
            '吸氣壓峰值 (cmH2O)', 
            labels=['session_id', 'device_model', 'patient_id']
        )
        metric_pinsp = GaugeMetricFamily(
            'ventilator_pinsp_cmh2o', 
            '吸氣壓力 (cmH2O)', 
            labels=['session_id', 'device_model', 'patient_id']
        )
        
        # === 腦氧監測 ===
        metric_rso2_left = GaugeMetricFamily(
            'cerebral_rso2_left_percentage', 
            '左腦氧飽和度 (%)', 
            labels=['session_id', 'device_model', 'patient_id']
        )
        metric_rso2_right = GaugeMetricFamily(
            'cerebral_rso2_right_percentage', 
            '右腦氧飽和度 (%)', 
            labels=['session_id', 'device_model', 'patient_id']
        )

        # 3. 填充數據
        for item in data_list:
            try:
                session_id = item.get('session_id', 'unknown')
                device_model = item.get('device_model', 'unknown')
                medical_values = item.get('medical_values', {})
                video_name = item.get('video_name')
                
                # patient_id 優先使用 video_name，否則用 session_id
                patient_id = video_name if video_name else session_id
                
                labels = [session_id, device_model, patient_id]
                
                # 獲取時間戳記 (關鍵步驟)
                analyzed_at = item.get('analyzed_at')
                timestamp = None
                if isinstance(analyzed_at, datetime):
                    timestamp = analyzed_at.timestamp()
                
                if not timestamp:
                    continue

                # Helper function to add metric if value exists
                def add_if_exists(metric_family, key, transform_func=None):
                    if key in medical_values:
                        val = self._safe_float_conversion(medical_values[key])
                        if val is not None:
                            metric_family.add_metric(labels, val, timestamp=timestamp)

                # --- 填充各項指標 ---
                add_if_exists(metric_hr, 'heart_rate')
                add_if_exists(metric_spo2, 'spo2')
                add_if_exists(metric_rr, 'respiration_rate')
                
                # 血壓特殊處理
                if 'blood_pressure' in medical_values:
                    bp_val = medical_values['blood_pressure']
                    if isinstance(bp_val, str) and '/' in bp_val:
                        try:
                            s, d = bp_val.split('/')
                            sys_val = self._safe_float_conversion(s)
                            dia_val = self._safe_float_conversion(d)
                            if sys_val: metric_bp_sys.add_metric(labels, sys_val, timestamp=timestamp)
                            if dia_val: metric_bp_dia.add_metric(labels, dia_val, timestamp=timestamp)
                        except:
                            pass

                # 呼吸機
                add_if_exists(metric_fio2, 'FiO2')
                add_if_exists(metric_peep, 'PEEP')
                add_if_exists(metric_pip, 'PIP')
                add_if_exists(metric_pinsp, 'Pinsp')
                
                # 腦氧
                add_if_exists(metric_rso2_left, 'rSO2_left')
                add_if_exists(metric_rso2_right, 'rSO2_right')

            except Exception as e:
                self.logger.error(f"處理單筆數據時發生錯誤: {e}")

        # 4. Yield 所有指標家族
        yield metric_hr
        yield metric_spo2
        yield metric_rr
        yield metric_bp_sys
        yield metric_bp_dia
        yield metric_fio2
        yield metric_peep
        yield metric_pip
        yield metric_pinsp
        yield metric_rso2_left
        yield metric_rso2_right


class MedicalMongoExporter:
    """MongoDB 醫療數據 Prometheus 導出器管理類別"""
    
    def __init__(self, 
                 mongodb_uri: str = "mongodb://localhost:27017",
                 database_name: str = "medical_monitor_db",
                 metrics_port: int = 8000,
                 interval: int = 15):
        self.mongodb_uri = mongodb_uri
        self.database_name = database_name
        self.metrics_port = metrics_port
        self.interval = interval
        
        self.client = None
        self.db = None
    
    def connect_to_mongodb(self) -> bool:
        """連接到 MongoDB"""
        try:
            logger.info(f"正在連接到 MongoDB: {self.mongodb_uri}")
            self.client = MongoClient(
                self.mongodb_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            
            # 測試連接
            self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            
            logger.info(f"成功連接到 MongoDB 數據庫: {self.database_name}")
            return True
            
        except (ServerSelectionTimeoutError, AutoReconnect) as e:
            logger.error(f"MongoDB 連接失敗: {e}")
            return False
        except Exception as e:
            logger.error(f"連接 MongoDB 時發生未知錯誤: {e}")
            return False
    
    def start(self):
        """啟動 Exporter"""
        if not self.connect_to_mongodb():
            return
        
        # 註冊自定義收集器
        try:
            # 清除默認的收集器 (可選，但為了乾淨通常保留系統指標)
            # REGISTRY.unregister(REGISTRY._names_to_collectors['python_gc_objects_collected_total']) 
            
            # 實例化並註冊我們的收集器
            collector = MedicalCustomCollector(self.db, logger)
            REGISTRY.register(collector)
            logger.info("已註冊 MedicalCustomCollector")
            
            # [新增] 啟動背景執行緒
            t = threading.Thread(target=collector.start_background_loop, args=(self.interval,))
            t.daemon = True
            t.start()
            
            # 啟動 HTTP 伺服器
            start_http_server(self.metrics_port)
            logger.info(f"Prometheus Exporter 已在 port {self.metrics_port} 啟動")
            
            # 保持運行
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("停止 Exporter")
        except Exception as e:
            logger.error(f"Exporter 運行錯誤: {e}")
            logger.error(traceback.format_exc())
        finally:
            if self.client:
                self.client.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical MongoDB Prometheus Exporter")
    parser.add_argument("--mongodb-uri", default="mongodb://localhost:27017",
                       help="MongoDB 連接 URI")
    parser.add_argument("--database", default="medical_monitor_db",
                       help="數據庫名稱")
    parser.add_argument("--port", type=int, default=8000,
                       help="Prometheus metrics 端口")
    parser.add_argument("--interval", type=int, default=15,
                       help="數據更新頻率(秒)")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日誌等級")
    
    args = parser.parse_args()
    
    # 設定日誌等級
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    exporter = MedicalMongoExporter(
        mongodb_uri=args.mongodb_uri,
        database_name=args.database,
        metrics_port=args.port,
        interval=args.interval
    )
    
    exporter.start()

if __name__ == "__main__":
    main()
