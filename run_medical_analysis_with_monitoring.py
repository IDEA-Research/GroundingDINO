#!/usr/bin/env python3
"""
整合的醫療影片分析與監控系統

自動啟動影片處理和監控功能，無需分別啟動
"""

import os
import sys
import time
import subprocess
import argparse
import signal
import atexit
from dotenv import load_dotenv

load_dotenv()

def log_info(msg):
    print(f"[INFO] {msg}")

def log_success(msg):
    print(f"[SUCCESS] ✅ {msg}")

def log_error(msg):
    print(f"[ERROR] ❌ {msg}")

class IntegratedMedicalSystem:
    """整合的醫療分析系統"""
    
    def __init__(self):
        self.monitoring_process = None
        self.cleanup_registered = False
    
    def register_cleanup(self):
        """註冊清理函數"""
        if not self.cleanup_registered:
            atexit.register(self.cleanup)
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            self.cleanup_registered = True
    
    def _signal_handler(self, signum, frame):
        """信號處理"""
        log_info("收到終止信號，正在清理...")
        self.cleanup()
        sys.exit(0)
    
    def start_monitoring(self):
        """啟動醫療監控服務"""
        try:
            # 檢查監控服務是否已經在運行
            result = subprocess.run(['pgrep', '-f', 'medical_mongodb_exporter.py'], 
                                  capture_output=True, text=True)
            if result.stdout.strip():
                log_info("醫療監控服務已在運行")
                return True
            
            log_info("啟動醫療監控服務...")
            
            # 啟動監控服務
            self.monitoring_process = subprocess.Popen([
                sys.executable, 'medical_mongodb_exporter.py',
                '--mongodb-uri', 'mongodb://localhost:27017',
                '--database', 'medical_monitor_db', 
                '--port', '8000',
                '--interval', '15',
                '--log-level', 'INFO'
            ], stdout=open('integrated_monitoring.log', 'w'), 
               stderr=subprocess.STDOUT)
            
            # 等待服務啟動
            time.sleep(5)
            
            # 檢查服務是否正常
            import urllib.request
            try:
                response = urllib.request.urlopen('http://localhost:8000/metrics', timeout=10)
                if response.getcode() == 200:
                    log_success("醫療監控服務啟動成功 - http://localhost:8000/metrics")
                    
                    # 保存 PID
                    with open('integrated_monitoring.pid', 'w') as f:
                        f.write(str(self.monitoring_process.pid))
                    
                    return True
                else:
                    log_error("監控服務回應異常")
                    return False
            except Exception as e:
                log_error(f"監控服務無法訪問: {e}")
                return False
                
        except Exception as e:
            log_error(f"啟動監控服務失敗: {e}")
            return False
    
    def start_mongodb_if_needed(self):
        """如果需要則啟動 MongoDB"""
        try:
            from mongo_manager import get_mongo_manager
            mgr = get_mongo_manager()
            
            if mgr.is_mongodb_running():
                log_info("MongoDB 已在運行")
                return True
            else:
                log_info("啟動 MongoDB...")
                return mgr.start_mongodb()
        except Exception as e:
            log_error(f"MongoDB 啟動失敗: {e}")
            return False
    
    def process_video_with_monitoring(self, video_path, api_key, **kwargs):
        """處理影片並啟動監控"""
        
        # 註冊清理函數
        self.register_cleanup()
        
        log_info("🏥 整合醫療分析系統啟動")
        log_info("=" * 40)
        
        # 1. 啟動 MongoDB
        if not self.start_mongodb_if_needed():
            log_error("MongoDB 啟動失敗")
            return False
        
        # 2. 啟動監控服務
        if not self.start_monitoring():
            log_error("監控服務啟動失敗")
            return False
        
        # 3. 處理影片
        log_info("開始處理醫療影片...")
        
        try:
            from video_screen_digit_extractor import VideoScreenDigitExtractor
            
            extractor = VideoScreenDigitExtractor(
                api_key=api_key,
                config_file=kwargs.get('config_file', 'groundingdino/config/GroundingDINO_SwinT_OGC.py'),
                checkpoint_path=kwargs.get('checkpoint_path', 'groundingdino_swint_ogc.pth'),
                model=kwargs.get('model', 'gpt-4o'),
                cpu_only=kwargs.get('cpu_only', False),
                target_data=kwargs.get('target_data', 'medical_values')
            )
            
            # 處理影片
            results = extractor.process_video(video_path, kwargs.get('output_dir', 'medical_analysis'))
            
            log_success("影片處理完成!")
            log_info("監控服務持續運行，可通過以下方式查看數據：")
            log_info("• 監控指標: curl http://localhost:8000/metrics")
            log_info("• 患者數據: curl -s http://localhost:8000/metrics | grep patient_")
            log_info("• 異常警報: curl -s http://localhost:8000/metrics | grep medical_alerts")
            
            return True
            
        except Exception as e:
            log_error(f"影片處理失敗: {e}")
            return False
    
    def cleanup(self):
        """清理資源"""
        if self.monitoring_process:
            log_info("停止監控服務...")
            try:
                self.monitoring_process.terminate()
                self.monitoring_process.wait(timeout=5)
                log_success("監控服務已停止")
            except:
                try:
                    self.monitoring_process.kill()
                    log_info("強制終止監控服務")
                except:
                    pass
            
            # 清理 PID 檔案
            try:
                os.remove('integrated_monitoring.pid')
            except:
                pass

def main():
    parser = argparse.ArgumentParser(description="整合的醫療影片分析與監控系統")
    
    parser.add_argument("--video_path", "-v", required=True, 
                       help="醫療影片檔案路徑")
    parser.add_argument("--api_key", "-k", default=os.getenv("OPENAI_API_KEY"),
                       help="OpenAI API 金鑰")
    parser.add_argument("--config_file", "-c",
                       default="groundingdino/config/GroundingDINO_SwinT_OGC.py",
                       help="GroundingDINO 配置檔案")
    parser.add_argument("--checkpoint_path", "-p",
                       default="groundingdino_swint_ogc.pth",
                       help="GroundingDINO 模型權重")
    parser.add_argument("--model", "-m", default="gpt-4o", 
                       help="OpenAI 模型")
    parser.add_argument("--output_dir", "-o", default="medical_analysis_with_monitoring",
                       help="輸出目錄")
    parser.add_argument("--target_data", "-t", default="medical_values",
                       help="目標數據類型")
    parser.add_argument("--cpu-only", action="store_true", 
                       help="只使用 CPU")
    parser.add_argument("--keep-monitoring", action="store_true",
                       help="處理完成後保持監控服務運行")
    
    args = parser.parse_args()
    
    # 檢查影片檔案
    if not os.path.exists(args.video_path):
        log_error(f"影片檔案不存在: {args.video_path}")
        return 1
        
    if not args.api_key:
        log_error("未設定 API Key。請設定 OPENAI_API_KEY 環境變數或使用 --api_key 參數。")
        return 1
    
    # 創建整合系統
    system = IntegratedMedicalSystem()
    
    try:
        # 處理影片與監控
        success = system.process_video_with_monitoring(
            video_path=args.video_path,
            api_key=args.api_key,
            config_file=args.config_file,
            checkpoint_path=args.checkpoint_path,
            model=args.model,
            output_dir=args.output_dir,
            target_data=args.target_data,
            cpu_only=args.cpu_only
        )
        
        if success:
            if args.keep_monitoring:
                log_info("影片處理完成，監控服務保持運行...")
                log_info("按 Ctrl+C 停止監控服務")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    log_info("收到停止信號")
            else:
                log_info("影片處理完成，5秒後自動停止監控服務...")
                time.sleep(5)
                system.cleanup()
            
            return 0
        else:
            return 1
            
    except KeyboardInterrupt:
        log_info("用戶中斷處理")
        return 1
    except Exception as e:
        log_error(f"系統執行錯誤: {e}")
        return 1

if __name__ == "__main__":
    exit(main())