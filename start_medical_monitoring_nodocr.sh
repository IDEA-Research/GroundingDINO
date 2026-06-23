#!/bin/bash

# 醫療監控系統啟動腳本（無 Docker 版本）
# 只啟動核心的 Medical MongoDB Exporter

set -e  

# 定義顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 定義日誌函數
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 主函數
main() {
    echo "🏥 醫療監控系統啟動腳本（無 Docker 版本）"
    echo "============================================="
    echo ""
    
    # 檢查 MongoDB 狀態
    log_info "檢查 MongoDB 連接狀態..."
    if python3 -c "
from mongo_manager import get_mongo_manager
try:
    mgr = get_mongo_manager()
    if mgr.is_mongodb_running():
        print('MongoDB 連接成功')
        exit(0)
    else:
        print('MongoDB 未運行，正在啟動...')
        if mgr.start_mongodb():
            print('MongoDB 啟動成功')
            exit(0)
        else:
            exit(1)
except Exception as e:
    print(f'MongoDB 檢查失敗: {e}')
    exit(1)
" 2>/dev/null; then
        log_success "MongoDB 狀態正常"
    else
        log_error "MongoDB 啟動失敗"
        exit 1
    fi
    
    # 檢查端口是否被佔用
    if netstat -tuln 2>/dev/null | grep -q ":8000 "; then
        log_warning "端口 8000 已被佔用，正在終止現有進程..."
        pkill -f medical_mongodb_exporter.py || true
        sleep 3
    fi
    
    # 啟動 Medical MongoDB Exporter
    log_info "啟動 Medical MongoDB Exporter..."
    python3 medical_mongodb_exporter.py \
        --mongodb-uri mongodb://localhost:27017 \
        --database medical_monitor_db \
        --port 8000 \
        --interval 15 \
        --log-level INFO > medical_exporter_simple.log 2>&1 &
    
    local exporter_pid=$!
    echo $exporter_pid > medical_exporter_simple.pid
    
    # 等待服務啟動
    log_info "等待服務啟動..."
    sleep 8
    
    # 檢查服務是否正常運行
    if curl -f http://localhost:8000/metrics > /dev/null 2>&1; then
        log_success "Medical MongoDB Exporter 啟動成功 (PID: $exporter_pid)"
    else
        log_error "Medical MongoDB Exporter 啟動失敗"
        exit 1
    fi
    
    echo ""
    echo "=========================================="
    echo "🎉 醫療監控系統啟動完成！"
    echo "=========================================="
    echo ""
    echo "📊 服務訪問地址："
    echo "  • Medical Exporter Metrics: http://localhost:8000/metrics"
    echo ""
    echo "📈 使用方法："
    echo "  • 查看所有指標：curl http://localhost:8000/metrics"
    echo "  • 查看患者數據：curl -s http://localhost:8000/metrics | grep patient_"
    echo "  • 查看異常警報：curl -s http://localhost:8000/metrics | grep medical_alerts"
    echo ""
    echo "🏥 處理醫療影片："
    echo "  python3 video_screen_digit_extractor.py \\"
    echo "    --video_path medSample/med20250812-9.mkv \\"
    echo "    --api_key 你的OpenAI_API_KEY"
    echo ""
    echo "📝 日誌檔案："
    echo "  • Medical Exporter: medical_exporter_simple.log"
    echo ""
    echo "⚠️  停止服務："
    echo "  • ./stop_medical_monitoring_nodocr.sh"
    echo ""
    echo "=========================================="
    
    log_success "醫療監控核心服務啟動完成！"
}

# 執行主函數
main "$@"