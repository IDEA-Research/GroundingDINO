#!/bin/bash

# 醫療監控系統啟動腳本
# 檢查 MongoDB 連接狀態，啟動 medical_mongodb_exporter.py，以及 Prometheus 和 Grafana 服務

set -e  # 遇到錯誤立即退出

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

# 檢查命令是否存在
check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "命令 '$1' 未找到。請安裝後重試。"
        return 1
    fi
}

# 檢查端口是否被佔用
check_port() {
    local port=$1
    local service_name=$2
    
    # 修改：只檢查 TCP 端口，忽略 UDP 端口
    if netstat -tln 2>/dev/null | grep -q ":$port "; then
        log_warning "$service_name 端口 $port 已被佔用"
        return 1
    fi
    return 0
}

# 檢查 MongoDB 連接
check_mongodb() {
    log_info "檢查 MongoDB 連接狀態..."
    
    # 嘗試連接 MongoDB
    if python3 -c "
import pymongo
import sys
try:
    client = pymongo.MongoClient('mongodb://localhost:27017', serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print('MongoDB 連接成功')
    sys.exit(0)
except Exception as e:
    print(f'MongoDB 連接失敗: {e}')
    sys.exit(1)
" 2>/dev/null; then
        log_success "MongoDB 連接正常"
        return 0
    else
        log_error "MongoDB 連接失敗"
        return 1
    fi
}

# 啟動本地 MongoDB
start_local_mongodb() {
    log_info "啟動本地 MongoDB..."
    
    # 檢查 MongoDB 是否已在運行
    if check_mongodb; then
        log_success "MongoDB 已在運行"
        return 0
    fi
    
    # 嘗試啟動 MongoDB
    if python3 -c "
from mongo_manager import start_local_mongodb
if start_local_mongodb():
    print('MongoDB 啟動成功')
else:
    exit(1)
" 2>/dev/null; then
        log_success "MongoDB 啟動成功"
        sleep 3  # 等待 MongoDB 完全啟動
        return 0
    else
        log_error "MongoDB 啟動失敗"
        return 1
    fi
}

# 檢查必要的文件
check_required_files() {
    log_info "檢查必要的檔案..."
    
    local required_files=(
        "medical_mongodb_exporter.py"
        "prometheus.yml"
        "docker-compose.monitoring.yml"
        "alerts/medical_alerts.yml"
        "grafana/provisioning/datasources/prometheus.yml"
        "grafana/dashboards/patient_monitoring.json"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "必要檔案不存在: $file"
            return 1
        fi
    done
    
    log_success "所有必要檔案檢查完成"
    return 0
}

# 安裝 Python 依賴
install_dependencies() {
    log_info "檢查並安裝 Python 依賴..."
    
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt --quiet
        log_success "Python 依賴安裝完成"
    else
        log_warning "未找到 requirements.txt，跳過依賴安裝"
    fi
}

# 啟動 Medical MongoDB Exporter
start_medical_exporter() {
    log_info "啟動 Medical MongoDB Exporter..."
    
    # 檢查端口 8000
    if ! check_port 8000 "Medical Exporter"; then
        log_warning "端口 8000 已被佔用，嘗試終止現有進程..."
        pkill -f "medical_mongodb_exporter.py" || true
        sleep 2
    fi
    
    # 啟動 exporter
    nohup python3 medical_mongodb_exporter.py \
        --mongodb-uri mongodb://localhost:27017 \
        --database medical_monitor_db \
        --port 8000 \
        --interval 15 \
        --log-level INFO > medical_exporter.log 2>&1 &
    
    local exporter_pid=$!
    echo $exporter_pid > medical_exporter.pid
    
    # 等待服務啟動
    sleep 5
    
    # 檢查服務是否正常運行
    if curl -f http://localhost:8000/metrics > /dev/null 2>&1; then
        log_success "Medical MongoDB Exporter 在 port 8000 啟動成功 (PID: $exporter_pid)"
    else
        log_error "Medical MongoDB Exporter 啟動失敗"
        return 1
    fi
}

# 使用 Docker Compose 啟動監控服務
start_monitoring_services() {
    log_info "啟動 Prometheus 和 Grafana 服務..."
    
    # 檢查 Docker
    check_command "docker" || return 1
    
    # 判斷使用 docker-compose 還是 docker compose
    local compose_cmd=""
    if command -v docker-compose &> /dev/null; then
        compose_cmd="docker-compose"
    elif docker compose version &> /dev/null; then
        compose_cmd="docker compose"
    else
        log_error "未找到 docker-compose 或 docker compose 命令"
        return 1
    fi
    
    log_info "使用 Compose 命令: $compose_cmd"
    
    # 啟動監控服務（跳過 medical-mongodb-exporter，因為我們已經本地啟動了）
    $compose_cmd -f docker-compose.monitoring.yml up -d prometheus grafana
    
    # 等待服務啟動
    log_info "等待服務啟動 (30秒)..."
    sleep 30
    
    # 檢查 Prometheus
    if curl -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
        log_success "Prometheus 在 port 9090 啟動成功"
    else
        log_error "Prometheus 啟動失敗"
        return 1
    fi
    
    # 檢查 Grafana
    if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
        log_success "Grafana 在 port 3000 啟動成功"
    else
        log_error "Grafana 啟動失敗"
        return 1
    fi
}

# 顯示服務 URL
display_service_urls() {
    echo ""
    echo "=========================================="
    echo "🎉 醫療監控系統啟動完成！"
    echo "=========================================="
    echo ""
    echo "📊 服務訪問地址："
    echo "  • Medical Exporter Metrics: http://localhost:8000/metrics"
    echo "  • Prometheus Web UI:        http://localhost:9090"
    echo "  • Grafana Dashboard:        http://localhost:3000"
    echo ""
    echo "🔐 Grafana 登入資訊："
    echo "  • 使用者名稱: admin"
    echo "  • 密碼:       medical123"
    echo ""
    echo "📈 重要儀表板："
    echo "  • 患者監控儀表板: http://localhost:3000/d/patient-monitoring-dashboard"
    echo "  • 系統總覽儀表板: http://localhost:3000/d/system-overview-dashboard"
    echo ""
    echo "📝 日誌檔案："
    echo "  • Medical Exporter: $(pwd)/medical_exporter.log"
    echo "  • Docker Logs:      docker-compose -f docker-compose.monitoring.yml logs"
    echo ""
    echo "⚠️  停止服務："
    echo "  • ./stop_medical_monitoring.sh"
    echo ""
    echo "=========================================="
}

# 主函數
main() {
    echo "🏥 醫療監控系統啟動腳本"
    echo "=================================="
    echo ""
    
    # 檢查必要檔案
    if ! check_required_files; then
        exit 1
    fi
    
    # 安裝依賴
    # install_dependencies
    
    # 檢查並啟動 MongoDB
    if ! check_mongodb; then
        if ! start_local_mongodb; then
            log_error "無法啟動 MongoDB，請檢查配置"
            exit 1
        fi
    fi
    
    # 啟動 Medical Exporter
    if ! start_medical_exporter; then
        exit 1
    fi
    
    # 啟動監控服務
    if ! start_monitoring_services; then
        log_error "監控服務啟動失敗"
        exit 1
    fi
    
    # 顯示服務信息
    display_service_urls
    
    log_success "醫療監控系統啟動完成！"
}

# 捕捉中斷信號
trap 'log_warning "腳本被中斷"; exit 1' INT TERM

# 執行主函數
main "$@"