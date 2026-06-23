#!/bin/bash

# 醫療監控系統整合測試腳本
# 處理測試影片、驗證 MongoDB 數據寫入、檢查 Prometheus 指標輸出、驗證 Grafana 儀表板顯示

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

# 載入 .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# 測試結果統計
TESTS_PASSED=0
TESTS_FAILED=0
TEST_START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

# 記錄測試結果
record_test_result() {
    local test_name="$1"
    local result="$2"  # "PASS" 或 "FAIL"
    local details="$3"
    
    if [ "$result" = "PASS" ]; then
        TESTS_PASSED=$((TESTS_PASSED + 1))
        log_success "✅ $test_name"
        [ -n "$details" ] && echo "    $details"
    else
        TESTS_FAILED=$((TESTS_FAILED + 1))
        log_error "❌ $test_name"
        [ -n "$details" ] && echo "    $details"
    fi
}

# 檢查必要的檔案
check_test_prerequisites() {
    log_info "檢查測試先決條件..."
    
    local required_files=(
        "medical_mongodb_exporter.py"
        "video_screen_digit_extractor.py" 
        "mongo_manager.py"
        "prometheus.yml"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            record_test_result "檢查必要檔案: $file" "FAIL" "檔案不存在"
            return 1
        fi
    done
    
    # 檢查測試影片
    local test_video="medSample/med20250812-9.mkv"
    if [ ! -f "$test_video" ]; then
        log_warning "測試影片不存在: $test_video"
        log_info "將跳過影片處理測試"
        TEST_VIDEO=""
    else
        TEST_VIDEO="$test_video"
        record_test_result "檢查測試影片" "PASS" "找到: $test_video"
    fi
    
    record_test_result "檢查測試先決條件" "PASS"
    return 0
}

# 測試 MongoDB 連接
test_mongodb_connection() {
    log_info "測試 MongoDB 連接..."
    
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
        record_test_result "MongoDB 連接測試" "PASS"
        return 0
    else
        record_test_result "MongoDB 連接測試" "FAIL" "無法連接到 MongoDB"
        return 1
    fi
}

# 測試影片處理
test_video_processing() {
    if [ -z "$TEST_VIDEO" ]; then
        log_warning "跳過影片處理測試（無測試影片）"
        return 0
    fi
    
    log_info "測試影片處理..."
    
    # 創建測試會話ID
    local session_id="test_session_$(date +%s)"
    local output_dir="test_video_analysis_$(date +%s)"
    
    # 處理影片（只處理前2分鐘以節省時間）
    if [ -z "${OPENROUTER_API_KEY:-}" ]; then
        record_test_result "影片處理測試" "SKIP" "未設定 OPENROUTER_API_KEY，請 export 後再跑測試"
        return 0
    fi
    if python3 video_screen_digit_extractor.py \
        --video_path "$TEST_VIDEO" \
        --output_dir "$output_dir" \
        --provider openrouter \
        --model openai/gpt-5.4 \
        --api_key "${OPENROUTER_API_KEY}" \
        --target_data "medical_values" > test_video_processing.log 2>&1; then
        
        record_test_result "影片處理測試" "PASS" "輸出目錄: $output_dir"
        
        # 清理測試檔案
        if [ -d "$output_dir" ]; then
            rm -rf "$output_dir"
        fi
        return 0
    else
        record_test_result "影片處理測試" "FAIL" "處理失敗，查看 test_video_processing.log"
        return 1
    fi
}

# 測試 MongoDB 數據寫入
test_mongodb_data() {
    log_info "測試 MongoDB 數據查詢..."
    
    local data_count=$(python3 -c "
from mongo_manager import get_mongo_manager
try:
    mongo_mgr = get_mongo_manager()
    if not mongo_mgr.connect_to_mongodb():
        print('0')
        exit(1)
    
    # 查詢最近的醫療數據
    count = mongo_mgr.db.screen_analysis.count_documents({
        'success': True,
        'medical_values': {'\$exists': True, '\$ne': {}}
    })
    print(count)
except Exception as e:
    print('0')
    exit(1)
" 2>/dev/null)
    
    if [ "$data_count" -gt 0 ]; then
        record_test_result "MongoDB 數據查詢測試" "PASS" "找到 $data_count 筆醫療數據"
        return 0
    else
        record_test_result "MongoDB 數據查詢測試" "FAIL" "未找到醫療數據"
        return 1
    fi
}

# 測試 Medical Exporter 服務
test_medical_exporter() {
    log_info "測試 Medical MongoDB Exporter..."
    
    local metrics_url="http://localhost:8000/metrics"
    
    # 檢查服務是否響應
    if curl -f "$metrics_url" > /tmp/metrics_test.txt 2>/dev/null; then
        record_test_result "Medical Exporter 服務響應" "PASS"
        
        # 檢查關鍵指標是否存在
        local key_metrics=(
            "patient_heart_rate_bpm"
            "patient_spo2_percentage"
            "medical_alerts_total"
            "active_medical_sessions_total"
            "last_successful_scrape_timestamp"
        )
        
        local metrics_found=0
        for metric in "${key_metrics[@]}"; do
            if grep -q "$metric" /tmp/metrics_test.txt; then
                metrics_found=$((metrics_found + 1))
            fi
        done
        
        if [ "$metrics_found" -ge 3 ]; then
            record_test_result "Prometheus 指標檢查" "PASS" "找到 $metrics_found/${#key_metrics[@]} 個關鍵指標"
        else
            record_test_result "Prometheus 指標檢查" "FAIL" "僅找到 $metrics_found/${#key_metrics[@]} 個關鍵指標"
        fi
        
        rm -f /tmp/metrics_test.txt
        return 0
    else
        record_test_result "Medical Exporter 服務響應" "FAIL" "服務無響應"
        return 1
    fi
}

# 測試 Prometheus 服務
test_prometheus() {
    log_info "測試 Prometheus 服務..."
    
    local prometheus_url="http://localhost:9090"
    
    # 檢查健康狀態
    if curl -f "$prometheus_url/-/healthy" > /dev/null 2>&1; then
        record_test_result "Prometheus 健康檢查" "PASS"
        
        # 檢查目標狀態
        if curl -f "$prometheus_url/api/v1/targets" > /tmp/targets_test.json 2>/dev/null; then
            local active_targets=$(python3 -c "
import json
import sys
try:
    with open('/tmp/targets_test.json', 'r') as f:
        data = json.load(f)
    active = sum(1 for target in data['data']['activeTargets'] if target['health'] == 'up')
    print(active)
except:
    print(0)
" 2>/dev/null)
            
            if [ "$active_targets" -gt 0 ]; then
                record_test_result "Prometheus 目標檢查" "PASS" "$active_targets 個目標正常"
            else
                record_test_result "Prometheus 目標檢查" "FAIL" "沒有正常的監控目標"
            fi
            
            rm -f /tmp/targets_test.json
        fi
        return 0
    else
        record_test_result "Prometheus 健康檢查" "FAIL" "服務無響應"
        return 1
    fi
}

# 測試 Grafana 服務
test_grafana() {
    log_info "測試 Grafana 服務..."
    
    local grafana_url="http://localhost:3001"
    
    # 檢查健康狀態
    if curl -f "$grafana_url/api/health" > /dev/null 2>&1; then
        record_test_result "Grafana 健康檢查" "PASS"
        
        # 檢查數據源
        if curl -f -u admin:medical123 "$grafana_url/api/datasources" > /tmp/datasources_test.json 2>/dev/null; then
            local prometheus_ds=$(python3 -c "
import json
import sys
try:
    with open('/tmp/datasources_test.json', 'r') as f:
        datasources = json.load(f)
    prometheus_count = sum(1 for ds in datasources if ds.get('type') == 'prometheus')
    print(prometheus_count)
except:
    print(0)
" 2>/dev/null)
            
            if [ "$prometheus_ds" -gt 0 ]; then
                record_test_result "Grafana 數據源檢查" "PASS" "找到 Prometheus 數據源"
            else
                record_test_result "Grafana 數據源檢查" "FAIL" "未找到 Prometheus 數據源"
            fi
            
            rm -f /tmp/datasources_test.json
        fi
        
        # 檢查儀表板
        if curl -f -u admin:medical123 "$grafana_url/api/search" > /tmp/dashboards_test.json 2>/dev/null; then
            local dashboard_count=$(python3 -c "
import json
import sys
try:
    with open('/tmp/dashboards_test.json', 'r') as f:
        dashboards = json.load(f)
    count = len([d for d in dashboards if d.get('type') == 'dash-db'])
    print(count)
except:
    print(0)
" 2>/dev/null)
            
            if [ "$dashboard_count" -gt 0 ]; then
                record_test_result "Grafana 儀表板檢查" "PASS" "找到 $dashboard_count 個儀表板"
            else
                record_test_result "Grafana 儀表板檢查" "FAIL" "未找到儀表板"
            fi
            
            rm -f /tmp/dashboards_test.json
        fi
        return 0
    else
        record_test_result "Grafana 健康檢查" "FAIL" "服務無響應"
        return 1
    fi
}

# 執行效能測試
test_performance() {
    log_info "執行效能測試..."
    
    # 測試 MongoDB 查詢效能
    local query_time=$(python3 -c "
import time
from mongo_manager import get_mongo_manager
try:
    mongo_mgr = get_mongo_manager()
    if not mongo_mgr.connect_to_mongodb():
        print('999')
        exit(1)
    
    start_time = time.time()
    # 執行複雜查詢
    list(mongo_mgr.db.screen_analysis.aggregate([
        {'\$match': {'success': True}},
        {'\$group': {'_id': '\$detected_model', 'count': {'\$sum': 1}}},
        {'\$sort': {'count': -1}}
    ]))
    end_time = time.time()
    
    query_duration = (end_time - start_time) * 1000  # 轉換為毫秒
    print(f'{query_duration:.2f}')
except Exception as e:
    print('999')
" 2>/dev/null)
    
    if (( $(echo "$query_time < 2000" | bc -l) )); then
        record_test_result "MongoDB 查詢效能測試" "PASS" "查詢時間: ${query_time}ms < 2s"
    else
        record_test_result "MongoDB 查詢效能測試" "FAIL" "查詢時間: ${query_time}ms >= 2s"
    fi
    
    # 測試 Medical Exporter 響應效能
    local response_time=$(curl -o /dev/null -s -w '%{time_total}' http://localhost:8000/metrics 2>/dev/null || echo "999")
    local response_ms=$(echo "$response_time * 1000" | bc -l 2>/dev/null || echo "999")
    
    if (( $(echo "$response_time < 5" | bc -l) )); then
        record_test_result "Medical Exporter 響應效能測試" "PASS" "響應時間: ${response_ms}ms < 5s"
    else
        record_test_result "Medical Exporter 響應效能測試" "FAIL" "響應時間: ${response_ms}ms >= 5s"
    fi
}

# 生成測試報告
generate_test_report() {
    local test_end_time=$(date '+%Y-%m-%d %H:%M:%S')
    local total_tests=$((TESTS_PASSED + TESTS_FAILED))
    
    echo ""
    echo "=========================================="
    echo "📋 醫療監控系統測試報告"
    echo "=========================================="
    echo ""
    echo "🕐 測試時間："
    echo "  開始時間: $TEST_START_TIME"
    echo "  結束時間: $test_end_time"
    echo ""
    echo "📊 測試結果："
    echo "  總測試數量: $total_tests"
    echo "  通過測試: $TESTS_PASSED"
    echo "  失敗測試: $TESTS_FAILED"
    echo "  通過率: $(( TESTS_PASSED * 100 / total_tests ))%"
    echo ""
    
    if [ $TESTS_FAILED -eq 0 ]; then
        echo "🎉 所有測試通過！醫療監控系統運行正常。"
    else
        echo "⚠️  有 $TESTS_FAILED 項測試失敗，請檢查相關服務。"
    fi
    
    echo ""
    echo "🔗 快速訪問連結："
    echo "  • Medical Exporter: http://localhost:8000/metrics"
    echo "  • Prometheus:       http://localhost:9090"
    echo "  • Grafana:          http://localhost:3001 (admin/medical123)"
    echo ""
    echo "📁 測試日誌："
    if [ -f "test_video_processing.log" ]; then
        echo "  • 影片處理日誌: test_video_processing.log"
    fi
    echo "  • Medical Exporter: medical_exporter.log"
    echo ""
    echo "=========================================="
}

# 主函數
main() {
    echo "🧪 醫療監控系統整合測試"
    echo "============================="
    echo ""
    
    # 檢查先決條件
    check_test_prerequisites || exit 1
    
    # 執行測試套組
    echo ""
    log_info "開始執行測試套組..."
    echo ""
    
    # 基礎服務測試
    test_mongodb_connection
    test_mongodb_data
    
    # 應用程式測試
    test_medical_exporter
    
    # 監控系統測試
    test_prometheus
    test_grafana
    
    # 效能測試
    test_performance
    
    # 影片處理測試（如果有測試影片的話）
    if [ -n "$TEST_VIDEO" ]; then
        test_video_processing
    fi
    
    # 生成測試報告
    generate_test_report
    
    # 返回適當的退出碼
    if [ $TESTS_FAILED -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
}

# 顯示幫助信息
show_help() {
    echo "醫療監控系統整合測試腳本"
    echo ""
    echo "用法: $0 [選項]"
    echo ""
    echo "功能:"
    echo "  • 測試 MongoDB 連接和數據"
    echo "  • 測試 Medical MongoDB Exporter"
    echo "  • 測試 Prometheus 和 Grafana 服務"
    echo "  • 執行效能測試"
    echo "  • 處理測試影片（如果存在）"
    echo ""
    echo "先決條件:"
    echo "  • 執行 ./start_medical_monitoring.sh 啟動服務"
    echo "  • 確保測試影片存在於 medSample/med20250812-9.mkv"
    echo ""
    echo "選項:"
    echo "  --help         顯示此幫助信息"
    echo ""
}

# 解析命令行參數
case "$1" in
    --help)
        show_help
        exit 0
        ;;
    "")
        main
        ;;
    *)
        echo "未知參數: $1"
        echo "使用 --help 查看幫助信息"
        exit 1
        ;;
esac