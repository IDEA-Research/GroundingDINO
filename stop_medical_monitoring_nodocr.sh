#!/bin/bash

# 醫療監控系統停止腳本（無 Docker 版本）
# 停止 Medical MongoDB Exporter

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

# 停止 Medical MongoDB Exporter
stop_medical_exporter() {
    log_info "停止 Medical MongoDB Exporter..."
    
    # 通過 PID 檔案停止
    if [ -f "medical_exporter_simple.pid" ]; then
        local pid=$(cat medical_exporter_simple.pid)
        if ps -p $pid > /dev/null 2>&1; then
            kill $pid
            sleep 3
            if ps -p $pid > /dev/null 2>&1; then
                log_warning "進程 $pid 仍在運行，強制終止..."
                kill -9 $pid
            fi
            rm -f medical_exporter_simple.pid
            log_success "Medical Exporter (PID: $pid) 已停止"
        else
            log_warning "PID $pid 的進程已不存在"
            rm -f medical_exporter_simple.pid
        fi
    else
        log_info "未找到 PID 檔案，嘗試按進程名稱終止..."
    fi
    
    # 按進程名稱停止
    local pids=$(pgrep -f "medical_mongodb_exporter.py" || true)
    if [ -n "$pids" ]; then
        log_info "發現運行中的 Medical Exporter 進程: $pids"
        pkill -f "medical_mongodb_exporter.py"
        sleep 2
        
        # 檢查是否還有進程運行
        local remaining_pids=$(pgrep -f "medical_mongodb_exporter.py" || true)
        if [ -n "$remaining_pids" ]; then
            log_warning "仍有進程運行，強制終止..."
            pkill -9 -f "medical_mongodb_exporter.py"
        fi
        log_success "Medical Exporter 進程已停止"
    else
        log_info "未發現運行中的 Medical Exporter 進程"
    fi
}

# 檢查端口狀態
check_port_status() {
    log_info "檢查端口狀態..."
    
    if netstat -tuln 2>/dev/null | grep -q ":8000 "; then
        log_warning "Medical Exporter (端口 8000) 仍在使用中"
    else
        log_success "Medical Exporter (端口 8000) 已釋放"
    fi
}

# 清理檔案
cleanup_files() {
    log_info "清理臨時檔案..."
    
    # 清理 PID 檔案
    rm -f medical_exporter_simple.pid
    
    # 詢問是否清理日誌
    if [ "$1" != "--keep-logs" ]; then
        read -p "是否要清理日誌檔案? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -f medical_exporter_simple.log
            log_success "日誌檔案已清理"
        else
            log_info "保留日誌檔案: medical_exporter_simple.log"
        fi
    else
        log_info "保留日誌檔案: medical_exporter_simple.log"
    fi
}

# 顯示停止狀態
display_stop_status() {
    echo ""
    echo "=========================================="
    echo "🛑 醫療監控系統已停止（無 Docker 版本）"
    echo "=========================================="
    echo ""
    echo "💡 後續操作："
    echo "  • 重新啟動系統: ./start_medical_monitoring_nodocr.sh"
    echo "  • 查看留存的日誌: cat medical_exporter_simple.log"
    echo ""
    echo "📁 保留的檔案："
    if [ -f "medical_exporter_simple.log" ]; then
        echo "  • 日誌檔案: medical_exporter_simple.log ($(du -h medical_exporter_simple.log | cut -f1))"
    fi
    echo ""
    echo "=========================================="
}

# 主函數
main() {
    echo "🛑 醫療監控系統停止腳本（無 Docker 版本）"
    echo "==========================================="
    echo ""
    
    # 停止 Medical Exporter
    stop_medical_exporter
    
    # 等待服務完全停止
    log_info "等待服務完全停止..."
    sleep 3
    
    # 檢查端口狀態
    check_port_status
    
    # 清理檔案
    cleanup_files "$1"
    
    # 顯示停止狀態
    display_stop_status
    
    log_success "醫療監控系統停止完成！"
}

# 顯示幫助信息
show_help() {
    echo "醫療監控系統停止腳本（無 Docker 版本）"
    echo ""
    echo "用法: $0 [選項]"
    echo ""
    echo "選項:"
    echo "  --keep-logs    保留所有日誌檔案，不進行清理"
    echo "  --help         顯示此幫助信息"
    echo ""
    echo "範例:"
    echo "  $0                # 正常停止系統（會詢問是否清理日誌）"
    echo "  $0 --keep-logs   # 停止系統但保留所有日誌"
    echo ""
}

# 解析命令行參數
case "$1" in
    --help)
        show_help
        exit 0
        ;;
    --keep-logs)
        main "$1"
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