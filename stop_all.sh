#!/bin/bash

# 定義顏色
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${RED}=======================================${NC}"
echo -e "${RED}🛑 正在停止全套醫療監控測試環境${NC}"
echo -e "${RED}=======================================${NC}"

# 定義要停止的腳本和進程關鍵字
TARGETS=(
    "run_rtsp_stream.sh"
    "video_screen_digit_extractor.py"
    "run_playlist.sh"
    "ffmpeg"
    "run.sh"
    "web_app.py"
    "start_medical_monitoring.sh"
    "medical_mongodb_exporter.py"
)

count=0

for target in "${TARGETS[@]}"; do
    # 檢查進程是否存在
    pids=$(pgrep -f "$target")
    
    if [ -n "$pids" ]; then
        echo -e "正在停止 ${target} (PID: $pids)..."
        # 使用 pkill 強制結束匹配的進程
        pkill -f "$target"
        count=$((count+1))
    fi
done

# 特別處理 Docker 容器 (Grafana & Prometheus)
echo "正在檢查並停止 Docker 監控服務..."
if [ -f "docker-compose.monitoring.yml" ]; then
    docker-compose -f docker-compose.monitoring.yml stop
    echo "Docker 服務已停止"
fi

echo -e "\n${GREEN}✅ 已停止 $count 個相關進程與 Docker 服務${NC}"
echo "系統已清理完畢。"
