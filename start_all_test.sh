#!/bin/bash

# 定義顏色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}🚀 啟動全套醫療監控測試環境${NC}"
echo -e "${BLUE}=======================================${NC}"

# 1. 啟動基礎監控服務 (MongoDB, Prometheus, Grafana, Exporter)
echo -e "\n${GREEN}[1/4] 啟動監控基礎設施...${NC}"
# 注意：start_medical_monitoring.sh 通常是前台執行，我們讓它在後台跑
if [ -f "./start_medical_monitoring.sh" ]; then
    nohup ./start_medical_monitoring.sh > monitoring.log 2>&1 &
    echo "  -> 監控服務已在後台啟動 (Log: monitoring.log)"
    echo "  -> 等待 10 秒讓資料庫與服務就緒..."
    sleep 10 
else
    echo -e "${RED}  ⚠️  找不到 start_medical_monitoring.sh，跳過監控服務啟動${NC}"
fi

# 2. 啟動 Web 應用程式
echo -e "\n${GREEN}[2/4] 啟動 Web 使用者介面...${NC}"
if [ -f "./run.sh" ]; then
    nohup ./run.sh > web.log 2>&1 &
    echo "  -> Web App 已在後台啟動 (Log: web.log)"
else
    echo -e "${RED}  ⚠️  找不到 run.sh，跳過 Web App 啟動${NC}"
fi

# 3. 啟動 RTSP 影片輪播
echo -e "\n${GREEN}[3/4] 啟動測試影片輪播...${NC}"
# if [ -f "./run_playlist.sh" ]; then
#     # 先檢查是否已經在跑
#     if pgrep -f "run_playlist.sh" > /dev/null; then
#         echo "  -> 影片輪播已在運行中，略過啟動"
#     else
#         nohup ./run_playlist.sh > stream.log 2>&1 &
#         echo "  -> 影片輪播已在後台啟動 (Log: stream.log)"
#     fi
# else
#     echo -e "${RED}  ⚠️  找不到 run_playlist.sh，跳過影片輪播${NC}"
# fi
echo "  -> ⚠️  已停用本地影片輪播，請確保 Omniverse 正在推流到 rtsp://localhost:8554/stream1"

# 4. 啟動 AI 串流分析
echo -e "\n${GREEN}[4/4] 啟動 AI 即時分析...${NC}"
if [ -f "./run_rtsp_stream.sh" ]; then
    # 先檢查是否已經在跑
    if pgrep -f "run_rtsp_stream.sh" > /dev/null; then
        echo "  -> AI 分析已在運行中，略過啟動"
    else
        nohup ./run_rtsp_stream.sh > analysis.log 2>&1 &
        echo "  -> AI 分析已在後台啟動 (Log: analysis.log)"
    fi
else
    echo -e "${RED}  ⚠️  找不到 run_rtsp_stream.sh，跳過 AI 分析${NC}"
fi

echo -e "\n${BLUE}=======================================${NC}"
echo -e "${GREEN}✅ 所有服務啟動流程結束！${NC}"
echo -e "${BLUE}=======================================${NC}"
echo "📊 服務狀態快速檢查 (關鍵進程)："
ps -ef | grep -E "python|ffmpeg|prometheus|grafana" | grep -v grep | awk '{printf "  %-8s %s\n", $2, $8}'
echo -e "\n💡 提示：使用 'tail -f <log_file>' 查看各服務日誌"
echo "  - 監控: monitoring.log"
echo "  - 網頁: web.log"
echo "  - 串流: stream.log"
echo "  - 分析: analysis.log"
echo -e "\n❌ 若要停止所有服務，請執行： ./stop_all.sh"
