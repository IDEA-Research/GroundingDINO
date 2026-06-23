#!/bin/bash

# UI Agent 使用 Conda 環境啟動腳本

set -e  # 錯誤時停止

echo "========================================="
echo "🚀 UI Agent - 使用 Conda 環境啟動"
echo "========================================="
echo ""

# 檢查 conda 是否已安裝
if ! command -v conda &> /dev/null; then
    echo "❌ 錯誤: 未找到 conda"
    echo "請先執行: ./setup-conda.sh"
    exit 1
fi

# 檢查環境是否存在
if ! conda env list | grep -q "^uiagent "; then
    echo "❌ 錯誤: 'uiagent' 環境不存在"
    echo "請先執行: ./setup-conda.sh"
    exit 1
fi

echo "✅ Conda 環境已就緒"
echo ""

# 初始化 conda
eval "$(conda shell.bash hook)"

# 啟動環境
echo "🔄 啟動 conda 環境..."
conda activate uiagent

# 檢查 Node.js
if ! command -v node &> /dev/null; then
    echo "❌ 錯誤: Node.js 未正確安裝在環境中"
    exit 1
fi

# 檢查 pnpm
if ! command -v pnpm &> /dev/null; then
    echo "📦 安裝 pnpm..."
    npm install -g pnpm
fi

echo "✅ Node.js 版本: $(node --version)"
echo "✅ pnpm 版本: $(pnpm --version)"
echo ""

# 檢查是否已安裝依賴
if [ ! -d "node_modules" ]; then
    echo "📦 首次執行，正在安裝依賴..."
    pnpm install
    echo ""
fi

# 建置所有 packages（包含前端）
echo "🔨 建置應用程式..."
pnpm run build
echo ""

# 檢查環境變數檔案
if [ ! -f "packages/api/.env" ]; then
    echo "📝 建立 API 環境變數檔案..."
    cp packages/api/.env.example packages/api/.env
    echo "✅ 已建立 packages/api/.env"
    echo "   提示: 可以編輯此檔案加入 OpenAI API Key"
    echo ""
fi

echo "========================================="
echo "🎯 啟動服務（生產模式）"
echo "========================================="
echo ""
echo "服務將在以下地址啟動："
echo "   📊 Web UI:    http://localhost:4000"
echo "   🔌 API Server: http://localhost:4001"
echo ""
echo "💡 提示: 使用生產構建以避免 proxy 環境下的路徑問題"
echo "   詳見: KUBEFLOW_PROXY.md"
echo ""
echo "按 Ctrl+C 停止服務"
echo ""
echo "----------------------------------------"
echo ""

# 檢查生產構建是否存在
if [ ! -d "packages/web/dist" ]; then
    echo "❌ 錯誤: 找不到前端生產構建"
    echo "建置過程可能失敗，請檢查上方的錯誤訊息"
    exit 1
fi

# 在後台啟動 API server
echo "🔌 啟動 API Server..."
cd packages/api
PORT=4001 node dist/index.js &
API_PID=$!
cd ../..

# 等待 API server 啟動
sleep 2

# 啟動靜態文件服務器提供前端
echo "📊 啟動 Web UI (靜態文件服務器)..."
cd packages/web/dist
python3 -m http.server 4000 &
WEB_PID=$!
cd ../../..

echo ""
echo "✅ 服務已啟動"
echo "   API Server PID: $API_PID"
echo "   Web UI PID: $WEB_PID"
echo ""

# 捕捉 Ctrl+C 信號並清理
trap "echo ''; echo '🛑 正在停止服務...'; kill $API_PID $WEB_PID 2>/dev/null; exit 0" INT TERM

# 等待進程
wait
