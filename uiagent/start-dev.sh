#!/bin/bash

# UI Agent 開發模式啟動腳本（僅用於本地開發）

set -e  # 錯誤時停止

echo "========================================="
echo "🚀 UI Agent - 開發模式"
echo "========================================="
echo ""
echo "⚠️  警告: 此模式僅適用於本地開發"
echo "   在 Kubeflow/JupyterHub proxy 環境下請使用:"
echo "   ./start-with-conda.sh (生產模式)"
echo ""

# 檢查是否在 conda 環境中
if [[ "$CONDA_DEFAULT_ENV" != "uiagent" ]]; then
    echo "🔄 啟動 conda 環境..."
    if ! command -v conda &> /dev/null; then
        echo "❌ 錯誤: 未找到 conda，請先執行 ./setup-conda.sh"
        exit 1
    fi
    eval "$(conda shell.bash hook)"
    conda activate uiagent
fi

# 檢查 pnpm
if ! command -v pnpm &> /dev/null; then
    echo "📦 安裝 pnpm..."
    npm install -g pnpm
fi

# 檢查是否已安裝依賴
if [ ! -d "node_modules" ]; then
    echo "📦 安裝依賴..."
    pnpm install
    echo ""
fi

# 建置必要的 packages
if [ ! -d "packages/types/dist" ] || [ ! -d "packages/validator/dist" ]; then
    echo "🔨 建置 packages..."
    pnpm run build
    echo ""
fi

# 檢查環境變數檔案
if [ ! -f "packages/api/.env" ]; then
    echo "📝 建立 API 環境變數檔案..."
    cp packages/api/.env.example packages/api/.env
    echo "✅ 已建立 packages/api/.env"
    echo ""
fi

# 確保使用本地開發配置
if [ ! -f "packages/web/.env.development" ]; then
    echo "VITE_BASE_PATH=" > packages/web/.env.development
fi

echo "========================================="
echo "🎯 啟動開發服務器"
echo "========================================="
echo ""
echo "服務將在以下地址啟動："
echo "   📊 Web UI:    http://localhost:4000"
echo "   🔌 API Server: http://localhost:4001"
echo ""
echo "✨ 功能: 熱模組替換 (HMR) 已啟用"
echo "   修改代碼後會自動重載"
echo ""
echo "按 Ctrl+C 停止服務"
echo ""
echo "----------------------------------------"
echo ""

# 啟動開發模式
pnpm dev
