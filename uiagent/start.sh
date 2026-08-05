#!/bin/bash

# UI Agent 快速啟動腳本

NPX_RUNTIME="npx -y -p node@20 -p pnpm@8 -c"

echo "🚀 啟動 UI Agent 系統..."
echo ""

# 檢查是否已安裝依賴
if [ ! -d "node_modules" ]; then
    echo "📦 未找到 node_modules，正在安裝依賴..."
    ${NPX_RUNTIME} 'pnpm install'
    echo ""
fi

# 建置 packages
echo "🔨 建置 packages..."
${NPX_RUNTIME} 'pnpm run build'
echo ""

# 檢查環境變數檔案
if [ ! -f "packages/api/.env" ]; then
    echo "📝 建立 API 環境變數檔案..."
    cp packages/api/.env.example packages/api/.env
    echo "✅ 已建立 packages/api/.env (沒有 OpenAI API Key 時會使用 mock 資料)"
    echo ""
fi

# 啟動服務
echo "🎯 啟動服務..."
echo "   - API Server: http://localhost:3001"
echo "   - Web UI: http://localhost:3000"
echo ""

${NPX_RUNTIME} 'pnpm run dev'
