#!/bin/bash

# UI Agent Anaconda 環境設定腳本

set -e  # 錯誤時停止

echo "========================================="
echo "🐍 UI Agent - Anaconda 環境設定"
echo "========================================="
echo ""

# 檢查 conda 是否已安裝
if ! command -v conda &> /dev/null; then
    echo "❌ 錯誤: 未找到 conda"
    echo "請先安裝 Anaconda 或 Miniconda"
    echo "下載連結: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ 找到 conda"
echo ""

# 建立 conda 環境
echo "📦 建立 conda 環境 'uiagent'..."
if conda env list | grep -q "^uiagent "; then
    echo "⚠️  環境 'uiagent' 已存在"
    if [ "$FORCE" = "true" ]; then
        REPLY="y"
    else
        read -p "是否要重新建立環境? (y/N) " -n 1 -r
        echo
    fi
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  移除舊環境..."
        conda env remove -n uiagent -y
        echo "📦 建立新環境..."
        conda env create -f environment.yml
    else
        echo "✅ 使用現有環境"
    fi
else
    conda env create -f environment.yml
fi

echo ""
echo "========================================="
echo "✅ Conda 環境設定完成！"
echo "========================================="
echo ""
echo "下一步："
echo "1. 啟動環境: conda activate uiagent"
echo "2. 安裝依賴: npm install"
echo "3. 啟動系統: npm run dev"
echo ""
echo "或直接執行: ./start-with-conda.sh"
echo ""
