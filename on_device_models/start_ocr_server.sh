#!/bin/bash
# 啟動本機微調螢幕 OCR 服務（供 web_app provider="ondevice" 使用）
# 用法: ./start_ocr_server.sh [--preload glm paddle]
cd "$(dirname "$0")"
export PYTHONNOUSERSITE=1
exec /home/cluster/miniforge3/envs/ondevice_ocr/bin/python ocr_server.py "$@"
