if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "$1" ]; then
    echo "錯誤: 請以第一個參數傳入 OpenRouter API Key。"
    echo "用法: $0 <OPENROUTER_API_KEY> [其他參數...]"
    exit 1
fi
API_KEY="$1"
shift

# =================================================================
# 🚀 核心修復區 (Final Fix)
# =================================================================

# 1. 清除所有設定
unset LD_PRELOAD
unset LD_LIBRARY_PATH

# 2. 設定順序：系統 CUDA 12.6 優先 -> 系統 AARCH64 庫 -> 最後才是 Conda
# 注意：我們把系統路徑放最前面
# export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/lib/aarch64-linux-gnu:/home/cluster/miniforge3/envs/webapp_mongo_gpu/lib:$LD_LIBRARY_PATH

# 3. 再次測試
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); x = torch.ones(2, 2).cuda(); print('運算成功:', torch.matmul(x, x))"

# 4. 暫時不要使用這個設定，看看原生的分配器是否正常
unset PYTORCH_CUDA_ALLOC_CONF

# 5. 強制 OpenCV 使用 TCP (維持原樣)
export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"

cd "$(dirname "$0")"

echo "=========================================="
echo "🚀 RTSP 流處理程序啟動"
echo "=========================================="
echo "✅ 記憶體策略: expandable_segments:True"
echo "✅ 核心庫: 系統原廠 cuBLAS"
echo ""

# 運行程序
python3 video_screen_digit_extractor.py \
  --camera \
  --rtsp_url rtsp://localhost:8554/stream1 \
  --provider openrouter \
  --model openai/gpt-5.4 \
  --api_key "$API_KEY" \
  --target_data medical_values \
  --interval 0 \
  "$@"


