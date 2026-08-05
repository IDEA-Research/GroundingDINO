#!/bin/bash

# Set video directory (default to Test_video in current directory)
VIDEO_DIR="./Test_video"
# RTSP URL
RTSP_URL="rtsp://localhost:8554/stream1"

# Check if directory exists
if [ ! -d "$VIDEO_DIR" ]; then
    echo "❌ Error: Directory $VIDEO_DIR does not exist"
    exit 1
fi

echo "=================================================="
echo "🎥 Starting optimized playlist streaming (Background Mode)"
echo "📂 Video Source: $VIDEO_DIR"
echo "📡 Stream URL: $RTSP_URL"
echo "⚙️  Encoding: libx264, veryfast, zerolatency, keyint=30"
echo "=================================================="

# Infinite loop to play the playlist repeatedly
while true; do
    # Find all video files in the directory, sort them, and play one by one
    find "$VIDEO_DIR" -type f \( -name "*.mp4" -o -name "*.mkv" -o -name "*.avi" \) | sort | while read video_file; do
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ▶️  Now Playing: $(basename "$video_file")"
        
        FFMPEG_CMD="/home/cluster/miniforge3/envs/webapp_mongo_gpu/bin/ffmpeg"

        # Use the optimized ffmpeg command provided by user
        # Note: -stream_loop -1 is removed to allow switching to the next file
        $FFMPEG_CMD -re -i "$video_file" \
          -c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p \
          -g 30 -keyint_min 30 -bf 0 \
          -an -f rtsp -rtsp_transport tcp "$RTSP_URL"

    done
    
    echo "🔄 Playlist finished. Restarting in 1 second..."
    sleep 1
done
