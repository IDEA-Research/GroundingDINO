#!/bin/bash
# 獨立啟動 NICU 即時監測 MCP（Streamable HTTP）。
# 不接 Cursor、不啟動/重啟 medical-webapp。

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv-mcp-stream"
REQ="$ROOT/requirements-mcp-stream.txt"
HOST="${MCP_STREAM_HOST:-127.0.0.1}"
PORT="${MCP_STREAM_PORT:-8765}"
export WEB_APP_BASE_URL="${WEB_APP_BASE_URL:-http://127.0.0.1:3001}"

if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f "$ROOT/.env.systemd" ]; then
  KEY_LINE="$(grep -E '^OPENROUTER_API_KEY=' "$ROOT/.env.systemd" | tail -n 1 || true)"
  if [ -n "$KEY_LINE" ]; then
    export OPENROUTER_API_KEY="${KEY_LINE#OPENROUTER_API_KEY=}"
    OPENROUTER_API_KEY="${OPENROUTER_API_KEY%\"}"
    OPENROUTER_API_KEY="${OPENROUTER_API_KEY#\"}"
    OPENROUTER_API_KEY="${OPENROUTER_API_KEY%\'}"
    OPENROUTER_API_KEY="${OPENROUTER_API_KEY#\'}"
    export OPENROUTER_API_KEY
  fi
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo "建立獨立 venv: $VENV（不會裝進 webapp_mongo_gpu）"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install -r "$REQ"
fi

if ! curl -fsS --max-time 2 "$WEB_APP_BASE_URL/api/mongodb/status" >/dev/null 2>&1; then
  echo "警告: 連不上 web_app ($WEB_APP_BASE_URL)。MCP 仍會啟動，但 tools 會失敗。"
  echo "請先確認 medical-webapp / python web_app.py 已在 :3001；這個 script 不會幫你啟動它。"
fi

echo "啟動 NICU stream MCP: http://${HOST}:${PORT}/mcp"
exec "$VENV/bin/python" "$ROOT/mcp_stream_monitor.py" --host "$HOST" --port "$PORT"
