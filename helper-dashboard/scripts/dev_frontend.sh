#!/usr/bin/env bash
# Start backend (uvicorn) + frontend (next) together.
# Sources .env if present. Traps EXIT so Ctrl-C kills both.
#
# Usage: ./scripts/dev_frontend.sh [--mode mock|opencode|auto] [--debug]
#
# Default mode: whatever is in .env (HELPER_DASHBOARD_OPENCODE), or
# `mock` if unset.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROOT="$(pwd)"

# --- 1. Load .env if present ------------------------------------------
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
  echo "[dev] loaded .env"
fi

# --- 2. CLI overrides --------------------------------------------------
MODE="${HELPER_DASHBOARD_OPENCODE:-mock}"
FRONTEND_DEBUG=0
while [ $# -gt 0 ]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --debug)
      FRONTEND_DEBUG=1
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [--mode mock|opencode|auto] [--debug]"
      exit 0
      ;;
    *)
      echo "[dev] unknown arg: $1" >&2
      exit 2
      ;;
  esac
done
export HELPER_DASHBOARD_OPENCODE="$MODE"

# --- 3. Runtime sanity checks ------------------------------------------
if [ "$MODE" = "opencode" ] || [ "$MODE" = "auto" ]; then
  BIN="${HELPER_DASHBOARD_OPENCODE_BIN:-./bin/opencode}"
  if [ ! -x "$BIN" ]; then
    echo "[dev] OpenCode binary not executable at $BIN" >&2
    exit 3
  fi
  if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[dev] no LLM API key set; bin/opencode will fail unless" \
         "OPENCODE_LLM_PROVIDER=heuristic."
  fi
fi


# --- 5. Start frontend -------------------------------------------------
echo "[dev] starting frontend on :3050"
(
  cd frontend
  if [ ! -d node_modules ]; then
    echo "[dev] first run: installing npm deps"
    npm install --no-audit --no-fund
  fi
  if [ "$FRONTEND_DEBUG" = "1" ]; then
    echo "[dev] frontend debug mode: next dev"
    NEXT_PUBLIC_API_BASE="http://127.0.0.1:8000" \
      npx next dev -p 3050
  else
    # Production build once for cleaner reload.
    if [ ! -d .next ]; then
      npm run build
    fi
    NEXT_PUBLIC_API_BASE="http://127.0.0.1:8000" \
      npx next start -p 3050
  fi

)
