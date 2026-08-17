#!/usr/bin/env bash
# Start backend (uvicorn) + frontend (next) together.
# Sources .env if present. Traps EXIT so Ctrl-C kills both.
#
# Usage: ./scripts/dev.sh [--mode mock|helpercode|auto]
#
# Default mode: whatever is in .env (HELPER_DASHBOARD_HELPERCODE), or
# `mock` if unset.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROOT="$(pwd)"
BACKEND_PORT="${HELPER_DASHBOARD_BACKEND_PORT:-8010}"
mkdir -p "$ROOT/logs"

export HELPER_DASHBOARD_HELPERCODE_TRACE="${HELPER_DASHBOARD_HELPERCODE_TRACE:-1}"
export HELPERCODE_DEBUG="${HELPERCODE_DEBUG:-1}"
export HELPERCODE_TRACE_FILE="${HELPERCODE_TRACE_FILE:-$ROOT/logs/helpercode-trace.log}"

# --- 1. Load .env if present ------------------------------------------
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
  echo "[dev] loaded .env"
fi

# --- 2. CLI overrides --------------------------------------------------
MODE="${HELPER_DASHBOARD_HELPERCODE:-mock}"
while [ $# -gt 0 ]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [--mode mock|helpercode|auto]"
      exit 0
      ;;
    *)
      echo "[dev] unknown arg: $1" >&2
      exit 2
      ;;
  esac
done
export HELPER_DASHBOARD_HELPERCODE="$MODE"

# --- 3. Runtime sanity checks ------------------------------------------
if [ "$MODE" = "helpercode" ] || [ "$MODE" = "auto" ]; then
  BIN="${HELPER_DASHBOARD_HELPERCODE_BIN:-./bin/helpercode}"
  if [ ! -x "$BIN" ]; then
    echo "[dev] helperCode binary not executable at $BIN" >&2
    exit 3
  fi
  if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[dev] no LLM API key set; bin/helpercode will fail unless" \
         "HELPERCODE_LLM_PROVIDER=heuristic."
  fi
fi

# --- 4. Start backend --------------------------------------------------
echo "[dev] starting backend on :$BACKEND_PORT (mode=$MODE)"
(
  cd backend
  PYTHONPATH=. python3 -m uvicorn app.main:app \
    --host 127.0.0.1 --port "$BACKEND_PORT" --log-level info \
    2>&1 | sed -u 's/^/[backend] /'
)
BACK_PID=$!

# --- 5. Start frontend -------------------------------------------------
echo "[dev] starting frontend on :3050"
(
  cd frontend
  if [ ! -d node_modules ]; then
    echo "[dev] first run: installing npm deps"
    npm install --no-audit --no-fund
  fi
  # Production build once for cleaner reload; use `next dev` instead
  # if you prefer HMR and don't mind the warning noise.
  if [ ! -d .next ]; then
    npm run build
  fi
  NEXT_PUBLIC_API_BASE="http://127.0.0.1:${BACKEND_PORT}" \
    npx next start -p 3050
 
