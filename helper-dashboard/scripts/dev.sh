#!/usr/bin/env bash 
# Start backend (uvicorn) + frontend (next) together.
# Sources .env if present. Traps EXIT so Ctrl-C kills both.
#
# Usage: ./scripts/dev.sh [--mode mock|opencode|auto]
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
while [ $# -gt 0 ]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [--mode mock|opencode|auto]"
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

# --- 4. Start backend --------------------------------------------------
# Tell the browser evaluator where the frontend actually lives. Without
# this, it defaults to `http://localhost:3000` and silently evaluates
# whatever else is on that port (or nothing) — every "missing widgets"
# report becomes a false positive that escalates to the rescue agent.
export HELPER_DASHBOARD_FRONTEND_URL="${HELPER_DASHBOARD_FRONTEND_URL:-http://127.0.0.1:3050}"
echo "[dev] starting backend on :8000 (mode=$MODE, frontend_url=$HELPER_DASHBOARD_FRONTEND_URL)"
(
  cd backend
  PYTHONPATH=. python3 -m uvicorn app.main:app \
    --host 127.0.0.1 --port 8000 --log-level info \
    2>&1 | sed -u 's/^/[backend] /'
) &
BACK_PID=$!

# --- 5. Start frontend -------------------------------------------------
# IMPORTANT: next.config.js's `rewrites()` is fully evaluated at
# `next build` time, not at `next start` time. A build done with one
# NEXT_PUBLIC_API_BASE will keep proxying to that backend URL forever,
# even if you `next start` later with a different env. We therefore:
#   1. Always pass NEXT_PUBLIC_API_BASE to both build and start.
#   2. Rebuild whenever the cached "baked" URL doesn't match the
#      current target (sentinel file under .next/).
NEXT_API_BASE_DEV="${NEXT_PUBLIC_API_BASE:-http://127.0.0.1:8000}"
echo "[dev] starting frontend on :3050 (backend=$NEXT_API_BASE_DEV)"
(
  cd frontend
  if [ ! -d node_modules ]; then
    echo "[dev] first run: installing npm deps"
    npm install --no-audit --no-fund
  fi
  BAKED_FILE=".next/.api_base_baked"
  NEED_BUILD=1
  if [ -d .next ] && [ -f "$BAKED_FILE" ]; then
    if [ "$(cat "$BAKED_FILE")" = "$NEXT_API_BASE_DEV" ]; then
      NEED_BUILD=0
    else
      echo "[dev] NEXT_PUBLIC_API_BASE changed since last build (was=" \
           "$(cat "$BAKED_FILE") now=$NEXT_API_BASE_DEV); rebuilding"
    fi
  fi
  if [ "$NEED_BUILD" = "1" ]; then
    NEXT_PUBLIC_API_BASE="$NEXT_API_BASE_DEV" npm run build
    mkdir -p .next
    printf '%s' "$NEXT_API_BASE_DEV" > "$BAKED_FILE"
  fi
  NEXT_PUBLIC_API_BASE="$NEXT_API_BASE_DEV" \
    npx next start -p 3050 \
    2>&1 | sed -u 's/^/[frontend] /'
) &
FRONT_PID=$!

# --- 6. Trap for clean shutdown ----------------------------------------
cleanup() {
  echo ""
  echo "[dev] shutting down..."
  kill "$BACK_PID" "$FRONT_PID" 2>/dev/null || true
  wait "$BACK_PID" "$FRONT_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- 7. Wait for health ------------------------------------------------
for i in $(seq 1 40); do
  if curl -fsS -m 1 http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    echo "[dev] backend healthy"
    break
  fi
  sleep 0.5
done

for i in $(seq 1 60); do
  if curl -fsS -m 1 http://127.0.0.1:3050/ >/dev/null 2>&1; then
    echo "[dev] frontend healthy"
    echo "[dev] -> http://127.0.0.1:3050"
    break
  fi
  sleep 0.5
done

# --- 8. Wait for either to exit ----------------------------------------
wait -n "$BACK_PID" "$FRONT_PID" 2>/dev/null || true
echo "[dev] one process exited; shutting down the other..."
