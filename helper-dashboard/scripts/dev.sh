#!/usr/bin/env bash 
# Start backend (uvicorn) + frontend (next) together.
# Sources .env when readable. Traps EXIT so Ctrl-C kills both.
#
# Usage: ./scripts/dev.sh [--mode mock|helpercode|auto]
#
# Default mode: whatever is in .env (HELPER_DASHBOARD_HELPERCODE), or
# `mock` if unset.

set -euo pipefail
# Job control: without this, backgrounded pipelines (the `( ... ) | sed ...
# &` blocks below) share THIS script's own process group, so `kill $PID`
# in cleanup() only signals the immediate subshell — any descendant that
# re-execs through a few layers (npx -> sh -c -> the actual next-server
# binary) survives and keeps the port bound. With `set -m`, each `&` job
# becomes its own process group (PGID == the job's first PID), so
# `kill -TERM -PGID` below reliably reaches the whole tree.
set -m
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROOT="$(pwd)"
BACKEND_PORT="${HELPER_DASHBOARD_BACKEND_PORT:-8010}"
BACKEND_URL="http://127.0.0.1:${BACKEND_PORT}"

# --- 1. Load .env if readable -----------------------------------------
# systemd loads a root-owned .env through EnvironmentFile before dropping
# privileges to User=cluster, so skip sourcing it again when it is unreadable.
if [ -r .env ]; then
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
# Tell the browser evaluator where the frontend actually lives. Without
# this, it defaults to `http://localhost:3000` and silently evaluates
# whatever else is on that port (or nothing) — every "missing widgets"
# report becomes a false positive that escalates to the rescue agent.
export HELPER_DASHBOARD_FRONTEND_URL="${HELPER_DASHBOARD_FRONTEND_URL:-http://127.0.0.1:3050}"
echo "[dev] starting backend on :$BACKEND_PORT (mode=$MODE, frontend_url=$HELPER_DASHBOARD_FRONTEND_URL)"
(
  cd backend
  PYTHONPATH=. python3 -m uvicorn app.main:app \
    --host 127.0.0.1 --port "$BACKEND_PORT" --log-level info \
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
NEXT_API_BASE_DEV="${NEXT_PUBLIC_API_BASE:-$BACKEND_URL}"
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
# Signal the whole PROCESS GROUP of each job (negative PID), not just the
# subshell PID, so descendants that re-exec through npx/sh -c/next-server
# actually die instead of surviving to squat on the port for the next
# cycle (this is what caused the crash-restart loop under systemd's
# Restart=always: a leaked next-server from cycle N held :3050, so
# cycle N+1's frontend failed EADDRINUSE, killed its sibling, and exited).
cleanup() {
  echo ""
  echo "[dev] shutting down..."
  kill -TERM -"$BACK_PID" -"$FRONT_PID" 2>/dev/null || true
  sleep 1
  kill -KILL -"$BACK_PID" -"$FRONT_PID" 2>/dev/null || true
  wait "$BACK_PID" "$FRONT_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- 7. Wait for health ------------------------------------------------
for i in $(seq 1 40); do
  if curl -fsS -m 1 "$BACKEND_URL/api/health" >/dev/null 2>&1; then
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
