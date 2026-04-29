#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cleanup() {
  if [ -n "${API_PID:-}" ] && kill -0 "$API_PID" 2>/dev/null; then
    kill "$API_PID" 2>/dev/null || true
  fi
  if [ -n "${WEB_PID:-}" ] && kill -0 "$WEB_PID" 2>/dev/null; then
    kill "$WEB_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

cd "$ROOT_DIR"
source scripts/activate.sh

echo "==> 启动 FastAPI (http://127.0.0.1:8000)"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
API_PID=$!

echo "==> 启动 Vite (http://127.0.0.1:5173)"
cd "$ROOT_DIR/frontend"
/opt/homebrew/bin/npm install >/dev/null
/opt/homebrew/bin/npm run dev -- --host 0.0.0.0 --port 5173 &
WEB_PID=$!

wait
