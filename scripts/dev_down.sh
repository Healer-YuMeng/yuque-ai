#!/usr/bin/env bash
set -euo pipefail

stop_port() {
  local port="$1"
  local pids
  pids="$(lsof -ti tcp:"$port" || true)"
  if [ -n "$pids" ]; then
    echo "==> 停止端口 ${port}: ${pids}"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
  else
    echo "==> 端口 ${port} 未发现进程"
  fi
}

stop_port 8000
stop_port 5173

echo "==> 完成"
