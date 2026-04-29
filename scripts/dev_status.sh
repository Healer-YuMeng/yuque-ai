#!/usr/bin/env bash
set -euo pipefail

check_port() {
  local port="$1"
  echo "==> 端口 ${port}"
  local rows
  rows="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN || true)"
  if [ -z "$rows" ]; then
    echo "未监听"
  else
    echo "$rows"
  fi
  echo
}

check_port 8000
check_port 5173
