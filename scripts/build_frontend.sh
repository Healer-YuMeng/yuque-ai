#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"

if [ ! -d "$FRONTEND_DIR" ]; then
  echo "frontend 目录不存在：$FRONTEND_DIR"
  exit 1
fi

echo "==> 安装前端依赖"
cd "$FRONTEND_DIR"
/opt/homebrew/bin/npm install

echo "==> 构建前端"
/opt/homebrew/bin/npm run build

echo "==> 完成：$FRONTEND_DIR/dist"
