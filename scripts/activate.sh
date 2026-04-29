#!/usr/bin/env sh
set -eu

# 用法：
#   source scripts/activate.sh
#
# 说明：
# - 必须用 source（或 .）执行，才能让虚拟环境在当前终端生效
# - 需要虚拟环境目录位于项目根目录下的 ./yuqueai

VENV_DIR="${VENV_DIR:-yuqueai}"

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "未找到虚拟环境：${VENV_DIR}/bin/activate"
  echo "请确认虚拟环境目录名是否为 '${VENV_DIR}'，或先执行：export VENV_DIR=你的目录名"
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
. "${VENV_DIR}/bin/activate"

echo "已激活虚拟环境：${VIRTUAL_ENV:-unknown}"
python -c "import sys; print('python =', sys.executable)"

