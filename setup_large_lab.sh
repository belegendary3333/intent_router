#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
if [ ! -x .venv-lab/bin/python ]; then
  echo '[ERROR] 请先运行 bash setup_lab.sh' >&2
  exit 1
fi
.venv-lab/bin/python download_large_model.py
echo '[OK] large 模型已校验。重新启动 bash run_lab.sh 即可比较 small / large。'
