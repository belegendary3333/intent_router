#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
if [ ! -x .venv-lab/bin/python ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python "${LAB_PYTHON:-3.13}" .venv-lab
  else
    task_python="${LAB_PYTHON:-python3}"
    "$task_python" -c 'import sys; assert (3,12) <= sys.version_info[:2] <= (3,13), "Use Python 3.12–3.13"'
    "$task_python" -m venv .venv-lab
  fi
fi
task_requirements=requirements-lab.txt
if [ -f requirements-lab.lock ]; then task_requirements=requirements-lab.lock; fi
if command -v uv >/dev/null 2>&1; then
  uv pip install --python .venv-lab/bin/python -r "$task_requirements"
else
  .venv-lab/bin/python -m pip install -r "$task_requirements"
fi
.venv-lab/bin/python download_lab_model.py
echo '[OK] 本地分析环境已准备。启动：bash run_lab.sh'
