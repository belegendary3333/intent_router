#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.11 .venv
  else
    task_python="${ROUTER_PYTHON:-python3.11}"
    if ! command -v "$task_python" >/dev/null 2>&1; then
      echo '[ERROR] Install Python 3.11 or uv, or set ROUTER_PYTHON to a Python 3.10–3.12 executable.' >&2
      exit 1
    fi
    "$task_python" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), "Use Python 3.10–3.12"'
    "$task_python" -m venv .venv
  fi
fi
if command -v uv >/dev/null 2>&1; then
  uv pip install --python .venv/bin/python -r requirements.txt
else
  .venv/bin/python -m pip install -r requirements.txt
fi
if [ ! -f .env ]; then
  cp .env.example .env
fi
set -a
source .env
set +a
.venv/bin/python export_bge.py "$@"
echo '[OK] Setup complete. Run: bash run.sh'
