#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
if [ ! -x .venv/bin/python ]; then
  echo '[ERROR] Missing local environment. Run: bash setup.sh' >&2
  exit 1
fi
echo "[INFO] BGE intent router: http://${ROUTER_HOST:-127.0.0.1}:${ROUTER_PORT:-8890}"
exec .venv/bin/python -m uvicorn app:app \
  --host "${ROUTER_HOST:-127.0.0.1}" \
  --port "${ROUTER_PORT:-8890}" \
  --workers 1
