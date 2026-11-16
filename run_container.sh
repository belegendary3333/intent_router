#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="$PWD/.offline-site"
mkdir -p logs
if [ ! -d .offline-site ] || [ ! -f model/onnx/model_quantized.onnx ]; then
  echo '[ERROR] Offline dependencies or model are missing.' >&2
  exit 1
fi
echo "[INFO] BGE intent router: http://${ROUTER_HOST:-127.0.0.1}:${ROUTER_PORT:-8890}"
exec python3 -m uvicorn standalone_app:app \
  --host "${ROUTER_HOST:-127.0.0.1}" \
  --port "${ROUTER_PORT:-8890}" \
  --workers 1
