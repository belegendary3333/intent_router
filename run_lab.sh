#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
task_port="${LAB_PORT:-8892}"
if [ ! -x .venv-lab/bin/python ]; then
  echo '[ERROR] 请先运行 bash setup_lab.sh' >&2
  exit 1
fi
.venv-lab/bin/python - "$task_port" <<'PY'
import socket, sys
port = int(sys.argv[1])
if not 1024 <= port <= 65535:
    raise SystemExit('[ERROR] LAB_PORT 必须在 1024–65535 之间')
with socket.socket() as sock:
    try:
        sock.bind(('127.0.0.1', port))
    except OSError:
        raise SystemExit(f'[ERROR] 本机 {port} 端口已占用；请打开现有网页，或用 LAB_PORT=8893 bash run_lab.sh。不会停止其他进程。')
PY
echo "本地意图分析台：http://127.0.0.1:${task_port}/"
echo '保留 small；large 已下载时自动构建双模型对照。首次重建可能较久，请看页面进度。Ctrl+C 停止。'
exec .venv-lab/bin/python -m uvicorn lab_app:app --host 127.0.0.1 --port "$task_port" --workers 1
