#!/bin/bash
# Start LongVideoAgent: backend (FastAPI, port 8123) + frontend (Vite, port 5173).
#
# Usage:
#   ./run.sh           # start both
#   ./run.sh backend   # start backend only
#   ./run.sh frontend  # start frontend only
#   ./run.sh stop      # stop both
set -e
cd "$(dirname "$0")"

CONDA_ENV="zjx_openvla"
BACKEND_PORT=8123
FRONTEND_PORT=5173

log() { echo "[$(date +%H:%M:%S)] $*"; }

stop() {
  log "stopping backend/frontend..."
  pkill -f "uvicorn api.app:app" 2>/dev/null || true
  pkill -f "vite" 2>/dev/null || true
  log "stopped."
}

start_backend() {
  log "starting backend on :$BACKEND_PORT (loads Qwen3-VL + video memory)..."
  source /home/ps/.miniconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV"
  nohup uvicorn api.app:app --port "$BACKEND_PORT" --log-level warning \
    > /tmp/lva-backend.log 2>&1 &
  log "backend started, health: http://127.0.0.1:$BACKEND_PORT/health"
}

start_frontend() {
  log "starting frontend on :$FRONTEND_PORT ..."
  (cd frontend && npm run dev > /tmp/lva-frontend.log 2>&1 &)
  log "frontend: http://localhost:$FRONTEND_PORT"
}

case "${1:-all}" in
  stop)
    stop
    ;;
  backend)
    start_backend
    ;;
  frontend)
    start_frontend
    ;;
  all|*)
    start_backend
    start_frontend
    log "done. open http://localhost:$FRONTEND_PORT"
    ;;
esac
