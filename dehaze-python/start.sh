#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR="$ROOT_DIR/logs"
APP_NAME="dehaze-python"
PID_FILE="$LOG_DIR/${APP_NAME}.pid"
XXLJOB_PID_FILE="$LOG_DIR/pyxxl.pid"
NOHUP_LOG="$LOG_DIR/${APP_NAME}.log"

PORT="${DEHAZE_PYTHON_PORT:-8000}"
HOST="${DEHAZE_PYTHON_HOST:-0.0.0.0}"
WORKERS="${DEHAZE_PYTHON_WORKERS:-1}"

mkdir -p "$LOG_DIR"

port_in_use() {
  ss -lnt "sport = :${PORT}" | awk 'NR>1{print $0}' | grep -q .
}

is_running_by_pid() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid=$(cat "$PID_FILE")
    if [[ -n "$pid" ]] && ps -p "$pid" > /dev/null 2>&1; then
      return 0
    fi
  fi
  return 1
}

stop_xxljob() {
  if [[ -f "$XXLJOB_PID_FILE" ]]; then
    local xxl_pid
    xxl_pid=$(cat "$XXLJOB_PID_FILE")
    if [[ -n "$xxl_pid" ]] && ps -p "$xxl_pid" > /dev/null 2>&1; then
      echo "stopping xxl-job executor (pid=${xxl_pid})"
      kill "$xxl_pid" || true
    fi
    rm -f "$XXLJOB_PID_FILE"
  fi
}

stop_running() {
  # 先停止 XXL-Job 子进程
  stop_xxljob

  if is_running_by_pid; then
    local pid
    pid=$(cat "$PID_FILE")
    echo "stopping ${APP_NAME} (pid=${pid})"
    kill "$pid" || true
    for _ in {1..10}; do
      if ps -p "$pid" > /dev/null 2>&1; then
        sleep 1
      else
        break
      fi
    done
  else
    local pids
    pids=$(pgrep -f "uvicorn app.main:app.*${PORT}" || true)
    if [[ -n "$pids" ]]; then
      echo "stopping ${APP_NAME} (pid=${pids})"
      kill $pids || true
    fi
  fi
  rm -f "$PID_FILE"
}

if is_running_by_pid || pgrep -f "uvicorn app.main:app.*${PORT}" > /dev/null 2>&1; then
  stop_running
fi

if port_in_use; then
  echo "port ${PORT} already in use, abort"
  exit 1
fi

# 激活虚拟环境
if [[ -d "$ROOT_DIR/.venv" ]]; then
  source "$ROOT_DIR/.venv/bin/activate"
elif command -v uv &> /dev/null; then
  echo "virtual environment not found, creating..."
  uv venv "$ROOT_DIR/.venv" --python 3.11
  source "$ROOT_DIR/.venv/bin/activate"
else
  echo "no .venv found and uv not installed, abort"
  exit 1
fi

# 同步依赖
if command -v uv &> /dev/null; then
  echo "syncing dependencies..."
  uv sync --project "$ROOT_DIR"
fi

echo "starting ${APP_NAME} on ${HOST}:${PORT} (workers=${WORKERS})..."

nohup uvicorn app.main:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers "$WORKERS" \
  > "$NOHUP_LOG" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

sleep 2
if ps -p "$PID" > /dev/null 2>&1; then
  echo "started ${APP_NAME} (pid=${PID}), log: ${NOHUP_LOG}"
  # 等待 XXL-Job 子进程 PID 文件生成
  for _ in {1..5}; do
    if [[ -f "$XXLJOB_PID_FILE" ]]; then
      echo "xxl-job executor (pid=$(cat "$XXLJOB_PID_FILE")), pid_file: ${XXLJOB_PID_FILE}"
      break
    fi
    sleep 1
  done
else
  echo "start failed, check log: ${NOHUP_LOG}"
  exit 1
fi
