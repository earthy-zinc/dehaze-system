#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR="$ROOT_DIR/log"
BIN_DIR="$ROOT_DIR/bin"
BIN_NAME="dehaze-go"
BIN_PATH="$BIN_DIR/$BIN_NAME"
PID_FILE="$LOG_DIR/${BIN_NAME}.pid"
NOHUP_LOG="$LOG_DIR/${BIN_NAME}.log"

PORT="${DEHAZE_PORT:-8990}"

mkdir -p "$LOG_DIR" "$BIN_DIR"

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

stop_running() {
  if is_running_by_pid; then
    local pid
    pid=$(cat "$PID_FILE")
    echo "stopping ${BIN_NAME} (pid=${pid})"
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
    pids=$(pgrep -f "$BIN_PATH" || true)
    if [[ -n "$pids" ]]; then
      echo "stopping ${BIN_NAME} (pid=${pids})"
      kill $pids || true
    fi
  fi
  rm -f "$PID_FILE"
}

if is_running_by_pid || pgrep -f "$BIN_PATH" > /dev/null 2>&1; then
  stop_running
fi

if port_in_use; then
  echo "port ${PORT} already in use, abort"
  exit 1
fi

echo "building ${BIN_NAME}..."
go build -o "$BIN_PATH" ./cmd/main.go

nohup "$BIN_PATH" > "$NOHUP_LOG" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

sleep 1
if ps -p "$PID" > /dev/null 2>&1; then
  echo "started ${BIN_NAME} (pid=${PID}), log: ${NOHUP_LOG}"
else
  echo "start failed, check log: ${NOHUP_LOG}"
  exit 1
fi
