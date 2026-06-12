#!/usr/bin/env bash
# 启动 / 重启本机 RAG API 后端，日志写入 logs/pipeline-api.log。
# 用法:  bash run_api.sh
# 看日志: tail -f logs/pipeline-api.log
#
# 注意: 若到服务器的 SSH 隧道断开重连过, 后端持有的 Milvus 连接会失效,
#       检索会静默返回 0 条。此时重新运行本脚本即可重建连接。

set -uo pipefail
cd "$(dirname "$0")"

PORT="${API_PORT:-8080}"
LOG_FILE="logs/pipeline-api.log"
mkdir -p logs

# 停掉占用端口的旧进程
OLD_PIDS="$(lsof -tnP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [ -n "$OLD_PIDS" ]; then
  echo "[run_api] 停止旧后端 PID: $OLD_PIDS"
  kill -9 $OLD_PIDS 2>/dev/null || true
  sleep 1
fi

echo "[run_api] 启动后端 :$PORT, 日志 -> $LOG_FILE"
PYTHONUNBUFFERED=1 \
CONFIG_PATH="${CONFIG_PATH:-local_api_config.yaml}" \
CORS_ORIGINS="${CORS_ORIGINS:-https://rag.hal9k.one,http://localhost:9527}" \
API_PORT="$PORT" \
nohup .venv-api/bin/python run_api.py >>"$LOG_FILE" 2>&1 &

sleep 2
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[run_api] 已启动。健康检查:"
  curl -s --max-time 20 "http://localhost:$PORT/api/v1/health"; echo
else
  echo "[run_api] 启动失败，请查看 $LOG_FILE" >&2
  tail -n 20 "$LOG_FILE" >&2
  exit 1
fi
