#!/usr/bin/env bash
# Backend launcher for launchd KeepAlive. Loads .env.local then runs uvicorn in foreground.
# ASCII-only + absolute path: launchd runs under C locale and the project path
# contains non-ASCII chars, so we must set UTF-8 locale and avoid $0/dirname.
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

PROJECT_DIR="/Users/dp/Desktop/工作文件/DP_rag_skill"
cd "$PROJECT_DIR" || exit 1

# launchd 给的 soft fd 上限默认仅 256, 批量上传 (multipart 落临时文件) +
# Milvus gRPC 连接很容易耗尽导致 400; 抬高到 8192 (hard 为 unlimited)。
ulimit -n 8192 2>/dev/null || true

if [ -f ./.env.local ]; then
  set -a
  . ./.env.local
  set +a
fi

mkdir -p logs
export PYTHONUNBUFFERED=1
export CONFIG_PATH="${CONFIG_PATH:-local_api_config.yaml}"
export CORS_ORIGINS="${CORS_ORIGINS:-*}"
export API_PORT="${API_PORT:-8080}"

exec .venv-api/bin/python run_api.py
