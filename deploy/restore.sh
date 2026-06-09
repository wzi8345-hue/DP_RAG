#!/usr/bin/env bash
# DP-RAG Postgres 全量恢复（与 backup.sh 配套）。
# 用法: ./restore.sh ./backups/dprag_YYYYmmdd_HHMMSS.sql.gz
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] && set -a && . ./.env && set +a

FILE="${1:?用法: ./restore.sh <backup.sql.gz>}"
POSTGRES_USER="${POSTGRES_USER:-dprag}"
POSTGRES_DB="${POSTGRES_DB:-dprag}"

echo "[restore] $FILE -> $POSTGRES_DB （将覆盖现有数据）"
gunzip -c "$FILE" | docker compose exec -T postgres \
  psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"
echo "[restore] done"
