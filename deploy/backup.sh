#!/usr/bin/env bash
# DP-RAG Postgres 全量备份（shell 脚本，不用迁移框架）。
# 默认通过 docker compose 的 postgres 服务执行 pg_dump，输出 gzip 压缩的全量 SQL。
#
# 用法:
#   ./backup.sh                 # 备份到 ./backups/dprag_YYYYmmdd_HHMMSS.sql.gz
#   BACKUP_DIR=/data/bk ./backup.sh
#
# 定时: 可加到 crontab, 例如每天 03:00
#   0 3 * * * cd /opt/dprag/deploy && ./backup.sh >> backup.log 2>&1
set -euo pipefail
cd "$(dirname "$0")"

# 载入 .env（POSTGRES_USER / POSTGRES_DB 等）
[ -f .env ] && set -a && . ./.env && set +a

POSTGRES_USER="${POSTGRES_USER:-dprag}"
POSTGRES_DB="${POSTGRES_DB:-dprag}"
OUT_DIR="${BACKUP_DIR:-./backups}"
KEEP="${BACKUP_KEEP:-14}"

mkdir -p "$OUT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="$OUT_DIR/dprag_${TS}.sql.gz"

echo "[backup] dumping $POSTGRES_DB -> $OUT"
docker compose exec -T postgres \
  pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean --if-exists \
  | gzip > "$OUT"

echo "[backup] done: $(du -h "$OUT" | cut -f1)"

# 仅保留最近 KEEP 份
ls -1t "$OUT_DIR"/dprag_*.sql.gz 2>/dev/null | tail -n +"$((KEEP + 1))" | xargs -r rm -f
