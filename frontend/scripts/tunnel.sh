#!/usr/bin/env bash
# 确保到 GPU/Milvus 服务器的 SSH 端口转发隧道处于连接状态（幂等）。
# 被 `npm run dev` 的 predev 钩子调用；也可手动执行：bash scripts/tunnel.sh
#
# 若 key 带密码：先执行一次（仅一次，需手动输入密码，存入 Keychain）：
#   ssh-add --apple-use-keychain ~/.ssh/id_ed25519
# 之后本脚本即可静默连接；否则会在当前终端提示输入密码。

set -uo pipefail

SSH_HOST="root@221.194.152.152"
SSH_PORT="8010"
SSH_KEY="$HOME/.ssh/id_ed25519"
# 需要转发的端口（本地:127.0.0.1:远程 一一对应）
FORWARD_PORTS=(8000 8001 8002 19530 3000)
# 用于探测隧道是否已建立的代表端口（Milvus）
PROBE_PORT="19530"

if nc -z -w2 127.0.0.1 "$PROBE_PORT" >/dev/null 2>&1; then
  echo "[tunnel] 已连接（127.0.0.1:${PROBE_PORT} 可达），跳过。"
  exit 0
fi

if [ ! -f "$SSH_KEY" ]; then
  echo "[tunnel] 错误：找不到 SSH key: $SSH_KEY" >&2
  exit 1
fi

L_ARGS=()
for p in "${FORWARD_PORTS[@]}"; do
  L_ARGS+=("-L" "${p}:127.0.0.1:${p}")
done

echo "[tunnel] 正在建立到 ${SSH_HOST} 的隧道 ..."
ssh -f -N \
  -p "$SSH_PORT" \
  -i "$SSH_KEY" \
  -o AddKeysToAgent=yes \
  -o UseKeychain=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -o StrictHostKeyChecking=accept-new \
  "${L_ARGS[@]}" \
  "$SSH_HOST"

# 等待端口可达（最多 ~10s）
for _ in $(seq 1 10); do
  if nc -z -w2 127.0.0.1 "$PROBE_PORT" >/dev/null 2>&1; then
    echo "[tunnel] 已连接。"
    exit 0
  fi
  sleep 1
done

echo "[tunnel] 警告：隧道进程已启动，但 127.0.0.1:${PROBE_PORT} 暂不可达。" >&2
echo "[tunnel] 若 key 带密码，请先执行：ssh-add --apple-use-keychain ${SSH_KEY}" >&2
exit 0
