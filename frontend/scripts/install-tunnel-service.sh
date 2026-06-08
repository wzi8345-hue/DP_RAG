#!/usr/bin/env bash
# 安装/卸载 launchd 隧道服务（开机自启 + 断线自动重连）。
# 用法:
#   bash scripts/install-tunnel-service.sh            # 安装并启动
#   bash scripts/install-tunnel-service.sh uninstall  # 停止并卸载

set -uo pipefail

LABEL="com.dprag.tunnel"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_PLIST="$SCRIPT_DIR/${LABEL}.plist"
DEST_DIR="$HOME/Library/LaunchAgents"
DEST_PLIST="$DEST_DIR/${LABEL}.plist"
UID_NUM="$(id -u)"

uninstall() {
  echo "[service] 停止并卸载 $LABEL ..."
  launchctl bootout "gui/${UID_NUM}/${LABEL}" 2>/dev/null || launchctl unload "$DEST_PLIST" 2>/dev/null || true
  rm -f "$DEST_PLIST"
  echo "[service] 已卸载。"
}

if [ "${1:-}" = "uninstall" ]; then
  uninstall
  exit 0
fi

# key 带密码时，确保已存入 Keychain，否则无人值守连接会失败。
if ! ssh-keygen -y -P "" -f "$HOME/.ssh/id_ed25519" >/dev/null 2>&1; then
  if ! ssh-add -l 2>/dev/null | grep -q .; then
    echo "[service] 提示：你的 key 带密码且 ssh-agent 为空。"
    echo "          请先执行一次（需手动输入密码，存入 Keychain）："
    echo "            ssh-add --apple-use-keychain ~/.ssh/id_ed25519"
    echo "          完成后再运行本安装脚本。"
  fi
fi

mkdir -p "$DEST_DIR"
cp "$SRC_PLIST" "$DEST_PLIST"
echo "[service] 已复制 plist 到 $DEST_PLIST"

# 先卸载旧的（忽略错误），再加载
launchctl bootout "gui/${UID_NUM}/${LABEL}" 2>/dev/null || launchctl unload "$DEST_PLIST" 2>/dev/null || true
if launchctl bootstrap "gui/${UID_NUM}" "$DEST_PLIST" 2>/dev/null; then
  :
else
  launchctl load -w "$DEST_PLIST"
fi
launchctl enable "gui/${UID_NUM}/${LABEL}" 2>/dev/null || true

echo "[service] 已加载。日志: /tmp/dprag-tunnel.log"
echo "[service] 验证: nc -z 127.0.0.1 19530 && echo UP"
