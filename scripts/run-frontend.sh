#!/usr/bin/env bash
# Frontend launcher for launchd KeepAlive. Runs Vite dev server in foreground.
# ASCII-only + absolute path + UTF-8 locale (launchd runs under C locale).
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# launchd PATH is minimal; include node location.
export PATH="/Users/dp/.hermes/node/bin:/Users/dp/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

PROJECT_DIR="/Users/dp/Desktop/工作文件/DP_rag_skill/frontend"
cd "$PROJECT_DIR" || exit 1

# Invoke node directly on vite.js (not the .bin/vite env-shebang wrapper) so that
# bash's Full Disk Access responsibility is inherited (single exec hop, like the
# backend python launcher). The env-shebang indirection breaks TCC inheritance.
exec /Users/dp/.hermes/node/bin/node node_modules/vite/bin/vite.js --host 0.0.0.0
