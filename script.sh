#!/usr/bin/env bash
# Dirtybird Go GPU Miner -- source/release launcher.
set -euo pipefail
cd "$(dirname "$0")"

if [ -f ./go-gpu-miner.exe ]; then
  BIN=./go-gpu-miner.exe
elif [ -f ./go-gpu-miner ]; then
  BIN=./go-gpu-miner
else
  command -v go >/dev/null 2>&1 || { echo "error: install Go 1.25+ and retry" >&2; exit 1; }
  COMMIT="$(git rev-parse --short=12 HEAD 2>/dev/null || printf unknown)"
  CGO_ENABLED=0 go build -trimpath -ldflags "-s -w -X main.commit=${COMMIT}" -o go-gpu-miner ./cmd/go-gpu-miner
  BIN=./go-gpu-miner
fi

[ -f config.json ] || [ ! -f config.example.json ] || cp config.example.json config.json
echo "Configure config.json or pass -d and -w on the command line."
exec "$BIN" "$@"
