#!/usr/bin/env bash
set -o pipefail
cd "$(dirname "$(readlink -f "$0")")" || exit 1
. ./h-manifest.conf

killall -9 go-gpu-miner >/dev/null 2>&1 || true
[[ -s $CUSTOM_CONFIG_FILENAME ]] || { echo "missing miner config; set pool and wallet in the flight sheet" >&2; exit 1; }
mkdir -p "$CUSTOM_LOG_BASEDIR"
read -ra args < "$CUSTOM_CONFIG_FILENAME"
GOMINER_FORCE_STATUS=1 ./go-gpu-miner "${args[@]}" "$@" 2>&1 | tee -a "${CUSTOM_LOG_BASENAME}.log"
