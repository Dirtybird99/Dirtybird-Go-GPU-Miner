#!/usr/bin/env bash
set -e
cd "$(dirname "$(readlink -f "$0")")"
. ./h-manifest.conf

: "${CUSTOM_URL:?flight-sheet pool URL is required}"
: "${CUSTOM_TEMPLATE:?flight-sheet wallet is required}"

wallet=$CUSTOM_TEMPLATE
[[ $CUSTOM_URL == wss://* ]] && wallet=${wallet%%.*}
mkdir -p "$(dirname "$CUSTOM_CONFIG_FILENAME")"
printf '%s%s\n' "-d $CUSTOM_URL -w $wallet" "${CUSTOM_USER_CONFIG:+ $CUSTOM_USER_CONFIG}" > "$CUSTOM_CONFIG_FILENAME"
