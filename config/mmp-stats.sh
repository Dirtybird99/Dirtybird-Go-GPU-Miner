#!/usr/bin/env bash
dir=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
. "$dir/h-stats.sh"
printf '{"hs":[%s],"hs_units":"khs","ar":[%s,%s],"algo":"ASTROBWT","miner_name":"%s","miner_version":"%s"}\n' \
    "$khs" "$acc" "$rej" "$CUSTOM_NAME" "$CUSTOM_VERSION"
