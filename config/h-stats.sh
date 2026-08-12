#!/usr/bin/env bash
# Sourced by HiveOS: exports total kH/s in $khs and dashboard JSON in $stats.
# HIVE_MANIFEST is the override hook test-h-stats.sh uses; rigs get the sibling
# file. BASH_SOURCE, not $0 -- the agent sources this, so $0 is the agent.
# shellcheck source=config/h-manifest.conf disable=SC1091
. "${HIVE_MANIFEST:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/h-manifest.conf}"

log="${CUSTOM_LOG_BASENAME}.log"
hs=0
acc=0
rej=0
if [[ -f $log ]]; then
    # `|| true` throughout: before the first status line the log exists but has
    # nothing to match, so grep exits 1. That is a normal state, not an error,
    # and it would abort the whole hook if the agent sourced it under `set -e`.
    line=$(tail -c 16384 "$log" 2>/dev/null | tr '\r' '\n' | grep 'accepted(miniblocks)=' | tail -n1 || true)
    if [[ -n $line ]]; then
        hs=$(grep -oE '[0-9]+([.][0-9]+)? H/s' <<<"$line" | head -n1 | grep -oE '[0-9]+([.][0-9]+)?' || true)
        acc=$(grep -oE 'accepted\(miniblocks\)=[0-9]+' <<<"$line" | grep -oE '[0-9]+' || true)
        rej=$(grep -oE 'rejected=[0-9]+' <<<"$line" | grep -oE '[0-9]+' || true)
    fi
fi
hs=${hs:-0}; acc=${acc:-0}; rej=${rej:-0}
khs=$(awk -v hs="$hs" 'BEGIN { printf "%.6f", hs / 1000 }')
# Rig uptime, not miner uptime. The status line this parser reads carries no
# elapsed field (cmd/go-gpu-miner/main.go), so there is nothing better to read
# without changing the miner's output; on a dedicated rig the two track closely.
uptime=$(cut -d. -f1 /proc/uptime 2>/dev/null || true); uptime=${uptime:-0}
stats=$(printf '{"hs":[%s],"hs_units":"khs","uptime":%s,"ar":[%s,%s],"algo":"ASTROBWT","ver":"%s"}' \
    "$khs" "$uptime" "$acc" "$rej" "${CUSTOM_VERSION:-dev}")
