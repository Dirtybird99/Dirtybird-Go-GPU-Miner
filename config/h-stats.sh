#!/usr/bin/env bash
# Sourced by HiveOS: exports total kH/s in $khs and dashboard JSON in $stats.
dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$dir/h-manifest.conf"

log="${CUSTOM_LOG_BASENAME}.log"
hs=0
acc=0
rej=0
if [[ -f $log ]]; then
    line=$(tail -c 16384 "$log" 2>/dev/null | tr '\r' '\n' | grep 'accepted(miniblocks)=' | tail -n1)
    if [[ -n $line ]]; then
        hs=$(grep -oE '[0-9]+([.][0-9]+)? H/s' <<<"$line" | head -n1 | grep -oE '[0-9]+([.][0-9]+)?')
        acc=$(grep -oE 'accepted\(miniblocks\)=[0-9]+' <<<"$line" | grep -oE '[0-9]+')
        rej=$(grep -oE 'rejected=[0-9]+' <<<"$line" | grep -oE '[0-9]+')
    fi
fi
hs=${hs:-0}; acc=${acc:-0}; rej=${rej:-0}
khs=$(awk -v hs="$hs" 'BEGIN { printf "%.6f", hs / 1000 }')
uptime=$(cut -d. -f1 /proc/uptime 2>/dev/null); uptime=${uptime:-0}
stats=$(printf '{"hs":[%s],"hs_units":"khs","uptime":%s,"ar":[%s,%s],"algo":"ASTROBWT"}' "$khs" "$uptime" "$acc" "$rej")
