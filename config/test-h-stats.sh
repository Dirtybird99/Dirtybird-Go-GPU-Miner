#!/usr/bin/env bash
# The HiveOS dashboard shows whatever h-stats.sh puts in $stats, and this miner
# has no stats API to fall back on -- a parser regression is invisible until a
# rig reports N/A. Pin the parse against captured status lines.
set -euo pipefail
cd "$(dirname "$0")/.."

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cat >"$tmp/manifest" <<EOF
CUSTOM_VERSION=9.9.9
CUSTOM_LOG_BASEDIR=$tmp
CUSTOM_LOG_BASENAME=$tmp/miner
EOF
export HIVE_MANIFEST="$tmp/manifest"

fail() { echo "FAIL: $1" >&2; echo "  stats=$stats" >&2; exit 1; }

# --- the status line as cmd/go-gpu-miner/main.go writes it. Two of them: the
# freshest must win. The trailing counters are traps for a loose `=[0-9]+` match.
cat >"$tmp/miner.log" <<'EOF'
[12:00:00] 21500.0 H/s  height=7212990 diff=311000000  accepted(miniblocks)=1990 rejected=3 submitted=1993 miscompute=0 dropped=0 sendfails=0
[12:00:10] 23760.0 H/s  height=7212998 diff=312000000  accepted(miniblocks)=1998 rejected=4 submitted=2002 miscompute=0 dropped=1 sendfails=2
EOF
(
    . config/h-stats.sh
    [[ $khs == 23.760000 ]]           || fail "khs should be 23.760000, got $khs"
    [[ $stats == *'"ar":[1998,4]'* ]] || fail "ar"
    [[ $stats == *'"ver":"9.9.9"'* ]] || fail "ver from manifest"
)

# --- no log at all: a live zero, never stale or malformed data
rm -f "$tmp/miner.log"
(
    . config/h-stats.sh
    [[ $khs == 0.000000 ]]          || fail "no-log khs should be 0, got $khs"
    [[ $stats == *'"ar":[0,0]'* ]]  || fail "no-log ar"
)

# --- log present but no status line yet (startup, or a reconnect). This is the
# case that makes grep exit 1; without the `|| true` guards, sourcing h-stats.sh
# under `set -e` aborts here instead of reporting zeros.
printf 'GPU: NVIDIA (vulkan)\nstartup gate passed in 1.2s (fused path)\n' >"$tmp/miner.log"
(
    . config/h-stats.sh
    [[ $khs == 0.000000 ]]         || fail "pre-status khs should be 0, got $khs"
    [[ $stats == *'"ar":[0,0]'* ]] || fail "pre-status ar"
)

if command -v jq > /dev/null 2>&1; then
    . config/h-stats.sh
    jq -e . > /dev/null <<< "$stats" || fail "malformed JSON"
fi

echo "h-stats.sh: all cases passed"
