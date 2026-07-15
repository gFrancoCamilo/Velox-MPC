#!/usr/bin/env bash
# Aggregate VELOX_PROFILE tables across per-node logs.
#
# The profiler (protocol/src/profiler.rs) makes each node log its own
# building-block span table and phase timeline whenever VELOX_PROFILE is set.
# This script pulls the LAST table each node emitted (the most complete one)
# for both the mpc and acss contexts, so you can spot the bottleneck and
# compare CPU vs GPU runs side by side.
#
# Usage:
#   VELOX_PROFILE=1 TYPE=release ./scripts/test.sh 10 <msgs> <comp>   # run
#   ./scripts/profile_report.sh logs                                  # report
#
# Arg 1: directory holding party-*.log files (default: logs).

set -euo pipefail
LOGDIR="${1:-logs}"

if [ ! -d "$LOGDIR" ]; then
    echo "no such log dir: $LOGDIR" >&2
    exit 1
fi

shopt -s nullglob
logs=("$LOGDIR"/party-*.log)
if [ ${#logs[@]} -eq 0 ]; then
    echo "no party-*.log files in $LOGDIR" >&2
    exit 1
fi

# Print the last profiler block of a given kind ("mpc node" / "acss node").
# A block is the header line plus the indented rows that follow it.
print_last_block() {
    local log="$1" kind="$2"
    local start
    start=$(grep -n "\[profile:.*${kind}" "$log" | tail -1 | cut -d: -f1 || true)
    [ -z "${start:-}" ] && return 0
    awk -v s="$start" '
        NR==s { print; next }
        NR>s {
            # stop at the first line that is neither an indented table row
            # nor another profile header
            if ($0 !~ /^  / && $0 !~ /\[profile:/) exit
            print
        }
    ' "$log"
    echo
}

for log in "${logs[@]}"; do
    grep -q "\[profile:" "$log" || continue
    echo "============================================================"
    echo "  $log"
    echo "============================================================"
    print_last_block "$log" "mpc node"
    print_last_block "$log" "acss node"
done
