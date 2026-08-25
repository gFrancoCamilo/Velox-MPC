#!/bin/bash
# Sequential batch sweep. Runs are NEVER concurrent -- stacking runs is what
# crashed the machine twice. Each run must fully finish and every node process
# must exit before the next one starts.
#
#   ./scripts/sweep.sh
set -u

N=4
MIN_FREE_MB=3000          # abort the sweep if available memory drops below this
POLL=3                    # seconds between completion checks
MAX_WAIT=900              # seconds before declaring a run stuck
                          # (CIFAR-10 MLP at b=2048 is ~18.3e9 field mults, ~4-5 min)

free_mb() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
# `pgrep -c` prints 0 AND exits non-zero when nothing matches, so `|| echo 0`
# appended a second line and every integer test against it failed silently --
# which disabled the concurrent-run guard entirely. Count lines instead.
nodes_up() { pgrep -x node 2>/dev/null | wc -l; }

run_one() {
    local layers=$1 batch=$2 det=$3
    local tag="n${N}_$(echo "$layers" | tr ',' '-')_b${batch}"

    if [ "$(nodes_up)" -ne 0 ]; then
        echo "ABORT: $(nodes_up) node processes still running before '$tag'"; return 1
    fi
    if [ "$(free_mb)" -lt "$MIN_FREE_MB" ]; then
        echo "ABORT: only $(free_mb) MB free, need $MIN_FREE_MB"; return 1
    fi

    rm -f "logs/syncer_$tag.log" logs/party-*-"$tag".log
    echo "--- $layers  b=$batch  (deterministic=$det, free=$(free_mb) MB)"
    NN_DETERMINISTIC=$det ./scripts/test.sh "$N" "$layers" "$batch" || return 1

    local waited=0 done=0
    while [ $waited -lt $MAX_WAIT ]; do
        sleep $POLL; waited=$((waited+POLL))
        if grep -q '"output"' "logs/syncer_$tag.log" 2>/dev/null; then done=1; break; fi
        if [ "$(free_mb)" -lt 800 ]; then
            echo "ABORT: memory critical ($(free_mb) MB) -- killing run"
            pkill -9 -f 'release/nod[e]'; return 1
        fi
    done

    pkill -9 -f 'release/nod[e]' 2>/dev/null
    sleep 2
    if [ "$done" -ne 1 ]; then echo "TIMEOUT after ${MAX_WAIT}s: $tag"; return 1; fi

    grep -h "NN CHECK" logs/party-0-"$tag".log 2>/dev/null | sed 's/.*NN CHECK/    NN CHECK/'
    grep "All n nodes" "logs/syncer_$tag.log" \
        | sed 's/.*latency \(\[[^]]*\]\), status \({[^}]*}\).*/    \2 \1/'
    echo
}

echo "=== correctness spot-check (b=1, deterministic) ==="
run_one 784,300,100,10     1 1 || exit 1
run_one 512,4096,4096,10   1 1 || exit 1
run_one 3072,2048,1024,512,10 1 1 || exit 1

echo "=== LeNet-300-100 [784,300,100,10] batch sweep ==="
for b in 1 8 32 128 512 2048; do run_one 784,300,100,10 "$b" 0 || exit 1; done

echo "=== VGG-16 head, CIFAR-10 variant [512,4096,4096,10] batch sweep ==="
for b in 1 8 32 128 256 512; do run_one 512,4096,4096,10 "$b" 0 || exit 1; done

echo "=== CIFAR-10 MLP baseline [3072,2048,1024,512,10] batch sweep ==="
for b in 1 8 32 128 512 2048; do run_one 3072,2048,1024,512,10 "$b" 0 || exit 1; done

echo "sweep complete; free memory: $(free_mb) MB, node processes: $(nodes_up)"
