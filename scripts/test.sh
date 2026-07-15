# A script to test quickly

killall {node} &> /dev/null
rm -rf /tmp/*.db &> /dev/null
vals=(27000 27100 27200 27300)

#rand=$(gshuf -i 1000-150000000 -n 1)
TESTDIR=${TESTDIR:="testdata/$1"}
TYPE=${TYPE:="release"}

# Optional 4th arg: number of random-sharing sub-batches (--rand-batches).
# When omitted, the node falls back to mpc::NUM_RAND_BATCHES.
RAND_BATCHES_ARG=${4:+--rand-batches $4}

# ---- Optional nvprof GPU profiling of a SINGLE node -------------------------
# Enable by setting PROFILE_NODE to the node index to profile (e.g. 0). Only
# that node is wrapped; the rest run normally so the protocol still makes
# progress. Modes (PROFILE_MODE):
#   trace   (default) low-overhead kernel timeline + geometry (safe whole-run)
#   summary           per-kernel time summary (safe whole-run)
#   metrics           achieved_occupancy + efficiency — REPLAYS kernels, which
#                     distorts timing and will desync this node. Bounded by
#                     PROFILE_TIMEOUT (default 90s): nvprof stops and flushes
#                     the report after that, capturing the early ACSS kernels.
#   nvvp              trace to a .nvvp file for the Visual Profiler.
# Output goes to logs/nvprof-<i>.log (or .nvvp). Occupancy is a property of
# launch geometry + resource use, so metrics numbers match an isolated harness
# at the same sizes — the live run just gives you the real sizes + idle gaps.
PROFILE_NODE=${PROFILE_NODE:-}
PROFILE_MODE=${PROFILE_MODE:-trace}
PROFILE_TIMEOUT=${PROFILE_TIMEOUT:-90}

prof_prefix() {
    local i="$1"
    [ -z "$PROFILE_NODE" ] && return 0
    [ "$i" != "$PROFILE_NODE" ] && return 0
    case "$PROFILE_MODE" in
        trace)   echo "nvprof --print-gpu-trace --log-file logs/nvprof-$i.log" ;;
        summary) echo "nvprof --log-file logs/nvprof-$i.log" ;;
        metrics) echo "nvprof --timeout $PROFILE_TIMEOUT --metrics achieved_occupancy,sm_efficiency,warp_execution_efficiency --log-file logs/nvprof-$i.log" ;;
        nvvp)    echo "nvprof -o logs/nvprof-$i.nvvp" ;;
        *)       echo "" ;;
    esac
}

# Run the syncer now
./target/$TYPE/node \
    --config $TESTDIR/nodes-0.json \
    --ip ip_file \
    --protocol sync \
    --syncer $TESTDIR/syncer \
    --messages $2 \
    --comp $3 \
    --byzantine false > logs/syncer_n_$1_$2_$3.log &

for((i=0;i<$1;i++)); do
PREFIX=$(prof_prefix $i)
$PREFIX ./target/$TYPE/node \
    --config $TESTDIR/nodes-$i.json \
    --ip ip_file \
    --protocol mpc \
    --syncer $TESTDIR/syncer \
    --messages $2 \
    --comp $3 \
    $RAND_BATCHES_ARG \
    --byzantine false > logs/party-$i-n_$1_$2_$3.log &
done

# Kill all nodes sudo lsof -ti:7000-7015 | xargs kill -9
