#!/bin/bash
# Launch a local NN-inference run.
#
#   ./scripts/test.sh <num_parties> <layer_widths> <nn_batch> [comp] [weight_chunk]
#
# layer_widths is a comma-separated list, e.g. 25088,4096,4096,1000 for the
# VGG-16 classifier head. The network is all-to-all and evaluated on nn_batch
# inputs. Config files are expected in testdata/hyb_<num_parties>/.

rm -rf /tmp/*.db &> /dev/null
export NN_DETERMINISTIC=${NN_DETERMINISTIC:-0}
# SKIP_INPUT=1 derives input shares locally (benchmark only: not secret)
SKIP=${SKIP_INPUT:-false}
[ "$SKIP" = "1" ] && SKIP=true

N=$1
LAYERS=${2:-512,512,512,128}
NN_BATCH=${3:-1}
COMP=${4:-10}
WEIGHT_CHUNK=${5:-250000}

TESTDIR=${TESTDIR:="testdata/hyb_$N"}
TYPE=${TYPE:="release"}

# The repo's `ip_file` describes a 16-party deployment. For a local run the
# net_map already in nodes-*.json is correct, so only pass --ip when an override
# is explicitly supplied: IPFILE=ip_file ./scripts/test.sh ...
IP_ARGS=""
if [ -n "$IPFILE" ]; then IP_ARGS="--ip $IPFILE"; fi

TAG="n${N}_$(echo $LAYERS | tr ',' '-')_b${NN_BATCH}"

# Refuse to launch a run that cannot fit in RAM. Every party holds a share of
# every weight regardless of N, and the preprocessing receive buffers scale with
# N, so total footprint grows roughly linearly in the party count. Launching
# N=16 at b=256 on a 5 GB box took the machine down.
AVAIL_MB=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
PER_PARTY_MB=$(python3 -c "
w=[int(v) for v in '$LAYERS'.split(',')]
weights=sum(w[i]*w[i+1] for i in range(len(w)-1))
ips=$NN_BATCH*sum(w[1:])
# Two measured points:
#   [512,512,512,128]   b=256, 0.59M weights -> 289 MB/party
#   [3072,2048,1024,512,10] b=128, 8.9M weights -> ~1000 MB/party
# Memory is dominated by the ACSS transient, which scales with the WEIGHT count
# rather than the batch: every serialized element is an individually heap-
# allocated Vec<u8> holding 8 bytes (~6x overhead), materialized n times by the
# dealer and once per receiving party. The final 8-byte-per-share storage is a
# small fraction of peak. Hence the large weight coefficient.
# THIS ESTIMATE HAS BEEN OPTIMISTIC TWICE. Treat it as a lower bound and rely on
# the live memory abort in sweep.sh, not on this number.
print(int(weights*8/1048576*14 + ips*48/1048576*2.5) + 200)")
EST_MB=$(( PER_PARTY_MB * N ))
echo "estimated footprint: ~${PER_PARTY_MB} MB/party x ${N} = ~${EST_MB} MB (available: ${AVAIL_MB} MB)"
# DRY=1 prints the estimate and exits without launching anything. Use this for
# any sizing question -- piping the normal path to `head` truncates the output
# but still launches every party, and setsid keeps them alive afterwards.
if [ -n "$DRY" ]; then exit 0; fi
if [ "$EST_MB" -gt $(( AVAIL_MB * 4 / 5 )) ] && [ -z "$FORCE" ]; then
    echo "REFUSING TO LAUNCH: estimate exceeds 80% of available memory."
    echo "Lower N / nn_batch / nn_x, or re-run with FORCE=1 to override."
    exit 1
fi

# setsid detaches each node so it survives the launching shell exiting.
setsid ./target/$TYPE/node \
    --config $TESTDIR/nodes-0.json \
    $IP_ARGS \
    --protocol sync \
    --syncer $TESTDIR/syncer \
    --nn_layers $LAYERS \
    --nn_batch $NN_BATCH \
    --weight_chunk $WEIGHT_CHUNK \
    --skip_input_phase $SKIP \
    --comp $COMP \
    --byzantine false > logs/syncer_$TAG.log 2>&1 &

for((i=0;i<$N;i++)); do
setsid ./target/$TYPE/node \
    --config $TESTDIR/nodes-$i.json \
    $IP_ARGS \
    --protocol mpc \
    --syncer $TESTDIR/syncer \
    --nn_layers $LAYERS \
    --nn_batch $NN_BATCH \
    --weight_chunk $WEIGHT_CHUNK \
    --skip_input_phase $SKIP \
    --comp $COMP \
    --byzantine false > logs/party-$i-$TAG.log 2>&1 &
done

# Kill all nodes: pkill -9 -f 'target/release/node'
