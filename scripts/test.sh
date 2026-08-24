#!/bin/bash
# Launch a local NN-inference run.
#
#   ./scripts/test.sh <num_parties> <nn_x> <nn_y> <nn_batch> [comp] [weight_chunk]
#
# The network is [nn_x, nn_x, nn_x, nn_y], all-to-all, evaluated on nn_batch
# inputs. Config files are expected in testdata/hyb_<num_parties>/.

rm -rf /tmp/*.db &> /dev/null
export NN_DETERMINISTIC=${NN_DETERMINISTIC:-0}

N=$1
NN_X=${2:-512}
NN_Y=${3:-128}
NN_BATCH=${4:-1}
COMP=${5:-10}
WEIGHT_CHUNK=${6:-250000}

TESTDIR=${TESTDIR:="testdata/hyb_$N"}
TYPE=${TYPE:="release"}

# The repo's `ip_file` describes a 16-party deployment. For a local run the
# net_map already in nodes-*.json is correct, so only pass --ip when an override
# is explicitly supplied: IPFILE=ip_file ./scripts/test.sh ...
IP_ARGS=""
if [ -n "$IPFILE" ]; then IP_ARGS="--ip $IPFILE"; fi

TAG="n${N}_x${NN_X}_y${NN_Y}_b${NN_BATCH}"

# setsid detaches each node so it survives the launching shell exiting.
setsid ./target/$TYPE/node \
    --config $TESTDIR/nodes-0.json \
    $IP_ARGS \
    --protocol sync \
    --syncer $TESTDIR/syncer \
    --nn_x $NN_X \
    --nn_y $NN_Y \
    --nn_batch $NN_BATCH \
    --weight_chunk $WEIGHT_CHUNK \
    --comp $COMP \
    --byzantine false > logs/syncer_$TAG.log 2>&1 &

for((i=0;i<$N;i++)); do
setsid ./target/$TYPE/node \
    --config $TESTDIR/nodes-$i.json \
    $IP_ARGS \
    --protocol mpc \
    --syncer $TESTDIR/syncer \
    --nn_x $NN_X \
    --nn_y $NN_Y \
    --nn_batch $NN_BATCH \
    --weight_chunk $WEIGHT_CHUNK \
    --comp $COMP \
    --byzantine false > logs/party-$i-$TAG.log 2>&1 &
done

# Kill all nodes: pkill -9 -f 'target/release/node'
