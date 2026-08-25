#!/bin/bash
# Repeat the configured benchmark three times, keeping each run's logs.
#
# `fab sweep` boots the nodes, waits for node 0's syncer to report the output
# phase, downloads the logs and kills the run. The previous version slept a
# fixed 30s before collecting, which silently truncated any run longer than
# that -- NN inference at realistic widths takes minutes, and the CIFAR-10 MLP
# spends most of it in the input phase.
set -u
for i in {1..3}; do
    echo "=== iteration $i ==="
    fab sweep || { echo "iteration $i failed"; exit 1; }
    mkdir -p logs/$i
    mv logs/*.log logs/$i/ 2>/dev/null
done
echo "done; results in logs/1, logs/2, logs/3"
echo "summarize with: cd logs/1 && python3 ../compile_results.py"
