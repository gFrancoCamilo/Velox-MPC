"""Summarize NN-inference benchmark runs from the syncer logs.

Reads every `syncer-*.log` in this directory. The syncer emits one line per
protocol phase:

    All n nodes completed the protocol for ID: 1 with latency [a, b, c],
    status {"Preprocessing"}, and value {[]}

Phases appear in execution order: Preprocessing -> Input -> Online -> output.
The reported latencies are cumulative from protocol start, so a phase's own
duration is the difference from the previous phase.

Log file names carry the configuration:
    syncer-n_{nodes}_{d0}-{d1}-...-{dL}_b{batch}_c{compression}.log
"""
import glob
import re
from collections import OrderedDict
from statistics import mean

LINE = re.compile(r'with latency \[([^\]]+)\], status \{"([^"]+)"\}')
NAME = re.compile(r'syncer-n_(\d+)_([\d-]+)_b(\d+)_c(\d+)\.log$')

PHASE_ORDER = ['Preprocessing', 'Input', 'Online', 'verification', 'output']


def phase_sort_key(name):
    return PHASE_ORDER.index(name) if name in PHASE_ORDER else len(PHASE_ORDER)


def parse(path):
    """-> (cumulative-latency-per-phase, config) for one syncer log."""
    phases = OrderedDict()
    with open(path) as f:
        for line in f:
            m = LINE.search(line)
            if not m:
                continue
            latencies = [int(x.strip()) for x in m.group(1).split(',')]
            phases.setdefault(m.group(2), []).extend(latencies)

    cfg = None
    m = NAME.search(path)
    if m:
        widths = [int(w) for w in m.group(2).split('-')]
        cfg = {
            'nodes': int(m.group(1)),
            'widths': widths,
            'batch': int(m.group(3)),
            'compression': int(m.group(4)),
            'weights': sum(widths[i] * widths[i + 1]
                           for i in range(len(widths) - 1)),
            'inner_products': int(m.group(3)) * sum(widths[1:]),
        }
    return phases, cfg


def report(path):
    phases, cfg = parse(path)
    if not phases:
        print(f'{path}: no phase lines found')
        return

    print(f'\n=== {path} ===')
    if cfg:
        print(f'  network {"-".join(map(str, cfg["widths"]))}  '
              f'n={cfg["nodes"]}  batch={cfg["batch"]}')
        print(f'  {cfg["weights"]:,} weights, {cfg["inner_products"]:,} '
              f'inner products (= degree-t double sharings)')

    ordered = sorted(phases.items(), key=lambda kv: phase_sort_key(kv[0]))
    prev_avg = 0.0
    total = 0.0
    print(f'  {"phase":16} {"cumulative":>12} {"duration":>12}')
    for name, latencies in ordered:
        avg = mean(latencies)
        duration = avg - prev_avg
        print(f'  {name:16} {avg:10.0f}ms {duration:10.0f}ms')
        prev_avg, total = avg, avg

    if cfg and cfg['batch']:
        print(f'  {"total":16} {total:10.0f}ms   '
              f'({total / cfg["batch"]:.1f} ms per inference)')


def main():
    paths = sorted(glob.glob('syncer-*.log'))
    if not paths:
        print('no syncer-*.log files in this directory')
        return
    for path in paths:
        report(path)

    # Batch-scaling view: group runs that share an architecture and node count.
    rows = []
    for path in paths:
        phases, cfg = parse(path)
        if not cfg or not phases:
            continue
        ordered = sorted(phases.items(), key=lambda kv: phase_sort_key(kv[0]))
        total = mean(ordered[-1][1])
        rows.append((cfg, total))

    groups = OrderedDict()
    for cfg, total in rows:
        key = ('-'.join(map(str, cfg['widths'])), cfg['nodes'])
        groups.setdefault(key, []).append((cfg['batch'], total,
                                           cfg['inner_products']))
    for (arch, nodes), entries in groups.items():
        if len(entries) < 2:
            continue
        print(f'\n=== batch scaling: {arch}, n={nodes} ===')
        print(f'  {"batch":>7} {"inner products":>16} {"total":>10} '
              f'{"per inference":>14}')
        for batch, total, ips in sorted(entries):
            print(f'  {batch:7,} {ips:16,} {total:8.0f}ms '
                  f'{total / batch:12.1f}ms')


if __name__ == '__main__':
    main()
