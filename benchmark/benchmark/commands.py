# Copyright(C) Facebook, Inc. and its affiliates.
from os.path import join

from benchmark.utils import PathMaker


class CommandMaker:

    @staticmethod
    def cleanup():
        return (
            f'rm -r .db-* ; rm .*.json ; mkdir -p {PathMaker.results_path()}'
        )

    @staticmethod
    def clean_logs():
        return f'rm -r {PathMaker.logs_path()} ; mkdir -p {PathMaker.logs_path()}'

    @staticmethod
    def compile():
        return 'cargo build --release'

    @staticmethod
    def generate_key(filename):
        assert isinstance(filename, str)
        return f'./node generate_keys --filename {filename}'

    @staticmethod
    def generate_config_files(bport, client_bport, client_run_port, num_nodes):
        # Velox's multiplication layer requires n = 3t+1 exactly (see lin_mult.rs:253-265
        # — the L2 reconstruction produces (n-t) coefficients per group while the
        # rand-sharing bookkeeping expects (2t+1); equality holds iff n = 3t+1).
        # The `config` binary defaults faults to (n-1)/2 when --faults is omitted,
        # which crashes the protocol at depth 0 for any n>4. Force t = (n-1)//3.
        num_faults = (num_nodes - 1) // 3
        return (
            f'./config --blocksize 100 --delay 100 --base_port {bport} '
            f'--client_base_port {client_bport} --NumNodes {num_nodes} '
            f'--faults {num_faults} '
            f'--target . --client_run_port {client_run_port} --local true'
        )

    @staticmethod
    def run_primary(key, mixing_batch_size, compression_factor, num_batches, debug=False):
        assert isinstance(key, str)
        assert isinstance(debug, bool)
        # Merge: Akhil's `--rand-batches` plumbing (paces preprocessing into
        # smaller ACSS instances; fixes the n=49 / k=32768 OOM) + the higher
        # ulimit (65k FDs needed for the cross-region TCP fan-out at large n).
        return (f'ulimit -n 65000; ./node --config {key} --ip ip_file '
                f'--protocol mpc --syncer syncer --messages {mixing_batch_size} '
                f'--comp {compression_factor} --rand-batches {num_batches} --byzantine false')

    @staticmethod
    def run_syncer(key, mixing_batch_size, compression_factor, num_batches, debug=False):
        assert isinstance(key, str)
        assert isinstance(debug, bool)
        return (f'ulimit -n 65000; ./node --config {key} --ip ip_file '
                f'--protocol sync --syncer syncer --messages {mixing_batch_size} '
                f'--comp {compression_factor} --rand-batches {num_batches} --byzantine false')

    @staticmethod
    def unzip_tkeys(fileloc, debug=False):
        return (f'tar -xvzf {fileloc}')

    @staticmethod
    def run_worker(keys, committee, store, parameters, id, debug=False):
        assert isinstance(keys, str)
        assert isinstance(committee, str)
        assert isinstance(parameters, str)
        assert isinstance(debug, bool)
        v = '-vvv' if debug else '-vv'
        return (f'./node {v} run --keys {keys} --committee {committee} '
                f'--store {store} --parameters {parameters} worker --id {id}')

    @staticmethod
    def run_client(address, size, rate, nodes):
        assert isinstance(address, str)
        assert isinstance(size, int) and size > 0
        assert isinstance(rate, int) and rate >= 0
        assert isinstance(nodes, list)
        assert all(isinstance(x, str) for x in nodes)
        nodes = f'--nodes {" ".join(nodes)}' if nodes else ''
        return f'./benchmark_client {address} --size {size} --rate {rate} {nodes}'

    @staticmethod
    def kill():
        return 'tmux kill-server'

    @staticmethod
    def alias_binaries(origin):
        assert isinstance(origin, str)
        # Velox produces `node` and `config` only — there is no `benchmark_client`
        # binary, so we drop the dangling symlink it used to create. `&&` instead
        # of `;` and `test -x` make a failed cargo build fail the deploy step
        # loudly instead of producing dangling symlinks that only surface later
        # as `./node: No such file or directory` on the remote run.
        node, config = join(origin, 'node'), join(origin, 'config')
        return (f'rm -f node config && '
                f'ln -sf {node} . && ln -sf {config} . && '
                f'test -x ./node && test -x ./config')
