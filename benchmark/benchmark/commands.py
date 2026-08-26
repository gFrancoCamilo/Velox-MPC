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
        return 'cargo build --quiet --release'

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
    def _nn_args(nn_layers, nn_batch, weight_chunk, compression_factor, skip_input=False):
        """Flags describing the network under test.

        `nn_layers` is the full width list `[d0, d1, ..., dL]` of an all-to-all
        feed-forward network, e.g. `[3072, 2048, 1024, 512, 10]`. The node
        derives everything else from it: weights = sum(d_{i}*d_{i+1}), and
        inner products = nn_batch * sum(d[1:]).

        The old `--messages` flag drove the anonymous-broadcast mixing circuit
        and is a no-op for NN inference. Passing it alone (as this file used to)
        left the node on its *default* architecture, so the benchmark silently
        measured a model nobody asked for.
        """
        assert isinstance(nn_layers, (list, tuple)) and len(nn_layers) >= 2
        assert all(isinstance(w, int) and w > 0 for w in nn_layers)
        assert isinstance(nn_batch, int) and nn_batch > 0
        assert isinstance(weight_chunk, int) and weight_chunk > 0
        layers = ','.join(str(w) for w in nn_layers)
        return (f'--nn_layers {layers} --nn_batch {nn_batch} '
                f'--weight_chunk {weight_chunk} --comp {compression_factor}'
                + f' --skip_input_phase {"true" if skip_input else "false"}')

    @staticmethod
    def run_primary(key, nn_layers, nn_batch, weight_chunk, compression_factor, skip_input=False, debug=False):
        assert isinstance(key, str)
        assert isinstance(debug, bool)
        args = CommandMaker._nn_args(nn_layers, nn_batch, weight_chunk, compression_factor, skip_input)
        return (f'ulimit -n 5000; ./node --config {key} --ip ip_file '
                f'--protocol mpc --syncer syncer {args} --byzantine false')

    @staticmethod
    def run_syncer(key, nn_layers, nn_batch, weight_chunk, compression_factor, skip_input=False, debug=False):
        assert isinstance(key, str)
        assert isinstance(debug, bool)
        args = CommandMaker._nn_args(nn_layers, nn_batch, weight_chunk, compression_factor, skip_input)
        return (f'ulimit -n 5000; ./node --config {key} --ip ip_file '
                f'--protocol sync --syncer syncer {args} --byzantine false')

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
        node, client, config = join(origin, 'node'), join(origin, 'benchmark_client'), join(origin,'config')
        return f'rm node ; rm benchmark_client ; rm config ; ln -s {node} . ; ln -s {client} . ; ln -s {config} .'
