# Running Benchmarks
Based on [Narwhal](https://github.com/facebookresearch/narwhal). 

This document explains how to benchmark the codebase and read benchmarks' results. It also provides a step-by-step tutorial to run benchmarks on [Amazon Web Services (AWS)](https://aws.amazon.com) across multiple data centers (WAN).

This branch benchmarks **dense neural-network inference**, not anonymous broadcast.
The network under test is configured near the top of `fabfile.py`:
```python
n = 64
nn_layers = [3072, 2048, 1024, 512, 10]   # full width list [d0, d1, ..., dL]
nn_batch  = [16, 32, 128]                 # a list sweeps these sequentially
weight_chunk = 250_000
compression_factor = 10
```
`nn_layers` is the complete width list of an all-to-all feed-forward network.
Everything else is derived from it:

| quantity | formula |
|---|---|
| dense weights | `sum(d[i] * d[i+1])` |
| inner products (= degree-t double sharings) | `nn_batch * sum(d[1:])` |
| local field multiplications | `nn_batch * weights` |
| sequential DN stages | `len(nn_layers) - 1` |

Note that the inner-product count depends only on the **output** widths — a wider
input layer costs more local GEMM but not one extra sharing.

## Setup
The core protocols are written in Rust, but all benchmarking scripts are written in Python and run with [Fabric](http://www.fabfile.org/). To run the remote benchmark, install the python dependencies:

```
$ pip install -r requirements.txt
```

You also need to install [tmux](https://linuxize.com/post/getting-started-with-tmux/#installing-tmux) (which runs all nodes and clients in the background). 

## AWS Benchmarks
This repo integrates various python scripts to deploy and benchmark the codebase on [Amazon Web Services (AWS)](https://aws.amazon.com). They are particularly useful to run benchmarks in the WAN, across multiple data centers. This section provides a step-by-step tutorial explaining how to use them.

### Step 1. Set up your AWS credentials
Set up your AWS credentials to enable programmatic access to your account from your local machine. These credentials will authorize your machine to create, delete, and edit instances on your AWS account programmatically. First of all, [find your 'access key id' and 'secret access key'](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html#cli-configure-quickstart-creds). Then, create a file `~/.aws/credentials` with the following content:
```
[default]
aws_access_key_id = YOUR_ACCESS_KEY_ID
aws_secret_access_key = YOUR_SECRET_ACCESS_KEY
```
Do not specify any AWS region in that file as the python scripts will allow you to handle multiple regions programmatically.

### Step 2. Add your SSH public key to your AWS account
You must now [add your SSH public key to your AWS account](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html). This operation is manual (AWS exposes little APIs to manipulate keys) and needs to be repeated for each AWS region that you plan to use. Upon importing your key, AWS requires you to choose a 'name' for your key; ensure you set the same name on all AWS regions. This SSH key will be used by the python scripts to execute commands and upload/download files to your AWS instances.
If you don't have an SSH key, you can create one using [ssh-keygen](https://www.ssh.com/ssh/keygen/):
```
$ ssh-keygen -f ~/.ssh/aws
```

### Step 3. Configure the testbed
The file [settings.json](https://github.com/akhilsb/Velox-MPC/blob/master/benchmark/settings.json) (located in [Velox-MPC/benchmarks](https://github.com/akhilsb/Velox-MPC/blob/master/benchmark)) contains all the configuration parameters of the testbed to deploy. Its content looks as follows:
```json
{
    "key": {
        "name": "aws",
        "path": "/absolute/key/path"
    },
    "port": 5000,
    "client_base_port": 7500,
    "client_run_port": 8000,
    "repo": {
        "name": "Velox-MPC",
        "url": "https://github.com/akhilsb/Velox-MPC.git",
        "branch": "master"
    },
    "instances": {
        "type": "c5.large",
        "regions": ["us-east-1"]
    }
}
```
The first block (`key`) contains information regarding your SSH key:
```json
"key": {
    "name": "aws",
    "path": "/absolute/key/path"
},
```
Enter the name of your SSH key; this is the name you specified in the AWS web console in step 2. Also, enter the absolute path of your SSH private key (using a relative path won't work). 


The second block (`ports`) specifies the TCP ports to use:
```json
"port": 5000,
"client_base_port": 7500,
"client_run_port": 8000,
```
The artifact requires a number of TCP ports for communication between the processes. Note that the script will open a large port range (5000-10000) to the LAN on all your AWS instances. 

The third block (`repo`) contains the information regarding the repository's name, the URL of the repo, and the branch containing the code to deploy: 
```json
"repo": {
    "name": "Velox-MPC",
    "url": "https://github.com/akhilsb/Velox-MPC.git",
    "branch": "master"
},
```
Remember to update the `url` field to the name of your repo. Modifying the branch name is particularly useful when testing new functionalities without having to checkout the code locally. 

The the last block (`instances`) specifies the [AWS instance type](https://aws.amazon.com/ec2/instance-types) and the [AWS regions](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html#concepts-available-regions) to use:
```json
"instances": {
    "type": "c5.large",
    "regions": ["us-east-1"]
}
```
The instance type selects the hardware on which to deploy the testbed. For example, `c5.large` instances come with 2 vCPU (2 physical cores), and 4 GB of RAM. The python scripts will configure each instance with 300 GB of SSD hard drive. The `regions` field specifies the data centers to use. If you require more nodes than data centers, the python scripts will distribute the nodes as equally as possible amongst the data centers. All machines run a fresh install of Ubuntu Server 24.04.

### Step 4. Create a testbed
The AWS instances are orchestrated with [Fabric](http://www.fabfile.org) from the file [fabfile.py](https://github.com/akhil-sb/Velox-MPC/blob/master/benchmark/fabfile.py) (located in [hashrand-rs/benchmarks](https://github.com/akhilsb/Velox-MPC/blob/master/benchmark)); you can list all possible commands as follows:
```bash
fab --list
```
The command `fab create` creates new AWS instances; open [fabfile.py](https://github.com/akhilsb/hashrand-rs/blob/master/benchmark/fabfile.py) and locate the `create` task:
```python
@task
def create(ctx, nodes=n):
    ...
```
The parameter `nodes`, set in the beginning of the `fabfile.py` file, determines how many instances to create in *each* AWS region. That is, if you specified 1 AWS region as in the example of step 3, setting `nodes=16` will create 16 machines:
```bash
fab create

Creating 16 instances |██████████████████████████████| 100.0% 
Waiting for all instances to boot...
Successfully created 16 new instances
```

You can then clone the repo and install rust on the remote instances with `fab install`:
```bash
fab install

Installing rust and cloning the repo...
Initialized testbed of 16 nodes
```

This may take a long time as the command will first update all instances.
The commands `fab stop` and `fab start` respectively stop and start the testbed without destroying it (it is good practice to stop the testbed when not in use as AWS can be quite expensive); and `fab destroy` terminates all instances and destroys the testbed. Note that, depending on the instance types, AWS instances may take up to several minutes to fully start or stop. The command `fab info` displays a nice summary of all available machines and information to manually connect to them (for debug).

### Step 5. Run a benchmark
After setting up the testbed, set the parameters near the top of `fabfile.py`:
1. Number of parties `n`
2. `nn_layers` — the width list of the network to evaluate
3. `nn_batch` — batch size, or a list of batch sizes to sweep
4. `weight_chunk` — secrets per ACSS instance when a party's weight block is split
5. `compression_factor` — rounds/computation tradeoff in the verification phase

There are **no input files to generate**. Weights and input activations are both
produced in-protocol: each party ACSS-shares an equal-sized block of the flat
weight array and of the `nn_batch * d0` activation array, and the online phase
begins only once all `n` parties have delivered every chunk. (The old
`inputs/inp_gen.py` text-payload generator has been removed along with the
node's text input loader.)

Some reference architectures, dense parts only:

| network | `nn_layers` | weights |
|---|---|---|
| LeNet-300-100 | `[784, 300, 100, 10]` | 266,200 |
| CIFAR-10 MLP | `[3072, 2048, 1024, 512, 10]` | 8,918,016 |
| VGG-16 head (CIFAR-10) | `[512, 4096, 4096, 10]` | 18,915,328 |
| AlexNet head | `[9216, 4096, 4096, 1000]` | 58,621,952 |
| VGG-16 head (ImageNet) | `[25088, 4096, 4096, 1000]` | 123,633,664 |

To configure the testbed and run a single benchmark:
```bash
fab remote                # uses nn_batch[0]
fab remote --batch=128    # override the batch size
```
This first updates all machines to the latest commit of the branch specified in
[settings.json](settings.json), then generates and uploads configuration files
and boots the protocol.

To run every batch size in `nn_batch` **sequentially** — booting, waiting for the
output phase, collecting logs, and killing each run before the next starts:
```bash
fab sweep
fab sweep --timeout=3600  # per-run timeout in seconds, default 1800
```
Runs must never overlap: every party holds a share of *every* weight, so two
concurrent runs contend for memory on all hosts.

`fab rerun` re-launches without regenerating or re-uploading the config files.

### Step 6: Download logs and compile results
`fab sweep` collects logs itself. After a `fab remote` or `fab rerun`, download
them once the protocol has terminated:
```bash
fab logs
fab logs --batch=128      # must match the batch that was run
```
Logs land in `benchmark/logs/` as
`syncer-n_{nodes}_{d0}-{d1}-...-{dL}_b{batch}_c{compression}.log`.
Then summarize:
```bash
cd logs/
python3 compile_results.py
```
The syncer reports **cumulative** latency at each phase boundary, so the script
reports both the cumulative figure and each phase's own duration, plus a
batch-scaling table when several batch sizes share an architecture:
```
=== syncer-n_4_3072-2048-1024-512-10_b128_c10.log ===
  network 3072-2048-1024-512-10  n=4  batch=128
  8,918,016 weights, 460,032 inner products (= degree-t double sharings)
  phase              cumulative     duration
  Preprocessing          4228ms       4228ms
  Input                 30501ms      26273ms
  Online                33996ms       3495ms
  output                34039ms         43ms
  total                 34039ms   (265.9 ms per inference)

=== batch scaling: 3072-2048-1024-512-10, n=4 ===
    batch   inner products      total  per inference
       16           57,504    24417ms       1526.1ms
       32          115,008    25794ms        806.1ms
      128          460,032    34039ms        265.9ms
```

The four phases are:

| phase | what it covers | scales with |
|---|---|---|
| Preprocessing | random double sharings (ACSS + Sh2t + AVSS masks), ACS | inner products, i.e. `batch` |
| Input | ACSS of the weight and activation blocks; barrier on **all `n`** dealers | weights (fixed in `batch`) |
| Online | the `L` DN inner-product stages | `batch * weights` |
| output | masked reconstruction, CTRBC, second ACS | `batch * d_L` |

Because weights are shared once regardless of batch size, the input phase is
roughly constant while everything else grows with `batch` — which is why
per-inference cost falls sharply as the batch grows.

**The verification phase is currently disabled** (`Context::verification_enabled`
is `false`), so no `verification` line appears and runs are semi-honest-only. See
`NN_INFERENCE_TODO.md` in the repository root.

If anything goes wrong during a benchmark, stop it with `fab kill`.

### Step 7: Cleanup
Be sure to kill the prior benchmark using the following command before running a new benchmark. 
Additionally, clean up the files created by the benchmark by running the `cleanup.sh` script.  
```bash
fab kill
./cleanup.sh
```
After running the benchmarks for a given number of nodes, destroy the testbed with the following command. 
```bash
fab destroy
```
This command destroys the testbed and terminates all created AWS instances.
For running a benchmark with a different testbed setup, execute the pipeline from Step 3. 

# Reproducing the results in the paper

**The configurations below belong to the anonymous-broadcast (mixing-circuit)
protocol, which this branch replaces with NN inference.** They are kept for
reference; to reproduce the paper's numbers, check out `master`, where
`num_messages` / `batch_size` / `compression_factor` still drive the mixing
circuit.

For NN-inference results on this branch, sweep `nn_batch` for a fixed
`nn_layers` and read the batch-scaling table from `compile_results.py`. Useful
axes to vary:

- **batch size** — separates the fixed cost of weight distribution from the
  per-inference cost.
- **architecture** — average inner-product dimension (`weights / IP-per-example`)
  decides whether a workload is preprocessing-bound or compute-bound. LeNet-300-100
  sits at ~649, the CIFAR-10 MLP at ~2,481, the ImageNet VGG-16 head at ~13,451.
- **party count `n`** — note the input phase waits for all `n` dealers, so it has
  no fault tolerance; a single crashed party stalls the run.

