# PVD disaggregation

PVD is opt-in. The existing PD path remains the default when
`--disaggregation-topology` is omitted.

## Current data-plane topology

- Gateway may route over independently sized pools of P and D worker groups.
  V is one physical node running both GPU storage shards and its coordinator
  in one Python process.
- Production dual rail: `P0 -> V0 -> D0` uses `mlx5_0`, and
  `P1 -> V1 -> D1` uses `mlx5_1`.
- CloudLab single-rail debug: both rank-local paths use `mlx5_0`, selected
  explicitly with `mlx5_0,mlx5_0`. This preserves rank sharding but provides
  neither rail redundancy nor aggregate dual-rail bandwidth.
- PVD protocol v2 supports P TP1 or TP2 and D TP2 or TP4. The first
  heterogeneous path is `P TP1 -> V TP2 -> D TP4`; balanced TP2 remains
  supported. PP/DP size is 1, speculative decoding and hierarchical/SWA cache
  modes are rejected. KV pools with SWA, DSA or Mamba auxiliary state
  buffers are also rejected until those state components have a versioned PVD
  wire layout. HiSparse, Prefill context parallelism and the legacy PD staging
  environment variable are rejected because PVD owns its staging layout.
- P and D radix caches are disabled. Every immutable Entry stores every page
  of the prompt KV shard, including padding in the final page.
- The V process owns two independent rank-local stores and transfer engines.
  The coordinator manages their lifecycle but neither shard owns the other's
  GPU memory.

`Entry`, `EntryShard` and `Delivery` have independent state machines. An Entry
is retained until its TTL and may serve multiple Delivery IDs. ACK releases
only Delivery resources; it does not release the Entry.

Supported compute/storage combinations in this phase:

| P worker-group TP | V storage TP | D worker-group TP | Status |
| ---: | ---: | ---: | --- |
| 2 | 2 | 2 | Supported; raw rank-local copy |
| 1 | 2 | 4 | Supported; V performs head-shard staging |
| 1 | 2 | 2 | Supported |
| 2 | 2 | 4 | Supported when total KV heads are divisible by 4 |

Worker-group count is a separate dimension: the Gateway may have any positive
number of P URLs and any positive number of D URLs. The table limits TP inside
one selected worker group, not the number of physical P or D workers.

## Start order

All roles require Linux, CUDA, Mooncake, CUDA memory registration, and a
successful GPU-memory transfer preflight. Production dual-rail mode requires
both HCAs to be active. Single-rail debug mode requires `mlx5_0` to be active
and runs the same strict registration/transfer check for both ranks. A process
exits before serving traffic if its configured rank/rail check fails.

Start the complete V worker group with one command. Replace addresses and pool
sizes:

```bash
# On the V node; rank 0 uses cuda:0 and rank 1 uses cuda:1
CUDA_VISIBLE_DEVICES=0,1 \
python -m sglang.srt.disaggregation.pvd.server \
  --world-size 2 --pvd-rank-devices 0,1 \
  --advertise-host 10.0.0.2 \
  --total-pages 131072 --page-bytes PAGE_BYTES
```

For the CloudLab single-rail debug topology, append the following option to the
V group command:

```bash
--pvd-rank-rails mlx5_0,mlx5_0
```

`--rails` remains available as a shorter V-only alias.

For a standard MHA/GQA KV pool, calculate `PAGE_BYTES` per rank as:

```text
2 * local_layer_count * kv_heads_per_rank * head_dim * dtype_bytes * page_size
```

Use the sum of `component_bytes_per_token` reported by the storage layout,
multiplied by `page_size`. V rejects a manifest whose per-prompt-page bytes
exceed its configured `PAGE_BYTES`.

Start P and D with the same coordinator URL and model instance ID:

```bash
# node-1 (P)
python -m sglang.launch_server --model-path MODEL --tp-size 2 \
  --disaggregation-mode prefill --disaggregation-topology pvd \
  --pvd-vector-coordinator-url http://10.0.0.2:9100 \
  --pvd-model-instance-id MODEL_INSTANCE

# node-3 (D)
python -m sglang.launch_server --model-path MODEL --tp-size 2 \
  --disaggregation-mode decode --disaggregation-topology pvd \
  --pvd-vector-coordinator-url http://10.0.0.2:9100 \
  --pvd-model-instance-id MODEL_INSTANCE
```

For the CloudLab single-rail debug topology, append the following option to
**both** P and D commands:

```bash
--pvd-rank-rails mlx5_0,mlx5_0
```

The P/D launcher then generates the Mooncake GPU mapping
`{"0":"mlx5_0","1":"mlx5_0"}`. The PVD argument validator accepts only the
production `mlx5_0,mlx5_1` mapping or an explicit all-`mlx5_0` debug mapping,
and logs a warning for the latter.

For the P TP1 to D TP4 single-rail topology, use one rail value on P and four
on D. The V command remains the same two-GPU command shown above:

```bash
# node-1 (P, one GPU)
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path MODEL --tp-size 1 \
  --disaggregation-mode prefill --disaggregation-topology pvd \
  --pvd-vector-coordinator-url http://10.0.0.2:9100 \
  --pvd-model-instance-id MODEL_INSTANCE \
  --pvd-rank-rails mlx5_0

# node-3 (D, four GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
  --model-path MODEL --tp-size 4 \
  --disaggregation-mode decode --disaggregation-topology pvd \
  --pvd-vector-coordinator-url http://10.0.0.2:9100 \
  --pvd-model-instance-id MODEL_INSTANCE \
  --pvd-rank-rails mlx5_0,mlx5_0,mlx5_0,mlx5_0
```

The v2 head mapping is `P0 -> {V0,V1}` and
`V0 -> {D0,D1}, V1 -> {D2,D3}`. This path currently requires an ordinary
MHA/GQA KV pool whose total KV-head count is divisible by 4. MLA, MQA head
replication, SWA and auxiliary state buffers are rejected rather than copied
with an ambiguous layout.

Finally launch Model Gateway with its existing P/D worker addresses plus the
PVD flag. The Gateway validates both V shards and their strict preflight
reports before accepting traffic, then injects a shared Entry transfer ID and
an independent Delivery ID into the P and D copies of each request. PVD
uses the Gateway's HTTP worker connection mode; gRPC worker routing is rejected
at configuration validation until its protobuf carries the two PVD IDs.

```bash
sglang-router --pvd-disaggregation \
  --pvd-vector-coordinator-url http://10.0.0.2:9100 \
  --prefill http://10.0.0.1:30000 none \
  --prefill http://10.0.0.4:30000 none \
  --decode http://10.0.0.3:30000 \
  --decode http://10.0.0.5:30000
```

The number of `--prefill` and `--decode` entries is independent. One URL
represents one complete worker group. All groups must expose the same model,
revision and semantic KV layout, but P and D group counts are independent and
their supported TP sizes may differ. Supplying `--rank` to the V launcher keeps
the old one-process-per-rank mode available for diagnostics.

The fake transport and CPU storage switches are test-only and require both
`--allow-fake-transport` and `--no-strict-rdma-preflight`. The Gateway's strict
PVD startup gate intentionally rejects such a V group.
