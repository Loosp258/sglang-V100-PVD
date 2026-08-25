# PVD disaggregation v1

PVD is opt-in. The existing PD path remains the default when
`--disaggregation-topology` is omitted.

## Fixed topology

- P, V and D each run one two-rank TP group on one physical node.
- Production dual rail: `P0 -> V0 -> D0` uses `mlx5_0`, and
  `P1 -> V1 -> D1` uses `mlx5_1`.
- CloudLab single-rail debug: both rank-local paths use `mlx5_0`, selected
  explicitly with `mlx5_0,mlx5_0`. This preserves rank sharding but provides
  neither rail redundancy nor aggregate dual-rail bandwidth.
- TP size is 2, PP/DP size is 1, speculative decoding and hierarchical/SWA
  cache modes are rejected. KV pools with SWA, DSA or Mamba auxiliary state
  buffers are also rejected until those state components have a versioned PVD
  wire layout. HiSparse, Prefill context parallelism and the legacy PD staging
  environment variable are rejected because PVD owns its staging layout.
- P and D radix caches are disabled. Every immutable Entry stores every page
  of the prompt KV shard, including padding in the final page.
- V rank 0 hosts its rank-local shard API and the group coordinator. It does
  not own rank 1's GPU memory or run rank 1 transfers.

`Entry`, `EntryShard` and `Delivery` have independent state machines. An Entry
is retained until its TTL and may serve multiple Delivery IDs. ACK releases
only Delivery resources; it does not release the Entry.

## Start order

All roles require Linux, CUDA, Mooncake, CUDA memory registration, and a
successful GPU-memory transfer preflight. Production dual-rail mode requires
both HCAs to be active. Single-rail debug mode requires `mlx5_0` to be active
and runs the same strict registration/transfer check for both ranks. A process
exits before serving traffic if its configured rank/rail check fails.

Start V rank 1 first, then V rank 0. Replace addresses and pool sizes:

```bash
# On node-2, rank 1 / GPU 1
python -m sglang.srt.disaggregation.pvd.server \
  --rank 1 --local-rank 1 --world-size 2 \
  --advertise-host 10.0.0.2 \
  --total-pages 131072 --page-bytes PAGE_BYTES

# On node-2, rank 0 / GPU 0
python -m sglang.srt.disaggregation.pvd.server \
  --rank 0 --local-rank 0 --world-size 2 \
  --advertise-host 10.0.0.2 \
  --rank1-shard-url http://10.0.0.2:9201 \
  --total-pages 131072 --page-bytes PAGE_BYTES
```

For the CloudLab single-rail debug topology, append the following option to
**both** V rank commands:

```bash
--pvd-rank-rails mlx5_0,mlx5_0
```

`--rails` remains available as a shorter V-only alias.

For a standard MHA/GQA KV pool, calculate `PAGE_BYTES` per rank as:

```text
2 * local_layer_count * kv_heads_per_rank * head_dim * dtype_bytes * page_size
```

For MLA or another supported tensor layout, use the sum of
`component_bytes_per_token` reported by the P/D layout multiplied by
`page_size`. V rejects a manifest whose per-prompt-page bytes exceed its
configured `PAGE_BYTES`.

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
`{"0":"mlx5_0","1":"mlx5_0"}`. The Gateway accepts only the production
`mlx5_0,mlx5_1` mapping or this explicit debug mapping and logs a warning for
the latter.

Finally launch Model Gateway with its existing P/D worker addresses plus the
PVD flag. The Gateway validates both V shards and their strict preflight
reports before accepting traffic, then injects a shared Entry transfer ID and
an independent Delivery ID into the P and D copies of each request. PVD v1
uses the Gateway's HTTP worker connection mode; gRPC worker routing is rejected
at configuration validation until its protobuf carries the two PVD IDs.

```bash
sglang-router --pvd-disaggregation \
  --pvd-vector-coordinator-url http://10.0.0.2:9100 \
  --prefill http://10.0.0.1:30000 none \
  --decode http://10.0.0.3:30000
```

The fake transport and CPU storage switches are test-only and require both
`--allow-fake-transport` and `--no-strict-rdma-preflight`. The Gateway's strict
PVD startup gate intentionally rejects such a V group.
