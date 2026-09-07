# PVD 3.0 disaggregation

PVD is opt-in. The existing PD path remains the default when
`--disaggregation-topology` is omitted.

## PVD 3.0 request flow

The accepted design is recorded in
[the design specification](../../../../../docs/superpowers/specs/2026-09-05-pvd3-design.md).
The Gateway chooses P, V and D, registers the identity-bearing prompt request
with V (`POST /v1/requests`), then dispatches the same identities to P and D.
The V coordinator accepts control request bodies up to 64 MiB.
V does not tokenize this announcement or build an index. P supplies the precise
token count and layout in its Entry manifest; V reserves the actual GPU bytes
and returns the registered receive regions before P writes Prompt KV.

After all P shard writes complete, `STORED` means `KV_READY`. D polls that
control state and obtains P's first-token metadata, then enters its waiting
queue without a V-to-D KV transfer. Once continuous batching selects the
request, D rank 0 submits a list of sequences and per-rank registered receive
regions to `POST /v1/retrieve`. V returns **the entire Prompt KV**. There is no
index construction (diagram step 7) or CAGRA search (step 12).

Before the first D forward, and after every M completed D-generated tokens,
all D ranks wait for transfer completion, overwrite only valid Prompt token
slots, synchronize the local GPU copy, and ACK that round's Delivery. P's
sampled first token is not counted in M. Each round has a unique Delivery ID;
late/mismatched replies cannot advance the refresh clock. D-generated KV stays
local, including generated tokens sharing the final Prompt page.

Add this option to the **D** launch command to select M (default 16):

```bash
--pvd-kv-refresh-interval 16
```

The receive staging allocation is reused across rounds. This first
implementation uses a synchronous batch barrier and automatically disables
D overlap scheduling; continuous batching still admits and removes requests.
It does **not** hide network latency or reduce Prompt KV memory on D. Budget
for Prompt KV, generated KV, and one full-prompt receive staging per active
sequence/rank. A future selector can replace `selection="full_prompt"` and
return logical `token_ranges`; D currently rejects unsupported selection/ranges.

D maintains renewable consumer leases while requests wait or run, preventing
Entry TTL eviction between refresh rounds. Finishing/aborting a request releases
its lease and staging, but does not destroy an Entry used by another consumer.
On an uncertain transfer timeout, D retains its staging until the coordinator
and every involved V shard confirm that writes are fenced. V rejects delayed
reserve/start messages for fenced IDs. If V stays unreachable, the receive
buffer remains retained until confirmation or process restart.

Upgrade the Gateway, P, V (both shards) and D together. The binary KV layout
remains protocol v2; the new control endpoints require this PVD 3.0 revision.

### Verification

The isolated CPU tests use real PyTorch tensors, HTTP handlers, coordinator
and stores with FakeTransferEngine. The TP test uses thread barriers, not
CUDA/NCCL or real Gloo networking:

```bash
python test/registered/disaggregation/run_pvd_cpu_tests.py \
  test/registered/disaggregation/test_pvd_core.py \
  test/registered/disaggregation/test_pvd3.py -q
```

On an RDMA machine, start V, P, D and the rebuilt Gateway using the commands
below, adding `--pvd-kv-refresh-interval 4` and `--log-level debug` to D. Verify:

1. A deterministic request with `max_new_tokens=18`, EOS ignored, generates
   17 D tokens and logs five refresh rounds (at D counts 0, 4, 8, 12, 16).
2. Test non-page-aligned prompt lengths and compare token IDs against an
   equivalent full-KV baseline; the generated tail must survive refreshes.
3. Run concurrent short/long requests to exercise batch membership changes.
4. Exercise P TP1 → V TP2 → D TP4 and the existing TP2 balanced topology with
   the configured rail mapping; repeat with an explicit single rail if needed.
5. Cancel waiting/running requests and interrupt V connectivity during a
   refresh; D must not run attention against incomplete KV or free an RDMA
   destination before fencing. Restore V and check retained buffers are released.
6. Run a request longer than V's Entry TTL and verify lease renewal keeps the
   Entry available. Measure throughput/ITL separately: CPU tests do not establish
   real GPU correctness, RDMA performance or network/compute overlap.

## Current data-plane topology

- Gateway may route over independently sized pools of P and D worker groups.
  V is one physical node running both GPU storage shards and its coordinator
  in one Python process.
- The legacy TP2 default uses `mlx5_0` for rank 0 and `mlx5_1` for rank 1.
  HCA names and order are configurable on P/V/D;
  for example, `--disaggregation-ib-device mlx5_2,mlx5_3` binds rank 0 to
  `mlx5_2` and rank 1 to `mlx5_3`. Configure matching names on corresponding
  transfer paths: the existing transport still checks local/remote rail-name
  equality, in addition to requiring end-to-end RDMA connectivity.
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
both configured HCAs to be active. Single-rail debug mode requires its selected HCA to be active
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

For one V group, start P and D with the same coordinator URL and model
instance ID. The legacy URL is normalized to a trusted group named `default`:

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
`{"0":"mlx5_0","1":"mlx5_0"}`. A shared HCA emits a single-rail warning
regardless of its name. The same configuration can be written as
`--disaggregation-ib-device mlx5_0` on P, V, and D.

### Configurable HCA selection

All three roles accept `--disaggregation-ib-device`. In **PVD topology**:

- One name shares that HCA across every local group rank, e.g. `mlx5_2`.
- A comma-separated list has exactly one HCA per rank, in rank order.
  TP2 / V example: `mlx5_2,mlx5_3`; D TP4 binding example:
  `mlx5_2,mlx5_3,mlx5_2,mlx5_3`. Entries are not deduplicated.
- `--pvd-rank-rails` remains supported (V also accepts `--rails`), and
  requires exactly one entry per rank. If both flags are explicit, their
  resolved mappings must agree; otherwise startup fails.
- When neither flag is supplied, the legacy TP2 default remains
  `mlx5_0,mlx5_1`. For TP1 / TP4, explicitly configure the desired mapping.
- The common flag also accepts a JSON object or `.json` file containing
  exactly string rank keys `0` through `world_size - 1`, with one HCA per
  value. This matches the generated Mooncake mapping with the default
  rank-to-local-GPU numbering; it is not a list of candidate HCAs per GPU.
- Empty names, malformed mappings, or incorrect entry counts fail early.
  Device presence, ACTIVE ports, and GPU registration/transfer preflight
  remain mandatory for real transport. Different names alone do not prove
  independent physical rails or cross-host GPUDirect RDMA capability.

These list semantics apply only to PVD; original PD/Mooncake selection is
unchanged. This change does not expand the supported P/V/D GPU counts or
remove the transport's existing matching-rail-name checks. For heterogeneous
TP configurations involving cross-rank transfers, continue to use one common
HCA name on all ranks/hosts; accepting a multi-rail list does not by itself
enable arbitrary cross-rail transfers.

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

## Multiple request-selectable V groups

The number of V worker groups is also independent. Each V group is still one
two-GPU storage group with its own coordinator, V0 and V1. Give every P and D
worker the same trusted ID-to-URL registry:

```bash
# Append these options to every P and D launch command. In multi-V mode omit
# the legacy --pvd-vector-coordinator-url option.
--pvd-vector-group vector-0=http://10.0.0.2:9100 \
--pvd-vector-group vector-1=http://10.0.0.6:9100
```

Register the identical group IDs with the Gateway:

```bash
sglang-router --pvd-disaggregation \
  --pvd-vector-group vector-0=http://10.0.0.2:9100 \
  --pvd-vector-group vector-1=http://10.0.0.6:9100 \
  --prefill http://10.0.0.1:30000 none \
  --decode http://10.0.0.3:30000
```

At startup the Gateway strictly validates every configured V group. It then
selects a complete V group round-robin for each routing attempt and injects
the trusted `pvd_vector_group_id` together with `pvd_transfer_id` and
`pvd_delivery_id` into both the P and D request copies. P and D reject a group
ID that is absent from their startup registry; request data is never accepted
as an arbitrary coordinator URL. An Entry and all of its Deliveries remain
bound to the selected V group. A retry creates fresh IDs and may select another
V group.

The same identity-bearing request is first announced to that V coordinator;
an admission failure prevents P/D dispatch for that attempt. The original
`pvd_delivery_id` is the Decode consumer identity and prefix for per-round IDs.

The fake transport and CPU storage switches are test-only and require both
`--allow-fake-transport` and `--no-strict-rdma-preflight`. The Gateway's strict
PVD startup gate intentionally rejects such a V group.
