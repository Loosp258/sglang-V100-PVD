# Oasis Decode alignment: formal CloudLab P/V/D pilot

Branch: `codex/pvd-oasiskv`. Measured serving implementation: `173a2ac0e`.
The user's selected scope is the OasisKV Decode pipeline, retaining V/CAGRA
and the selected fast graph. Paper: <https://arxiv.org/html/2608.08097v1>.

## Implemented path

The formal SGLang Scheduler uses the loaded Qwen2.5-7B target with its dedicated
EAGLE3 head. Actual and lookahead tokens share one two-row target forward.
Each layer publishes the lookahead's target-space post-RoPE Q immediately;
the next step consumes/waits for its own layer's KV bank. The ordinary Scheduler
sampler and result processor commit only actual tokens, KV and target features.
Lookahead tokens are always rejected. There is no steady Decode prefix probe.

Per-head resident intersections are retained, replacements are capped, and only
D CPU-cache misses are fetched through native Mooncake sparse delivery from V.
The bounded contiguous CPU cache is request-owned. Native terminal proof and the
existing conservative GPUDirect device fence precede D2H/cache installation;
CPU misses subsequently populate private GPU banks by H2D. Unknown native
completion retains owners and budget rather than releasing them on timeout.

V preserves the selected graph implementation: two ranks, four heads/group,
14 graphs/rank, degree16 (14 KNN edges + 2 same-head ring edges), fixed centering,
2048-row prefix + 111-row tail, batched preparation, planned tail, score reuse,
early final update, native filtered CAGRA search with `itopk_size=2048`.
This uses custom immutable-KV edge maintenance, not native `cagra.extend`.
Per-layer requests cover only two local heads/rank and therefore use the existing
partial-group filtered search. No layer waits for a later layer to fill a batch.

**Admission remains different from the paper:** full initial P→V→D KV is kept,
with graph-gated delivery and P→D direct bootstrap disabled. D performs one
private actual-Prompt forward to seed EAGLE features/root Q. Both costs are
charged to startup. This pilot does not implement sparse-only admission or
the paper's D-side block-summary selector.

## Fair live comparison

Run: `oasis_formal_abba02`, serialized A → overlap A → overlap B → serialized B.
Both modes use the same actual target/EAGLE checkpoint, graph sources, retrieval
budget, query schedule, native transport and ordinary greedy sampler. The only
configuration difference is `overlap` (defer next-step jobs until the paired
forward ends, versus publish them immediately after each layer's projection).

- CloudLab: P=node0, V=node1 on both V100S GPUs, D=node2 on one V100S GPU; TP1,
  one live request, no Scheduler overlap or CUDA graph. GPUs were idle at start.
- Two identical 2159-token synthetic Prompts, cases 99401/99402; 16 output tokens,
  EOS ignored. Four measured requests/mode, eight total.
- Two separate warmup requests per arm, excluded from formal statistics.
- P/V stay running; D/Gateway restart for each arm. Raw source hashes validate
  the deployed V/D files against the local LF-normalized source before launch.
- Live selection, no teacher-forced banks or token trajectory. All four arms
  produced identical output IDs and text on both cases.

| Formal request wall time, seconds | Case 99401 | Case 99402 |
|---|---:|---:|
| Serialized A | 13.445 | 13.774 |
| Overlap A | 13.620 | 13.841 |
| Overlap B | 13.807 | 13.887 |
| Serialized B | 14.192 | 13.922 |

| Metric, median | Serialized paired | Per-layer overlap |
|---|---:|---:|
| Client completion | 13.8479 s | 13.8243 s |
| First streamed event | 2.7545 s | 2.7733 s |
| Stream duration after first event | 11.0591 s | 11.0193 s |
| D initialization (Prompt seed + initial sparse banks) | 1.6678 s | 1.6856 s |
| First paired Decode forward (banks already primed) | 33.56 ms | 40.98 ms |
| Subsequent paired forward | 783.82 ms | 783.61 ms |
| Sum of foreground layer waits per subsequent forward | 704.98 ms | 701.53 ms |
| Forward elapsed minus foreground waits | 80.76 ms | 79.76 ms |
| EAGLE proposal after initialization | 2.75 ms | 2.67 ms |
| Layer job search/delivery RPC region | 52.10 ms | 52.37 ms |
| Layer worker service | 56.32 ms | 56.27 ms |
| Sparse network payload/request, including priming | 2,562,816 B | 2,563,072 B |
| Initial complete KV/request, logical payload | 123,805,696 B | 123,805,696 B |

Forward statistics exclude step0; each request has 15 actual paired forwards
(the first output token comes from P), followed by 392 consumed lookahead layer
futures. There are 420 layer transport jobs/request: 28 initial-bank jobs plus
392 future jobs. Each issues two rank search batches: **840 search RPCs/request**,
plus sparse delivery control RPCs. The last forward does not enqueue unused
future work. Proposal timings exclude EAGLE's initial Prompt-cache creation.
Table entries are separate medians and need not sum exactly. First streamed
event is a client-visible metric; it is not a measurement of first D generation.
The logical initial KV count excludes page-padding/control overhead.

**No reliable client speedup is demonstrated.** Combined medians differ by
23.59 ms (0.17%), smaller than order-to-order drift. The first overlap arm is
slower than serialized A on both requests; serialized B is slower than overlap B.
This bounded ABBA experiment does not isolate temperature/system drift.

## Bottleneck and practical implications

In overlap mode, about 702 ms of a 784 ms subsequent target step is foreground
layer waiting. EAGLE proposal costs about 2.67 ms. The remaining roughly 80 ms
includes projections, sparse attention, FFN, publication and local synchronization;
it is **not** a measurement of Q projection alone. Later-layer Q still depends
on earlier-layer attention and its incoming KV.

The current two-worker queue cannot hide all 28 layer jobs. Each layer opens
clients/receive state, sends two small rank search batches, then performs native
delivery control and conservative GPU visibility/cache copies. Existing full
four-head grouped-search batching is unavailable to these partial-layer jobs.
These are observed implementation costs; the run does not assign a separate
causal cost to CAGRA kernels, client setup, network RTT or the device fence.
Queue duration is longer when Q is published earlier, so it must not be read
as extra client delay by itself.

The prior ordinary-probe mode's 5.742-second result is historical context only.
It refreshes every four tokens with a different query schedule/budget, uses a
different Draft and whole-prefix target-Q capture, and was not rerun as this
experiment's contemporaneous control. The present numbers measure the overlap
change within the new paired pipeline, not a speedup over that older mode.

## Verification, negative evidence and limits

- **118 tests passed** on CloudLab: paired causality/ownership, ordered layer
  futures, actual-only commits, native-terminal/ACK cache copying over controlled
  localhost transport, Prompt indexes and chunks. These protocol tests alone
  are not RDMA proof; the formal run supplies real cross-node Mooncake evidence.
- The real-model paired smoke previously verified 28 shared two-row QKV/MLP
  calls, actual argmax agreement with ordinary single-row execution, and
  bitwise unchanged actual logits/features when only the lookahead token changes.
- A six-output-token live native smoke completed before the ABBA run. An earlier
  failed admission attempted bootstrap after waiting-queue handoff; its traceback
  is retained. `173a2ac0e` moves admission before that handoff and explicitly
  requires the installed waiting-queue bootstrap receipt.
- After all eight formal requests completed and all D/Gateway logs were saved,
  uncompressed V-log collection timed out. The runner exited 1 during collection,
  not Decode. Recovery terminated only recorded P/V process groups, retrieved
  shared logs compressed, and verified **0 MiB GPU use on every GPU of all three
  nodes**. Recovery evidence is retained. The runner now compresses collection
  and attempts every cleanup even if a collection fails; serving sources were
  unchanged by this post-run harness fix.

Identical outputs here establish mode consistency on two synthetic fixtures,
not full-KV output quality or a general retrieval-recall guarantee. Quality
benchmarks, TP2, concurrent requests, cancellation under load, peak-memory
measurement, pressure and graph-not-ready startup remain open. Mode is explicit
and default-off; the existing modes/defaults are unchanged.

## Reproduction and preserved evidence

```text
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/run_pvd_oasis_serving_cloudlab.py --tag FRESH_TAG --cases 99401,99402 --tokens 16
wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_serving.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/FRESH_TAG
```

Tracked evidence directory: `benchmark/results/pvd_oasis_formal_cloudlab_20261002/`.
`raw.tar.gz` contains original outputs, warmups, logs, argv, launch commands,
source hashes, configs and recovery status. `smoke.tar.gz` retains failed/successful
native pilot stages. `summary.json`, `source_hashes.json`, `checkout_heads.json`,
`checkpoint_manifest.json`, `unit.txt`, and final GPU status are also available
directly. Remote checkout HEADs are launch-gate identities; P/V had the selected
source snapshot overlaid, so file hashes are the authoritative implementation
identity. The original chosen graph document is `PVD_FASTEST_GRAPH_HANDOFF.md`.
