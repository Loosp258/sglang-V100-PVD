# Exact per-head KNN seed for chunked CAGRA — CloudLab V100S

This is an offline, real Qwen2.5-7B K/Q probe. It does **not** measure P→V
transfer, production graph readiness, D first refresh, or client completion.
The serving backend still uses its original `ivf_pq` builder.

## Hypothesis and method

The first KV chunk contains only 512 K vectors per local KV head. For this
bounded prefix, a GPU matrix multiplication can find exact inner-product
neighbors without running an IVF-PQ training/build pipeline for every graph.
The probe builds 14 four-head graphs per V rank (56 local heads), using a
separate 512×512 similarity matrix per head and the best 16 or 32 non-self
neighbors for each K (also testing 8 to match the reference graph degree). It
passes the resulting uint32 graph and the same
centered K vectors to cuVS `cagra.from_graph`. Subsequent KV chunks still use
native `cagra.extend`, and search still uses native filtered CAGRA. The exact
graph and native index stay alive together. The reference arm uses the same
K/Q, chunk boundaries, centering, filters, and `itopk_size=2048`, but builds
the initial graphs with `cagra.build(build_algo="ivf_pq", graph_degree=8,
intermediate_graph_degree=16)`.

The K/Q are real model tensors: Prefill supplies K; one next-token forward
supplies Q from two query heads per KV head after RoPE. Both arms are compared
to GPU exact inner-product Top-10 on the original full K. Per-head mean centering uses only
the first chunk and is identical between arms. The timed `build` includes
the KNN matrix multiplication, top-neighbor selection, `from_graph`, and CUDA
synchronization. `extend` is the sum of native calls across all 14 graphs.
K preparation/centering, graph scheduling, transfer, and model inference are
outside `build + extend`.

CloudLab node0 `clgpu020.clemson.cloudlab.us`, two V100S GPUs, one rank per
GPU, isolated cuVS 25.10.0 environment. Online Case 40 uses 2156 tokens
with chunk boundaries 0/512/1024/1536/2048/2156. Natural-language and
seeded-random Prompts use 2048 tokens and 512-token chunks. Each rank has
56 heads and 112 K/Q retrieval cases. Build and search are warmed before
timing. The two ranks ran concurrently for each new arm. Times below are
seconds per rank, except search, which is the median per head in milliseconds.

## Results

| Prompt | Rank | Initial graph | Build | Extend | Build + extend | Mean/min Top-10 recall | Search ms/head |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Online Case 40 | 0 | IVF-PQ, degree 8 | 4.305 | 1.017 | 5.323 | .9982/.95 | 1.118 |
| Online Case 40 | 0 | Exact block, degree 8 | .090 | 1.032 | 1.122 | .9866/.85 | 1.112 |
| Online Case 40 | 0 | Exact block, degree 16 | .094 | 1.342 | 1.436 | .9982/.95 | 1.134 |
| Online Case 40 | 0 | Exact block, degree 32 | .093 | 1.708 | 1.800 | 1.000/1.00 | 1.221 |
| Online Case 40 | 1 | IVF-PQ, degree 8 | 4.313 | 1.222 | 5.535 | .9946/.90 | 1.106 |
| Online Case 40 | 1 | Exact block, degree 8 | .090 | 1.172 | 1.263 | .9902/.90 | 1.101 |
| Online Case 40 | 1 | Exact block, degree 16 | .091 | 1.231 | 1.322 | .9955/.90 | 1.114 |
| Online Case 40 | 1 | Exact block, degree 32 | .094 | 1.678 | 1.771 | .9982/.95 | 1.231 |
| Natural language | 0 | IVF-PQ, degree 8 | 3.860 | .849 | 4.709 | .9911/.90 | 1.095 |
| Natural language | 0 | Exact block, degree 32 | .097 | 1.340 | 1.437 | .9973/.95 | 1.138 |
| Natural language | 1 | IVF-PQ, degree 8 | 3.845 | .948 | 4.793 | .9955/.90 | 1.109 |
| Natural language | 1 | Exact block, degree 32 | .087 | 1.247 | 1.334 | .9973/.95 | 1.139 |
| Seeded random | 0 | IVF-PQ, degree 8 | 4.437 | .995 | 5.432 | .9982/.95 | 1.102 |
| Seeded random | 0 | Exact block, degree 32 | .090 | 1.438 | 1.528 | .9991/.95 | 1.189 |
| Seeded random | 1 | IVF-PQ, degree 8 | 4.348 | .888 | 5.236 | .9991/.95 | 1.063 |
| Seeded random | 1 | Exact block, degree 32 | .089 | 1.288 | 1.377 | .9991/.95 | 1.143 |

All arms returned zero invalid IDs. The exact degree-32 seed cuts measured
`build + extend` by 3.0–3.8× across these paired fixtures. The initial build
is about 40–49× faster; native `extend` sometimes becomes slower, but does
not erase the net gain. Search rises by about 0.01–0.13 ms per head. Degree
8 matches the reference adjacency size, but loses recall on Case 40. Degree
16 uses twice the reference adjacency size and matches or improves mean recall
on that fixture; degree 32 uses four times and improves recall on these three
fixtures. GPU memory and later refresh costs must be checked in production.

Simply changing native CAGRA's build algorithm was not helpful on Case 40:
`nn_descent` took 8.41/7.91 s to build on ranks 0/1, and
`iterative_cagra_search` took 7.16/7.83 s, versus IVF-PQ's 4.31/4.31 s.

### If all KV arrives before building

The same Case 40 K/Q was also replayed with `PVD_CAGRA_GROUP_PREFIX=0`,
`PVD_CAGRA_GROUP_CHUNK_ROWS=0`: all 2156 K rows per head were present before
one build per graph, with no `extend`. These are again sums over 14 four-head
graphs **per rank**, not production READY or D waiting times. In this arm the
per-head mean is computed from the full K, as a complete-build serving path
would do; the exact Top-10 oracle remains on the original full K.

| Complete graph builder | Rank 0 build | Rank 1 build | Mean Top-10 recall, rank 0/1 | Worst head, rank 0/1 |
| --- | ---: | ---: | ---: | ---: |
| IVF-PQ, degree 8 | 4.250 s | 4.125 s | 1.000 / 1.000 | 1.00 / 1.00 |
| Exact block, degree 16 | .362 s | .361 s | .9866 / .9902 | .80 / .90 |
| Exact block, degree 32 | .380 s | .364 s | .9973 / .9964 | .90 / .90 |

All complete-build arms returned zero invalid IDs. On this one Prompt, the
complete degree-32 exact graph is faster to make after the last KV arrives,
but loses some recall versus both the complete IVF-PQ graph and the
degree-32 prefix-plus-extend arm. The latter reached mean recall
1.000/.9982 with worst-head recall 1.00/.95. If D receives initial full KV
directly, its first sparse search waits for V only when it reaches the search
before V finishes; this probe does not measure that overlap or the time
between final KV arrival and native build start.

## Limits and next integration step

These are one Prompt per input style and two next-token Q rows per head, not
a broad query workload. Four-head blocks initially have no cross-head edges;
native `extend` changes graph topology. High observed recall therefore does
not establish a general guarantee. The production backend needs explicit
ownership of the graph buffer, bounded GPU scratch allocation, cleanup on
failure, and a fair P/V/D run measuring READY and D first refresh. In
particular, this offline 3.0–3.8× reduction must **not** be subtracted from
client or D waiting time without a full-system measurement.

Probe code: `test/registered/disaggregation/run_pvd_qwen_grouped_cagra_gpu.py`.
Set `PVD_CAGRA_GROUP_BUILD_ALGO=exact_block_knn` and
`PVD_CAGRA_EXACT_GRAPH_DEGREE=8`, `16`, or `32`; omitting the build-algorithm
variable keeps the IVF-PQ reference. Raw logs are on node0 under
`$SGLANG_PVD_ROOT/validation/logs/group4-*-20260929.log`.

cuVS API reference: https://docs.nvidia.com/cuvs/api-reference/python-api-neighbors-cagra

cuVS indexing guide: https://docs.nvidia.com/cuvs/user-guide/api-guides/indexing-guide/cagra
GPU, isolated cuVS 25.10.0 environment. Online Case 40 uses 2156 tokens
