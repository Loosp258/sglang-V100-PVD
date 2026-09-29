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

### Independent heads versus four-head grouping

An additional Case 40 degree-16 exact-seed replay compares 56 independent
one-head indexes with 14 four-head indexes on each rank. It uses the same
K/Q, 512-token first chunk, four native `extend` calls per index, and
`itopk_size=2048`. Both orders were run to expose order effects. Times are
the sums across all graphs on one rank, in seconds.

| Run order | Rank | One-head build | One-head extend | One-head total | Four-head build | Four-head extend | Four-head total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| One-head then four-head | 0 | .116 | 2.304 | 2.420 | .091 | 1.164 | 1.255 |
| One-head then four-head | 1 | .122 | 2.213 | 2.334 | .088 | 1.010 | 1.098 |
| Four-head then one-head | 0 | .124 | 1.894 | 2.018 | .090 | 1.325 | 1.415 |
| Four-head then one-head | 1 | .124 | 1.947 | 2.071 | .093 | 1.439 | 1.532 |

One-head graphs reached mean and worst-head Top-10 recall 1.0 on both ranks
in both runs; four-head graphs reached mean .9982/.9955 and worst-head
.95/.90 on ranks 0/1. All invalid-ID counts were zero. Four-head grouping
also incurred about .11–.12 s per rank of filter setup, excluded from the
table; one-head indexes require no such filter. Most of the construction
difference is native `extend`: 56 independent indexes receive 224 calls,
versus 14 grouped indexes receiving 56 calls. The observed four-head
build-plus-extend speedup is 1.35–2.13× depending on run order. These are
offline index times, not production READY or client latency. Raw logs are
`exact16-group1-4-case40-rank{0,1}-20260929.log` and
`exact16-group4-1-case40-rank{0,1}-20260929.log` on node0.

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

### If V gates initial KV delivery on graph READY

The Case 40 online V log (`v-group4a_20260929.log` on node1) recorded three
accepted chunk commits and an immediate Entry commit. Its first 512-row
prefix is explicit in the index log. The remaining upload is consistent with
512 more rows and a coalesced final 1132 rows; the HTTP access log does not
print payload ranges directly. Rank 0 chunk commits were at 11:18:49.672,
50.899, and 51.921 UTC; final Entry commit was at 51.923. Rank 1 commits
were at 49.682, 51.251, and 52.253; final Entry commit was at 52.256.
`conn.py` can skip a nonfinal publish while the previous one is still running,
which explains why the online final insert need not be only 108 rows.

An offline replay with the inferred `0/512/1024/2156` boundaries, degree-16
exact graph seed, and the same Case 40 K/Q measured the following per-rank
aggregate native times across 14 graphs:

| Rank | Build first 512 | Extend next 512 | Extend final 1132 | Mean/min Top-10 recall |
| --- | ---: | ---: | ---: | ---: |
| 0 | .090 s | .319 s | .522 s | 1.000/1.00 |
| 1 | .089 s | .296 s | .503 s | 1.000/1.00 |

At the recorded arrival intervals, each early build and first extend can
finish before that rank's final chunk. If V starts native work immediately
when eligible and both modes incur comparable scheduling overhead, the final
chunk leaves about .522/.503 s of incremental work, versus .362/.361 s for
one complete-build degree-16 graph. The **slower rank's** readiness is then
about .14 s earlier with complete-build. If V sends initial KV to D only
after both ranks are READY, this projects about .14 s less D waiting, before
any V→D transfer or service overhead. This is a trace-driven projection, not
an online D measurement; the exact degree-16 complete-build arm also had
lower recall (.9866/.9902 mean and .80/.90 worst head) than this replay.
The previously reported .264/.250 s final `extend` applies only to a
five-chunk `512×4+108` replay, not to this coalesced arrival shape.

To isolate the role of four-head grouping in the **complete-KV, no-extend**
case, the degree-16 exact-seed arm also ran both group sizes in both orders.
Each row uses the same Case 40 K/Q and cuVS setup; seconds are sums across
all 56 heads on one rank. `build` includes exact adjacency construction,
`from_graph`, and GPU synchronization, but excludes data assembly and filter
setup.

| Run order | Rank | 56 one-head graphs: build | 14 four-head graphs: build | Difference |
| --- | ---: | ---: | ---: | ---: |
| One-head then four-head | 0 | .379 s | .351 s | .029 s |
| One-head then four-head | 1 | .376 s | .350 s | .026 s |
| Four-head then one-head | 0 | .380 s | .362 s | .019 s |
| Four-head then one-head | 1 | .394 s | .376 s | .018 s |

Thus grouping saves only .018–.029 s (about 5–8%) in this no-extend shape.
Unlike the incremental case, there are no repeated native insertion calls
for grouping to eliminate. Four-head search needs per-head bitset filters;
their setup cost .103–.108 s per rank in this probe, excluded above. If
filter preparation is on the readiness critical path, that extra work can
outweigh the pure build saving. One-head mean recall was .9902/.9884 and
worst-head .90/.90 on ranks 0/1. Four-head mean recall was .9866/.9902 and
worst-head .80/.90. All arms had zero invalid IDs. Raw logs are
`exact16-full-group1-4-case40-rank{0,1}-20260929.log` and
`exact16-full-group4-1-case40-rank{0,1}-20260929.log` on node0.

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
For the coalesced replay, set `PVD_CAGRA_GROUP_PREFIX=512`,
`PVD_CAGRA_GROUP_CHUNK_ROWS=0`, and
`PVD_CAGRA_GROUP_BOUNDARIES=0,512,1024,2156`.

cuVS API reference: https://docs.nvidia.com/cuvs/api-reference/python-api-neighbors-cagra

cuVS indexing guide: https://docs.nvidia.com/cuvs/user-guide/api-guides/indexing-guide/cagra
