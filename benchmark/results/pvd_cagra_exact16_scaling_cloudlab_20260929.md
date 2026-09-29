# Exact degree-16 CAGRA build and extend scaling on real Qwen K

This offline CloudLab node0 probe measures GPU build and native insertion on two V100S GPUs,
one hypothetical V rank per GPU. It does not include Prefill wall time,
P→V transfer, index scheduling, V→D delivery, or D waiting.

## Method

The probe uses Qwen2.5-7B-Instruct K from seeded-random token Prompts and
the same cuVS 25.10 setup as the exact-KNN report. Each rank owns 56 KV
heads grouped into 14 four-head graphs. K is float32 and centered per head.
Each graph starts with exact inner-product, degree-16 per-head edges imported
through `cagra.from_graph`; a later chunk uses native `cagra.extend`.
The timed initial build includes KNN matrix multiplication, top-neighbor
selection, graph import, and CUDA synchronization. Build and search are
warmed before timing. The random generator is reset to the same seed for
each length, so shorter token sequences are prefixes of longer ones.

`N` means Prompt tokens and also K rows **per head**. Grouped graph points
are four times that number. The reported seconds are sums across all 14
graphs **on one rank**, not a single graph call. Each row is one run per
rank; the 2048-token rank-0 run was restarted once after model startup
could not reserve enough GPU memory. The successful rerun was on an idle GPU.

## Initial graph size

The first four rows below are the initial-build stages of the fixed-512-tail
runs. The final row is a separate full-build run on the same seeded-random
2156-token Prompt. The 1644-row point is an offline split, not a current
512-token Prefill publication boundary.

| Initial K rows/head | Points/graph | Rank 0 build | Rank 1 build |
| ---: | ---: | ---: | ---: |
| 512 | 2048 | .089 s | .098 s |
| 1024 | 4096 | .137 s | .132 s |
| 1536 | 6144 | .217 s | .216 s |
| 1644 | 6576 | .236 s | .234 s |
| 2156 | 8624 | .374 s | — |

For these five rank-0 measurements only, a descriptive least-squares fit is
`B(p) ≈ 0.068 + 6.46e-8 * p²` seconds (R²=.998). The quadratic term is
consistent with the exact per-head `p × p` similarity matrices. It is **not**
a calibrated serving predictor outside 512–2156 rows: CUDA allocation,
memory pressure, GPU occupancy, graph topology, and workload concurrency
may change the curve. The Case 40 full-build cross-check was .362/.361 s on
ranks 0/1. Native IVF-PQ is a different builder: its complete Case 40 graph
took 4.250/4.125 s and should not be modeled by this quadratic fit.

## Fixed 512-row insertion as the original graph grows

Every row inserts exactly 512 new K rows/head in one native `extend`. The
Prompt and original graph get longer together. These runs use the same
seeded-random token family and exact degree-16 builder.

| Prompt `N` | Initial `p=N-512` | Rank 0 extend | Rank 1 extend | Rank 0/1 initial build |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 512 | .344 s | .340 s | .089/.098 s |
| 1536 | 1024 | .375 s | .373 s | .137/.132 s |
| 2048 | 1536 | .371 s | .373 s | .217/.216 s |
| 2156 | 1644 | .387 s | .372 s | .236/.234 s |

Across this tested range, adding a fixed 512 rows took about .34–.39 s per
rank. It rises much more slowly than the initial exact-KNN build and is not
strictly monotonic in these single runs. A fixed-cost component is plausible
because 14 native insertion calls are made, but these data cannot isolate
native setup from graph traversal or allocation costs.

## Tail length as a second independent variable

The earlier real Case 40 sweep holds total `N=2156` constant and varies
the initial graph and final tail together. It measures:

| Initial `p` | Tail `N-p` | Final extend, rank 0/1 |
| ---: | ---: | ---: |
| 1024 | 1132 | .484/.514 s |
| 1536 | 620 | .382/.399 s |
| 1792 | 364 | .352/.358 s |
| 1920 | 236 | .297/.312 s |
| 2048 | 108 | .242/.230 s |

The two tables show why `extend` must be treated as a function of both
existing graph rows and new rows, `E(p, N-p)`, rather than of total `N`
alone. The Case 40 and seeded-random Prompt families differ, so a single
smooth two-dimensional formula is not justified. All arms above returned
zero invalid IDs, but recall varies by split and must remain a separate
admission constraint; the exact-KNN report records the per-head results.

For any deployment threshold, measure `B(p)` and `E(p,m)` on both V ranks,
then add real P→V arrival timestamps. The 25 Gb/s CloudLab `mlx5_0` wire
rate does not capture Prefill, pack, upload queueing, and terminal proof.
The current test runner has context length 2304, so these timings do not
establish scaling for longer supported model contexts.

Reproduction: `test/registered/disaggregation/run_pvd_qwen_grouped_cagra_gpu.py`
through `benchmark/run_pvd_grouped_multichunk_cloudlab.sh`, setting
`PVD_CAGRA_GROUP_SIZES=4`, `PVD_CAGRA_GROUP_BUILD_ALGO=exact_block_knn`,
`PVD_CAGRA_EXACT_GRAPH_DEGREE=16`, `PVD_CAGRA_GROUP_ITOPK=2048`,
`PVD_CAGRA_RECALL_PROMPT=random`, `PVD_CAGRA_RECALL_ROWS=N`,
`PVD_CAGRA_GROUP_PREFIX=p`, `PVD_CAGRA_GROUP_CHUNK_ROWS=0`, and
`PVD_CAGRA_KV_HEAD_START=0` or `2`. Raw node0 logs are
`validation/logs/group4-exact16-scale-n*-20260929.log`.
