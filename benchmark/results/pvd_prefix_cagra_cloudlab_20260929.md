# CloudLab V100S prefix CAGRA reuse experiment — 2026-09-29

This is an isolated **native CAGRA build/search microbenchmark**, not a live
P/V/D/Gateway latency result. The prefix cache prototype is not connected to
`PromptIndexManager` or the V upload protocol.

## Matched setup

- V host: `clgpu021.clemson.cloudlab.us`, GPU 0, Tesla V100S-PCIE-32GB.
- Isolated checkout: `b35752ffb6d01d3d8a0ba41c4ad7425e216d320d`.
  Its `cagra_backend.py` and `index_search.py` are byte-for-byte unchanged
  from local HEAD `b6e0ae90f`; the copied prefix prototype's SHA-256 matched
  the local file (`fad2628709f2a8b8991afc0953901bba03da3e8d9b8779a384d8678774e760b8`).
- Same backend instance, prebuilt K input, CUDA device and parameters in both
  arms: cuVS 25.02, float32 IP, IVF-PQ, graph degree 8, intermediate degree
  16, itopk 64, per-index native cap 512 MiB, shared native cap 640 MiB.
- One unmeasured full-shape warmup precedes `baseline → shared → shared →
  baseline`. Each arm processes four sequential Entries of 56 heads × 2304
  rows × 128 dimensions. Every Entry has its **own GPU K buffer**. The data
  are synthetic; equal prefixes are byte-identical by construction and are
  checked against actual K before sharing. Inputs are allocated before timing.
- "Ready" includes native build completion and prefix verification. It
  excludes K production/extraction, graph disposal, P→V transfer, HTTP,
  scheduler work and Decode. Each arm retires all native indexes; the parent
  RMM limiter reports zero retained bytes afterwards.

## Native build results

| Four-Entry scenario | Independent full graphs, two runs | Shared prefix, two runs | Median saved | Native graph builds |
| --- | ---: | ---: | ---: | ---: |
| All 2304 K rows identical | 44.132 / 44.129 s | 11.113 / 11.015 s | **33.066 s (74.9%)** | 224 → 56 |
| First 2048 rows identical, each 256-row suffix differs; both parts use CAGRA | 44.410 / 41.324 s | 39.751 / 33.723 s | **6.130 s (14.3%)** | 224 → 280 |

The first Entry in the split case took 18.385 / 15.995 s versus the
corresponding full-build baseline 12.555 / 10.310 s. The four-Entry total
improves only after later Entries reuse the prefix. For identical complete K,
the three later Entries each became graph-ready in roughly 2–3 ms, while their
baseline native builds each took roughly 10–12 s. This measures a guaranteed
cache hit, not its frequency in real traffic.

## Warm search cost and synthetic retrieval

One separate matched run built both alternatives over the same 56-head K and
searched seven independent random Q rows per head at Top-4. After warmups,
the order was `full → split → split → full`.

| Path | 56-head search, two runs | Median | Exact Top-4 overlap |
| --- | ---: | ---: | ---: |
| One full graph per head | 48.857 / 48.573 ms | **48.715 ms** | 0.3884 |
| Prefix and suffix graph per head | 105.454 / 110.037 ms | **107.746 ms** | 0.4783 |

The split path adds about **59.0 ms per 56-head search round** in this
microbenchmark. Its exact Top-4 overlap was higher on these synthetic random
queries, but both values are low and from one graph build; they say nothing
about recall on real target-model Q. Full-versus-split Top-4 overlap was
0.5230. The prefix cache does not change the search path for identical
complete K: each query uses the same one full graph.

## Interpretation and limits

Full-Prompt reuse is a clear build-time win **when identical K recurs** and
the graph survives between Entries. It adds retained graph memory and needs
model/vector-space identity, per-Entry ID mapping, reference counting, budget
admission and retirement fences before serving integration. V's current
manifest does not carry a trusted token-prefix identity. Bytewise K checking
is safe for this experiment but its real-model hit rate has not been measured.

Strict prefix-plus-suffix graphing helped the four-Entry *aggregate* build
time despite more graph builds, because the 256-row graph built faster. Its
first Entry was slower and each later search costs more. A rough break-even
using these medians is `6.130 / 0.059 ≈ 104` additional 56-head search rounds
across the four Entries; this omits HTTP, batching, memory pressure and real
query behavior. It is not a production admission rule.

The two CloudLab build logs and search log are retained under
`$SGLANG_PVD_ROOT/validation/prefix-cagra-spike-20260929/` on V as
`cagra-identical56-independent-buffers.log`, `cagra-split56.log`, and
`cagra-search56.log`. The scripts are
`benchmark/pvd_prefix_cagra_cloudlab_bench.py`,
`benchmark/pvd_prefix_cagra_search_cloudlab_bench.py`, and
`benchmark/run_pvd_prefix_cagra_cloudlab.sh` in this checkout.
