# Chunk arrival / CAGRA build overlap feasibility — CloudLab, 2026-09-29

This is a native V100S CAGRA cost experiment, not a live P→V→D result. It
measures graph builds over the same preallocated K on the same GPU/backend and
models the gap between the first and final KV chunk. It does not simulate RDMA,
Prefill, CUDA contention between upload and build, or the serving protocol.

## Setup

- V host: `clgpu021.clemson.cloudlab.us`, GPU 0, Tesla V100S-PCIE-32GB.
- cuVS 25.02, float32 IP, IVF-PQ CAGRA build, graph degree 8,
  intermediate degree 16, one shared 640 MiB native cap.
- Same immutable 56-head × 2304-row × 128-dimension K for both arms. One
  unmeasured warmup, then `full → chunked → chunked → full`. Each full arm
  builds 56 indexes; each chunked arm builds 112 indexes. Backend build and
  synchronization are timed; input allocation and graph disposal are excluded.
- Two page-aligned boundaries tested: 1024/1280 and 2048/256 rows. The
  CloudLab wheel's `cagra.pyx` exposes `build` and `search`, with no Python
  `extend` binding. These arms use independent graphs, not in-place insertion.

## Measured build cost

| First / final rows | Full build observations | First graph observations | Final graph observations | Median excess work | Minimum inter-arrival gap for a ready-time win |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024 / 1280 | 12.233 / 10.953 s | 8.257 / 7.159 s | 9.404 / 8.025 s | 4.829 s | >4.829 s |
| 2048 / 256 | 13.632 / 10.016 s | 11.541 / 11.682 s | 6.411 / 5.958 s | 5.972 s | >5.972 s |

Let `Δ` be the time from the first chunk becoming safely readable on V until
the final chunk becomes safely readable. The full-build path finishes at
`Δ + T_full`; the independent-graphs path finishes at
`max(Δ, T_first) + T_final`. At Δ=0, chunked graphing loses 4.8–6.0 seconds.
At Δ=5 seconds, 1024/1280 saves only 0.171 seconds in this optimistic model,
while 2048/256 still loses 0.972 seconds. At Δ=10 seconds, the modeled savings
are 2.879 and 4.028 seconds respectively. In both runs, all native allocations
were released after each arm and the final native retained byte count was zero.

The previous matched 2048/256 search experiment measured 48.715 ms per
56-head round for one full graph versus 107.746 ms for two graphs, with only
52.3% Top-4 overlap between the two approximate search paths on synthetic Q.
Its retrieval and serving implications still require real-model Q and traffic.

## Serving compatibility

- `PVDKVSender.should_send_kv_chunk` currently returns true only for the last
  chunk, and the sender packs/transfers the complete Prompt once.
- V publishes the shard as STORED and authorizes indexing only after complete
  bytes and a successful native transport terminal are proved.
- Any streaming protocol must provide page-aligned, ordered chunk identity;
  per-chunk native terminal proof; non-overlapping destination offsets;
  pinned allocation lifetime during an in-flight build; and final-only index
  publication to sparse search. An abort must retire provisional graphs only
  after native completion. The existing one-shot upload authorization cannot
  simply be reused for multiple PUTs.
- cuVS 25.02 has no Python CAGRA `extend`. An actual one-graph incremental
  implementation needs a compatible newer cuVS binding or a separately
  validated C/C++ bridge. Its API requires a caller-owned, padded contiguous
  dataset containing old and new vectors and a stable lifetime through search.

Raw logs are on V under
`$SGLANG_PVD_ROOT/validation/prefix-cagra-spike-20260929/` as
`chunked-overlap-1024.log` and `chunked-overlap-2048.log`.
