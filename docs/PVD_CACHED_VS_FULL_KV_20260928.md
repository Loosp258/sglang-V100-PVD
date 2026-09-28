# Cached PVD versus full KV without periodic refresh (2026-09-28)

## Matched setup

CloudLab V100S: P uses TP2 on two 32 GiB GPUs, V uses two 32 GiB GPUs,
and D uses GPU1 with TP1. Both modes use the same Qwen2.5-7B-Instruct
weights, P/V/Gateway processes, TorchNative target backend, 18,432-token
context, 512-token P prefill chunks, one RDMA rail (`mlx5_0`), and local
commit `ae7e8c709` on D. The target model reserves the same `mem_fraction_static=0.5`
in both D modes. PVD additionally loads its 0.5B draft on D GPU1 and enables
the private draft (384 MiB) and target-Q (3 GiB) prefix caches. Its refresh
policy is M64, lead32, predict32. The full-KV control has no predictive
pipeline and uses `--pvd-kv-refresh-interval 4096`, larger than every tested
completion, so it imports full Prompt KV once and performs **zero periodic
refreshes**. Both still use the same initial full-Prompt fan-in from V.

The 16k single-request run uses the identical seed, prompt SHA-256
`489223019dc08934ff78b586cc86f53e9c1fb1297ff8d0757b97f5a6de9046d4`,
15,875 actual prompt tokens, temperature 0, `ignore_eos=true`, and 256 output
tokens. Each D mode had a separate, excluded 16k warmup request. The P/V
services remained running across modes. P's logs show zero cached prompt
tokens on the compared requests. Times below are seconds. The 16k result is
one measured request per mode; it is a controlled observation, not a
statistical throughput estimate.

| End-to-end 16k request | Cached PVD | Full KV, no refresh | PVD minus full KV |
| --- | ---: | ---: | ---: |
| First token (TTFT) | 8.508 | 8.146 | +0.362 |
| Decode, first to last token | 68.758 | 21.596 | +47.162 |
| Client wall | **77.266** | **29.742** | **+47.524** |
| Largest streamed inter-token gap | 38.359 | 0.086 | +38.272 |
| Formal batch count | 255 | 255 | 0 |
| Formal batch sum | 14.954 | 21.337 | **-6.383** |
| Time between formal batches | 53.801 | 0.258 | **+53.542** |

The batch and gap rows partition the D formal-Decode span and reconstruct the
client Decode time within 0.003 s. They do not include TTFT. The PVD timer
includes executor bind, target run, and result handling; the full-KV timer
includes `run_batch` and result handling. They are formal batch wall times,
not isolated CUDA attention-kernel times.

### Initial Prefill and admission

| Observed stage inside TTFT | Cached PVD | Full KV, no refresh |
| --- | ---: | ---: |
| P Prefill + routing + V storage/index + D admission + first output, composite TTFT | 8.508 | 8.146 |
| V initial full-Prompt fan-in, slower of two parallel rank writers | 0.3245 | 0.3249 |
| V exact-index build, slower of two parallel ranks | 0.1669 | 0.1537 |
| Initial KV bytes written to D, per rank | 455,168,000 | 455,168,000 |

P Prefill and D KV installation were not separately timestamped in this
build. The rank writer and index rows are nested/overlapping work within TTFT;
they must not be added to TTFT. V built its exact index in both modes, even
though full KV does not later search it.

### Formal Decode, by 32-batch segment

The gap column includes idle time before the segment's first batch, except
segment 0. Batch indices follow output ordinal; PVD logged counters 0–254,
and full KV logged 1–255.

| Batch indices | PVD formal | Full-KV formal | PVD gaps | Full-KV gaps |
| --- | ---: | ---: | ---: | ---: |
| 0–31 | 4.150 | 2.669 | 0.217 | 0.032 |
| 32–63 | 4.146 | 2.667 | **38.518** | 0.032 |
| 64–95 | 1.079 | 2.671 | 0.239 | 0.033 |
| 96–127 | 1.103 | 2.675 | **7.081** | 0.032 |
| 128–159 | 1.111 | 2.679 | 0.224 | 0.033 |
| 160–191 | 1.124 | 2.683 | **7.083** | 0.033 |
| 192–223 | 1.132 | 2.687 | 0.231 | 0.033 |
| 224–254 | 1.110 | 2.607 | 0.208 | 0.032 |
| **Total** | **14.954** | **21.337** | **53.801** | **0.258** |

Before the first sparse-bank install, PVD's first 64 formal batches cost
8.296 s versus full KV's 5.336 s. After it, PVD's remaining formal batches
cost 6.659 s versus full KV's 16.002 s. Sparse attention therefore saves
formal batch time, but the prediction pauses dominate total latency.

### Each PVD refresh

These stage durations overlap the formal Decode window; **do not add them to
the batch/gap or client totals**. Capture includes the private draft and
target-Q probe; this build does not time those two subparts separately.
`pause after lead` is the observed gap while this single request is withheld
from formal Decode for prediction. `ready before boundary` is positive when
V's result arrived before the M64 boundary, negative when late.

| Boundary | Capture | V search | Union | Delivery | Refresh path total | Pause after lead | Ready before boundary | Boundary batch gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M64 | 38.376 | 1.530 | 0.002 | 1.110 | 41.018 | **38.229** | +1.641 | 0.023 |
| M128 | 6.853 | 0.727 | 0.002 | 0.597 | 8.180 | **6.803** | -0.001 | 0.010 |
| M192 | 6.863 | 0.778 | 0.002 | 0.569 | 8.213 | **6.813** | -0.010 | 0.018 |

The target-Q cache logged `prefill` on the first refresh and `append` on the
next two; the draft cache likewise appended. V search and delivery were
largely hidden by the remaining formal tokens. The large gaps occurred at
lead positions because the only request pauses while its private prediction
captures Q. With other requests, their formal batches can continue, but this
does not erase the GPU cost or A's own pause. Full KV has no search, sparse
union, delivery, or periodic boundary work after initial admission.

## Two concurrent requests

Two matched rounds use 5,009/1,009 prompt tokens, a 1-second start stagger,
160 output tokens each, temperature 0, and a unique suffix per round. P/V/Gateway
were unchanged; only D mode changed. All four corresponding output hashes
matched exactly between cached PVD and full KV.

| Round | Cached PVD pair wall | Full-KV pair wall | PVD extra |
| --- | ---: | ---: | ---: |
| 1 | 22.382 | 11.000 | 11.382 |
| 2 | 22.491 | 10.940 | 11.551 |
| Median | **22.437** | **10.970** | **11.467** |

The paired client script records wall time and hashes, but does not stream
per-request TTFT or token intervals. Those substeps are not claimed here.
For the 16k single request, both modes produced the expected first fact code,
but their full output hashes differed. Exact answer parity at 16k remains
unverified.

## Evidence and interpretation

The 16k client reports are V `validation/logs/compare_cached_pvd_16k_256_measured.json`
and `compare_full_norefresh_16k_256_measured.json`; pair reports are
`compare_cached_pvd_pair_160.json` and `compare_full_norefresh_pair_160.json`.
D traces are `validation/logs/d-comparecachedpvd.log` and
`d-comparefullnorefresh.log` on clgpu019. V's fan-in and index timings are in
`validation/logs/v-comparecachedfull.log` on clgpu021. The reproducible
single-request D parser is `scripts/pvd_compare_decode_logs.py` with
`--event target_batch|full_kv_batch --run-index 1`. The full-KV log advertises
interval 4096 and has no periodic refresh events for these requests.

The measured 16k Decode gap is **47.162 s**. PVD's formal batches are
**6.383 s faster**, but its between-batch time is **53.542 s longer**.
The immediate optimization target is the private long-prefix capture and
the pause it imposes on A; faster V search or network transfer alone cannot
remove most of this difference in the measured configuration.
