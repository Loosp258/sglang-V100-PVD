# NSA-inspired contiguous KV delivery

2026-10-04, branch `codex/pvd-oasiskv`. Each completed stage is committed locally;
no GitHub push. Keep temporary outputs inside this worktree's `artifacts/`.

## Scope

Keep Qwen2.5-7B, paired actual/lookahead Decode, immediate per-layer Q publication,
V/CAGRA, four-head fast graphs, Top4, capacity32, max_new16 and two workers.
This experiment borrows NSA's contiguous access principle. It does not add
NSA's learned compression/gates or claim native NSA model quality.

The original component-major Entry interleaves two KV heads at each token.
Consecutive tokens for one head therefore cannot become one contiguous RDMA
source slice. The qualified baseline packs selected rows into a contiguous
staging buffer and sends that buffer; preserve that wire contract and all native
terminal, byte-count, identity, fence, ownership and UNKNOWN handling.

## Sequence

1. Replay the two saved real Decode trajectories without GPU work. Count
   contiguous runs in exact bank/miss order and sorted miss order. Account for
   monotonic D CPU caching, exclude never-consumed terminal prefetch, and report
   bootstrap separately. Also quantify whole-block expansion at sizes4/8/16
   against the existing32-token/head cap; expansion is not implemented.
2. Add a default-off contiguous packing option. Preserve manifest order and
   payload bytes; copy each maximal consecutive token run into the existing
   staging buffer with a strided source view. Sort only missing wire token IDs
   under that option; preserve selected bank order and resident policy. Validate
   paired mode activation on D and V rather than silently falling back.
3. Prove exact bytes and selected bank semantics on CPU, including reversed
   IDs, disjoint runs, page boundaries, multiple heads/layers, dtype differences,
   invalid later groups, alias rejection and a failed partial copy.
4. On restored CloudLab access, validate real CUDA packing and native receive
   against an independent byte oracle, then run matched base/opt/opt/base full
   P/V/D requests with identical resources and warmups. Record actual mode,
   run/row counts, bytes, outputs, V service, D wait, READY ratio, TPOT and cleanup.
   Short saved trajectories do not establish full-KV quality or general speedup.
5. Keep the option off unless live results support it. A whole-block selector or
   compressed coarse index is a separate quality experiment requiring equal
   token/byte budgets and real-Q/output-quality comparisons.

## Current external dependency

The user confirmed that the CloudLab lease has expired and no GPU is currently
available. Do not retry those nodes. Local implementation and evidence can
proceed; CUDA qualification and fair live timing await new GPU resources.

## Completed local stages

- Layout analysis committed as `d1305209f`: bootstrap copy-call opportunity
  about35%, steady opportunity about7%; whole-block4 expansion exceeds
  capacity32 in about79% of captured banks.
- Implementation committed as `28e223be6`: opt-in strided run packing plus
  wire-only missing-token sorting; serving defaults remain off.
- Final CPU gate:105passed,6actual CUDA cases skipped,1existing config warning.
  Saved real-model KV verifies840distinct consumed layer banks bitwise, with
  equal token counts, payload bytes and delivery counts. No GPU timing claim.
- See `benchmark/results/pvd_nsa_contiguous_kv_local_20261004.md` for exact
  scopes, earlier failed fixture, source proof and the pending live gates.

## Next local experiment: batch D bank installation

1. Keep selected IDs/order, resident intersections, CPU-cache misses, network
   payload and Prompt-bank capacity identical. Aggregate four heads' resident
   gathers/scatters and CPU-miss H2D into one bounded installation. Preserve
   the caller's CUDA stream, completion fence and UNKNOWN owner retention.
2. Add an explicit default-off `batched_bank_install` D option. Reject mixing
   with GPU-backup, staged transport or attention workspace experiments; charge
   a conservative allocation bound against existing request scratch admission.
   The baseline install path stays unchanged.
3. Validate the actual helper on CPU against an independent per-head oracle,
   then replay the captured real banks, including resident hits and CPU-cache
   reuse. Record exact KV bytes and planned Torch call counts, not GPU timing.
   Add real CUDA tests for nondefault streams and failure-owner retention.
4. Commit implementation and evidence locally. On new GPU resources, run the
   CUDA gate and matched `d-batch-install` ABBA full-path comparison, changing
   only this option. Require actual installation profiles, output/byte budgets,
   D wait, TPOT and cleanup; do not infer a latency win from fewer calls.

### Batch-install local stages completed

- Plan committed as `dff9c8a5b`; implementation as `d6ac2fb61`.
- CPU gate:133passed,14real CUDA cases skipped,1existing config warning.
- Two real trajectories:840distinct banks bitwise exact; resident/local KV
  rows and bytes unchanged. Steady KV H2D submissions fall1189/1183 to387/388,
  resident gather and scatter submissions each fall3136 to784 per case.
- `d-batch-install` fair ABBA/proof entry is prepared. Native/CUDA tests,
  allocator memory, live installation time, D wait and TPOT are still pending.
- Evidence and reproduction scope:
  `benchmark/results/pvd_oasis_batched_bank_local_20261004.md`.
