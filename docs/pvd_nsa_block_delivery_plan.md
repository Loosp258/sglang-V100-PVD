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

The previously used P/V/D nodes reject the existing project SSH key with
`Permission denied (publickey)`. Local implementation and evidence can proceed;
CUDA/native qualification and fair live timing require restored node access.
