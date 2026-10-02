# OasisKV alignment, 2026-10-02

User requirement: align this branch with OasisKV's Decode method. Work in the
isolated `codex/pvd-oasiskv` checkout; commit each completed stage. Existing
whole-prefix target probes must not be used by an Oasis-mode request.

Paper: https://arxiv.org/html/2608.08097v1, sections 4.2 and 4.4.

## Required execution contract

1. A target-specific EAGLE3 head proposes one lookahead token using only
   committed target features and its separately owned cached draft state.
2. A single target forward processes the actual and lookahead rows together
   through QKV projection, attention, and FFN. Both attend to the same resident
   sparse history; causal masking prevents the actual row seeing the future.
3. Immediately after each layer's RoPE projection, publish that layer's future
   query. Run selection and data transfer concurrently with foreground work.
4. The next target step waits for a layer's own incoming working set when that
   layer consumes it. Do not put an all-layer barrier at the start of the step.
5. Commit only actual token states, actual KV, and actual auxiliary features.
   The prototype always rejects the lookahead token even when it is correct.
6. Retain predicted/resident intersections, cap replacement per KV head, and
   send only D CPU-cache misses. Keep a bounded per-request cache and preserve
   request/incarnation/step/layer identity through transfer and retirement.
7. Initial admission receives sparse working sets plus draft initialization
   state, not a complete Prompt KV replica on D. P/V keeps the immutable full KV.

## Selector scope

The paper performs head-wise block ranking against min/max K summaries on D.
The existing experiment queries token-wise CAGRA on V. This distinction must
stay explicit. The user has been asked whether alignment includes moving the
selector; shared paired-forward and ownership work does not depend on that
answer. Preserve the selected fast graph if V remains the selector.

## Implementation order and evidence

1. Audit the experimental paired path and paper, record all discrepancies.
2. Make paired execution compatible with the serving target's existing weights,
   while preserving residual/norm semantics, causal isolation and actual-only
   commits. Enforce the execution contract; no full-prefix probe fallback.
3. Add explicit request-owned per-layer working sets and transfers. A whole
   forward bank lease must not be repurposed as independently replaceable layers.
4. Install explicit Oasis admission/Decode wiring and paired model execution in
   the formal Scheduler. Validate actual model/TP/device/backend support before
   admitting requests. Keep the existing serving modes and default unchanged.
5. Test row isolation, shared projection calls, per-layer publication/consumption,
   foreign/stale replies, cancellation, memory limits and transfer drainage.
6. On idle CloudLab GPUs, use the same model/Prompt/KV budget/graphs and warmup
   to compare ordinary probe, serialized paired, and overlapped paired paths.
   Include startup, Q generation, selection, delivery, consumer waits and client
   completion. Use live selection for free Decode; record quality independently.

## Current evidence

The standalone HF experiment already implements steps 1-6 of the execution
contract. It is not a formal Scheduler integration and does not use the paper's
D-side summaries. The latest 1.01-second-Q formal Decode measurement was made
on the separate `codex/pvd-search-opt` branch's ordinary probe mode. It cannot
be reported as OasisKV performance. Existing report:
`benchmark/results/pvd_oasis_cloudlab_20261002.md`.
