# OasisKV alignment, 2026-10-02

User requirement: align this branch with OasisKV's Decode method. Work in the
isolated `codex/pvd-oasiskv` checkout; commit each completed stage. Existing
whole-prefix target probes must not be used during Oasis Decode/refresh.

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
7. The paper's initial admission receives sparse working sets plus draft seed.
   The formal pilot in this branch retains the existing complete P->V->D
   bootstrap because the user's selected scope is Decode execution. D performs
   one private actual-Prompt pass to initialize EAGLE features/root Q. Charge
   both costs to startup and label this admission difference in every report;
   never report it as paper-equivalent sparse-only admission. P->D stays off.

## Selector scope

The user selected Decode-pipeline alignment with V/CAGRA and the current fast
graph retained. The paper performs head-wise block ranking against min/max K
summaries on D. This project continues querying token-wise CAGRA on V and must
identify that selector adaptation explicitly. Do not replace CAGRA with Quest.

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
   to compare serialized paired and overlapped paired paths. Ordinary probe
   uses a different refresh schedule; label its evidence separately rather
   than attributing that schedule change to overlap.
   Include startup, Q generation, selection, delivery, consumer waits and client
   completion. Use live selection for free Decode; record quality independently.

## Current evidence

The standalone HF experiment already implements steps 1-6 of the execution
contract. It is not a formal Scheduler integration and does not use the paper's
D-side summaries. The latest 1.01-second-Q formal Decode measurement was made
on the separate `codex/pvd-search-opt` branch's ordinary probe mode. It cannot
be reported as OasisKV performance. Existing report:
`benchmark/results/pvd_oasis_cloudlab_20261002.md`.

## Formal pilot implementation

`--pvd-oasis-config /absolute/config.json` installs a fail-closed, single-request
TP1 Qwen2.5-7B binding. The normal Scheduler sampler and result processor commit
only one actual token. Both candidate/actual target rows share existing weights.
The native per-layer path snapshots/publishes Q before waiting for its incoming
bank; bank selection/retention receives a separate foreground handoff. A bounded
two-future slot per layer permits current and next layer work to coexist.

The pilot keeps P->V->D initial full KV and separately charges one private actual
Prompt pass for EAGLE initialization. Steady Decode has no whole-prefix target
probe. Native sparse Mooncake misses arrive in private D GPU staging, pass the
existing conservative GPUDirect device fence, then fill a monotonic D CPU cache.
Resident intersections are copied on GPU; H2D transfers use CPU cached rows.
The retained device fence is an overlap limitation to measure explicitly.

The selected V source files match the actual source snapshot from the prior
fast-graph/search comparison: 14 four-head graphs/rank, degree16 ring2, prefix2048,
exact immutable KV tail update, itopk2048. Commit `80a91419c` brings that snapshot
into this branch. Per-layer partial groups use existing filtered CAGRA search;
they never wait for another layer to fill a grouped search batch.

Admission JSON has exactly these keys (no inferred budgets or model paths):
`eagle_source`, `eagle_checkpoint`, `eagle_manifest`, `vector_space`, `capacity`,
`max_new`, `top_k`, `workers`, `timeout_seconds`, `max_sequence_tokens`,
`max_decode_steps`, `request_budget_bytes`, `request_scratch_bytes`,
`bootstrap_budget_bytes`, `bootstrap_transient_bytes`, `overlap`.
`overlap=false` is the serialized paired control; both use identical layer KV
selection/replacement/transport.

## Completed bounded formal validation

Implementation `173a2ac0e` passed 118 tests and the real native Gateway/P/V/D
path. Admission runs before the waiting-queue handoff; the exact native bootstrap
receipt is required. Two 2159-token Prompts with 16 output tokens were measured
in serialized/overlap/overlap/serialized order, two warmups per arm and four
formal requests/mode. All modes' actual output IDs/text matched.

Client medians were 13.8479 s serialized and 13.8243 s overlap. The 0.17% difference
is below observed order drift and does not demonstrate reliable speedup. Overlap
subsequent forward median was 783.61 ms, of which 701.53 ms was layer waiting;
EAGLE proposal was 2.67 ms. Per-layer remote search/delivery fragmentation is
the main remaining measured issue, not a slow EAGLE proposal. No Q-only timing
is inferred from total forward minus layer waiting.

The initial full KV and one private seed Prompt pass remain separately charged.
Default-off TP1/single-request restrictions remain; broader quality, memory and
failure-under-load gates are open. Full report/raw evidence:
`benchmark/results/pvd_oasis_formal_cloudlab_20261002.md`.
