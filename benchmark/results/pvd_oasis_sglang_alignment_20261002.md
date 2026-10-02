# Oasis Decode alignment: existing SGLang target weights

Branch `codex/pvd-oasiskv`, CloudLab node2, one otherwise idle V100S GPU1.
The user selected Oasis Decode execution with V/CAGRA and the selected fast
graph preserved. D-side Quest summaries are outside that selected scope.

## Completed paired-model stage

`SGLangQwenPairedDecode` uses the already loaded Qwen2.5-7B target object.
There is no duplicate target model and no complete-prefix Q probe per step.
Each of 28 layers performs one two-row QKV projection and one two-row MLP.
Its ordinary SGLang norms, residual handling, RoPE and output projection remain
in use. Paired sparse attention bypasses dense KV writes and retains the actual
row only. The paper's lookahead is always rejected.

Predicted Q is published after RoPE per layer. A callable bank provider waits
only for the consuming layer. Contexts retain banks, native input/output tensors
and pending actual KV through a completion fence. Actual history is appended
only after a complete successful forward; an unknown fence quarantines the
owner and retains its model execution lease and native references.

## Real model verification

One bounded 8-token Prompt and one actual token, FP16, actual checkpoint on
CloudLab. Full resident Prompt banks provide the dense-reference oracle;
this is a causal/model consistency check, not a sparse quality benchmark.

| Check | Observation |
|---|---:|
| Actual QKV projection calls | 28, each with 2 rows |
| Actual MLP calls | 28, each with 2 rows |
| Actual logits max difference from ordinary single-row forward | 0.0234375 |
| Actual auxiliary features max difference | 0.125 |
| Actual argmax | same |
| Change only predicted token | actual logits/features bitwise unchanged |
| Committed actual history | 1 row/layer |
| Formal pool mappings, free rows, actual-token KV canaries | unchanged |
| Per-layer publication/consumption | all 28, in order |

Six paired-attention ownership/causality tests plus four existing pipeline
tests passed on CloudLab (10 total). Native smoke entry:
`test/registered/disaggregation/run_pvd_oasis_sglang_smoke.py`.
The smoke precedes the final retention-only hardening; that hardening changes
owner lifetimes and completion validation, not attention or model arithmetic.

## Formal wiring stage (separate from the model smoke above)

The explicit `--pvd-oasis-config` pilot now wires admission, the native paired
adapter, pinned EAGLE3, request-owned layer banks, ordinary Scheduler sampling
and actual-only result commits. Its Q is submitted before that layer's bank
consume; two bounded layer futures permit current and next work to coexist.
Only native terminal-success GPU destinations may populate the D CPU cache.
The existing GPUDirect device fence remains, followed by owned D2H/cache/H2D.
V/CAGRA is the selected two-rank fast graph (14 graphs/rank), carried in commit
`80a91419c`. Partial per-layer groups use existing filtered native search.

118 tests passed on CloudLab: paired causality, request/future ownership,
foreign commits, deferred consumer release, actual CPU terminal/ACK cache
copy over localhost control, Prompt index and chunks. These tests do not prove
real RDMA, live Scheduler admission or client latency by themselves. The later
bounded native Scheduler/ABBA validation is recorded separately in
`pvd_oasis_formal_cloudlab_20261002.md`; it demonstrates no reliable overlap gain.

This pilot retains the existing full P->V->D initial KV, then performs one
separately charged private Prompt pass for EAGLE initialization. It does not
claim the paper's sparse-only admission. Steady Decode/refresh cannot invoke
that probe. Existing ordinary predictive mode still uses its original probe;
it is not an Oasis mode. Defaults remain unchanged.
