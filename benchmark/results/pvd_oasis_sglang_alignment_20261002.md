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

## Remaining serving stage

This stage is an explicit native target adapter. Formal Scheduler admission,
request-owned layer bank replacement, EAGLE startup/seed routing and ordinary
Scheduler sampler/result handling still need wiring and online validation.
It must not be reported as a completed serving conversion or a latency gain.
Existing ordinary predictive mode still uses its original probe; it is not an
Oasis mode. Keep defaults unchanged while the explicit Oasis mode is installed.
