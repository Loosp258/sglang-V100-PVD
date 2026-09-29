# Direct P→D initial Prompt KV, with V graph construction in parallel

## Implementation

The optional `SGLANG_PVD_DIRECT_PD_BOOTSTRAP=1` path requires TP1 P/D and
chunked P→V upload. P still sends all chunks to V. At the final Prefill step,
P packs its full Prompt KV into a separate buffer, publishes the immutable
Entry metadata through its bootstrap HTTP process, and performs a native
Mooncake PUT directly into D's canonical full-KV staging. D checks the exact
Entry, byte count, sender/receiver epochs and destination generation; it
installs the KV and ACKs before it is runnable. P's registered source lives
until native terminal completion. V graph construction continues independently.

Predictive D receives V's selected shard route snapshot from P and can begin
formal Decode before `INDEX_READY`. Its first sparse search uses the existing
bounded retry on V's retryable index-not-ready response. D still obtains a V
consumer lease after V reaches `STORED`, for later search and refresh.

The feature is off by default. The existing V→D initial fan-in remains the
comparison path. Later periodic refreshes still use V.

## CloudLab setup and fairness

- Node0: Qwen2.5-7B-Instruct P, V100S; node1: two V ranks on two V100S GPUs
  plus Gateway; node2: Qwen2.5-7B-Instruct D, V100S.
- Both arms use the same base commit `34513f8c`, cuVS 25.10 on V, 512-token
  Prefill chunks, native `build+extend`, two heads per graph, `itopk_size=2048`,
  predictive D with the same draft and retrieval configuration, 2155-token
  Prompts and 6 output tokens. Only the direct-bootstrap flag changes on P/D.
- The same four Prompt texts are used on both arms. Each P/D restart has a
  separate warmup request before recorded cases. V/Gateway stay running. Case
  order differs between arms to reduce order bias. First event measures the
  first nonempty Gateway SSE `data:` event; completion measures the full SSE
  response. All recorded requests returned HTTP 200, six events, six output
  tokens, and matching final-text SHA-256 for each Prompt.

| Case | Baseline first event (s) | Direct first event (s) | Baseline complete (s) | Direct complete (s) |
|---:|---:|---:|---:|---:|
| 5 | 10.065 | 1.850 | 11.643 | 10.692 |
| 7 | 3.902 | 1.319 | 10.823 | 10.671 |
| 9 | 10.359 | 1.888 | 11.984 | 11.325 |
| 11 | 9.908 | 1.652 | 11.549 | 11.255 |

The median first event falls from 9.987 to 1.751 seconds (8.236 seconds).
Median completion falls from 11.596 to 10.974 seconds (0.623 seconds). The
paired completion reductions are 0.951, 0.152, 0.659 and 0.294 seconds.
Four Prompts are a bounded feasibility test, not a latency distribution.

The direct path also completed a full-KV D request with identical output hash.
Its first request took 29.048 seconds because of cold startup; the next took
1.470 seconds. Those cold and warm requests are not included in the table.
For a predictive request, D's first search waited 8.507 seconds while the two
V graphs were finishing; both became READY at 10:30:11.895 and 10:30:12.113,
and D completed the search/refresh at 10:30:12. A previous predictive request
spent 15 seconds initializing its draft path, so it likewise was excluded.

The baseline's case 7 delivered initial KV before its graph was READY. V fan-in
finished at 10:34:19.570, and the two graphs were READY at 10:34:26.166 and
10:34:26.317. Thus this comparison measures the direct path against the
actual existing fan-in implementation; it does not assume V always delays KV
delivery until graph construction completes. The direct path chiefly improves
time to first event; the first sparse retrieval still waits for the graph.

## Validation and remaining gates

Three CloudLab CPU tests passed: exact native fake-transfer bytes and terminal
proof, destination-generation replay rejection plus short-success rejection,
and bounded rendezvous capacity. A further 49 existing Decode fan-in and
lifecycle tests passed. `compileall` and `git diff --check` passed.
The direct path is still experimental. The current implementation and online
measurements cover TP1, one request at a time, and a single Prompt length.
Mixed-request scheduling, abort during P→D PUT, capacity pressure, broader
Prompt lengths, and TP2 are not validated. Keep the feature opt-in.
