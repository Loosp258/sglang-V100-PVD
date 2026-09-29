# PVD streaming KV upload and incremental Prompt graph work

This file records the implementation order for overlapping P→V Prompt KV
arrival with V-side graph construction. The requested serving outcome is that
V begins a provisional graph after a proven-complete page-aligned KV prefix,
extends it as later chunks arrive, and publishes a searchable index only after
the complete immutable Entry is stored. Keep the current complete-Prompt path
as the default until the streaming path passes the gates below.

## Current constraints and implementation state

- The complete-shard path remains the default. Set
  `SGLANG_PVD_CHUNKED_CAGRA_UPLOAD=1` on P and `--chunked-cagra-upload` on V
  to opt in; V requires Mooncake, native CAGRA and a cuVS `extend` binding.
- Each chunk now has an independent write authorization and native terminal
  proof. `STORED` still requires the complete byte count and a successful
  aggregate terminal after all chunks on each shard.
- The packed shard is component-major: each K/V layer component contains all
  Prompt pages. A new page chunk has a separate destination interval in each
  component; a single append at the end of the existing buffer is incorrect.
- The original CloudLab cuVS 25.02 Python CAGRA wrapper has `build/search`
  but no `extend`. Two independent graphs were measured in
  `benchmark/results/pvd_chunked_graph_overlap_cloudlab_20260929.md`; they
  have extra build and search cost and are not a substitute for insertion.
- The cuVS 25.10 candidate remains in an isolated environment. CloudLab
  experimental V runs load this environment explicitly; the default serving
  dependency and upload mode are unchanged.

## Progress as of 2026-09-29

- An isolated cuVS 25.10 environment on CloudLab V100S passed raw
  `build+extend` measurement and one bounded-backend build/extend/search/
  dispose check; see `benchmark/results/pvd_cagra_extend_cloudlab_20260929.md`.
  The backend now exposes `extend` only for the validated 25.10 call shape.
  A real-model Q probe and online arrival-gap measurements now exist.
- `sharding.plan_prompt_chunk_puts()` now checks and maps page chunks into
  the existing component-major destination. Its reconstruction test passed.
- `prompt_chunks.PromptChunkIdentity` binds each chunk to the Entry, both
  worker epochs, destination generation, page interval and expected byte
  count. `PromptChunkProgress` rejects gaps, overlap and replay and advances
  only after caller-supplied terminal-success and exact-byte proof. V enforces
  this through its begin/terminal/commit endpoints.
- `extract_prompt_k(..., readable_pages=..., first_page=...)` copies only the
  completed K rows from the full component-major allocation. The CAGRA-auto
  wrapper now preserves its owner registry across native `extend`.
- Real Qwen2.5-7B weights were found on CloudLab node0, and a bounded
  real-Q quality probe now compares full/extended CAGRA on identical K/Q.
  On one Prompt, extended Top-10 recall was 0.838 (1024 tokens) and 0.913
  (2048 tokens), versus 0.525 and 0.538 for full graphs. The sets overlap
  by only 0.425 and 0.500, so this is a quality signal, not a deployment
  gate. See `benchmark/results/pvd_cagra_extend_real_q_cloudlab_20260929.md`.
  This remains a bounded quality probe, not a production recall guarantee.
- P now packs each accepted complete-page Prefill chunk; V validates its
  component spans and native PUT proof. The provisional graph builds on a
  proven prefix, extends with later rows and is published only after `STORED`.
  The P/V/Gateway path completed 2155-token requests on CloudLab with the
  same output hashes as complete upload. Unit integration and one native
  V100S manager acceptance passed.
- An initial 2048+107-token online run exposed that the 1-second index reaper
  missed the approximately 0.6-second arrival gap. A later reaper run started
  one 2048-row prefix build before the final chunk, and both STORED commits
  returned while that build was still running. The shard HTTP handler and the
  group-mode local client now both trigger index progress after a successful
  nonfinal chunk commit. No end-to-end graph-ready speedup has
  been demonstrated for this short gap. The paired comparison below used
  smaller Prefill chunks with the same model, requests and cuVS 25.10 on both arms.
- The 512-token paired CloudLab check is recorded in
  `benchmark/results/pvd_chunked_online_cloudlab_20260929.md`. Two matched
  prompts showed 1.2–3.6 seconds less time from final V commit to both
  indexes READY, but client completion was 4.1–5.5 seconds slower. From
  request start to both READY, one pair improved and one regressed. Keep the
  feature opt-in. The paired run exposed that group-mode `LocalShardClient`
  bypassed the shard HTTP index kick, leaving sequential reaper progress.
  The local client now kicks each rank after chunk and final commits. A
  follow-up CloudLab run confirmed concurrent early builds on both GPUs, but
  V→D fan-in stalled until they finished; investigate same-process build/
  delivery interference before attempting default enablement. TP2 and
  failure-under-load gates remain open.
- A small grouped-head CAGRA feasibility probe is recorded in
  `benchmark/results/pvd_cagra_small_group_cloudlab_20260929.md`. Native
  bitset search and grouped `build+extend` can preserve per-head ID mapping,
  and fewer graphs reduce isolated construction time. The uncentered two-head
  option was the closest grouping candidate, but one head lost 0.5 exact
  Top-10 recall, and default-width filtered search returned invalid IDs.
  A subsequent independent-graph diagnosis in
  `benchmark/results/pvd_cagra_independent_recall_diagnosis_20260929.md`
  found large per-head common K components; subtracting a fixed head mean
  preserved exact Top-10 and raised several low-recall native CAGRA heads to
  0.95–1.0 on three small real-Qwen fixtures. Re-evaluate grouped graphs
  against this stronger independent baseline; keep per-head graphs in serving
  until broader real-Q quality, grouped extension, and two-rank online latency
  gates pass.
- The follow-up full-shard real-Q benchmark is recorded in
  `benchmark/results/pvd_cagra_centered_group_real_q_cloudlab_20260929.md`.
  On all 56 local heads, centered two-head graphs roughly halved isolated
  native construction time versus centered one-head graphs. Full-build
  quality at `itopk_size=256` was close on sampled Prompts, but the 2048-token
  `build(512)+extend(1536)` shape dropped mean exact Top-10 recall from 0.953
  to 0.890 and one head from 1.0 to 0.4. Four-or-more-head full graphs also
  returned invalid filtered IDs at `itopk_size=128`. Grouping remains a
  benchmark candidate, not a serving option. Preserve the centered one-head
  quality baseline and require worst-head, multi-extend, memory and online
  end-to-end gates before any grouped implementation.
- The bounded follow-up in
  `benchmark/results/pvd_grouped_multichunk_online_cloudlab_20260929.md`
  used real 512×4+107 Prefill chunks, four native `extend` calls in the
  quality replay, and a default-off two-head grouped V implementation. At
  `itopk_size=2048`, two online-text fixtures on rank0 reached 1.0 exact
  Top-10 for every sampled head; one rank1 fixture reached 0.998 mean and
  0.95 worst head. With D in full-KV mode, two matched reverse-order online
  pairs shortened client completion from 19.807/19.459 s with 56 graphs/rank
  to 10.376/10.580 s with 28 graphs/rank. Both V ranks built concurrently.
  The online build occupied the entire arrival interval, so V coalesced the
  remaining 1643 rows into one native extend. The four-extend quality replay
  and the live two-rank client timing are separate observations. Grouping
  remains opt-in; predictive D, grouped search under live load, memory peak,
  more prompts and failure races are still open.
- The opt-in direct initial-KV path now lets P keep chunked P→V upload while
  sending a separate complete Prompt KV buffer to D at final Prefill. D may
  start Decode before V graph READY; its first sparse search uses the existing
  bounded index-readiness retry. The TP1 CloudLab comparison in
  `benchmark/results/pvd_direct_initial_kv_cloudlab_20260929.md` used four
  matched, warmed, 2155-token predictive requests with six output tokens.
  Median first streamed event improved from 9.987 to 1.751 seconds, while
  median client completion improved from 11.596 to 10.974 seconds. The former
  V→D fan-in can sometimes finish before graph READY, so do not attribute all
  baseline waiting to an enforced graph barrier. Keep
  `SGLANG_PVD_DIRECT_PD_BOOTSTRAP=1` default-off; TP2, cancellation races,
  capacity pressure and broader Prompt shapes remain open.

## Implementation sequence

1. **Prove the native incremental path.** In an isolated CloudLab dependency
   environment, select a cuVS Python version with CAGRA `extend` that runs on
   V100S. Benchmark `build(prefix) + extend(new rows)` against one complete
   `build` on identical real-shaped K. Measure build/extend time, temporary
   and retained GPU memory, search latency and recall on real-model Q. Verify
   the API's padded contiguous dataset and lifetime requirements. If the
   current serving dependency cannot support it, keep streaming disabled;
   never advertise the feature while silently rebuilding or splitting graphs.

2. **Specify chunk identity and layout.** A chunk is a monotonically ordered,
   page-aligned Prompt token range, except the final partial page. Give each
   chunk a unique write identity tied to Entry, shard rank, sender/receiver
   epochs, destination generation, range and expected bytes. Derive checked,
   non-overlapping destination spans for every component of the existing
   component-major layout. Reject gaps, overlap, replay with changed bytes,
   stale incarnations and out-of-bounds writes before native submission.

3. **Extend the upload lifecycle.** Authorize and pin each chunk independently.
   A V chunk becomes readable only after its native PUT is terminal-success,
   its exact byte count is confirmed and the matching authorization is closed.
   Keep unresolved/unknown native writes pinned and quarantined. Cancellation,
   timeout and sender disappearance must prevent future submissions while
   retaining every in-flight source and destination until terminal proof.
   `STORED` and D delivery still require every chunk on both V shards and the
   final first-token metadata.

   Implement this as an explicit manifest upload mode in `protocol.py`, a
   per-shard chunk gate in `vector_store.py`, and matching begin/terminal/
   commit RPCs through `control_server.py`, `coordinator.py` and `client.py`.
   Keep the complete-shard authorization as the default wire contract. The
   streaming mode may be admitted only when V reports native `extend` support;
   otherwise P must use the original complete-shard path for that Entry.

4. **Stream from P.** After each eligible Prefill chunk, pack only its new
   complete pages for each storage shard and submit bounded asynchronous
   PUTs. Advance `start_send_idx` only after this sender accepts the chunk;
   do not hold the scheduler on remote terminal waits. Retain staging memory
   and transfer slots under the existing budget until each PUT drains. Send
   the final partial page and first-token metadata on the last chunk.

   The scheduler's `start_send_idx` is advanced after `send()` accepts pages;
   the sender must own those pages until packing finishes. In `conn.py`, keep
   a bounded queue of chunk publication futures, and in `runtime.py` use the
   component spans from `plan_prompt_chunk_puts()` with one bounded native
   batch PUT per chunk. `PVDUploadManager` owns each child handle and its
   staging registration after the request sender disappears.

5. **Build provisionally on V.** On each chunk's readable transition, pin its
   Entry allocation, copy only completed K rows into an owned, budgeted
   contiguous index dataset and run `build` once, then `extend` in token order.
   Synchronize native work before exposing the next prefix boundary. Keep
   provisional indexes unavailable to sparse search and D. After the final
   complete Entry and all extends succeed, atomically publish a versioned
   full-length ID mapping and index. On failure, retire provisional native
   owners only after completion proof, retain unknown operations and preserve
   dense full-KV delivery.

   Add a provisional owner to `PromptIndexManager` that is separate from its
   searchable `IndexGate`. `VectorKVStore.progress_prompt_indexes()` can then
   pick a proven prefix while upload continues. Use
   `extract_prompt_k(..., readable_pages=N, first_page=previous_N)` for each
   new range, retaining the copied tensors through native disposal. On the
   complete `STORED` transition, check every head's row count and mapping,
   then install a new full-length descriptor without re-building the graph.

6. **Validate and compare fairly.** Exercise TP1/TP2, final partial pages,
   replay, reordered/duplicate chunks, failed/unknown PUT, abort, eviction,
   index close/search races and bounded-memory pressure. On CloudLab compare
   streaming versus current complete-Prompt upload with the same model,
   requests, arrival schedule, chunk size, P/V/D resources and warmup. Record
   timestamps for P chunk ready, each V chunk terminal, first build start,
   every extend, `STORED`, `INDEX_READY`, first search, TTFT and total latency.
   Report cold cases, repeated cases, memory, query latency and real-Q recall;
   enable the path only for shapes/arrival gaps that show an end-to-end gain.
