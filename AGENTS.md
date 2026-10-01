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
- An offline algorithm probe in
  `benchmark/results/pvd_cagra_exact_kv_graph_cloudlab_20260929.md` replaces
  only the initial IVF-PQ graph builder with exact per-head KNN over each
  512-row KV prefix, then imports that graph through cuVS `from_graph` and
  continues with native `extend`. On Case 40, degree-16 graph seeds retained
  or improved mean Top-10 recall while reducing measured 14-graph build plus
  extend from 5.32/5.53 s to 1.44/1.32 s on ranks 0/1. Degree 32 gave
  1.80/1.77 s and higher recall in three sampled Prompt styles, at four
  times the degree-8 adjacency size. Degree 8 was fast but lost recall.
  Waiting for all 2156 K rows and building once took 4.25/4.12 s with
  IVF-PQ, or 0.380/0.364 s with exact degree 32 on the same Case 40 fixture;
  the latter lost some recall relative to both complete IVF-PQ and the
  degree-32 incremental arm.
  A Case 40 trace replay for exact degree 16 found that online upload can
  coalesce the final chunk to 1132 rows. Its final `extend` then took
  0.522/0.503 s on ranks 0/1, versus 0.362/0.361 s for a complete build.
  With graph-gated V→D delivery and immediate progress, complete-build is
  projected to make the slower rank READY about 0.14 s earlier, but had worse
  sampled recall; this is not a live D-wait measurement.
  A later-prefix sweep at exact degree 16 found that 1024/1536-row tails
  still cost more than a complete build. If their first builds are fully
  overlapped, 1920/2048-row prefixes can save at most about 0.05/0.12 s on
  the slower rank. A 2048-row prefix must reach V roughly 0.32–0.34 s before
  final KV to finish its build by then; the observed commit sequence is
  consistent with coalescing this prefix into the final upload. The
  late-prefix rank-0 worst-head recall was
  0.85 in the sampled fixture. See the exact-kv-graph report for thresholds.
  The production backend is unchanged. Graph-buffer ownership, memory,
  broader real-Q quality and online P/V/D timing are required before serving.

## OasisKV experiment branch (2026-10-01)

Work only on `codex/pvd-oasiskv` in its isolated worktree. Commit each completed
stage. This branch first provides an explicit experimental P/V/D runner; it does
not enable the existing Scheduler automatically. Existing whole-forward bank
leases cannot provide per-layer replacement without a separate ownership design.

1. Pin the already validated Qwen2.5-7B/EAGLE3 pair. P prepares immutable Prompt
   KV, target auxiliary features and a known root token. D receives its initial
   sparse KV from V, never directly from P. Account separately for feature seed
   bytes and startup; do not hide full-KV admission in D's bootstrap.
2. Implement a bounded step/layer prefetch state machine. Publish target-space
   predicted Q immediately after that layer's RoPE projection. Associate every
   operation with request, incarnation, next Decode step and layer. Accept only
   one future token, reject replay/stale replies, and drain transfers on close.
   Keep the current resident intersection and cap newly admitted rows per head.
3. Implement a TP1 Qwen paired forward. Current and predicted tokens share target
   projections/MLP and the current layer's resident sparse Prompt KV. The predicted
   row may attend to current actual K/V and its own private K/V; actual attention
   must never see future K/V. Commit only actual Decode K/V and actual tokens.
   EAGLE uses committed target features, not features from its rejected branch.
4. Keep V's centered four-head exact-seeded CAGRA, with native filtered search;
   this is an OasisKV pipeline adaptation, not the paper's D-side Quest-summary
   selector. Return only missing layer/head KV; maintain a bounded D CPU cache and
   transfer only its misses. Include query serialization, network, V search,
   selection, CPU cache hits, H2D and per-layer consumer waiting in the trace.
5. Test causal isolation, request/step/layer identity, delayed replies, bounded
   admission, cache reuse and failure drainage. Run one idle GPU first. Compare
   serial versus overlapped paired forwarding on identical model, Prompt, token
   trajectory, budget, graph, query schedule and warmed resources, in reversed
   order. Separately report free-running outputs and a full-KV reference; do not
   equate approximate paired Q with exact full-attention Q.
6. Preserve raw timing/output/traffic evidence and negative outcomes. The first
   bounded cross-node runner uses an explicit experiment protocol, not Mooncake;
   its CPU-serialized transport overhead must not be advertised as production
   RDMA timing. Full Scheduler admission, TP2, concurrent requests, cancellation
   races and graph-not-ready startup remain later serving gates.

## Implementation sequence

### Target-specific pretrained EAGLE3 pair, 2026-10-01

The user redirected quality work to a pretrained target-specific drafter and
authorized one idle GPU first. Use Qwen2.5-7B-Instruct with Thoughtworks's
Qwen2.5-7B EAGLE3 checkpoint pinned to ff17dda64a036cf5bd7bc56c0ab728325f1c0d0b.
Do not continue custom six-layer training or launch all P/V/D GPUs for this step.

1. Fingerprint the downloaded config and weights, pin the author inference
   source, validate vocabulary mapping and every loaded tensor. The actual
   checkpoint has seven draft KV heads, despite its card describing four.
2. On one idle 32 GB V100S, load target and drafter together in FP16. Use the
   published auxiliary target blocks {1,13,24}, whose post-block states are
   HF hidden-state slots {2,14,25}. Validate finite outputs and shifted token
   alignment; the root token supplied by the target must not count as a
   successful draft prediction.
3. Compare eight-token greedy lookahead against the existing six-layer weights
   on identical calibration prefixes, EOS masks and repetition policies.
   Report positional agreement and consecutive-prefix agreement separately.
   Time target feature extraction, draft prefix processing and warm rollout
   separately, and record peak GPU memory and the single visible GPU.
4. Preserve raw results and commit the completed bounded step. These results
   measure token prediction only; no all-layer target Q prediction, native
   CAGRA recall or full P/V/D output-quality claim follows. Later integration
   must account for initial target-feature delivery and sparse D feature drift.

The one-GPU probe completed on node0 GPU1. FP16 loading and the author's
eight-step width-one tree alignment pass. On eight calibration questions and
23 prefixes, position-weighted token agreement improves 19.88% to32.30%, but
math improves21.88% to39.84% while reading regresses12.12% to3.03%. Conditional
2155+root+8 Draft time is36.89ms versus96.28ms for the old token trunk, excluding
Q readout and feature acquisition. Producing target features with a fresh
full Prefill costs737.38ms; reuse and initial feature handoff are required.
Peak allocated inference memory is17.94GiB. Do not promote on aggregate token
agreement; resolve the reading regression and measure real target-Q retrieval
and full-path quality first. Report:
`benchmark/results/pvd_eagle3_pair_cloudlab_20261001.md`.

### Draft-Q quality follow-up, 2026-09-30

The joint six-layer Draft-Q output-quality experiment (`cf447f525`) lost
12 correct GSM8K answers and two HotpotQA answers versus the real target-Q
probe after format normalization. Keep the six-layer architecture, baseline
checkpoint and serving retrieval budget unchanged while testing improvements.

1. Freeze a new train/calibration split from official GSM8K train and separate
   HotpotQA questions. Exclude every question in the fixed 40-question output
   benchmark and the known repository ReAct source. Save source hashes and
   exact chat prompts before capture. Do not train on benchmark answers.
2. Capture real teacher Decode trajectories at early and later committed
   boundaries. The search dataset is the original Prompt K only; generated
   committed tokens belong to the causal model prefix, not to that dataset.
   Label both true-token and current trained Draft-token branches. Stop real
   trajectories at EOS and never treat post-EOS padding as generated answers.
3. From identical baseline weights and captures, compare a frozen six-layer
   token trunk with trainable Q readout against joint token/readout updates.
   Supervise target Q and its scores on true Prompt K; keep candidate budgets
   fixed. Report true-token Q error, token-branch error, combined Top-10 recall,
   selected-position Top4 coverage at K16, worst-layer quality and token agreement.
4. Choose checkpoints using only the new calibration split. Measure cached
   eight-token inference and native retrieval for any recall improvement, then
   repeat the frozen 40-question full P/V/D output benchmark at identical
   generation/retrieval settings, P→D bootstrap KV disabled. Keep the existing
   target-Q probe as default unless output quality passes. Commit each completed
   step and preserve failures and negative results.

Completed one causal adaptation round on disjoint48 training/8 calibration
questions. New joint weights improve combined exact Top10 from0.4606 to0.5836
on25 calibration prefixes, native serving-shaped Top10 from0.4043 to0.5692
on8 early fixtures, and native Top4 coverage@16 from0.6104 to0.7817. Frozen
token-trunk/readout-only control does not improve the generated-token path.
New2155+8 cached inference takes107.7ms median on an idle V100S. The unchanged
40-question full-path benchmark improves math3/16 to8/16 and reading strict
EM13/24 to14/24, with no P→D KV, all predicted refreshes and zero target-Q
forwards. Math still trails the real target-Q reference15/16; preserve the
default and all existing gates. Reports:
`benchmark/results/pvd_draft_q_decode_cloudlab_20260930.md` and
`benchmark/results/pvd_output_quality_decode_cloudlab_20260930.md`.

Next quality iteration starts from checkpoint75a0c48a (math8/16), retains
six blocks and the existing Top16/128-token serving budget. Freeze the prior
eight calibration questions and40 output questions. Collect exact output IDs
on12 training-only requests through actual sparse D, with V graph-gated
initial KV and no P→D KV; release every completed Entry and save serving logs.
Do not infer token IDs by re-tokenizing returned text. On training questions
only, capture longer EOS-bounded teacher continuations and full-attention
teacher recovery labels on observed sparse-D prefixes. Original Prompt K is
the retrieval dataset; observed Decode K is excluded. Match teacher serving
repetition penalty1.05. Compare all-position versus Decode-only token CE
on identical new captures/optimizer budgets, retaining Q and K-score losses.
Choose with the frozen calibration protocol, then measure native retrieval,
cached latency and the unchanged40-question full-path quality test. Preserve
negative results and commit each completed stage. No serving-default change
until output quality reaches the reference gate.

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
