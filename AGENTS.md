# PVD streaming KV upload and incremental Prompt graph work

## Local output directory constraint

- 用户明确要求：工作时不得污染 `D:\code` 根目录。项目生成的文件必须
  放在对应的 `sglang*` 项目目录或其 worktree 内；临时文件、传输 bundle、
  压缩包、日志和实验产物统一放在项目内的 `artifacts/`。除非用户明确
  指定其他位置，创建文件前必须检查输出路径，不得写入项目外的目录。

- Keep all files generated for this project inside its own checkout or worktree.
- Do not write bundles, archives, scripts, logs, evidence, caches or temporary
  files directly into `D:\code` or another directory outside the project unless
  the user explicitly requests that destination.
- Use the project-local, Git-ignored `artifacts/` directory for temporary outputs
  and transfer packages. Keep intended source and reports in their normal
  repository paths. Use explicit output paths and check their resolved location
  before running a command that creates files.

## Current publication authorization (2026-10-05)

- Latest follow-up: implement four further steps, each with a local commit only:
  fused READY before ACK/cleanup; zero-miss absent-write proof in the search
  response; encode a frozen binary Q once; bounded request async scheduling.
  No GitHub push. See `docs/pvd_async_delivery_plan_20261005.md`.

- Latest user instruction: the next five optimizations are local commits only;
  do not push to GitHub. This supersedes the push authorization below for new
  work. Order: binary Q plus fused delivery; fused D receive-slot reuse; compact
  cache snapshots; pinned scratch and event-based bank readiness; request-scoped
  binary control channel. See `docs/pvd_delivery_followup_plan_20261005.md`.

- The user now requests committing and pushing all outstanding work on
  `codex/pvd-oasiskv`, then optimizing in the agreed order and committing/pushing
  each completed step. This supersedes earlier local-only/no-GitHub directions
  for this branch. Preserve the historical experiment records below.
- Order: bounded V source staging/MR reuse; D READY separated from owned cleanup;
  scoped CUDA source completion; binary Q transport; fused search/delivery.
  Keep each experiment isolated and default-off until its gates pass.
- CloudLab is expired and no GPU is available. Continue implementation and
  genuine CPU/byte/lifecycle verification; do not retry expired nodes or present
  CPU policy doubles as CUDA/native or full-path latency qualification.


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

### V search follow-up (2026-10-02)

#### V latency follow-up: native search versus candidate handling

The user requests a further reduction in V retrieval time, including testing
whether its time scale can change. Use the validated partial-head cache as the
new baseline. Keep the fast graph, native CAGRA, itopk2048, Top4, seven live Q
rows/head, resident budget and P/D pipeline unchanged. Do not promise a tenfold
gain from isolated kernel timings or reduce recall/budgets to manufacture one.

1. Instrument manager admission/lock, query placement, finite proof, native
   submissions/completion, candidate mapping/mean restoration/materialization
   and retirement. Measure real K/Q on one and two V GPUs, then live serving;
   distinguish host wall stages, CUDA execution and end-to-end D latency.
2. Optimize the measured hot path behind a default-off flag. For tiny native
   candidate sets, evaluate batched host materialization and CPU mapping/union,
   preserving float32 score restoration, head/chunk mapping and tie policies.
   Retain all input/output/index owners and budget until completion; preserve
   quarantine on an unknown fence. Avoid extra queries or delayed layer batches.
   Also measure a bounded per-rank RMM pool underneath the existing global and
   per-index limiters: native CAGRA currently allocates/frees CUDA scratch on
   every search. The pool's physical maximum must fit the already-reserved
   global native cap, start at zero, retain its resource through every owner,
   and never change search parameters or replace CAGRA with exact search.
3. Validate fixed candidate semantics (invalid/cross-head/duplicate/nonfinite
   outputs, ties, mappings, reversed subsets), stale versions, cache replacement,
   close/search races and unknown completion. Run same-graph real-Q native recall
   before live comparison; preserve approximate-search candidate jitter.
4. Compare cached baseline/new/new/baseline on CloudLab with identical P/D,
   requests, warmups, graph/selection budgets and resources. Save raw stage
   timings, outputs, traffic/source hashes and cleanup evidence. Report V time,
   D wait and client time independently; commit completed stages. Broader
   concurrency/quality/memory gates remain open and defaults stay off.

Completed latency follow-up (`c9512e4c0`): bounded per-rank RMM pool under the
existing640 MiB reservation, plus bounded candidate download/CPU mapping with
unchanged GPU float32 mean-score restoration and native filters. 196 tests
passed (2 explicit native opt-in cases skipped); real V100S/cuVS25.10 probes
verified identical IDs/pages/scores/ordering for fixed native candidates on all
28 layers/both ranks. Independent native calls retain rank0 candidate jitter:
pooled old/new mean Top4 union coverage0.999575/0.999150, worst0.857143 in both;
rank1 all1.0. Pool retains320 MiB/rank in this probe within its640 MiB physical
maximum; Entry close drains native live allocations but retains pool capacity.

Formal `oasis_v_latency_abba01` compares the prior two-head cache against the
two new flags, base/opt/opt/base, four2159-token/16-output requests per mode,
identical overlapped P/D configs and actual output IDs/text. V batch14.374→7.526
ms (1.91x), ID mapping5.119→0.062 ms, D layer-wait sum556.487→422.846 ms,
client11.9547→10.1937 s (14.73% reduction). Native submit+completion must be
combined per observation:2.729→1.993 ms; pooling shifts implicit waits into the
explicit fence. Complete queries remain millisecond-scale; no overall tenfold
or submillisecond claim. Both orders/cases improve; no sparse-only admission,
Q-count reduction, pure-kernel timing or serial/overlap gain is inferred.
All services drained, owned/cleanup_errors empty, all GPUs0 MiB. Full evidence:
`benchmark/results/pvd_oasis_v_latency_cloudlab_20261002.md`. Keep the new flags
default-off and retain the broader quality/concurrency/pressure gates.

Keep the selected fast graph, CAGRA `itopk_size=2048`, Top4, 32-token bank and
the current paired Decode/arrival schedule fixed. Current Oasis sends two
head items per rank/layer, so the existing complete-four-head search cache
falls back to per-item searches. Optimize already-arrived head subsets without
waiting for another layer or issuing fake queries. Keep this opt-in.

1. Measure existing formal V batch stages and preserve their request counts.
2. Extend the Entry-owned filtered CAGRA workspace to selected head subsets,
   reusing filters/params/output storage and one submission-completion fence.
   Preserve every head's mapping, scores, version pins and reader/budget owner.
3. Test subsets on both ranks, reversed item order, cache reuse across adjacent
   layers, shape replacement, stale identities, close/search and unknown work.
   Validate native search quality on identical K/Q before online comparison.
4. Deploy only to an isolated V checkout. Compare baseline/optimized/optimized/
   baseline with identical P/D, graph settings, requests, warmups and resources.
   Attribute V processing, D layer waits and client time separately. Retain
   outputs/raw hashes/traffic and negative results; commit completed stages.

The opt-in subset workspace is implemented (`--prompt-index-partial-group-search`,
launcher `PVD_PARTIAL_GROUP_SEARCH=1`), with 150 tests passing. Native real-K/Q
two-head probes on identical fast graphs reduced isolated manager time from
about 3.1 to 2.35 ms. Fixture Q has two rows/head, unlike the live seven-row GQA
shape. One rank0 layer showed CAGRA candidate jitter; 100 measurements/mode on
that layer reproduced it in both baseline and cache (union Top4 mean coverage
0.9893/0.9879, single-observation minimum 0.8571 in both). Do not claim exact
candidate equivalence or a production recall guarantee. Online seven-row
ABBA evidence follows below; full graph/width/arrival/Decode settings stay fixed.

The live seven-row trial now completed (`oasis_v_search_abba01`, implementation
`005f874cc`): same two2159-token Prompts,16 output tokens, four requests/mode,
base/opt/opt/base, identical actual IDs/text. V per-rank layer batch median
20.039→13.995 ms, D foreground layer-wait sum687.889→554.700 ms, client
13.5966→11.9948 s (11.78% reduction), with gains in both orders. Both modes use
overlap=true; this isolates the V cache, not the gain from Decode overlap.
150 tests passed; all GPUs drained to0 MiB. Report and native jitter evidence:
`benchmark/results/pvd_oasis_v_search_cloudlab_20261002.md`. Keep defaults off,
budget/width unchanged and broader quality/concurrency/pressure gates open.

### User-requested paper alignment (2026-10-02)

Follow `docs/pvd_oasis_alignment.md` for the next stages. The user requires
Oasis-mode Decode to use the paper's shared actual/lookahead forward and
per-layer prefetch/consumption. A complete-prefix target probe is not an Oasis
implementation. Preserve exact actual-only commits, causal masking, bounded
request-owned layer banks, cache-miss transfers and terminal drainage. The
user selected Decode-pipeline alignment while retaining V-side CAGRA and the
current fast graph. Do not migrate selection to D-side block summaries. Keep
the selector adaptation explicit in measurements and documentation.

The new formal pilot is a separate explicit `--pvd-oasis-config` mode. The
user's clarified scope aligns Decode and retains the existing fast-graph PVD
bootstrap: initial full P->V->D KV plus one charged private actual-Prompt pass
seeding EAGLE/root Q. This is an admission difference from the paper. Never
hide it, call it sparse-only admission, or use a prefix probe during steady
Decode/refresh. The original sparse-seed standalone experiment remains separate.

Completed stages: `32f3971f0` existing-weight paired target; `706f2fb0e`
request-local actual commits/layer futures; `80a91419c` measured fast V source
snapshot; `033b9974e` opt-in formal Scheduler sampler/result integration and
native layer misses; `173a2ac0e` admission before waiting-queue handoff and bounded
contiguous CPU cache. 118 bounded causal/protocol/index checks passed in the
isolated CloudLab checkout. The formal native Gateway/P/V/D path completed the
bounded TP1 ABBA trial: two 2159-token Prompts, 16 output tokens, four requests
per mode, identical actual IDs/text. Client medians 13.8479 s serialized versus
13.8243 s overlap do not establish a reliable gain (0.17%, below order drift).
Overlap subsequent steps took 783.61 ms with 701.53 ms foreground layer waits;
EAGLE proposals took 2.67 ms. Report/raw evidence:
`benchmark/results/pvd_oasis_formal_cloudlab_20261002.md`. Defaults remain off;
broader quality, TP2, concurrent/load/cancellation/memory gates remain open.

Work only on `codex/pvd-oasiskv` in its isolated worktree. Commit each completed
stage. This branch first provides an explicit experimental P/V/D runner; it does
not enable the existing Scheduler automatically. Existing whole-forward bank
leases cannot provide per-layer replacement without a separate ownership design.

The numbered sequence below records the original standalone experiment. The
formal pilot uses the admission contract and completed validation above.

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

### OasisKV first bounded result (2026-10-02)

The explicit P/V/D experiment is implemented in `benchmark/pvd_oasis_experiment.py`
with role launchers, matched trace analysis and a live native stability probe.
On one V GPU and one D GPU, fixed-selection comparisons over 16 teacher-forced
steps reduced median per-step time by 15.01%, 11.95% and 11.13% on 141/1372/2155
tokens. Native searches still execute and Q hashes, banks and KV bytes match.
This is relative to serialized paired forwarding, not existing serving. Before
EOS, working-set true-Q Top10 coverage was 0.99094/0.92439/0.90645; worst query
coverage reached zero. One completed reading output matched the full-KV reference;
math was truncated. Native identical-Q selection jitter is preserved, including
the initial failed fairness gate. Four ownership/selection CPU checks and actual
target paired-row isolation passed. The seed serializer now excludes future-Q
backing storage and excludes future teacher tokens from free generation, with
a CPU fixture check. Report: `benchmark/results/pvd_oasis_cloudlab_20261002.md`.
Keep this as an explicit isolated entrypoint. Scheduler bank ownership, Mooncake,
two V ranks, bootstrap before graph READY and the 40-question quality gate remain
unvalidated. The experiment's serving and staging processes were stopped.

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

#### Verify pooled host candidates and move finite-Q proof to host (2026-10-02)

Use c9512e4c0/33a81091e as the measured baseline, with both bounded native pool
and host candidates enabled. Keep the selected fast immutable KV graph,
itopk2048, Top4, resident/traffic budget, Q counts, P/D and paired per-layer
Oasis Decode unchanged. Work only in the codex/pvd-oasiskv isolated worktree.

1. Recheck saved evidence and lifecycle/selection tests; diagnose the remaining
   finite-Q GPU wait and D search/delivery interval. Preserve negative findings.
2. Add a default-off host_query_validation path for already-host-resident Q.
   The native workspace must own a private float32 CPU snapshot, validate its
   finite values before copying, and submit exactly that snapshot. Do not trust
   a caller boolean/proof or disable device/shape/head/identity validation.
   GPU callers retain the original finite proof. Keep snapshots, copies, native
   results, reader and scratch budget until proven completion; unknown work
   retains owners and quarantines. Restore mean scores with identical GPU math.
3. Test malformed/nonfinite Q, caller-alias mutation, reversed subsets, close,
   native failure/unknown fences and budget refund. Run real two-rank immutable
   K/Q probes with identical graphs/queries; record exact union coverage and
   candidate jitter. Commit each completed stage.
4. Compare latest baseline/new/new/latest baseline on CloudLab: identical
   services/resources/warmups/live requests/search parameters and P/D configs.
   Save full raw logs, actual output IDs, source hashes, traffic and cleanup.
   Report V processing, D wait and client timing separately. Retain default-off
   until broader quality, concurrency, cancellation and memory gates pass.

#### Separately measure per-request HTTP connection reuse (2026-10-02)

Saved formal traces show D search HTTP ~11.7 ms versus V batch7.5 ms, while
layer search/delivery ~35.3 ms. Each layer currently closes its search/control
sessions; control already reuses connections within a layer. Do not attribute
all delivery time to TCP handshakes or promise a magnitude gain from reuse.

1. After host-query single-knob ABBA, add default-off request-scoped search and
   control clients on the existing D manager I/O loop. Route calls through a
   bounded loop-affine proxy; retain each worker's own receive registry/stream
   and their existing terminal, native drain and cache-copy contracts.
2. Keep per-layer request/incarnation/step/layer/index identities unchanged.
   Drain HTTP futures on cancellation, join the lookahead workers before
   closing shared clients, retain close futures/cache/budget on unknown or
   timed-out closure, and refuse blocking close on the owning I/O thread.
3. Test multi-worker session reuse, foreign/stale identities, cancel/close
   unwind and unknown owner retention. Add actual session creation/reuse counts
   and search/delivery stage evidence, without asserting network-only timings.
4. Compare latest V baseline with/without only request-scoped I/O on matched
   live CloudLab ABBA requests. Keep Q counts, retrieval budgets, graph, workers
   and all other generation controls fixed; save IDs/traffic/logs/cleanup and
   preserve negative results. Commit separately; defaults remain disabled.


Host-Q single-knob ABBA (9869bfbbd, oasis_v_host_query_abba01) completed with
226 tests passed/2 native opt-in skips and same-graph two-rank real-Q probes.
Private device copies equal the proven CPU snapshots and fixed candidate
semantics remain identical. Offline ~1.48->1.42 ms does not transfer online:
V batch7.268->7.482 ms; client10.1576->10.3243 s; layer waits413.382->436.585 ms.
The finite proof itself1.0735->0.034 ms is not a total-query improvement.
All8 formal outputs match; native rank0 approximate candidate jitter persists.
Keep host_query_validation off. For the separate I/O-reuse ABBA, both V arms
use the proven pooled host-candidate path with original GPU finite-Q proof;
only D reuse_io differs. No host-Q/HTTP combination or cross-run subtraction
will be used to manufacture a gain. All host-Q services drained and6 GPUs0MiB.

Request-scoped I/O single-knob ABBA oasis_io_reuse_abba01 completed: local16
passed; CloudLab314 passed/2 native opt-in skips. Actual per-request search
sessions840->2, control770/772->2; all8 outputs match and close proof holds.
Client10.0965->9.8086 s (-2.85%); D cumulative layer waits419.218->409.987 ms
(-2.20%); D search12.042->11.509 ms. Two workers still~99.8% busy. Small pilot
with order drift; keep default off and do not claim a magnitude reduction.
Next sparse-pack ablation fixes reuse_io=false, pooled host candidates and GPU
finite-Q proof, so one knob changes and gains are not added across runs.
Report benchmark/results/pvd_oasis_io_reuse_cloudlab_20261002.md preserves
all raw configs/logs/IDs/session counters/source hashes and6-GPU clean exit.

#### Measure existing fused sparse KV packing on V (2026-10-02)

1. Keep the current fast immutable four-head graph and pooled host candidates
   with GPU finite-Q proof. Fix reuse_io=false and all P/D/search budgets, Q
   counts, worker counts and initial P->V->D graph gate. Only the existing
   experimental-triton-sparse-packing flag varies; defaults stay disabled.
2. Prove old Torch copies and fused byte gather produce identical staging on
   both V100S GPUs with the other GPU current. Cover full FP16 Entry shape
   (28 layers, 2 heads/rank, dim128, 2159 valid tokens plus page padding),
   first/last layer/head, unordered unequal selections and final valid token.
   Reject padding before any destination write. Charge workspace metadata;
   retain source/destination/metadata until a successful device fence.
3. Report warm wall and GPU event microbenchmarks with metadata construction
   and release included. These do not include registration/RDMA/Decode and
   cannot by themselves demonstrate reduced D waits. Preserve test failures.
4. Deploy only the isolated checked-out sources after hashes match. Run live
   base/opt/opt/base with identical per-arm role restarts and warmups. Verify
   actual sparse_pack_kernel on both ranks, output IDs, byte traffic, exact
   search counts/path, close proof and owned-process/6-GPU cleanup.
5. Report V search separately from D search+delivery/consumer wait and client
   wall. Packing is outside the V batch search timer; never call its gain a
   CAGRA search-kernel improvement. Preserve full evidence and commit each
   completed gate. Do not add gains from independent experiments.

V sparse packing native gate oasis_v_sparse_pack_native01 passed143 regressions,
24 standalone CUDA cases and both-GPU page1/page2 byte equality/padding gates.
Triton large60-row pack~2.67->1.23ms; small8-row~0.89->1.20ms including metadata
and fences. No online gain inferred. b61c671a4 also retains index lease after
metadata UNKNOWN despite later successful sync; CPU policy faultgate passes.
See benchmark/results/pvd_oasis_v_sparse_pack_native_cloudlab_20261002.md.
Live single-knob ABBA oasis_v_sparse_pack_abba01 running with I/O reuse false.

V sparse-pack live single-knob ABBA oasis_v_sparse_pack_abba01 completed.
Client10.0111->10.6680s (+6.56%), cumulative KVwait
415.997->463.596ms. Actualbothrank
kernels and15 V sourcehashes proved; all8 outputs match, no retries/fallback,
420jobs and840RPC perrequest with fixed reuse_io=false. Ownedservices drained
and6GPUs0MiB. Default off; no magnitude reduction proved or cross-run gains
added. Fullraw/native/runtimehealth evidence preserved in
benchmark/results/pvd_oasis_v_sparse_pack_cloudlab_20261002.md.

#### Independently test combined submission and receive registration reuse (2026-10-02)

1. Keep fast four-head degree16/ring2 graph, pooled host candidates, GPU finite-Q
   proof, Q counts, Top4, capacity32/max_new16, workers2, and graph-gated initial
   P->V->D delivery. Fix reuse_io=false and Triton sparse packing=false.
2. Add default-off combine_reserve_start: one shard RPC runs existing reserve
   validation followed by existing start, returning identical terminal proof.
   Publish destination locally before first await. Keep exact replay identity,
   changed-destination rejection, fenced tombstones, cancellation/unknown pins,
   and ACK only after successful cache installation.
3. Add per-rank delivery times/counters: allocate/register, reserve/start or
   combined, polls/count, cache copy, ACK, close/unregister. Existing RPC range
   is not pure network; do not infer substage savings from unrelated medians.
4. Test lost reply, identical/changed replay, cancellation before submission,
   sticky UNKNOWN and exact destination byte proof. Deploy checked isolated
   sources; run full ABBA changing only combine_reserve_start and preserve raw
   IDs, traffic, stages, source hashes, cleanup. Commit each completed stage.
5. Separately add default-off reuse_receive_slots with bounded request-owned
   physical registrations. Each logical write keeps fresh delivery/generation,
   exact manifest/extent, exclusive slot ownership and identical byte proof.
   A slot can recycle only after remote fence, local copy and business ACK.
   Unknown registration/write/local ordering/unregister quarantines slot and
   retains physical owner and budget; no force-free or reuse on mere timeout.
6. Preserve worker thread ownership of record operations and per-job CUDA
   streams. Close physical registrations only after all lookahead workers join
   and every slot is proven idle; retain request resources if any close fails.
   Reserve bounded staging/native registration slots without widening traffic
   caps. Test changed extents, successive generations/stale replies, busy slot,
   partial final pages, failed/unknown operations and close races.
7. Run native real-RDMA multi-round slot reuse plus independent live ABBA:
   combine_reserve_start=false fixed, only reuse_receive_slots differs. Prove
   actual register/unregister reductions, unchanged Q/selection/byte counts and
   same outputs; report D waits and client time separately. No cross-run adding
   of gains; defaults stay off until wider shape/load/failure gates pass.

#### Delivery fixed-cost implementation and native gates (2026-10-02)

- Added independent default-off combine_reserve_start and reuse_receive_slots.
  The former uses existing reserve/start lifecycle in one HTTP worker; the
  latter keeps at most workers slots per rank for one request, with exact
  logical destination generations and manifests. CPU finite-Q, HTTP reuse and
  Triton packing remain separate knobs. Every rank delivery now records actual
  allocation/registration, control RPC, cache-copy and local-close costs/calls.
- Receive slots charge their maximum physical capacity even while idle; live
  records retain the existing transfer-slot admission charge without charging
  physical bytes twice. Normal recycling requires exact terminal/ACK and local
  copy completion; cancelled paths need the exact identity-bound fence. UNKNOWN
  never recycles or replaces a slot. Only joined request retirement unregisters
  the original physical MR. CUDA ordering exceptions now remain sticky UNKNOWN
  even if a later stream synchronization succeeds, on both benchmark arms.
- Local CPU gates passed 105 tests including actual HTTP/FP16 record reuse,
  old delayed authorization after reuse, and ordering UNKNOWN. CloudLab broad
  CPU gates and a 48-case Mooncake local-session CUDA1 probe passed: four
  physical receive MRs serviced two separate executors and 48 fresh generations,
  exact bytes matched, original MRs retired and budgets returned to zero. This
  is native local PUT proof; online V->D latency still needs independent ABBA.
- A prelaunch comparison rejected an old D client.py through its source gate;
  no service or formal request started and all six GPUs remained empty. The
  deploy helper now aligns the complete V/D source-gate set. Preserve this
  rejection separately from timed arms and rerun with a fresh tag.

#### Combined reserve/start full-path pilot (2026-10-02)

- Completed clean CloudLab base_a/opt_a/opt_b/base_b on two matched 2159-token
  Prompts and 16 outputs, all-role restarts and two warmups per arm. Fixed
  receive slots=false, HTTP reuse=false, fast four-head V/GPU finite-Q/pool/
  host candidates, and graph-gated initial P->V->D KV. All eight actual output
  IDs/text match. Exact profiles keep 840 searches/request and unchanged Q.
- Client median 10.075738->9.886828 s (-1.87%); request cumulative steady D KV
  wait 6040.889->5780.557 ms (-4.31%); per-step wait median 426.381->414.268 ms.
  Layer RPC 34.931->33.924 ms. Per-delivery joint reserve/start submission
  16.802->16.275 ms; these ranges include source work and are not pure RTT.
  Across four requests/mode, 3084 deliveries each; combined removes 3084
  reserve/start calls but six additional polls leave 3078 fewer control RPCs.
- Client change is comparable to the -1.54%/-3.08% within-mode arm drift;
  treat timing as a small pilot, keep the option default-off. Per-step waiting
  and per-request cumulative waiting are different statistics. Full raw, native
  gate, source identities, failed prelaunch, zero-GPU cleanup and portable
  evidence are in pvd_oasis_combined_delivery_cloudlab_20261002. Registration
  reuse is being tested separately; independent effects cannot be added.

#### Receive registration reuse full-path pilot (2026-10-03)

- Completed independent CloudLab base_a/opt_a/opt_b/base_b on the same two
  2159-token Prompts and 16 outputs, with full role restarts and two warmups
  per arm. Only reuse_receive_slots differs; combine_reserve_start=false,
  HTTP reuse=false, GPU finite-Q proof and Triton packing=false stay fixed.
  All eight actual outputs match; query counts, graph and traffic caps stay
  fixed. Initial graph-gated P->V->D KV remains charged to the client.
- Actual physical registrations and retirements fall from 3084 to 16 across
  four requests/mode (four per optimized request, -99.48%). Every logical
  delivery retains fresh generation/exact bytes; all leases returned, original
  MRs unregistered and pool budgets zero. Steady per-rank prepare median
  1.072->0.111 ms, physical register 0.913->0 ms, close 0.410->0.015 ms.
- Client median 10.202355->10.109606 s (-0.91%), while request cumulative
  steady D KV wait 6133.970->6159.638 ms (+0.42%) and per-step wait median
  423.600->424.259 ms. This does not show reduced D waiting. Client change
  is smaller than the +1.92% optimized arm drift; no stable end-to-end gain
  or latency magnitude reduction is claimed. Independent gains cannot be added.
- Full source/native/cross-node evidence, actual calls, all eight outputs and
  cleanup are preserved in pvd_oasis_receive_slots_cloudlab_20261003. Formal
  date is derived from the first request timestamp in UTC+8. Owned services
  drained, cleanup_errors=[] and all six GPUs are zero. Options stay default
  off pending broader shapes, load and failure gates. Both requested independent
  fixed-cost experiments are complete and their local reductions are measured.

#### Independently test two versus four background workers (2026-10-03)

1. The user authorized continuing with the next proposed fair experiment.
   Test workers=2 versus workers=4 using the existing bounded serving config;
   no serving algorithm changes. Keep combine_reserve_start=false,
   reuse_receive_slots=false and reuse_io=false on both arms. Keep the fast
   four-head degree16/ring2 graph, GPU finite-Q proof, pooled host candidates,
   Top4/capacity32/max_new16 and graph-gated initial P->V->D KV identical.
   Triton sparse packing and direct P->D remain off. Keep memory budgets,
   epochs, exact native byte proofs, ACK and UNKNOWN retirement unchanged.
2. Audit four-worker admission and per-layer handoffs before deployment. Add
   v-workers to the bounded runner and evidence gates as the only differing
   config key. Check actual serving source hashes against the same native gate
   image; do not redeploy or mix source changes while a timed arm is running.
3. Run clean base_a/opt_a/opt_b/base_b with all-role restarts, the same two
   warmups per arm, two 2159-token Prompts and 16 actual output tokens. Record
   source/config/launch identities, all events, queries, native deliveries,
   output IDs and final owned-process/six-GPU cleanup. Freeze query and traffic
   caps but allow native candidate jitter; do not fake a cached selection.
4. Report client TPOT from actual first-to-last token timestamps divided by
   the 15 output intervals, plus the 14 later steady intervals separately.
   Report actual arithmetic mean D KV wait/token, cumulative request waiting,
   layer service and queue times, ready-before-consume and actual callback
   overlap. Compute workload windows using each arm's actual worker count.
   Callback slot occupancy includes blocking RPC and is not CPU/GPU usage.
5. More workers can increase V lock contention or full-device synchronization.
   Treat fixed-service four-worker capacity calculations as diagnostic only,
   never predict waiting will halve. Compare actual V query/delivery and D
   waiting/client timing; preserve any negative result or failed gate. Keep
   the existing default and commit the completed fair evidence. Independent
   prior RPC/registration gains cannot be added to this new experiment.

#### Two versus four background workers full-path pilot (2026-10-03)

- Completed CloudLab base_a/opt_a/opt_b/base_b with all-role restarts, two
  identical warmups per arm, two 2159-token Prompts and 16 actual output tokens.
  Only workers=2/4 differs; graph, Q, retrieval and traffic caps, budgets and
  delivery/source image remain fixed. Combined RPC, receive registration reuse,
  HTTP reuse, Triton packing and direct P->D stay disabled on both arms.
- Actual callback peaks are 2/4/4/2. Every request has 420 jobs, 840 searches
  and 392 consumed steady callbacks. All eight formal Prompt/output identities
  match. Every native delivery has exact row/byte proof and ACK; original MRs
  and request owners retire. Candidate jitter changes payload by at most two
  rows (0.040%) for one Prompt; do not claim byte-identical traffic across arms.
- Arithmetic mean later-token D KV wait is 427.322->412.722 ms (-3.42%);
  request cumulative wait median is 5901.056->5758.343 ms (-2.42%). Actual
  client TPOT from 15 stream intervals is 499.343->509.332 ms (+2.00%);
  the 14 later intervals are 531.238->541.488 ms (+1.93%). Client completion
  median is 9.987313->10.211263 s (+2.24%). No end-to-end gain is demonstrated.
- Mean consumed callback service rises 38.091->77.652 ms, while queue falls
  492.415->458.177 ms and ready-before-consume rises 3.763%->32.015%.
  Observed service/workers/steady-token is 533.278->543.563 ms; doubling
  worker count did not preserve task service time. Callback occupancy includes
  blocking HTTP/CUDA work and is not CPU/GPU utilization or a latency bound.
- On each actual steady forward, total minus measured KV wait averages
  101.919->125.673 ms. The 23.754 ms increase exceeds the 14.600 ms wait
  decrease, leaving total forward 529.241->538.395 ms. This residual includes
  concurrent pipeline work and synchronization; it does not isolate Q compute.
  V steady batch wall/manager wait/candidate download also increase; native
  search timing alone cannot identify the cause or explain end-to-end time.
- Same-config client arm drift is -3.86% baseline and +10.18% optimized,
  exceeding the aggregate client difference. Four requests/mode and two
  Prompts cannot establish stable waiting gains or statistical significance.
  Keep default workers=2 and do not add gains from independent earlier trials.
- The existing 48-test worker/lifecycle CPU recheck and online cross-node
  four-worker acceptance passed against the frozen serving image. These CPU
  tests overlap the historical gate and are not extra unique coverage. No
  real native failure injection, long Decode, TP2 or multi-request load was
  performed. Evidence is in pvd_oasis_workers_cloudlab_20261003; owned={},
  cleanup_errors=[] and all six GPUs returned to 0 MiB.

#### Balance ready-KV Decode with V prefetch: authorized order (2026-10-03)

The user authorized implementing and measuring the following steps in order.
After completing each step, commit it locally on codex/pvd-oasiskv and then
proceed to the next step. Do not push to GitHub. Root owns CloudLab/GPU runs
and commits; delegated independent reviews must not start other GPU jobs.
Preserve unrelated working-tree changes. Keep all generated temporary outputs
in this worktree or the corresponding remote checkout's artifacts directory.

1. Measure the true ready-KV D baseline before changing serving behavior.
   Capture two live 2159-token, 16-output requests using the current SGLang
   Qwen2.5-7B/EAGLE3 paired target. Retain the exact actual/predicted token,
   positions, per-step/per-layer resident PromptBank contents/IDs/valid mask,
   and sufficient draft/feature state for replay. Capture is untimed diagnostic
   work; do not mix its D2H copies into performance observations. Replay the
   same token/bank trajectory with every bank on D and proven READY before
   timing; no V query, transfer or future backlog may run in this replay.
   Keep the actual loaded SGLang target kernels, causal current/lookahead mask,
   actual-only history, EAGLE proposal semantics, 28 formal KV writes, fences
   and ordinary sampler. Reject mismatched outputs/positions/bank identities.
   Separate target-only and full foreground wall/event time and record timing
   scope explicitly. Use warmed repeated/reverse-order runs, identical input
   trajectories and source identities. GPU event readings must be deferred;
   do not add a synchronize at every layer to measure a different pipeline.
   The previous total-minus-consumer-wait residual is a comparator, not pure
   model compute or proof of a no-wait run. Preserve capture/replay identity,
   memory admission and drainage, all logs, outcomes and GPU cleanup. Commit
   this measurement and its evidence before implementing step 2.

2. Implement default-off V sparse batch PUT from the existing registered
   immutable Prompt Entry. Derive checked component-major K/V row slices and
   compact destination offsets from the validated SparseDeliveryManifest;
   do not assume source rows are contiguous across heads/components. Preserve
   Entry/index reader pins, sender/receiver epochs, exact destination generation,
   authorization, aggregate expected bytes, terminal proof, ACK, cancellation,
   quarantine and UNKNOWN retention. Keep the existing source/RDMA ordering
   fences initially; direct batch PUT does not itself prove they are redundant.
   Compare source staging/registration and batch descriptor/small-write costs,
   not just CAGRA kernel time. Cover final partial pages, changed/stale/replayed
   manifests, overlapping/out-of-bounds spans, abort/close and native UNKNOWN.
   Prove exact bytes with native Mooncake and run fair full-path ABBA with only
   this option differing, same fast graph/Q/Top4/cap32/max_new16/workers2 and
   current delivery options fixed. Report negative results; keep default off
   unless the relevant gates pass. Commit code and complete measured evidence.

3. Implement and independently test D GPU receive-to-bank installation.
   Use terminal-success private GPU destinations to populate the next bounded
   bank without making CPU cache D2H then H2D a dependency of bank READY.
   Preserve the CPU historical cache policy with owned asynchronous backup;
   mark cache rows valid only after its copy is proven complete. Carry events,
   exact source/destination generations and owners through both GPU use and
   backup/retirement. Pending backup, UNKNOWN, cancellation or failed ordering
   must never recycle physical receive memory or publish an unproven cache row.
   Keep actual/lookahead causality, capacity/missing-row limits and budgets.
   Test the native lifecycle and compare one variable at a time using the same
   complete path and output/traffic gates. Commit before starting step 4.

4. Separate bounded search, delivery and bank-install stages so one worker is
   not occupied through the whole chain. Use persistent stage resources and
   explicit per-layer stream/event dependencies, retaining deadline, handoff,
   exact ticket and joined retirement rules. Benchmark sustained completion
   intervals and each layer's publish-to-consume deadline; matching one search
   to a whole token is insufficient. Preserve shared RMM/workspace ownership;
   do not delete locks or GPUDirect fences without replacement proof. Reuse
   bounded D attention workspaces and evaluate graph-safe GPU subsegments with
   fixed shapes/owners; leave HTTP/future waits outside captured CUDA graphs.
   Verify logits/actual tokens, publication order and native owners before
   fair end-to-end timing. Keep unsupported capture paths disabled and preserve
   the actual failure evidence. Commit each completed implementation/experiment.

For every comparison, derive dates from formal timestamps in UTC+8, record
warmup/order and actual token events, and distinguish arithmetic means from
medians. Keep initial graph-gated P->V->D KV/private EAGLE seed charged; P->D
direct KV remains off. Do not add gains from separately measured experiments.
Defaults remain unchanged until measured gates support the change. Capacity
calculations from service/workers are diagnostic and not latency guarantees.

#### Ready-KV Decode diagnostic completed (2026-10-03)

- Step 1 captured two real 2159-token/16-output fast-graph P/V/D requests and
  replayed all 15 actual paired target steps on the same loaded SGLang runner,
  target weights, dedicated EAGLE3 closure and ordinary greedy sampler. Each
  case retained 420 immutable GPU banks; actual/predicted tokens, positions,
  logits, features, actual KV and all 420 private formal writes matched bitwise.
- After original native/session/formal retirement, each case ran two excluded
  warmups, three unprofiled wall trials and a separate CUDA-event trial. All
  measured callbacks were READY before consumption. Steady foreground means
  were 28.865/28.861 ms/token, aggregate 28.863. Ordinary access to already-ready
  futures still cost 0.084/0.082 ms/token; it is not a network/KV-arrival stall.
  Paired-target wall averaged about 25.472 ms, EAGLE about 2.455 ms and 28 formal
  KV writes about 0.809 ms/token. The owner fence is nested inside target time.
- Query clone/event, two executor workers, tickets and handoffs remain. Replay
  removes V/network/native receive/CPU backup/background GPU contention and
  includes sampler item()/actual commit. It is a measured counterfactual, not
  an online gain or an exact subtraction from the old 101.919 ms residual.
  The old residual must not be reported as pure Q/model compute.
- The diagnostic had a separate 256 MiB admission and retained 84364892/
  84410036 GPU bytes plus bounded scratch. All references and private rows were
  released after completion proof, and the original initial-KV close future was
  joined before timing. All six GPUs returned to 0 MiB; owned/cleanup empty.
- The first diagnostic startup failed on a wrong hook class name, before any
  formal performance result. The corrected hook has 11 CPU tests including
  real-source class/method checks. Preserve this failed attempt. A later
  240-second archive collection timeout occurred after both cases passed;
  smaller CPU-only SSH ranges recovered the exact original archive, SHA-256
  1ca5f788d857c56a60fd1ef5bb73ab66886921af52cfdc7fa976566b0899a9a2.
  No performance rerun or serving restart was used for artifact recovery.
- Evidence is benchmark/results/pvd_oasis_ready_kv_cloudlab_20261003.md and its
  result directory, including complete hashed trajectories, raw service logs,
  code/config identities, callback traces, failure and cleanup evidence.
  This is TP1, two synthetic-text Prompts and short greedy Decode; it does not
  establish broader quality, load, TP2 or failure-under-load behavior.
- At 28 layers the foreground budget is 1.031 ms/layer, or 2.062 ms/callback
  with two full-chain workers. This is a capacity diagnostic, not a V latency
  guarantee; compare complete service and per-layer deadlines. Step 2 may now
  implement the registered-Entry sparse batch PUT and its independent gates.
  Serving behavior/defaults remain unchanged. Commit locally only; no push.

#### Direct registered-Entry sparse PUT completed (2026-10-03)

- Step 2 adds default-off --experimental-direct-sparse-batch-put on V. A
  checked CPU scatter plan reads only the selected immutable Entry's original
  registered pool. It bounds 128 slices, validates component-major/padded
  layout, current Entry/index/mapping identity, per-shard heads and exact bytes.
  It retains CUDA readiness fences, Entry/index/source owners, write
  authorization, cancellation and UNKNOWN quarantine; no staging fallback.
- CloudLab CPU qualification: 176 passed, 6 CUDA skipped, 1 existing warning.
  A separate native local-session gate passed 48 exact scatter cases over two
  original GPU pools, four receive MRs and two 2-worker executor rounds. It
  proved exact bytes/sentinels, shared source pins and full native/budget
  retirement, with zero staging registration. This is not cross-node timing.
- The live ABBA kept the complete fast graph, target/EAGLE, Top4/capacity32,
  workers2, per-job HTTP, separate reserve/start and per-delivery D MRs fixed.
  All eight 2159-Prompt/16-output formal requests had identical actual outputs.
  Initial graph-gated P->V->D KV/private seed remains charged; P->D stays off.
- Direct scatter regressed: mean client TPOT 489.666 -> 573.654 ms/token;
  mean steady KV wait 413.390 -> 515.459 ms/token; median completion
  9.938 -> 11.607 s (+16.79%). Mean complete callback service 37.344 ->
  43.758 ms/layer. Many 256-byte native slices and extra poll costs outweighed
  avoided staging work. Preserve the negative result; keep direct scatter off.
- Both rank runtime modes and exact formal batch/slice increments were checked
  against D completed delivery profiles. Source/deployment hashes, all raw
  logs/configs/events, 48 native observations, local CPU failure attempts and
  two source-gate prelaunch failures are preserved. No formal request was
  counted from those failed starts. P keeps its separate historical checkout
  fixed by the actual step-1 source hashes, not this later D/V source tree.
- Evidence: benchmark/results/pvd_oasis_direct_sparse_cloudlab_20261003.md
  and its complete result directory. Owned/cleanup are empty and all six GPUs
  are 0 MiB. The unchanged two-worker packed-PUT baseline is used for step 3.
  TP2, load, broader quality and real failure-under-load remain open. Commit
  locally only. Step 3 GPU receive-to-bank with owned asynchronous CPU backup
  may now proceed; do not add these independent experiment gains.

#### GPU receive-to-bank and asynchronous CPU backup completed (2026-10-03)

- Step 3 adds default-off D gpu_receive_to_bank. Proven terminal GPU receive
  rows are privately cloned and fenced before ACK/receive MR retirement.
  Next-layer banks consume owned GPU rows directly; asynchronous historical
  CPU backup publishes valid rows only after its own completed copy proof.
  Independent bank/backup pins keep storage charged through actual retirement.
  Borrowed aliases clear before the final unpin, including a concurrent refund.
  Failed/UNKNOWN fences retain owners and budget; no silent fallback or repair.
- The final frozen qualification passed 110 CPU tests and 48 real Mooncake
  local-session cases. The latter force delayed CPU publication, retire a
  logical receive lease and poison its original slot; GPU bank and CPU backup
  remain byte-exact. Four physical receive MRs are unregistered only after all
  native cases. These are separate observations from cross-node online timing.
- Final live base_a/opt_a/opt_b/base_b restarts all roles and excludes two
  identical warmups per arm. Only gpu_receive_to_bank differs; V keeps staging
  packed PUT, the same fast graph/Q/Top4/capacity32/max_new16 and workers2.
  Two owned CPU backup threads are additional D resources in the optimized arm.
  Initial graph-gated full KV/private EAGLE seed stays charged; P->D remains off.
- Arithmetic mean steady KV wait 425.105 -> 440.957 ms/token;
  client TPOT 502.199 -> 518.285 ms/token;
  median client completion 10.147 -> 10.310 s.
  Mean complete callback service 38.303 -> 39.523 ms/layer.
  Preserve means, medians and ABBA drift separately; four requests/mode are
  insufficient to establish broader quality or production gains. Default off.
- All eight actual output ID/text identities match. Actual completed backup
  counts/rows agree with every native delivery; pending owners and charges are
  zero at request close. Original receive registrations/ACKs retire exactly.
  The first prototype and gate are archived separately. A later start was
  interrupted before formal requests to strengthen alias-before-unpin order;
  it and its superseded gate are preserved, excluded from primary statistics.
- Evidence: benchmark/results/pvd_oasis_gpu_bank_cloudlab_20261003.md and its
  result directory, including full raw logs/configs/source identities/native
  observations and portable hashes. Owned/cleanup are empty; six GPUs 0 MiB.
  TP2, long Decode, load, more Prompts and native failure injection remain open.
  Commit locally only, no push. Step 4 staged search/delivery/install may now
  proceed using unchanged packed-PUT V and default CPU-cache D as its baseline.
  Do not add independent earlier experiment gains.

#### Bounded persistent layer stages measured (2026-10-03)

- Step 4's transport experiment adds default-off staged_transport with two
  search, two delivery and one install thread. Every thread owns a persistent
  loop/stream and only its own stage HTTP clients/native registry. At most 56
  jobs may be admitted; exact bootstrap/lookahead tickets and publication
  deadlines are retained. Published futures cannot abandon active owners.
  Close joins upstream before downstream and retires resources on their
  creating threads. Unknown/failing drainage retains owners and reservations.
- Qualification: 116 CloudLab CPU tests and 48 real native scatter cases,
  dispatched as 24 jobs on the actual stage executor. Isolated search/install
  are dispatch/proof checks; CAGRA and actual bank installation are exercised
  separately by complete online requests. The 53 local CPU repeats overlap
  the CloudLab gate; their initial project-local tmp parent error is preserved.
- Live ABBA uses the same two Prompts, 16 outputs, two warmups/arm, hardware,
  fast graph, Q/Top4/capacity32/max_new16, packed V PUT and CPU history cache.
  Only staged_transport differs. Persistent stage clients inherently reuse
  their own sessions, while the older manager-shared reuse_io option is off.
  Extra CPU threads are part of the architecture variable, explicitly counted.
  GPU direct-bank, combined reserve/start, receive slots and P->D remain off.
- All eight actual outputs match; each request has 420 completed jobs/840
  searches, 392 consumed banks and actual phase peaks 2/2/1. Five persistent
  owners retire, versus 420 per-job loops; four search/control clients retire.
  Native registrations/ACKs/byte proofs match every delivery. All deadline
  and consumption timestamps are checked. No abandoned queue/owner remains.
- No stable end-to-end gain: mean KV wait 419.790 -> 446.477 ms/token;
  mean client TPOT 494.631 -> 492.505 ms; median completion 9.894 -> 9.998 s.
  Mean phases are search27.299, delivery37.541 and install5.119 ms; delivery
  queue445.461 ms. Actual bank completion interval18.842 -> 18.642 ms still
  exceeds the 1.031 ms/layer ready-KV diagnostic budget. V batch wall increases
  7.180 -> 14.372 ms with simultaneous search/delivery; individual lock,
  readiness-fence and network contributions were not causally isolated.
- The optimized mean full-chain service516.995 ms includes inter-stage queue,
  unlike baseline37.725 ms callback service. Do not divide chain service by two
  as stage capacity or report it as GPU work. Invalid worker-capacity fields
  are null; phase service/queue and exact sustained intervals are preserved.
- Evidence: benchmark/results/pvd_oasis_stages_cloudlab_20261003.md and its
  hashed raw/deployment/native/phase/event records. Owned/cleanup are empty;
  all six GPUs 0 MiB. Default remains off; broader quality, TP2, load and native
  failure injection remain open. Commit locally only; no GitHub push.
- Step 4's bounded attention workspace and CUDA-graph-safe subsegment
  evaluation remain to do; completing the transport trial does not complete
  the full authorized sequence. Use unchanged two-worker/packed/cache serving
  as the next independent baseline and do not add previous experiment gains.

#### Bounded attention workspace and graph subsegment measured (2026-10-03)

- Step 4's remaining experiment adds default-off attention_workspace on D.
  Request-owned flat buffers preserve original variable span, contiguous GQA
  expansion, mask and SDPA. Actual-only history and publication order remain.
  Static workspace storage is 820160 bytes for the qualified capacity32/steps15
  shape, inside the already charged 32 MiB scratch; close fences before release.
- CloudLab gate: 121 CPU tests and 48 real Mooncake scatter cases. Received
  bytes pass actual serving bank installation and compare to an independent
  oracle; original/workspace CUDA attention matches bitwise at histories0/3/14.
  These use proxy random Q, not real-model quality. A first fixture omitted
  receive-lease return and aborted the next acquisition barrier; preserved
  separately, excluded from qualification. Final owners/budgets/MRs retire.
- The actual P/V/D capture keeps the same loaded SGLang target, real EAGLE
  closure and ordinary sampler for three replay modes on two live trajectories.
  Two warmups and three wall trials per mode/case use alternating reverse order;
  CUDA-event trials are separate. All logits/features/actual KV, predicted and
  actual tokens, positions and420 formal writes match bitwise. Every callback
  is READY before consumption. Native/session/formal original owners retire
  before replay; query clone/event,2workers,tickets and handoffs remain.
- READY-KV steady means: original 28.836,
  workspace 28.604, SDPA graph
  28.543 ms/token. Graph setup/capture
  costs 20.043 ms/trial outside that interval.
  Graph captures only SDPA on39 explicit actual spans; waits/publication/EAGLE/
  sampling/formal writes remain outside. Private graph allocated increment
  560128 bytes; explicit32MiB bound and owned post-fence disposal pass. This
  reduces foreground by about1%, not a magnitude. Short16-output Decode does
  not amortize graph preparation; no serving CUDA graph flag is enabled.
- Independent complete-path base_a/opt_a/opt_b/base_b switches only workspace.
  Hardware/fast graph/Q/Top4/cap32/max_new16/workers2 stay fixed, all roles restart
  with identical two warmups. Stage/GPU backup/IO reuse/combine/receive slots
  stay off. Initial graph-gated P->V->D KV/private EAGLE seed remain charged.
  All8 actual output IDs/text match. Mean KV wait 420.369 ->
  411.835 ms/token; client TPOT 494.448 ->
  487.835 ms; median completion 9.983 ->
  9.870 s. Means/medians/order drift remain separate;
  four requests/mode do not establish broad quality or production gains.
- Evidence: benchmark/results/pvd_oasis_attention_workspace_cloudlab_20261003.md and result directory;
  benchmark/results/pvd_oasis_attention_replay_cloudlab_20261003.md and complete
  hashed live trajectories, exact replay source, stages, configs and logs.
  Portable verification covers deployment/bytes/native/outputs/timestamps and
  archival hashes. Local24-test and11-hook-test repeats are not added to121.
  Owned/cleanup empty, all6GPU0. Default remains off; broader quality, TP2,
  longer Decode, load and native failure injection are still open.
- The authorized four-step bounded experiment sequence is now complete.
  It has not made V complete delivery fit D's roughly1ms/layer foreground
  budget. Preserve negative results and do not sum independent experiments.
  Commit completed code/evidence locally only; no GitHub push.
