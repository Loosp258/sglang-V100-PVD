# PVD Predictive KV Retrieval and Prefetch Pipeline: AI Development Handoff

Updated: 2026-09-20.

This document is intended for an AI taking over without access to the previous conversation. Read it in full before inspecting or modifying the implementation. It consolidates the user's current requirements; do not reconstruct the design from guesses about earlier discussions.

Explicit subsequent user instructions take precedence. Update this document when the user changes a decision.

The implemented asynchronous initial-delivery contract and its limits are recorded in
[the bilingual waiting-queue bootstrap goal](PVD_Waiting_Queue_Bootstrap_CN_EN.md).

## 1. Objective

Extend the existing SGLang PVD framework with a predictive KV prefetch pipeline:

1. A user-configurable, separate small draft model predicts future tokens.
2. An isolated probe execution of the target model produces retrieval queries Q.
3. V runs CAGRA search ahead of time and transfers relevant Prompt KV to D.
4. Retrieval and transfer overlap as much as possible with D's ongoing committed decoding.

Predicted tokens are never emitted as committed output. Each request refreshes independently after M committed D-generated tokens. Admitting a new request must not force existing requests to refresh, reset their clocks, or cancel their in-flight prefetches.

Initial KV uses a waiting-queue-triggered pull: D waits until its scheduler places the request into the final waiting queue (`scheduler.waiting_queue`), then initiates delivery of the complete Prompt KV into a registered staging buffer, which D unpacks into the KV pages already preallocated for that request. V still performs the authorized RDMA WRITE; "pull" means D is the initiator, not that the transport direction changes. The request stays in the waiting queue marked not-runnable and is admitted to the running batch only after validated installation and ACK. This path is implemented behind `--pvd-waiting-queue-bootstrap`; real hardware acceptance remains pending.

The outcome to evaluate is end-to-end refresh waiting time, TPOT, throughput, memory usage, and output quality—not merely whether one CAGRA search or RDMA WRITE works.

In this document, “committed” means the real target-model generation path, as opposed to the prediction/probe branch. It does not refer to Git commits.

## 2. Confirmed design decisions

| Topic | User-confirmed requirement |
| --- | --- |
| Committed generation | D's target model generates the actual output tokens. |
| Prediction model | A separate small model, selected by configurable model name or local path; do not hard-code a model. |
| Model revision | Optional. Record the resolved revision when available for reproducibility; it is not a prerequisite for development. |
| Query source | Draft-predicted tokens → isolated target-model probe → queries in the target model's Q/K space. |
| Approximation | Sparse retrieval approximation is allowed, but quality degradation must be measured. |
| Refresh clock | Independent per request, based on committed D token counts. |
| Initial bootstrap | D waits until the request reaches the final waiting queue, then pulls the complete Prompt KV into a registered staging buffer and unpacks it into its already-preallocated final KV pages. V performs the authorized WRITE. The request stays in the waiting queue as not-runnable; admission to the running batch follows installation. |
| New request admission | Admit only when ready, without repeating the initial transfer; preserve existing periods and prefetches. |
| Periodic waiting policy | Already-running requests may retain a synchronous due-refresh barrier. An uninitialized newcomer stays outside the running batch and does not add an initialization barrier for existing requests. |
| V hardware target | V100S. That experimental hardware is currently unavailable locally. |
| Local development | Missing V100S, cuVS, or particular model weights must not stop hardware-independent implementation and CPU tests. |
| Retrieval scope | Search the current request's own Prompt KV, within its selected V group for the first implementation. |
| Generated KV | Remains on D for now; do not continuously write it back to V. |
| Safety | Preserve existing Mooncake/PVD native-handle, fencing, epoch/generation, and resource-lifecycle protections. |

“Configurable model” does not mean every architecture is automatically supported. At actual loading/enabling time, validate tokenizer compatibility, target-architecture Q capture, device, dtype, and budgets. Reject unsupported configurations explicitly rather than silently changing semantics.

## 3. Repository and version state

- Project: SGLang V100 PVD disaggregation framework.
- Original workspace: `D:\code\sglang-V100-PVD`. On Linux/WSL, use the actual repository root.
- Target branch: `pvd-disaggregation`.
- HEAD when this handoff was created: `7beb1cc58c7ddfed59cbf139588e3cd8364ae23e`.
- There are uncommitted prefetch foundations, tests, and documents beyond that HEAD. A GitHub-only checkout may not contain them.
- The user's untracked `Claude outputs/` directory is outside these changes. Do not overwrite, remove, or include it in a commit without authorization.

Start by checking `git status --short`, the current branch, and HEAD. Read applicable `AGENTS.md` instructions. Do not assume the recorded commit is still the current state when you receive this handoff.

Do not automatically commit, push, switch branches, or reset the working tree unless explicitly requested. Preserve existing user and other-developer changes. Do not upgrade the entire SGLang/CUDA/Mooncake environment merely to try installing CAGRA.

If the local foundation files listed below are absent, report the discrepancy and reconcile the checkout. Do not claim those files or their tests exist in a remote-only copy.

## 4. Responsibilities of P, V, D, and Router

- **Router:** selects P, a V worker group, and D; propagates consistent request identities and V association to P and D.
- **P:** computes complete Prompt KV and uploads it to the selected V using the supported layout.
- **V:** stores complete Prompt KV. The target implementation adds request-scoped indexing and CAGRA retrieval.
- **D:** performs committed decoding, retains generated KV, and manages its active Prompt KV working set and next prefetch.

A V physical node, V worker group, and V rank/shard are different concepts. One V group can serve multiple D instances. Router selection does not create a permanent one-to-one V/D ownership relationship.

Do not introduce cross-V-group sharding simply because CAGRA is being added.

Route source and destination data by layer, KV head, and token/page ownership, not by assuming matching rank numbers. Existing documentation describes selected paths involving P TP1/TP2, two V storage shards, and D TP2/TP4. Verify the actual current support in code; do not claim arbitrary M→N or GQA/MQA combinations already work.

## 5. What is actually implemented

### 5.1 Existing serving path

- V stores complete Prompt KV. An Entry may be reused by multiple Deliveries.
- D fetches complete Prompt KV before its first forward and when each request's refresh becomes due.
- Each request has its own `RefreshClock`.
- The committed D token count excludes the first output token sampled by P.
- A request's clock advances only after successful refresh, installation, and acknowledgement.
- `full_prompt` is the currently integrated retrieval mode.
- The refresh path currently waits synchronously and can stall the next forward of the current batch.
- This path is not sparse retrieval and does not already hide network latency.

### 5.2 Recently added foundations, not integrated into serving

1. `prefetch.py`: model-independent `PrefetchClock` and `PrefetchTicket`.
   - Fixes each request's target installation boundary.
   - Allows an early start, but not early installation or period advancement.
   - Allows only one pending ticket per request.
   - Rejects wrong identities, stale rounds, and committed counts advancing past an unfulfilled boundary.
   - `close()` retains the pending identity; it does not release memory or terminate RDMA.
2. A CAGRA diagnostic script: inventory by default; GPU execution only in explicit smoke mode.
3. `bootstrap.py`: model-independent `BootstrapGate` and `BootstrapTicket` for the
   waiting-queue-triggered initial pull.
   - Entry into the final waiting queue is the only trigger; nothing earlier may authorize.
   - The waiting-queue trigger and KV_STORED are independent and may arrive in either order.
   - Exactly one authorization per request; duplicate control requests reuse the first ticket.
   - RECEIVED is not RUNNABLE, and AUTHORIZED cannot jump to INSTALLED.
   - `handoff()` yields committed-token count 0 exactly once, so round 0 is never re-fetched.
   - `close()` retains the pending identity and refuses late completions; it releases,
     fences and drains nothing.
4. Waiting-queue bootstrap wiring behind `--pvd-waiting-queue-bootstrap` (default off):
   - `PVDKVManager.open_bootstrap_gate/bootstrap_runnable/enter_waiting_queue/close_bootstrap_gate`.
   - `PVDKVReceiver` opens the gate at enqueue and reports KV_STORED when the Entry validates.
   - `decode.py` progresses the whole final waiting queue on every scheduler pass, and the
     batch builder now counts admitted requests instead of queue positions so a not-runnable
     request is skipped without consuming a batch slot. With the flag off nothing is ever
     skipped and the count is identical to the previous index comparison.
   - Initial delivery and ACK use asynchronously polled control futures. TP agreement and
     unpack remain on the scheduler thread. Periodic refresh still uses the blocking driver
     of the same `full_prompt` protocol. Deferred waiters retry without requiring arrivals.
   - Ranks agree on source readiness and staging headroom before selecting a wave. A wave
     failure can fail its other newcomers; it does not include already-running requests.
5. `prediction.py`: the draft/probe interfaces, with fakes and no model loading.
   - `DraftConfig` takes a model name or local path, optional revision, device, dtype and
     token budget. No model is defaulted; a missing revision records `local/unknown`
     rather than an invented one.
   - `CommittedPrefix` / `snapshot_committed` hand the prediction branch an immutable,
     detached copy of committed state, so it cannot reach committed ids, positions or KV.
   - `run_isolated` forks the torch RNG, so a sampling draft model cannot change what the
     committed sampler produces next.
   - `QueryVectors` carries an explicit vector space, version, layer, head range,
     positions and valid length. `PredictionPipeline` refuses a query whose space is not
     the configured target model, so draft-space Q can never be searched against target K.
   - The pipeline also refuses a prediction for another request, one made against a stale
     prefix, one over budget, and a probe returning unrequested or duplicate layers.
   - `FakeDraftProvider` / `FakeTargetProbe` allow development without weights.
6. `draft_hf.py`: the first concrete `DraftProvider`, backed by a Hugging Face causal LM.
   - `transformers` is imported lazily inside the loader, so it is not a new hard dependency,
     and the loading step is injectable so everything below is testable without weights.
   - `VocabularySignature` compares vocab size, BOS/EOS and a fingerprint of a fixed probe
     encoding. A draft whose tokenizer disagrees with the target's is refused: predicted ids
     would mean different text to the probe, and no translation step exists.
   - Device and dtype are validated against what the model actually loaded as, not assumed.
   - The token budget is enforced on what the model returns, not only on what it was asked
     for, so a generate() that overshoots is still truncated.
   - The revision the loader resolved is recorded; a local path with none reports
     `local/unknown`.
   - Nothing constructs it yet; it is opt-in and not wired into serving.
7. `index_lifecycle.py`: the V-side retrieval-index state machine, with no vectors.
   - ABSENT, BUILDING, READY and FAILED are distinguishable: "not ready" is not "failed".
   - An index may only be built after the complete Prompt KV is stored and visible.
   - `deliverable` is deliberately independent of index state, so full initial delivery
     never acquires an INDEX_READY dependency.
   - A built Prompt index is immutable and its readiness is not consumed by a search, so
     one index serves many Delivery rounds.
   - `authorize_search` compares the query's vector space and the caller's id-mapping
     version; index version, mapping version and vector space stay separate identities.
   - Failed builds are retried up to a bound and then refused; `close()` retains the
     descriptor and frees no vectors, graph storage or mappings.
   - Not wired into the V control server; nothing builds or searches an index yet.
8. `index_search.py`: the retrieval backend contract, an exact reference implementation,
   logical selection and the merge policy. CAGRA becomes a swap, not a rewrite.
   - `IndexBackend` is the seam: `build` and `search`. `BruteForceIndexBackend` is exact,
     CPU-only and needs no cuVS, so it is also the ground truth a CAGRA backend's recall
     will be measured against rather than a throwaway stub.
   - `select` returns **logical** token and page ids through a versioned `IdMapping`, never
     raw addresses; whoever gathers KV resolves addresses under its own bounds checks.
   - A token chosen by several queries is selected once, keeping its best score.
   - `merge_selections` requires a named policy (`per_layer`, `union`, `intersection`) and
     refuses anything else. Scores from different layers are never ranked against each
     other, because a global Top-K would be a silent modelling claim.
   - Ties break deterministically on the lower row, so tests do not depend on kernel order.
9. `prompt_vectors.py`: Prompt K extraction from the real stored EntryShard.
   - Input is `kv_packer`'s packed byte buffer plus the validated storage
     `KVLayoutSignature` and `KVShardManifest` -- not a pre-extracted tensor.
   - K only. The V components are skipped, and padding in the final page is excluded
     using `last_page_valid_tokens`, so no vector comes from a token the prompt lacks.
   - Layers and KV heads stay separate; nothing averages or concatenates heads. Output is
     one vector set per (global layer, global KV head), where the global head is
     `manifest.rank * kv_heads_per_rank + local`. Filters take global ids and reject a
     peer shard's head rather than returning nothing.
   - dtype, component offsets, shapes and head ownership are all read from the layout
     metadata and cross-checked against the buffer's actual size.
   - **Positional encoding**: stored K is post-RoPE (models rotate before the attention
     layer writes k; see `models/llama.py`). Extraction applies no transformation, so
     nothing is rotated twice, and `require_compatible_query` refuses a Q declaring a
     different encoding. The value is an explicit input, never inferred from metadata.
   - `QueryHeadMapping` defines the query-head -> KV-head grouping for MHA, GQA and MQA
     and rejects non-divisible layouts. It takes the query-head count as a parameter,
     because `KVLayoutSignature` does not carry it.
   - Vectors **own a copy**: stored KV is never mutated, and the index never borrows
     memory inside the Entry's registered region. An optional budget/owner charges the
     copy; V passes none yet.
   - `Selection` gained an optional `kv_head`, so head identity survives selection and
     merging. Existing per-layer callers are unchanged.
   - Not wired into the V control server.
10. `prompt_index.py` + `VectorKVStore` wiring: the first thing that drives `IndexGate`.
    - The store takes an optional `prompt_index`. **None by default**, so a store built
      without one behaves exactly as before and the feature is entirely off.
    - `_publish_stored_locked` marks the gate KV-readable -- the only point an index may
      be built from. Nothing else happens under the store lock.
    - `progress_prompt_indexes()` is one bounded, caller-driven step, matching how uploads
      and decode closes are progressed. It pins each candidate's `allocation_guard` for the
      copy, so an Entry whose release has begun is skipped rather than read (`pin` refuses
      once release is requested), extracts outside the store lock, and unpins either way.
    - A build failure is recorded on the gate and reported, never raised: an Entry that
      cannot be indexed stays STORED and deliverable. Retries stop at the gate's bound.
    - `_free_allocation` closes the gate before the pages return to the allocator. That
      drops only the index's own copies; pages, registration and MR are released by their
      existing owners, unchanged.
    - `PromptIndexManager.search` goes through `IndexGate.authorize_search` and the
      stored vectors' `require_compatible_query`, then returns a logical `Selection`
      carrying layer and KV head.
    - Vector copies are refunded on close and on a failed build, and a build that lands
      after its Entry was closed is discarded rather than installed. The registry is
      locked, because `close()` can run from a guard-release callback on another thread.
    - Delivery never consults any of this, so no INDEX_READY dependency is introduced.
11. CPU tests covering clocks, the bootstrap gate, the waiting-queue trigger, the decode.py
   scheduler hooks, the draft/probe interfaces, new-request isolation, the existing
   refresher's selection scope, and diagnostic behavior.
12. Design and implementation-status documents.

### 5.3 Not yet implemented

- Running a **real** draft model. `draft_hf.py` implements loading, vocabulary and placement validation (see 5.2), but it has never been run against actual weights, so nothing is known about its speed, memory or prediction quality. It is also not constructed by any server path yet.
- Actual target-model probe **execution**: architecture-specific Q capture, prefix realignment and hidden-state extraction. Only the interface and its safety checks exist.
- A **CAGRA backend**. Everything above it now exists and runs (see 5.2): a stored Entry is indexed through the exact CPU backend and searched under the gate's identity checks. Nothing calls cuVS, and no recall comparison against the exact reference has been made.
- **Serving integration of retrieval.** The store can build and search, but no control-server route exposes it, no scheduler calls `progress_prompt_indexes()`, and no D request produces a real query. The V launcher constructs no `PromptIndexManager`, so on a running server the feature is off.
- Page-level representative vectors, deliberately.
- Sparse KV selection, packing, D installation, and attention integration.
- Actual active/next GPU buffers and prefetch Scheduler integration.
- Hardware validation and performance measurement of the implemented asynchronous initial pull (see 5.2). Network/ACK waits yield to the scheduler, but TP coordination and GPU installation still cost time.
- (Dropped, not pending.) Direct-to-final-page delivery is no longer a goal; see section 16. The pull reuses the existing `full_prompt` staging-and-unpack path by decision, not by omission.
- Real-model, V100S, multi-GPU TP, RDMA, output-quality, and performance acceptance testing.

There is no new launch argument that already enables the complete predictive pipeline. Replacing the current clock with the standalone `PrefetchClock` is not sufficient to implement the pipeline.

## 6. Exact per-request timeline

Definitions:

- `n`: committed tokens generated by D's target model, excluding P's first token.
- `M`: this request's refresh interval.
- `r`: configurable prefetch lead in tokens. The current foundation clock requires `0 <= r < M`.
- `boundary`: committed-token position at which the next KV working set must be installed.
- `round`: this request's refresh round, not a batch-wide round.

Initial installation occurs at `n=0`. Example with `M=16, r=4`:

```text
n=0:  install initial KV
      ↓ committed decoding
n=12: draft prediction → target-model probe Q → V search and transfer
      ↓ D continues committed tokens 13–16 using active KV
n=16: check next KV; install if ready, otherwise wait/handle failure
      ↓ installation and required acknowledgements complete; advance this request
Next cycle: prefetch at n=28, install at n=32
```

Represent “started,” “transferred,” “safe to consume,” and “installed” separately. Early arrival must not switch the working set early. Network duration must not change the token-count-based period.

Predicted tokens never advance `n`. Committed decoding must not cross an unfulfilled refresh boundary.

### Example: a new request awaits admission (target behavior)

| Request | D tokens since its last refresh | Action now |
| --- | ---: | --- |
| A | 10 | No refresh. |
| B | 16 | Due; install or wait for its own KV. |
| C | 3 | No refresh. |
| New request D, not in the running batch | Not initialized | Reserve capacity and receive complete initial KV outside the running batch. |

Only B's periodic refresh and new request D's initial preparation require KV work; they do not share one completion barrier. A and C keep their counters, rounds, and prefetches. The running batch may still wait for B's due refresh, but must not wait solely because newcomer D is uninitialized. Admit D only after installation.

Asynchronous release of already-running requests at periodic refresh boundaries remains a separate future optimization. The newly confirmed change is asynchronous initial preparation of queued newcomers.

### Initial waiting-queue pull (implemented, opt-in; hardware acceptance pending)

```text
Router selects P, V, D
  ├─ P computes/uploads full Prompt KV → V: KV_STORED
  └─ D: prealloc queue → transfer queue → FINAL WAITING QUEUE
                         ↓ entry into scheduler.waiting_queue is the trigger
D registers a staging buffer, authorizes it and requests delivery
                         ↓ V must already be KV_STORED; otherwise D waits here
V performs the authorized RDMA WRITE into that staging buffer
                         ↓ D unpacks it into its already-preallocated KV pages
                         ↓
D: native completion/identity checks → GPU visibility and installation
                         ↓
RUNNABLE → admission into the running batch
```

- The trigger is entry into the final waiting queue, not entry into the prealloc or transfer queue. Earlier stages do no KV transfer work.
- D is the initiator; V remains the writer. This reuses the existing authorized-destination WRITE path, write identities, epochs/generations and fences. It is not an RDMA READ and introduces no new transport direction.
- This is still a receiver-authorized write, never an unsolicited write to D memory. D does not wait for admission into the running batch to request delivery.
- The destination is a registered staging buffer, pinned before its descriptor is published; D then unpacks it into the KV pages `DecodePreallocQueue` already allocated. Writing straight into those final pages is **not** a current goal (see section 16).
- Staging bytes are charged to the worker's `--pvd-transfer-staging-budget-bytes`, so an initial pull competes for capacity with running requests' refreshes. A request that does not fit is left in the waiting queue and retried on a later pass; running out of staging headroom is backpressure, not a failure, and must never abort a queued newcomer.
- The first implementation uses complete Prompt KV for bootstrap: no bootstrap query or initial CAGRA search is required.
- V may build the index in parallel after KV becomes safely readable. Full initial delivery must not acquire an unnecessary INDEX_READY dependency.
- Subsequent periodic retrieval still requires a usable index and follows draft → target probe → CAGRA; specify failure policy separately.
- Bound the number of concurrently pulling requests. Bytes are bounded twice: prealloc admission bounds the final pages, and the staging budget bounds the in-flight receive buffers.
- A request that cannot be preallocated never reaches the waiting queue, so its KV stays on V and no D memory is committed ahead of demand.
- Account for the staging buffer, the final KV pages and the in-flight transfer budget. Pulling must not exhaust resources needed by running requests or create a capacity deadlock.
- Arrival is not readiness: mark RECEIVED, not RUNNABLE. Native terminal state, identity checks, GPU visibility and required TP agreement are all required before the request becomes runnable.
- Staging-and-unpack is the decided bootstrap path; a published address does not establish safe installation or runnable status.
- Bootstrap is performed exactly once. Complete initial clock initialization before admission; do not fetch round 0 again when the request joins. Thereafter count only committed D tokens.
- Cancellation/timeouts must close further submissions and safely drain native WRITEs. Removing a queue entry does not authorize immediate descriptor reuse.
- Full bootstrap requires capacity for complete Prompt KV on D. Do not claim support for prompts that D can never hold in full.
- While pulling, the request sits inside `scheduler.waiting_queue` in a not-runnable state: the batch builder must skip it and must not count it against the batch token budget. It is not a completion barrier for any running request.
- Because the trigger is later than the earlier pre-push proposal, less of the transfer overlaps with queueing. This is the accepted trade for committing D memory only to requests that have reached the final waiting queue. Complete hiding is not guaranteed.

## 7. Draft and probe interface requirements

Do not require the user to select a fixed model before generic development can proceed. Implement configurable interfaces, fake providers, and isolation tests first; load concrete models during experiments.

Configuration should express model name/local path, optional revision, placement, dtype, budget, and prediction length. These are interface requirements, not already-available CLI flags. Choose names and actual integration consistent with the repository.

The confirmed data path is:

```text
Read-only snapshot of the committed prefix
    → separate draft model predicts tokens
    → isolated target-model probe computes Q at defined positions
    → queries carrying layer/head/position semantics
    → V searches the target model's Prompt K
```

Do not directly search target-model K with draft-model Q. Sending token IDs alone is not an attention-vector retrieval implementation. Do not pass token IDs from one incompatible vocabulary directly to another model.

Prediction and probing must not mutate committed output IDs, committed positions, committed KV, or the committed sampler/RNG state. Prediction branches own temporary state and budgets and clean them up through the appropriate lifecycle.

Deeper-layer Q obtained by probing with the currently available sparse KV may itself be approximate. Document and evaluate this rather than treating it as full-attention ground truth.

Record the loaded model identifier and resolved revision when available. For local weights without a revision, record a local/unknown source; do not invent a revision or require downloading/hashing all weights merely to generate a report.

Model revision, index version, query version, and memory generation are distinct identities and must not be conflated.

## 8. V-side search and delivery

- Build the index only after complete Prompt KV upload and safe GPU visibility.
- Distinguish KV_STORED, INDEX_BUILDING, INDEX_READY, and INDEX_FAILED.
- Manage raw KV, retrieval vectors, graph storage, and ID mappings separately.
- Map results to Entry, layer, KV head, original token/page, and physical storage.
- Reuse an immutable Prompt index across multiple Delivery rounds.
- Do not merge scores from unrelated layers/heads into one global Top-K without a defined policy.
- Raw dot product, cosine similarity, and page representative vectors are different strategies, not interchangeable defaults.
- Return logical token/page selections and payload descriptions, not only raw addresses.
- Gather/pack selected KV and deliver it only into authorized destination regions.
- Search completion, HTTP success, WRITE submission, native transfer completion, and D installation are different states.

Enforce response-byte limits and backpressure. A selection larger than the authorized destination region must not cause an out-of-bounds write.

Batching can reduce control overhead, but must not merge request identities or clocks.

## 9. D-side sparse KV and transfer safety

Sending less KV while D still reads the full Prompt layout is incorrect. Implement the corresponding D behavior:

- Map original tokens/pages to local slots.
- Preserve position semantics, valid lengths, attention masks, and payload layout.
- Combine selected Prompt KV with locally committed generated KV.
- Support or explicitly reject missing-page, tail-page, per-layer/head selection, and GQA/MQA cases.
- Preserve generated KV, including tokens sharing the final Prompt page.

Keep active and next logically isolated. Never overwrite data concurrently read by GPU attention. Install next only after identity checks, native transfer completion, GPU visibility, and required TP agreement.

Preserve existing MR/metadata-consistency protections and native-handle lifetimes:

- Allocate and pin receive space before publishing its descriptor.
- Bind Entry, request incarnation, round, query/index versions, rank, epoch, generation, and byte ranges.
- An application-level generation does not automatically make the RNIC reject stale WRITEs.
- Canceling a request or prediction does not stop submitted RDMA.
- Reuse memory only after preventing further submissions, establishing native terminal state for submitted work, and completing GPU consumption.
- When safety cannot be proven, retain draining/quarantine resources. A timeout is not permission to release memory.
- Keep separate references and lifetimes for Entry, index, Delivery, prediction branch, and buffer.

Budget for models, draft/probe temporary KV, V index build/search scratch, active/next, pack/staging, generated KV, and retained draining resources.

A bounded Prompt working set does not bound generated KV indefinitely. Preserve normal admission and capacity limits.

## 10. Layered validation: missing hardware must not block generic development

### A. Local CPU/fake validation

Continue implementing interfaces, protocols, state machines, mappings, and isolation. These tasks do not require particular weights, V100S, CuPy, or cuVS.

Fake execution does not establish real GPU or RDMA correctness.

### B. Environment inventory: default command

Run from the repository root:

```bash
python scripts/pvd/check_cagra.py
```

The default is `--mode inventory`: no CuPy/cuVS import, no GPU kernels, and no dependency installation.

It reports `status=collected, cagra_test=not_run`. Exit code zero means inventory collection succeeded, not that CAGRA passed.

### C. Explicit hardware smoke test when an experimental machine is available

```bash
python scripts/pvd/check_cagra.py --mode smoke
```

By default, this checks the selected GPU for V100S and executes actual CAGRA build/search. Failure returns a nonzero exit code.

Validate the installed cuVS combination by actual API, GPU architecture, dtype, and metric capabilities, and record the tested environment. A version string or successful import does not prove V100S support. Do not require a single predetermined version merely to continue generic development.

This flexibility does not authorize removing existing Mooncake version or safety restrictions.

The default synthetic recall threshold of 0.90 is a small smoke-test criterion, not an output-quality acceptance threshold. Tests explicitly run on a different development GPU do not replace V100S acceptance testing.

## 11. Recorded validation evidence

Most recent run before this handoff was created:

- 2026-09-20, Linux/WSL venv against the Windows checkout, twenty-three PVD CPU test files:
  **786 passed in 7.4s** (776 plus 10 regression tests for three defects found by review
  and reproduced before fixing: a budget leak on close, a budget leak on a failed build,
  and a non-2-D query producing a nonsense `head_dim -1` message). Four mutations
  re-confirmed the fixes: not refunding on close failed 3, not refunding on a failed
  build failed 2, installing a build that finished after close failed 1, and accepting a
  non-2-D query failed 1.
- 2026-09-20, intermediate: twenty-three PVD CPU test files, **776 passed in 9.0s** (754 plus 22 store/index integration tests, which drive a real
  `VectorKVStore` through create, fill, commit, build, search and release). Five mutations
  confirmed those bite: reading an entry whose release has begun failed 1, building an
  entry that left STORED failed 1, letting a build failure escape failed 2, serving an
  index after release failed 1, and skipping the search authorization failed 1. The
  entry-state mutation initially failed nothing, because a gate only exists after STORED;
  a test that changes state with the gate already open was added.
- 2026-09-20, intermediate: twenty-two PVD CPU test files, **754 passed in 6.1s** (691 plus 63 Prompt K extraction tests, including a round trip
  from the real packed buffer through the exact index back to the original token and page).
  Six mutations confirmed those bite: including padded tokens failed 7, extracting the V
  components failed 6, using the local head index as the global one failed 4, accepting a
  mismatched positional encoding failed 1, rounding a non-divisible GQA layout failed 3,
  and borrowing storage instead of copying failed 1. That last one initially failed
  nothing, because the default float16 -> float32 conversion already copies; a test that
  extracts at the stored dtype was added, which is the case where a view would alias.
  **Queries in these tests are synthetic rows taken from the extracted vectors. They
  establish mapping and identity correctness, not real-model retrieval quality.**
- 2026-09-20, intermediate: twenty-one PVD CPU test files, **691 passed in 18.3s** (633 plus 58 index-backend and selection tests). Five mutations
  confirmed those bite: allowing an unnamed merge policy failed 5, unstable tie ordering
  failed 9, skipping the mapping-covers-index check failed 1, merging across different id
  mappings failed 1, and aliasing the caller's vectors failed 1. That last one initially
  failed nothing, because the first aliasing test still produced the same winner; it was
  rewritten to assert the stored score instead.
- 2026-09-20, intermediate: twenty PVD CPU test files, **633 passed in 16.4s** (584 after the asynchronous initial-pull work, plus 49 V-side
  index-lifecycle tests). Five mutations confirmed the new tests bite: making delivery wait
  for INDEX_READY failed 5, building before the KV is readable failed 1, dropping the
  vector-space check failed 1, unbounded rebuild attempts failed 1, and treating a failed
  build as absent failed 1.
- 2026-09-19, intermediate: eighteen PVD CPU test files,
  **552 passed in 6.2s** (516 plus 36 Hugging Face draft-provider tests). Four mutations
  confirmed those bite: skipping the vocabulary check failed 4, skipping placement
  validation failed 2, not truncating to the budget failed 1, and dropping the
  out-of-vocabulary prefix guard failed 1. The budget mutation initially failed nothing,
  because the first fake model respected `max_new_tokens`; a fake that ignores it was added
  so the guard is actually exercised.
- 2026-09-19, intermediate: seventeen files, **516 passed in 5.8s** (510 plus 6
  staging-backpressure tests). Three mutations confirmed
  those bite: ignoring staging headroom failed 3, not charging headroom down within a pass
  failed 2, and reporting a deferral as a failure failed 2.
- 2026-09-19, intermediate: **510 passed in 5.4s** (445 plus 65 draft/probe interface tests). Two mutations confirmed
  those bite: dropping the vector-space check failed 1, and removing the RNG fork failed 2.
  A third mutation (aliasing the snapshot token sequence) did **not** fail anything, because
  `CommittedPrefix` already rejects a non-tuple; the copy test is redundant with that check.
- 2026-09-19, intermediate, sixteen PVD CPU test files: **445 passed in 6.6s** (377 baseline, +38 bootstrap-gating, +15 waiting-queue wiring,
  +15 decode.py scheduler hooks). The scheduler-hook tests extract
  `get_new_prebuilt_batch` and `_pvd_enter_waiting_queue` from the shipped source with
  `ast` and execute them against fakes, so both decode.py edits are now covered. Three
  injected mutations confirmed they bite: position counting instead of admission counting
  failed 1, removing the not-runnable guard failed 4, and leaving a failed pull in the
  waiting queue failed 1.
  `ruff check --select E9,F401,F821,I` and `ruff format --check` pass on every file changed.
  `decode.py` already failed `I001` and `ruff format --check` at HEAD; that is pre-existing
  and was not introduced or fixed here.
- 2026-09-19, intermediate: fourteen files, **415 passed in 5.75s**
  (377 before `bootstrap.py`, +38 new bootstrap-gating tests).
  Two mutations were injected to confirm the new tests bite: treating RECEIVED as runnable
  failed 2 tests, and letting a closed gate accept a late completion failed 1.
- Historical, before this change set: thirteen PVD CPU test files, **377 passed in 3.38s**.
- Relevant changes passed Ruff formatting and E9/F401/F821/I checks.
- Local inventory succeeded with CAGRA marked `not_run`.
- Local smoke failed at `import_cupy` because CuPy is absent, with exit code 1.
- The development machine is Windows / RTX 4060 Laptop, not V100S.
- There is no successful V100S build/search evidence and no real-model or RDMA performance conclusion.

Reproduce the CPU regression in PowerShell at the repository root:

```powershell
$pvdTestFiles = @(rg --files test/registered/disaggregation -g 'test_pvd*.py')
& .venv/Scripts/python.exe test/registered/disaggregation/run_pvd_cpu_tests.py @pvdTestFiles -q --tb=short
```

These are historical results. Rerun relevant tests after your changes; do not cite an earlier count as evidence that your new implementation passed.

## 12. Implementation phases and deliverables

### Phase 0: inspect and implement generic interfaces

- Inspect actual code and uncommitted changes.
- Maintain request-local clocks and explicit state-machine design.
- Implement configurable draft-provider and target-probe interfaces with fake tests.
- Define snapshot, query, selection, and Delivery identities and capacity contracts.
- Record inventory separately from hardware testing.

Do not wait for a fixed model choice or local V100S availability to start generic work.

### Phase 1: shadow prediction and retrieval

- Preserve the implemented independent bootstrap: waiting-queue trigger, authorization over registered staging, V readiness, native completion, installation into preallocated final pages, ACK, and RUNNABLE admission. Extend hardware validation; this does not depend on CAGRA.
- Keep committed generation on the full-KV baseline.
- Implement replaceable draft, probe, index, and search interfaces, progressively connecting real backends.
- Do not yet let selection results change attention.
- Measure differences against real queries/exact retrieval and the added execution cost.

Without hardware, implement and fake-test the code while recording real-model execution as pending. Shadow mode is an intermediate diagnostic step, not the final objective, and does not establish reduced full-Prompt memory consumption on D.

### Phase 2: sparse delivery and attention

- Connect logical selection, packing, transfer, D installation, and attention.
- Initially use synchronous refresh to isolate correctness issues.
- Verify source/destination K/V correspondence and preservation of generated KV.
- Evaluate quality and memory on real hardware.

### Phase 3: request-independent prefetch pipeline

- Trigger prediction/prefetch at each request's M-r position; install or wait at its M boundary.
- Connect actual active/next buffers, transport protection, and Scheduler behavior.
- Test admission, departure, EOS, cancellation, late results, and failures.
- Retain the running batch's periodic-refresh barrier without refresh-all behavior; newcomer initialization waits outside that running batch.

### Phase 4: end-to-end optimization

- Tune M, r, retrieval budgets, concurrency, and packing based on measurements.
- Analyze draft/probe GPU contention with committed decoding.
- Optimize submission and progress without removing safety protections.
- Do not silently introduce asynchronous per-request release, a different search algorithm, or new hardware requirements.

For each phase, distinguish “implemented,” “CPU/fake verified,” and “real model/GPU accepted.” Hardware-dependent acceptance can remain pending while independent development continues. Do not enable an unvalidated production path by default.

## 13. Minimum interface semantics

These are required semantics, not existing APIs or mandatory final names:

- **Prediction input:** read-only committed-prefix snapshot, request identity, committed position, prediction length, and budget.
- **Probe input:** predicted tokens, associated committed prefix, target-model identity, and layer/head/position selection.
- **Query:** explicitly identified vector space, position semantics, version, and valid length.
- **Prefetch request:** Entry, round, target installation boundary, query/index identity, output budget, and per-D-rank grants.
- **Initial pull authorization:** Entry/request incarnation, initial Delivery identity, authorized per-rank staging regions, capacity, and validity/cancellation state. The authorization is published only once the request has entered the final waiting queue and its pages are registered and pinned; WRITE submission requires validated layout and actual capacity.
- **Selection:** original token/page IDs, layer/head ranges, layout, and actual byte count.
- **Delivery:** request identity, state, write identity, and linkage to native completion evidence.

TP ranks must agree on shared request sets, rounds, and errors. Legitimate rank-local queries, heads, and addresses may differ.

Retries must be idempotent; duplicate control requests must not cause unbounded duplicate WRITEs. A batch-membership change does not automatically invalidate a request-local snapshot.

## 14. Acceptance checklist

### Correctness and lifetimes

- Prediction does not modify committed output, count, RNG, or KV.
- New requests do not affect existing clocks or in-flight prefetches.
- The waiting-queue trigger and KV_STORED may arrive in either order; transfer starts without requiring admission into the running batch.
- An unready newcomer does not add a completion barrier to existing decoding. Shared-resource contention is still possible.
- No published authorization means no WRITE; final-page and in-flight budgets remain bounded under queued load.
- RECEIVED is not RUNNABLE; install before admission, without repeating initial delivery after admission.
- Waiting-queue cancellation, late WRITE, lease renewal/expiry, and authorization retries do not cause premature release or duplicate writes.
- Early completion does not install early; late completion does not permit crossing a boundary with stale or partially written data.
- Stale rounds, wrong Entries/heads/positions, and duplicates are handled correctly.
- TP collective ordering and control flow agree.
- Cancellation/timeouts do not release memory prematurely; tail pages and generated KV remain intact.
- Memory and in-flight work are bounded.
- The original `full_prompt` baseline remains usable.

### Quality

Record task quality, perplexity where appropriate, recall under actual queries, useful-prefetch rate, and corrective-fetch rate.

Token prediction match rate is not KV retrieval quality. Approximation being allowed does not make arbitrary degradation acceptable. Ask the user to confirm evaluation metrics and acceptance thresholds before judging quality.

### Performance

Compare at least:

1. Full-KV synchronous refresh.
2. Sparse retrieval without early prefetch.
3. Early prefetch driven by recent actual Q.
4. Early prefetch driven by separate draft prediction plus target-model probe.

Record TTFT, TPOT p50/p95/p99, throughput, refresh waiting, V queuing, search, packing, transfer, installation, draft/probe overhead, network bytes, wasted prefetch, and peak memory.

Test both stable batches and workloads with continuous new admissions. Also record bootstrap queue/transfer overlap, time spent ready but not admitted, residual admission waiting, existing-request TPOT changes, and peak preparation memory. Shared resources can still affect existing requests; do not claim zero impact. Do not manufacture speedups by skipping required refreshes or concealing quality losses.

## 15. Explicit non-goals and prohibited changes

- Do not restore “a new request joins, therefore every request in the batch updates.”
- Do not introduce one shared batch-wide M counter.
- Do not emit predicted tokens or treat prediction-branch KV as committed generation KV.
- Do not require a standard speculative-sampling acceptance/rejection loop.
- Do not hard-code a draft model or require a pinned revision before development can proceed.
- Do not substitute KV across requests or automatically introduce cross-V-group search.
- Do not claim arbitrary TP layouts or model architectures already work.
- Do not write D-generated KV back to V by default.
- Do not implement direct-to-final-page bootstrap delivery, or change the allocator to make it possible, without a new decision and a measurement.
- Do not expand the task into a full Host/NVMe tiering product or replace the whole transport stack.
- Do not silently replace CAGRA with another algorithm.
- Do not remove existing MR, metadata-cache, fencing, or native-handle safeguards.
- Do not stop hardware-independent work because hardware is unavailable, and do not fabricate hardware acceptance.

## 16. Remaining decisions that must not block all work

For these decisions, propose a design and its tradeoffs, ask the user where necessary, and continue independent work:

- Architecture-specific probe execution and how predicted positions produce queries for the next window.
- Token-level versus page-level indexing and independent versus shared selection across layers/heads.
- Bootstrap is already decided: waiting-queue-triggered complete-KV pull, D-initiated and V-written into registered staging, then unpacked into preallocated final pages. Preserve not-runnable gating and asynchronous network progress; do not silently switch to direct final-page writes.
- **Direct-to-final-page bootstrap delivery is decided: keep staging (2026-09-19).**
  `unpack_full_prompt_kv` scatters a contiguous packed buffer into per-layer K and V
  components at token indices derived from page indices the allocator does not guarantee to
  be contiguous, and one RDMA WRITE lands in one contiguous range. Writing straight into the
  final pages would require either contiguous whole-prompt page allocation, which constrains
  the allocator and fails under fragmentation, or one WRITE per (component x contiguous page
  run), which multiplies transfer slots and authorized regions and so changes the budget
  model. Neither is worth it now. Bootstrap uses the existing staging-and-unpack path, and
  removing the staging hop is not a current implementation goal. Do not reopen this without
  a measurement showing the extra copy matters.
- Index-not-ready, prediction-deviation, and failure policies for subsequent periodic retrieval.
- Mandatory initial/recent-token retention and retrieval capacity limits.
- Formal output-quality acceptance thresholds.

Concrete model names, optional revisions, deployment devices, M/r, and budgets are user configuration. The user does not have to supply fixed values now for interfaces to be designed. Explain proposed defaults and keep them configurable.

## 17. Files to inspect first

All paths below are relative to the actual repository root:

| File or directory | Purpose |
| --- | --- |
| `python/sglang/srt/disaggregation/pvd/README.md` | Existing serving flow, support matrix, development status. |
| `python/sglang/srt/disaggregation/pvd/retrieval.py` | Current full_prompt contract and RefreshClock. |
| `python/sglang/srt/disaggregation/pvd/prefetch.py` | Standalone request-local prefetch timing, not wired into serving. |
| `python/sglang/srt/disaggregation/pvd/bootstrap.py` | Request-local initial-pull gating; wired into `conn.py` and `decode.py` behind `--pvd-waiting-queue-bootstrap`. |
| `python/sglang/srt/disaggregation/decode.py` | Decode queue chain: prealloc -> transfer -> `scheduler.waiting_queue` -> running batch. The final waiting queue is the bootstrap trigger point. |
| `python/sglang/srt/disaggregation/pvd/decode_refresh.py` | Receive buffers, due selection, waiting, unpack, ACK. |
| `python/sglang/srt/disaggregation/pvd/conn.py` | PVD integration with P/D runtime. |
| `python/sglang/srt/disaggregation/pvd/runtime.py` | Upload and transfer lifecycle. |
| `python/sglang/srt/disaggregation/pvd/coordinator.py` | Entry/Delivery coordination. |
| `python/sglang/srt/disaggregation/pvd/vector_store.py` | V storage and delivery. |
| `python/sglang/srt/disaggregation/pvd/prompt_index.py` | Per-Entry gates, vectors and search on a V rank; driven by `VectorKVStore.progress_prompt_indexes()`. Off unless a manager is supplied. |
| `python/sglang/srt/disaggregation/pvd/prompt_vectors.py` | Prompt K extraction from a stored shard: K only, padding excluded, per layer and global KV head, post-RoPE, owns its copy. |
| `python/sglang/srt/disaggregation/pvd/index_search.py` | Backend seam, exact CPU reference, logical selection, explicit merge policy. No cuVS. |
| `python/sglang/srt/disaggregation/pvd/index_lifecycle.py` | V-side index state machine: build ordering, delivery independence, search identity. Builds nothing. |
| `python/sglang/srt/disaggregation/pvd/draft_hf.py` | Hugging Face `DraftProvider`: lazy import, injectable loader, vocabulary/placement/budget guards. Never run against real weights. |
| `python/sglang/srt/disaggregation/pvd/prediction.py` | Draft/probe interfaces, snapshot and RNG isolation, vector-space enforcement; fakes only, no model loading. |
| `python/sglang/srt/disaggregation/pvd/selector.py` | Current identity lookup, not a complete vector-search interface. |
| `python/sglang/srt/disaggregation/pvd/kv_packer.py` | KV layout and packing. |
| `scripts/pvd/check_cagra.py` | Inventory versus explicit GPU smoke testing. |
| `test/registered/disaggregation/test_pvd*.py` | CPU regressions and lifecycle tests. |
| `docs/superpowers/specs/2026-09-20-pvd-prefetch-design.md` | Detailed design constraints; currently Chinese. |
| `docs/superpowers/reports/2026-09-20-pvd-prefetch-foundation.md` | Foundation changes and validation history; currently Chinese. |

Also trace the actual Scheduler, Decode queues, model attention backends, and Router call paths. This table is a starting point, not sufficient evidence for modifying inference behavior.

## 18. Ongoing handoff requirements

After each change set, record the current commit, uncommitted files, phase, confirmed decisions, test commands/results, actual model/library versions, unverified items, blockers, and next concrete task.

Documents and historical discussion do not replace code evidence. Existing code also must not silently redefine the user's final objective.

Ask about material unresolved design choices. Do not repeatedly ask the user to decide requirements already confirmed above.

Final self-check:

> A configurable separate draft model predicts only. An isolated target-model probe produces Q. The target model generates committed output. V targets V100S and searches within each request. Every request independently refreshes every M committed tokens. New requests neither reset old clocks nor cancel old prefetches. Once Decode places a request into its final waiting queue, pull the complete initial KV into that request's already-preallocated pages under a published authorization, install it, then admit the request into the running batch. Generic development continues without local hardware. Hide retrieval and transfer waiting safely and measurably, without confusing plans, implementation, and acceptance evidence.
