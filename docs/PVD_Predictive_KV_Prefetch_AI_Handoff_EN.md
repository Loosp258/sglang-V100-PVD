# PVD Predictive KV Retrieval and Prefetch Pipeline: AI Development Handoff

Updated: 2026-09-20.

This document is intended for an AI taking over without access to the previous conversation. Read it in full before inspecting or modifying the implementation. It consolidates the user's current requirements; do not reconstruct the design from guesses about earlier discussions.

Explicit subsequent user instructions take precedence. Update this document when the user changes a decision.

## 1. Objective

Extend the existing SGLang PVD framework with a predictive KV prefetch pipeline:

1. A user-configurable, separate small draft model predicts future tokens.
2. An isolated probe execution of the target model produces retrieval queries Q.
3. V runs CAGRA search ahead of time and transfers relevant Prompt KV to D.
4. Retrieval and transfer overlap as much as possible with D's ongoing committed decoding.

Predicted tokens are never emitted as committed output. Each request refreshes independently after M committed D-generated tokens. Admitting a new request must not force existing requests to refresh, reset their clocks, or cancel their in-flight prefetches.

Initial KV uses a waiting-queue-triggered pull: D waits until its scheduler places the request into the final waiting queue (`scheduler.waiting_queue`), then initiates delivery of the complete Prompt KV into the KV pages already preallocated for that request. V still performs the authorized RDMA WRITE; "pull" means D is the initiator, not that the transport direction changes. The request stays in the waiting queue marked not-runnable and is admitted to the running batch only after validated installation. This design is confirmed but not yet implemented.

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
| Initial bootstrap | D waits until the request reaches the final waiting queue, then pulls the complete Prompt KV into its already-preallocated final KV pages. V performs the authorized WRITE. The request stays in the waiting queue as not-runnable; admission to the running batch follows installation. |
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
   - `decode.py` triggers the pull right after `waiting_queue.extend(transferred_reqs)`, and the
     batch builder now counts admitted requests instead of queue positions so a not-runnable
     request is skipped without consuming a batch slot. With the flag off nothing is ever
     skipped and the count is identical to the previous index comparison.
   - The pull itself is the existing synchronous `full_prompt` refresher.
5. CPU tests covering clocks, the bootstrap gate, the waiting-queue trigger, new-request isolation, the existing refresher's selection scope, and diagnostic behavior.
6. Design and implementation-status documents.

### 5.3 Not yet implemented

- Actual configurable draft-model loading and provider integration.
- Target-model probe execution, prefix realignment, and Q capture.
- Server-side CAGRA index lifecycle and real-request search.
- Sparse KV selection, packing, D installation, and attention integration.
- Actual active/next GPU buffers and prefetch Scheduler integration.
- Asynchronous initial pull. The waiting-queue trigger, gating and admission are wired (see 5.2), but the pull runs synchronously on the scheduler thread, so transfer time is moved rather than hidden. Overlapping it belongs to the prefetch pipeline.
- Direct registration/pinning of the preallocated final pages as the RDMA destination. The current pull reuses the existing `full_prompt` refresh path, which stages and unpacks; the direct-to-final-page destination is the decided target but is not yet implemented.
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

### Initial waiting-queue pull (confirmed, not implemented)

```text
Router selects P, V, D
  ├─ P computes/uploads full Prompt KV → V: KV_STORED
  └─ D: prealloc queue → transfer queue → FINAL WAITING QUEUE
                         ↓ entry into scheduler.waiting_queue is the trigger
D authorizes its already-preallocated final KV pages and requests delivery
                         ↓ V must already be KV_STORED; otherwise D waits here
V performs the authorized RDMA WRITE into those pages
                         ↓
D: native completion/identity checks → GPU visibility and installation
                         ↓
RUNNABLE → admission into the running batch
```

- The trigger is entry into the final waiting queue, not entry into the prealloc or transfer queue. Earlier stages do no KV transfer work.
- D is the initiator; V remains the writer. This reuses the existing authorized-destination WRITE path, write identities, epochs/generations and fences. It is not an RDMA READ and introduces no new transport direction.
- This is still a receiver-authorized write, never an unsolicited write to D memory. D does not wait for admission into the running batch to request delivery.
- The destination is the request's already-preallocated final KV pages, registered and pinned before their descriptor is published. There is no staging copy on the bootstrap path and therefore no separate preparation-bytes credit: `DecodePreallocQueue` admission already bounds this memory.
- The first implementation uses complete Prompt KV for bootstrap: no bootstrap query or initial CAGRA search is required.
- V may build the index in parallel after KV becomes safely readable. Full initial delivery must not acquire an unnecessary INDEX_READY dependency.
- Subsequent periodic retrieval still requires a usable index and follows draft → target probe → CAGRA; specify failure policy separately.
- Bound the number of concurrently pulling requests. Byte-level bounding is inherited from prealloc admission, because the destination is the final pages rather than an extra staging pool.
- A request that cannot be preallocated never reaches the waiting queue, so its KV stays on V and no D memory is committed ahead of demand.
- Account for the final KV pages and any in-flight transfer budget. Pulling must not exhaust resources needed by running requests or create a capacity deadlock.
- Arrival is not readiness: mark RECEIVED, not RUNNABLE. Native terminal state, identity checks, GPU visibility and required TP agreement are all required before the request becomes runnable.
- Direct-to-final-page delivery is the decided bootstrap path; a published address does not establish safe installation or runnable status.
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

- 2026-09-19, Linux/WSL venv against the Windows checkout, fifteen PVD CPU test files:
  **430 passed in 5.0s** (377 baseline, +38 bootstrap-gating, +15 waiting-queue wiring).
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

- Add an independent bootstrap subtask: waiting-queue entry as trigger, destination authorization over the preallocated final pages, V readiness, native completion, D installation, and RUNNABLE admission. Validate first with existing full_prompt and fake transport; this does not depend on CAGRA.
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
- **Initial pull authorization:** Entry/request incarnation, initial Delivery identity, authorized per-rank final-page regions, capacity, and validity/cancellation state. The authorization is published only once the request has entered the final waiting queue and its pages are registered and pinned; WRITE submission requires validated layout and actual capacity.
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
- Do not expand the task into a full Host/NVMe tiering product or replace the whole transport stack.
- Do not silently replace CAGRA with another algorithm.
- Do not remove existing MR, metadata-cache, fencing, or native-handle safeguards.
- Do not stop hardware-independent work because hardware is unavailable, and do not fabricate hardware acceptance.

## 16. Remaining decisions that must not block all work

For these decisions, propose a design and its tradeoffs, ask the user where necessary, and continue independent work:

- Architecture-specific probe execution and how predicted positions produce queries for the next window.
- Token-level versus page-level indexing and independent versus shared selection across layers/heads.
- Bootstrap is already decided: waiting-queue-triggered complete-KV pull, D-initiated, V-written, direct into the preallocated final pages. Do not ask whether to adopt it again; design the not-runnable gating and fair admission among concurrently pulling requests explicitly.
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
