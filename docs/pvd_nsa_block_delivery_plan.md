# NSA-inspired contiguous KV delivery

2026-10-04, branch `codex/pvd-oasiskv`. Each completed stage is committed locally;
no GitHub push. Keep temporary outputs inside this worktree's `artifacts/`.

## Scope

Keep Qwen2.5-7B, paired actual/lookahead Decode, immediate per-layer Q publication,
V/CAGRA, four-head fast graphs, Top4, capacity32, max_new16 and two workers.
This experiment borrows NSA's contiguous access principle. It does not add
NSA's learned compression/gates or claim native NSA model quality.

The original component-major Entry interleaves two KV heads at each token.
Consecutive tokens for one head therefore cannot become one contiguous RDMA
source slice. The qualified baseline packs selected rows into a contiguous
staging buffer and sends that buffer; preserve that wire contract and all native
terminal, byte-count, identity, fence, ownership and UNKNOWN handling.

## Sequence

1. Replay the two saved real Decode trajectories without GPU work. Count
   contiguous runs in exact bank/miss order and sorted miss order. Account for
   monotonic D CPU caching, exclude never-consumed terminal prefetch, and report
   bootstrap separately. Also quantify whole-block expansion at sizes4/8/16
   against the existing32-token/head cap; expansion is not implemented.
2. Add a default-off contiguous packing option. Preserve manifest order and
   payload bytes; copy each maximal consecutive token run into the existing
   staging buffer with a strided source view. Sort only missing wire token IDs
   under that option; preserve selected bank order and resident policy. Validate
   paired mode activation on D and V rather than silently falling back.
3. Prove exact bytes and selected bank semantics on CPU, including reversed
   IDs, disjoint runs, page boundaries, multiple heads/layers, dtype differences,
   invalid later groups, alias rejection and a failed partial copy.
4. On restored CloudLab access, validate real CUDA packing and native receive
   against an independent byte oracle, then run matched base/opt/opt/base full
   P/V/D requests with identical resources and warmups. Record actual mode,
   run/row counts, bytes, outputs, V service, D wait, READY ratio, TPOT and cleanup.
   Short saved trajectories do not establish full-KV quality or general speedup.
5. Keep the option off unless live results support it. A whole-block selector or
   compressed coarse index is a separate quality experiment requiring equal
   token/byte budgets and real-Q/output-quality comparisons.

## Current external dependency

The user confirmed that the CloudLab lease has expired and no GPU is currently
available. Do not retry those nodes. Local implementation and evidence can
proceed; CUDA qualification and fair live timing await new GPU resources.

## Completed local stages

- Layout analysis committed as `d1305209f`: bootstrap copy-call opportunity
  about35%, steady opportunity about7%; whole-block4 expansion exceeds
  capacity32 in about79% of captured banks.
- Implementation committed as `28e223be6`: opt-in strided run packing plus
  wire-only missing-token sorting; serving defaults remain off.
- Final CPU gate:105passed,6actual CUDA cases skipped,1existing config warning.
  Saved real-model KV verifies840distinct consumed layer banks bitwise, with
  equal token counts, payload bytes and delivery counts. No GPU timing claim.
- See `benchmark/results/pvd_nsa_contiguous_kv_local_20261004.md` for exact
  scopes, earlier failed fixture, source proof and the pending live gates.

## Next local experiment: batch D bank installation

1. Keep selected IDs/order, resident intersections, CPU-cache misses, network
   payload and Prompt-bank capacity identical. Aggregate four heads' resident
   gathers/scatters and CPU-miss H2D into one bounded installation. Preserve
   the caller's CUDA stream, completion fence and UNKNOWN owner retention.
2. Add an explicit default-off `batched_bank_install` D option. Reject mixing
   with GPU-backup, staged transport or attention workspace experiments; charge
   a conservative allocation bound against existing request scratch admission.
   The baseline install path stays unchanged.
3. Validate the actual helper on CPU against an independent per-head oracle,
   then replay the captured real banks, including resident hits and CPU-cache
   reuse. Record exact KV bytes and planned Torch call counts, not GPU timing.
   Add real CUDA tests for nondefault streams and failure-owner retention.
4. Commit implementation and evidence locally. On new GPU resources, run the
   CUDA gate and matched `d-batch-install` ABBA full-path comparison, changing
   only this option. Require actual installation profiles, output/byte budgets,
   D wait, TPOT and cleanup; do not infer a latency win from fewer calls.

### Batch-install local stages completed

- Plan committed as `dff9c8a5b`; implementation as `d6ac2fb61`.
- CPU gate:133passed,14real CUDA cases skipped,1existing config warning.
- Two real trajectories:840distinct banks bitwise exact; resident/local KV
  rows and bytes unchanged. Steady KV H2D submissions fall1189/1183 to387/388,
  resident gather and scatter submissions each fall3136 to784 per case.
- `d-batch-install` fair ABBA/proof entry is prepared. Native/CUDA tests,
  allocator memory, live installation time, D wait and TPOT are still pending.
- Evidence and reproduction scope:
  `benchmark/results/pvd_oasis_batched_bank_local_20261004.md`.

## Next local experiment: batch D CPU-cache installation

1. Preserve the same V payload and synchronous receive-ordering/D2H proof.
   Replace per-token clone/copy/valid writes with one indexed KV copy and one
   validity write per head into the existing charged monotonic CPU cache.
   Validate all groups before any cache mutation; retain sources on failures.
2. Add default-off `batched_cache_install`, isolated from batch-bank, GPU-backup,
   stages and attention-workspace experiments. Keep native receiver identities,
   ACK-after-install, quarantine, registration and terminal proofs unchanged.
3. Check real payload bytes, cache validity, candidate order and bank bits on
   CPU and saved real trajectories. Test duplicate/alias/later-group rejection,
   partial writes, ACK refusal and UNKNOWN ownership. CUDA/native qualification
   remains a separate gate when GPU resources return.
4. Commit plan, implementation and evidence locally. Prepare a fair
   `d-cache-install` ABBA entry changing only this option. Report operation
   counts separately from live cache-copy time, D wait and TPOT; no inferred
   latency gain from CPU-only saved-trajectory verification.

### CPU-cache-install local stages completed

- Plan committed as `c3d038206`; implementation as `1b581757d`.
- Gate:173passed,20real CUDA cases skipped,1existing config warning.
- Actual record-method CPU replay:840distinct real banks exact; unchanged
  manifest/wire hashes, cache rows, payload and local KV H2D bytes. Steady
  row clones3815/3796 to0; KV copies3815/3796 to1170/1169.
- Local CPU five-round matched ABBA:steady cache method means210.9/205.5us
  to113.3/108.5us per rank delivery. D2H/native/network, live D wait/TPOT and
  GPU contention excluded. These numbers are not CloudLab performance claims.
- `d-cache-install` full-path fair entry is prepared; default remains off.
- See `benchmark/results/pvd_oasis_cache_install_local_20261004.md` for source
  proof, failed assertion fixture, evidence and native/CUDA pending gates.

## Next local experiment: V sparse source fences

1. Instrument bounded per-delivery source phases: owned staging allocation,
   index pin, pack launch, pack completion fence, memory registration, outer
   preparation fence and adapter submission. Use monotonic wall time; launch
   time is not GPU execution time and adapter time includes its own CUDA fence.
   Forward diagnostics to D without using them as native completion proof.
   Commit this baseline instrumentation after CPU lifecycle/HTTP tests.
2. Add default-off `reuse_sparse_pack_fence` for ordinary Torch CUDA staging.
   Record a private completion marker only after this delivery's pack fence
   succeeds. Skip only the outer store fence when that marker is valid and the
   delivery is not UNKNOWN. Preserve pack/index/Entry ownership, cancellation,
   registration quarantine and the Mooncake source-readiness fence. Keep dense,
   direct batch, Triton and contiguous experiments outside this first pilot.
3. Test exact payloads, two V ranks, repeat start, cancellation before native
   submission, partly failed copies, both CUDA-fence failures, registration
   uncertainty and metadata uncertainty using explicitly labelled CPU policy
   doubles. Add actual CUDA byte/stream checks that skip when CUDA is absent.
   Commit the implementation and scoped local evidence independently.
4. Prepare a `v-pack-fence` matched ABBA pilot changing only the V option.
   Require actual per-delivery fence counts and phase timings, identical D
   configuration, fast CAGRA settings, wire/bank budgets and cleanup. On restored
   GPU resources, qualify native RDMA and compare V service, D wait and TPOT.
   CPU fence counts do not establish GPU savings; leave the option off meanwhile.
5. Use the new phase measurements to decide whether V staging registration
   reuse or narrower stream completion fences warrant a separate experiment.
   Do not combine earlier pilots or presume their gains add together.

### V source-fence local stages completed

- Plan `56e689006`; source profiling `13c8f38ae`; isolated reuse pilot `05dec10f2`.
- Final gate:233passed,22actual CUDA cases skipped,1existing config warning.
- Both rank byte oracles and failure lifetimes passed. Ordinary store fences
  fall2 to1 under explicit CPU CUDA-policy tests; the adapter fence stays1.
  This does not measure GPU or live Decode savings.
- Historical baseline adapter counters average0.183ms for its source fence and
  1.880ms for single native submission, including startup fan-in. Those counters
  do not cover the two store fences or justify subtracting unrelated medians.
- `v-pack-fence` ABBA/proof entry is ready. Actual CUDA/native timing, profile
  overhead, D wait and TPOT require new GPU resources; the option remains off.
- Next decisions follow measured source phases: V staging/MR reuse, precise
  stream completion, or native poll/retirement/control scheduling. Implement
  each as an isolated experiment, preserving the existing release proofs.
- Report: `benchmark/results/pvd_v_sparse_pack_fence_local_20261004.md`.

## Next local experiment: prepare only selected K/V component views

1. Historical baseline has6extra polls in2860steady deliveries, so additional
   terminal-wait RPC work has little demonstrated opportunity. The packed
   source currently constructs56Torch component views for every28-layer Entry
   delivery, although an Oasis layer job uses only its own K and V components.
   Freeze this finding and the bounded implementation scope in a local commit.
2. Keep validation of every component's dtype/shape/byte metadata on each call.
   Add default-off selected component preparation to ordinary Torch packing:
   construct only the unique selected K/V views, shared across local heads.
   Keep per-request identity, layer/head/token bounds, destination nonaliasing,
   full metadata validation, byte order, kernels, fences and budgets unchanged.
   Do not cache mutable layout dictionaries or combine other experiment flags.
3. Check both ranks, nonzero layer starts, multiple selected layers, partial
   pages, all supported dtypes and independent uint8 byte oracles. Invalid
   metadata in an unselected component must still fail before any write. Keep
   the existing failed-copy and UNKNOWN ownership gates; add real CUDA cases.
4. Replay selected bytes and timing on saved real Decode KV using the actual
   helper with a declared reconstructed CPU source; identify uncaptured rows.
   Record56-to2view preparation separately from CPU wall time. Preserve exact
   Prompt-bank IDs, network rows/bytes and source immutability.
5. Prepare an independent `v-selected-views` ABBA comparison with identical D,
   fast V/CAGRA, Torch packing and budgets; only the new V option may differ.
   Commit implementation and scoped evidence locally. CUDA/native, live V
   service, D wait and TPOT qualification await new GPU resources.

### Selected-component local stages completed

- Plan `e80d93f1d`; implementation `dfa4c8f9e`; default remains off.
- Final gate:333passed,34real CUDA cases skipped,1existing config warning.
  Actual source-storage reshape calls verify56to2views for single-layer jobs;
  unselected metadata still fails before writing. Original fences and UNKNOWN
  source/index/staging/MR/budget retention remain intact.
- Two matched real captures,840consumed banks:actual selected payload uint8
  bytes and manifest order exact. The declared CPU source has uncaptured rows
  poisoned and never selected; this is not native Entry/CAGRA qualification.
- Five-round CPU ABBA:steady helper means1.063/1.054ms to0.487/0.480ms per
  rank delivery; excludes source preparation,CUDA,registration,network,D wait.
  GPU savings and TPOT improvement cannot be inferred or added to prior pilots.
- Historical baseline extra polling is6calls in2860steady rank deliveries;
  no wait-on-start change implemented. That scope is only the saved fixture.
- `v-selected-views` independent full-path ABBA/proof entry is prepared.
  First restore new GPU hosts/model paths, qualify actual CUDA/native, then
  measure source phases,D wait and TPOT. Keep all other experiment flags off.
- Further staging/MR reuse or stream/event work depends on measured phase
  ownership and cost. Report and evidence:
  `benchmark/results/pvd_v_selected_component_views_local_20261004.md`.

## Next local experiment: remove discarded layout metadata deep copies

1. Profile the actual selected-layer CPU packing helper on the same frozen KV
   capture. KVLayoutSignature.to_dict currently recursively copies extra using
   dataclasses.asdict, then overwrites that copy with dict(self.extra). Record
   this repeated work separately from instrumented timing; do not cache a
   mutable layout or infer serving latency from cProfile cumulative time.
2. For atomic top-level fields, construct the same dictionary directly and
   retain the existing shallow extra copy. Preserve the original asdict path
   for non-atomic/custom/subclass field values. Fingerprint JSON, hash, protocol,
   live metadata reads, full validation and all device/transport proofs stay
   unchanged. This equivalent serialization does not introduce a serving mode.
3. Compare dictionary/wire/hash results against an independent old serializer:
   nested metadata, mutation, tuple/list values, fallback dataclasses/custom
   fields and subclass fields. Re-run affected protocol, HTTP, pack and UNKNOWN
   lifecycle gates, plus saved real KV bytes with unchanged manifest and row
   budgets. Commit implementation after these local gates pass.
4. Run fresh CPU ABBA on both frozen captures using identical selected-layer
   packing, preallocated jobs and Torch threads, changing only the serializer.
   Report fingerprint time separately from actual helper wall time. This is
   explicitly a controlled local monkeypatch reference, not a native V run.
5. Commit evidence locally with source/input proofs. No GPU is available:
   CUDA/native, full V service, D wait and TPOT remain pending. Future serving
   comparison should pin the baseline and candidate commits with identical
   launch options and inputs; no combined timing claims across earlier pilots.

### Layout-serialization local stages completed

- Plan `8334dc410`; implementation `3da9d9ed8`. Equivalent metadata serialization
  is used directly; the existing selected-component serving flag stays off.
- Gate:612passed,5subtests passed,16actual CUDA cases skipped,1existing warning.
  One Linux direct-bootstrap module could not collect on Windows because of
  resource; its original failed log is retained and Linux qualification pending.
- Dictionary/key order/JSON/fingerprint and complex-field fallback match the
  old serializer. Mutable extra is still read each call, with no cached hashes.
  Both ordinary and selected packing reject mismatched metadata before writes.
- Same frozen captures,840banks:wire/layout/manifest bits, row/byte budgets and
  source hash unchanged. Reconstructed uncaptured rows remain unqueried poison.
- Fresh controlled CPU ABBA with only to_dict changed:steady helper means
  363.36/377.01us to312.04/312.04us; fingerprints82.06/81.02us to37.30/37.93us.
  Case99402bootstrap regressed1.8percent; retain it rather than generalize gains.
  Fingerprint time is already part of helper time; neither can be added to
  prior experiments or translated into GPU/D wait/TPOT savings.
- Next local probes:manifest hash/wire construction and per-row copy dispatch.
  Native/CUDA and fixed-revision full-path comparisons require new GPU hosts.
- Report:`benchmark/results/pvd_layout_serialization_local_20261004.md`.

## Next experiment: budgeted indexed Torch row packing

1. Frozen-capture CPU probe:manifest fingerprint saves only2to3us; indexed
   gather prototype reduces actual validated pack from244to248us to183to184us.
   Prioritize row batching; no manifest serializer change in this experiment.
2. Add a bounded CPU/CUDA index workspace per delivery. Flatten logical token
   plus local head into source row IDs, preserving every original ID and wire
   order. Single-row groups retain copy_; multi-row groups use two index_select
   calls with exact preallocated out shapes. Retain host/device indexes and
   budget through success/partial failure fences, quarantine constructor UNKNOWN.
   No new KV gather buffers, stream switches, RDMA modes or completion proofs.
3. Add default-off indexed packing to V, requiring the selected-component mode
   on both baseline and candidate. Keep original store/native fences, Entry/index
   leases and registration/PUT/ACK rules. Isolate from Triton/direct/contiguous/
   fence-reuse experiments. Include index preparation in V pack wall time and
   record actual copy/index calls plus index bytes, never as terminal proof.
4. Validate exact uint8 bytes across both ranks/dtypes/layers/pages, all-group
   pre-write rejection, output-storage preservation, singleton fallback, budget
   and constructor/copy/fence/cancellation/registration/submit UNKNOWN lifetimes.
   Real CUDA nondefault-stream cases remain skips without GPU. Commit locally.
5. Fresh CPU ABBA includes actual index preparation, budget and release; same
   captures/selected views/validation, no native or Decode claim. Prepare the
   independent v-indexed-pack full-path comparison with identical D and selected
   V views on all arms; only indexed option differs. Freeze source/input proofs
   and commit reports. GPU/native/D wait/TPOT gates require new GPU resources.

### Indexed-row local stages completed

- Plan `df4df2a83`; implementation `bcab18ac6`. Indexed packing stays default off,
  with selected component views required on both arms. Singleton groups retain
  copy_; owned/budgeted multi-row indexes feed exact preallocated index_select out.
- Initial gate exposed indexed workspace forwarded as fused workspace; fixed
  conditional forwarding, retained the failed log. Final gate:476passed,
  46actual CUDA skipped,1existing warning. Twelve new nondefault-stream CUDA
  cases remain unexecuted; CPU policy doubles are not CUDA qualification.
- Same two frozen captures,840banks:actual uint8 payload, manifests, token order,
  rows/bytes and source hash unchanged. Source views remain2per rank delivery;
  uncaptured CPU reconstruction rows remain poison and never selected.
- Fresh five-round CPU ABBA includes workspace admission/preparation/release.
  Bootstrap606.39/638.55us to219.19/225.79us; steady235.70/246.76us to
  217.49/217.91us. Final steady gain is7.7/11.7percent, below the simpler prototype.
  Source preparation,staging/MR allocation,CUDA/H2D,native/network/D wait excluded.
- Steady API calls7630/7592to2340/2338. Extra CPU index bytes28064/27944 over
  each capture,peak scratch admission2368bytes per delivery. These are actual
  Torch API counts and CPU metadata,not measured GPU kernels or H2D cost.
- Original store/native fences and UNKNOWN Entry/index/staging/MR/budget retention
  preserved. Source proof matches171files against the implementation commit.
- Independent `v-indexed-pack` full-path ABBA/proof entry prepared,not executed.
  Restore GPU hosts/model paths,run actual CUDA/native two-rank gates,then measure
  index upload,V source phases,D wait and TPOT with identical budgets/outputs.
- No inference of live savings or addition to earlier pilots. Report:
  `benchmark/results/pvd_indexed_sparse_pack_local_20261004.md`.
