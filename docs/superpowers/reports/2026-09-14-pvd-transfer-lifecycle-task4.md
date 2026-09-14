# PVD lifecycle Task 4 verification

Date: 2026-09-14. Branch: `codex/pvd-transfer-lifecycle`.
Base: `4d9e341df`. Scope: V allocation/source lifetime and asynchronous delivery
control. This checkpoint is not yet an end-to-end deployable lifecycle fix.

## Changes

- Each Entry allocation has a ResourceGuard and pins the registered V pool.
  An upload pin is acquired before publishing the target descriptor. Cancel,
  expiry and close request release; they cannot bypass upload/delivery pins.
- V has a worker-generated epoch and per-allocation generation. A lifecycle
  destination reservation creates a stored WriteAuthorization using that V
  epoch and the destination's receiver epoch/generation. Legacy destinations
  keep source pins but cannot obtain lifecycle fence success.
- Delivery reservations pin Entry source pages. Heterogeneous TP packing has a
  separately guarded staging registration. Packing, CUDA synchronization,
  native submit/poll, unregister and engine health do not execute while holding
  the V business lock. A per-delivery poll lock excludes competing pollers.
- Cancellation during packing/submission keeps the source pin. If no native
  submit occurred, cleanup waits for local GPU packing to finish. Failed GPU
  synchronization, a lost submission handle or lost poll status retains the
  resources and isolates new store admission.
- Logical failure does not stop native progress. Late success/failure releases
  safe transport ownership without resurrecting the request. Unregister
  failures remain retryable; allocation free is idempotent.
- Full identity fences now reach the real VectorKVStore gate. In-flight gates
  return false; closed terminal gates return true. Old ID-only fences can block
  late starts but always return false. Wrong identities do not cancel a current
  write. Several Delivery objects can independently retain the same Entry.
- Added poll_delivery to coordinator, local/HTTP shard clients and the public
  client, with /v1/deliveries/poll and /internal/v1/deliveries/poll routes.
  Pending start/retrieve returns v_writing, not a failed delivery. Retrieval
  responses include saved write identities. Coordinator shard RPCs do not run
  under its business lock, and retrieve does not hold the fence lock across
  submission. Public reserve/start retain their idempotency gates.
- Pool close stops admission and keeps the MR registered while any allocation
  remains protected. Progress can finish reclamation after close. Snapshot
  reports worker epoch, isolation, closed/release-requested and upload status.

## Verified failure cases

The new tests use the actual store, allocator, registered CPU regions and
WriteAuthorization. DelayedTransferEngine records PUTs without copying until
finish(handle) is explicitly called. Real aiohttp TestServer exercises a
coordinator with a local V shard and an HTTP V shard.

1. First RED run: **8 failed**. Cancellation immediately returned Entry pages;
   heterogeneous staging was unregistered before completion; upload TTL/close
   reclaimed the destination; progress/full-identity fence APIs were absent.
2. After initial protection: **8 passed**.
3. Expanded tests: **1 failed, 14 passed**. A thrown poll call retained resources
   but did not isolate new admission. Fixed by marking tracking UNKNOWN and
   stopping new V work, without retrying an untrustworthy handle.
4. Concurrency/expiry RED: **2 failed, 16 passed**. Late upload begin/commit could
   resurrect an expired allocation, and retrieve held the fence lock while
   submit was blocked. Both now reject/wait safely without blocking fence.
5. Final V suite contains **21 passing cases**, including both normal and
   cancelled HTTP completion, wrong epochs, shared Entry consumers, MR cleanup
   retry, pool close, adapter pre-native rejection and local packing fences.

One existing test_pvd3 assertion was corrected: an ID-only tombstone is not
positive MR safety proof. It still verifies rejection of delayed reservation.

## Exact verification commands

Working directory for all commands:
`D:/code/sglang-V100-PVD/.worktrees/pvd-transfer-lifecycle`.

Targeted RED/GREEN command:

```powershell
& D:/code/sglang-V100-PVD/.venv/Scripts/python.exe test/registered/disaggregation/run_pvd_cpu_tests.py test/registered/disaggregation/test_pvd_vector_lifecycle.py -q --tb=short
```

Combined regression (final result: **213 passed in 5.35s**):

```powershell
& D:/code/sglang-V100-PVD/.venv/Scripts/python.exe test/registered/disaggregation/run_pvd_cpu_tests.py test/registered/disaggregation/test_pvd_core.py test/registered/disaggregation/test_pvd3.py test/registered/disaggregation/test_pvd_rails.py test/registered/disaggregation/test_pvd_mooncake_metadata.py test/registered/disaggregation/test_pvd_transfer_lifecycle.py test/registered/disaggregation/test_pvd_mooncake_lifecycle.py test/registered/disaggregation/test_pvd_transfer_authorization.py test/registered/disaggregation/test_pvd_vector_lifecycle.py -q --tb=short
```

Static checks (all selected rules pass; six files formatted):

```powershell
& D:/code/sglang-V100-PVD/.venv/Scripts/python.exe -m ruff check --select E9,F401,F821,I python/sglang/srt/disaggregation/pvd/vector_store.py python/sglang/srt/disaggregation/pvd/coordinator.py python/sglang/srt/disaggregation/pvd/client.py python/sglang/srt/disaggregation/pvd/control_server.py test/registered/disaggregation/test_pvd_vector_lifecycle.py test/registered/disaggregation/test_pvd3.py
& D:/code/sglang-V100-PVD/.venv/Scripts/python.exe -m ruff format --check python/sglang/srt/disaggregation/pvd/vector_store.py python/sglang/srt/disaggregation/pvd/coordinator.py python/sglang/srt/disaggregation/pvd/client.py python/sglang/srt/disaggregation/pvd/control_server.py test/registered/disaggregation/test_pvd_vector_lifecycle.py test/registered/disaggregation/test_pvd3.py
git -c safe.directory=D:/code/sglang-V100-PVD/.worktrees/pvd-transfer-lifecycle diff --check
```

## Remaining boundaries and handoff

- **Task 5:** The upload pin is currently released on a successful legacy
  commit_p_write report. Failure, cancel, TTL or missing report cannot release
  it. P epoch identity and explicit closed terminal confirmation still need to
  replace this legacy success path. Cancelled uploads intentionally retain
  pages; no timeout-only reclamation or new unverified cleanup API was added.
- **Task 6:** D must supply and retain process-owned receiver identities, pin
  its destination before publication, poll pending delivery and fence its
  close path. Existing production D code is not migrated by this checkpoint.
- **Task 7:** Explicit staging/in-flight budgets and bounded background progress
  are not wired yet. progress_transfers is a one-step driver also called from
  poll/cancel/reap/close; it is not a new autonomous worker. Record/tombstone
  bounding, capability admission and complete readiness/shutdown reporting
  remain required. Isolated/unconfirmed resources cannot simply be timed out.
- Heterogeneous packing still makes component chunks and a final concatenation.
  Peak accounting must include those temporary buffers, or packing must be
  refactored into one preallocated destination. No shared staging pool added.
- Adapter NOT_SUBMITTED after begin means the one-shot sender gate was consumed
  but the adapter proved local rejection before native submit. Its authorization
  closes as a failed gate; the native handle remains NOT_SUBMITTED. This does
  not permit an already begun authorization to claim it never began.
- CPU tests and a simulated CUDA synchronization boundary do not prove actual
  GPUDirect/RDMA/GPU ordering. No real hardware, link-failure or long-running
  benchmark validation was performed. Native package/fresh-metadata policy,
  Router selection, full-prompt behavior, TP/rail configuration and ordinary PD
  code are unchanged. No independent subagent review in this continuation.

Self-review covered source pins before submit, cancellation during preparation,
terminal-versus-business state, allocator/MR ownership, callback lock ordering,
unknown-state retention, late upload messages and HTTP pending/fence behavior.
