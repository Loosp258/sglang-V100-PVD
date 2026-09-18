# PVD transfer lifecycle — verification record

Plan: `docs/superpowers/plans/2026-09-09-pvd-transfer-lifecycle.md`.
Design: `docs/superpowers/specs/2026-09-09-pvd-transfer-lifecycle-design.md`.
Branch: `codex/pvd-upload-lifecycle`. Base: `2a66bc477`.
Last updated: 2026-09-15.

Follow-up: the 2026-09-19 integration verification is recorded in §8 below.
The original 303-test result did not cover the complete preflight-to-KV-manager
startup sequence or distinct P TP2 uploaders. Both paths had blockers in
`488856615`; the configuration requirements below were not proof of startup.

This record states what was executed and what was not. It is not a claim that
the lifecycle bug is fixed on real hardware.

## 1. Task status

| Task | Scope | State |
| --- | --- | --- |
| 1 | Transport state, resource guards, capacity primitives | Implemented before this branch |
| 2 | Native async handles, shared lifecycle manager | Implemented before this branch |
| 3 | Write identities, authorization gates, fence protocol | Implemented before this branch |
| 4 | V Entry page and staging protection, async delivery | Implemented before this branch |
| 5 | P→V upload identity, terminal confirmation, cancellation | Implemented and tested here |
| 6 | D receive lifetime, pending refresh, safe cleanup | Implemented and tested here |
| 7 | Explicit capacity config, bounded progress, admission | Implemented and tested here |
| 8 | Acceptance documentation and release checks | This document |

## 2. Executed test results

Environment: WSL2 Ubuntu 24.04.4 LTS on `ishmael`, kernel
`6.6.87.2-microsoft-standard-WSL2`, Python 3.12.3, torch 2.14.0+cpu,
aiohttp 3.14.3, pytest 9.1.1, numpy 2.5.3, ruff 0.16.7. No GPU, no RDMA.

```bash
cd <worktree>
T=test/registered/disaggregation
.venv-linux/bin/python $T/run_pvd_cpu_tests.py \
  $T/test_pvd_core.py $T/test_pvd3.py $T/test_pvd_rails.py \
  $T/test_pvd_mooncake_metadata.py $T/test_pvd_transfer_lifecycle.py \
  $T/test_pvd_mooncake_lifecycle.py $T/test_pvd_transfer_authorization.py \
  $T/test_pvd_vector_lifecycle.py $T/test_pvd_upload_lifecycle.py \
  $T/test_pvd_decode_lifecycle.py $T/test_pvd_transfer_admission.py \
  -q --tb=line
```

**Result: 303 passed, 0 failed, in 2.19s.**

| File | Passed |
| --- | --- |
| test_pvd_core.py | 14 |
| test_pvd3.py | 29 |
| test_pvd_rails.py | 25 |
| test_pvd_mooncake_metadata.py | 19 |
| test_pvd_transfer_lifecycle.py | 29 |
| test_pvd_mooncake_lifecycle.py | 23 |
| test_pvd_transfer_authorization.py | 54 |
| test_pvd_vector_lifecycle.py | 21 |
| test_pvd_upload_lifecycle.py | 30 |
| test_pvd_decode_lifecycle.py | 23 |
| test_pvd_transfer_admission.py | 36 |

Progression: 213 at `2a66bc477` (reported by the previous developer) → 243
after Task 5 → 267 after Task 6 → 303 after Task 7.

Static checks, all passing over the 15 changed files:

```
ruff check --select E9,F401,F821,I   -> All checks passed
ruff format --check                  -> 15 files already formatted
git diff --check                     -> clean
```

The repository's own pre-commit gate is `ruff --select=F401,F821` plus isort
(profile=black) and black. This branch follows the Task 3/4 convention of
`ruff check --select E9,F401,F821,I` and `ruff format`. `black --check` already
failed on `transfer_lifecycle.py` and `transfer_engine.py` at `2a66bc477`; that
divergence predates this work and was not changed.

## 3. Executed RED evidence

Both were run against a pristine `2a66bc477` worktree.

### Task 5 — the old publish path cancelled a live upload

```
submitting shard 0 with a transport that stays IN_FLIGHT...
  -> cancel_entry called: 'pending'
RESULT: publish_shard raised PVDDataPlaneError: ... failed: pending
  V shard0 state          = cancelled
  V release_requested     = True
  in-flight PUTs on wire  = 1
```

After: the shard stays `p_writing` with the write in flight, and becomes
`stored` only once the native WRITE completes.

### Task 6 — the old Decode close never terminated

```
RESULT: close() STILL LOOPING after 3s (never terminates)
  fence HTTP requests actually issued = 0
  registration still held             = True
  MR still live                       = True
```

Zero fence requests were issued: `fence_retrieval` was called with one argument
while the client required two, and the `TypeError` was swallowed inside
`while True: ... sleep(1)` before any HTTP call.

### Task 7 — config and admission

Suite-level RED: 29 existing tests failed once the two budget arguments became
required, because none of them supplied a budget. They now pass by supplying
explicit values.

## 4. Coverage of the design's safety invariants

| Design §3 invariant | Where enforced |
| --- | --- |
| Receiver pins before the descriptor is published | `vector_store.create_entry`, `decode_refresh.prepare` |
| Timeout/cancel/TTL/lease loss is not drain proof | `sync_upload`, `progress_close`, `_release_resources_locked` |
| Only NOT_SUBMITTED or an observed terminal may unpin | `TransportState.is_locally_safe_to_release`, `WriteAuthorization.observe_terminal` |
| Receive memory needs sender terminal plus a closed gate | `progress_close`, `coordinator.fence_retrieval` |
| One resource lifetime covers MR, tensor, pages, staging | `ResourceGuard` owners in all three roles |
| Business state may not act as a resource pin | separate `active_delivery_count` and guard owners |
| A late success only reclaims, never republishes | `_try_publish_stored_locked`, `release_refresh` |

## 5. What is NOT verified

- **No GPU, no RDMA, no GPUDirect.** Every result above is CPU tensors with a
  fake native transport and a simulated CUDA synchronization boundary. Ordering
  between GPUDirect reads and PyTorch packing kernels is not exercised.
- **No link-failure injection**, no long-running soak, no memory-curve capture.
- **No real Mooncake binary.** The native submit/poll boundary is a stub. The
  pinned `mooncake-transfer-engine==0.3.13.post1` version check and the
  `MC_DISABLE_METACACHE=1` ordering are safeguards, not proof that all
  stale-rkey risk is eliminated.
- **No independent subagent review** was performed on Tasks 5 through 8.
- The real-machine acceptance sequence in the plan's §8.7 is **NOT RUN**: no
  CloudLab GPU/RDMA access was available in this session.

### Hardware acceptance sequence still to run

1. Single normal request end to end; check P/V/D logs share one transfer id.
2. Sequential address reuse across refreshes; confirm no stale-write corruption.
3. Concurrent refreshes across several sequences.
4. HTTP cancellation mid-transfer; confirm V retains pages until the terminal
   report and releases exactly once afterwards.
5. Link fault and timeout injection, then recovery — only in an experiment
   environment the operator has authorised.
6. For each: record transfer ids, terminal-state logs, pin counts returning to
   zero, memory not growing, and generated-token correctness.

## 6. Operational notes

### Required configuration

PVD startup now requires two explicit positive integers on every role. There is
no default: a staging budget that fits one GPU can be fatal on another.

P and D model servers:

```
--pvd-transfer-staging-budget-bytes <bytes>
--pvd-transfer-max-inflight <count>
```

V launcher (`python -m sglang.srt.disaggregation.pvd.server`):

```
--transfer-staging-budget-bytes <bytes>
--transfer-max-inflight <count>
```

`None`, `0`, negatives, booleans and floats are all rejected at startup.
Ordinary PD is unaffected and never grows these attributes.

### Health and readiness

The V shard snapshot reports `capabilities`, `ready`, `draining`,
`worker_epoch`, `isolated_reason`, allocator counts and the transport health
block, which carries the lifecycle manager's budget, in-flight, unknown and
quarantine state. The P/D worker exposes the same shape through
`PVDKVManager.transfer_health()`.

### Recovery

A transfer whose native status is lost becomes `UNKNOWN`. That quarantines the
worker's engine: no new PVD transfer is admitted, and the isolated resources
are never refunded. First-version recovery is a coordinated restart of the
affected P/V/D workers after stopping the senders. Restarting only D, or
shortening a TTL, is not a recovery procedure.

## 7. Remaining boundaries

- Staging byte accounting covers the D receive buffer and the V repacking peak
  (charged at twice the destination size, because the per-component chunks and
  the final concatenation are live together). **P packing bytes are not yet
  charged**; only the transfer slot is.
- `VectorKVStore.entries`, `VectorCoordinator.entries` and
  `VectorCoordinator.deliveries` still grow without bound. Upload tombstones
  are bounded by domain, and `pending_decode_closes` is bounded only by the
  number of live requests.
- Progress is driven by the scheduler calling `progress_uploads()` /
  `progress_decode_closes()`. A worker that stops calling them retains
  resources rather than releasing them — the safe direction — but there is no
  self-starting thread.
- `require_capability` exists and is tested, but the handshake is not yet wired
  into the V control server's startup exchange.
- V's worker epoch is per store instance rather than the shared
  `worker_epoch()` helper.
- TP1 Decode against TP2 storage is unsupported: compute-rank KV heads cross a
  V shard boundary.

## 8. 2026-09-19 startup and TP2 uploader integration fixes

Base: `pvd-disaggregation` at `488856615`. These changes do not implement a new
transport, streaming protocol or retrieval algorithm.

### Changes

- P/D KV managers reuse the budget already installed by ModelRunner preflight
  on the shared Mooncake engine and verify its configured limits. The adapter's
  distinct-budget rejection remains intact; missing preflight still fails.
- Entry creation normalizes a complete storage-shard-to-uploader-epoch map.
  TP1 assigns both shards to one epoch; TP2 exchanges the two process epochs
  along with the existing manifest collective. Each V store still receives
  only its own uploader epoch, and full write identities remain mandatory.
- P runtimes own only their assigned uploads, including during abort/clear.
  They cannot submit another rank's shard or report its terminal state.
- A prior test expected an invalid foreign-epoch lease to be reported as
  NOT_SUBMITTED by the local process. That expectation was unsafe: the foreign
  authorization now stays pinned on V rather than accepting a false drain
  report. Valid local authorizations opened before a later validation failure
  are still abandoned and drained by their actual owner.

### Executed evidence

Environment: local Windows CPU venv (`.venv/Scripts/python.exe`), no GPU/RDMA.
The same eleven-file suite listed in §2 passed: **324 passed in 3.61s**.
This includes real HTTP coordinator-to-shard control traffic and real PVD
runtime/store code, but simulated native transport.

Before the fix, extending the existing ModelRunner initialization test into
the actual `PVDKVManager` constructor reproduced `ValueError: shared Mooncake
engine already has a different budget`. New TP2-map tests initially failed
because the runtime did not accept the map. After the fix the suite covers:

- P and D preflight followed by actual KV-manager initialization, identity of
  the shared accounting object, mismatched limits and missing initialization.
- Two independent P runtimes/managers with distinct epochs creating the same
  Entry concurrently, uploading separate shards, and publishing STORED.
- Cancellation of P rank0 before submission while rank1's WRITE is in flight:
  rank0's pages may drain, rank1's pages remain pinned until its native terminal;
  late success does not revive the cancelled Entry.
- Actual TP1/TP2 sender constructors forwarding the agreed map and local shard
  ownership (the TP allgather is simulated, not a multi-process launch).
- Map validation before allocation, mutation isolation, idempotency, changed
  or stale uploader rejection, non-owned submission refusal, and HTTP map
  propagation through the coordinator and a remote shard service.

Focused Ruff checks (`E9,F401,F821,I`), Ruff format checks and `git diff --check`
passed for the changed files. Hardware acceptance remains **NOT RUN**:
these tests do not launch real TP worker processes or verify CUDA/GDR ordering,
native Mooncake completion semantics, rail behavior, link faults or soak tests.
The remaining boundaries in §7 still apply. Deploy matching P/V versions;
this is not a rolling-upgrade protocol for workers with outstanding writes.
