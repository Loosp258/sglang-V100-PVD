# Task 2 implementation report

## Ownership of inherited partial work

This task resumed an uncommitted implementation from a stopped agent.  I did
not treat that code or its tests as trusted.  The first reproducible inherited
run was:

```powershell
& 'D:\code\sglang-V100-PVD\.venv\Scripts\python.exe' `
  'test\registered\disaggregation\run_pvd_cpu_tests.py' `
  'test\registered\disaggregation\test_pvd_transfer_lifecycle.py' `
  'test\registered\disaggregation\test_pvd_mooncake_lifecycle.py' `
  'test\registered\disaggregation\test_pvd_mooncake_metadata.py' -q
```

It produced `55 passed, 9 failed`.  Eight failures were because the old
metadata native double only implemented synchronous APIs and no explicit
budget was supplied; one was an incomplete deregistration-retry lifetime
test.  An earlier direct pytest invocation without the repository CPU runner
failed at collection because Windows lacks the POSIX `resource` module; that
command is not considered test evidence.

The original pre-implementation RED is not available from this replacement
agent, so I do not claim it.  The inherited partial production code predated
my work.  I did add one independent RED/GREEN cycle below.

## Implemented behavior

- Added a shared `TransferLifecycleManager` stored only on the shared
  Mooncake wrapper.  First construction requires an explicit `TransferBudget`;
  `from_existing` reuses that manager without a budget and rejects a different
  injected budget.
- `attach` reserves exactly `(0 bytes, 1 slot)`.  It deliberately does not
  charge transfer bytes as staging allocation bytes; a future staging
  allocation must reserve its own owner/byte lifetime.
- PVD writes validate bounds, source ownership/rail, metadata policy, CUDA
  synchronization, and budget admission before entering native code.  Any
  such failure remains `NOT_SUBMITTED`.  Once native submit is invoked, an
  exception or non-positive handle becomes `UNKNOWN` and retains its guard and
  slot.
- Native submit and the transition to `UNKNOWN` are covered by a single shared
  submission gate.  The gate holds admission through native submit; an
  `UNKNOWN` transition quarantines the manager.  A concurrent or later handle
  cannot get through to native submission; any previously admitted-but-not-yet
  submitted handle is discarded locally as `NOT_SUBMITTED` and releases its
  slot.  The unknown original remains pinned for coordinated restart.
- Polling is serialized per handle.  `0` stays `IN_FLIGHT`, `-2` becomes
  `DRAINING`, `1` becomes `TERMINAL_SUCCESS`, `-1` becomes
  `TERMINAL_FAILED`; unexpected results or poll exceptions become `UNKNOWN`.
  Terminal native status is consumed once.  Business cancellation remains
  `CANCELLED` even if a late native success arrives.
- A terminal guard unpin frees only after a successful release callback.
  Failed deregistration retains the registration/tensor, lets a later terminal
  poll retry local cleanup without another native status check, and does not
  retain exception traceback objects in logging records.
- The fake transport remains synchronous but now marks both terminal success
  and terminal failure transport states.
- Updated the metadata CPU double to expose async submit/check behavior while
  preserving the fresh `0.3.13.post1` policy tests and ordinary-PD behavior.

## Exact state transitions

```text
NOT_SUBMITTED --validated+attached+native id--> IN_FLIGHT
NOT_SUBMITTED --pre-submit validation/admission failure--> NOT_SUBMITTED
IN_FLIGHT --native 0--> IN_FLIGHT
IN_FLIGHT --native -2--> DRAINING
IN_FLIGHT|DRAINING --native 1--> TERMINAL_SUCCESS
IN_FLIGHT|DRAINING --native -1--> TERMINAL_FAILED
after native submit --exception/nonpositive id--> UNKNOWN (manager quarantine)
IN_FLIGHT|DRAINING --poll exception/unexpected status--> UNKNOWN (quarantine)
CANCELLED + native terminal success/failure --> CANCELLED business status,
    with terminal transport state recorded and source released only when safe
```

`UNKNOWN` has no release transition in this task.  It pins source ownership and
slot capacity until the specified isolate/coordinated-restart procedure.

## TDD evidence

### RED observed during this takeover

After adding `test_fake_copy_failure_marks_terminal_failure`, before changing
the fake engine, the focused command was:

```powershell
& 'D:\code\sglang-V100-PVD\.venv\Scripts\python.exe' `
  'test\registered\disaggregation\run_pvd_cpu_tests.py' `
  'test\registered\disaggregation\test_pvd_mooncake_lifecycle.py::test_fake_copy_failure_marks_terminal_failure' -q
```

It failed as expected:

```text
assert <TransportState.NOT_SUBMITTED: 'not_submitted'>
       == <TransportState.TERMINAL_FAILED: 'terminal_failed'>
1 failed
```

### GREEN

After setting `FakeTransferEngine` failure to `TERMINAL_FAILED`, the focused
lifecycle/adapter/metadata command passed:

```text
69 passed in 2.39s
```

## Verification

Focused command:

```powershell
& 'D:\code\sglang-V100-PVD\.venv\Scripts\python.exe' `
  'test\registered\disaggregation\run_pvd_cpu_tests.py' `
  'test\registered\disaggregation\test_pvd_transfer_lifecycle.py' `
  'test\registered\disaggregation\test_pvd_mooncake_lifecycle.py' `
  'test\registered\disaggregation\test_pvd_mooncake_metadata.py' -q
```

Result: `69 passed in 2.39s`.

The new lifecycle suite includes submit `0`/exception, poll `0/-2/1/-1`,
terminal-once polling, cancellation plus late success, unknown quarantine,
race-gated concurrent submission, shared-manager reuse/mismatched-budget
rejection, failed unregister retry without another native poll, byte-only and
slot-only budget overflow, and weak-reference callback/registration lifetime
coverage.

Existing regression command:

```powershell
& 'D:\code\sglang-V100-PVD\.venv\Scripts\python.exe' `
  'test\registered\disaggregation\run_pvd_cpu_tests.py' `
  'test\registered\disaggregation\test_pvd_core.py' `
  'test\registered\disaggregation\test_pvd_mooncake_metadata.py' `
  'test\registered\disaggregation\test_pvd_rails.py' `
  'test\registered\disaggregation\test_pvd3.py' -q
```

Result: `86 passed in 4.30s`.

Ruff syntax/undefined-name gate:

```powershell
& 'D:\code\sglang-V100-PVD\.venv\Scripts\python.exe' -m ruff check --select E9,F821 `
  'python\sglang\srt\disaggregation\pvd\mooncake_engine.py' `
  'python\sglang\srt\disaggregation\pvd\transfer_engine.py' `
  'python\sglang\srt\disaggregation\pvd\transfer_lifecycle.py' `
  'test\registered\disaggregation\test_pvd_transfer_lifecycle.py' `
  'test\registered\disaggregation\test_pvd_mooncake_lifecycle.py' `
  'test\registered\disaggregation\test_pvd_mooncake_metadata.py'
```

Result: `All checks passed!`  `git diff --check` found no whitespace errors;
Git did emit CRLF-conversion advisories for modified files, so its output was
not otherwise pristine.

## Files changed

- `python/sglang/srt/disaggregation/pvd/mooncake_engine.py`
- `python/sglang/srt/disaggregation/pvd/transfer_engine.py`
- `python/sglang/srt/disaggregation/pvd/transfer_lifecycle.py`
- `test/registered/disaggregation/test_pvd_transfer_lifecycle.py`
- `test/registered/disaggregation/test_pvd_mooncake_lifecycle.py` (new)
- `test/registered/disaggregation/test_pvd_mooncake_metadata.py`

## Remaining Task 7 wiring

This task intentionally leaves construction sites in `conn.py`, `server.py`,
and `model_runner.py` without an implicit budget.  They now fail closed when
they would create the first manager without the required explicit budget.
Task 7 must parse and validate the positive
`--pvd-transfer-staging-budget-bytes` and `--pvd-transfer-max-inflight`
settings, construct one budget per shared wrapper, reserve true staging-memory
allocation owners before allocation, and pass the same object to all adapters.
It must not make ordinary PD require those PVD settings.

## Self-review

- Checked that every path which has entered native submit either stores a
  positive native id or enters `UNKNOWN`; pre-submit paths do not call native.
- Checked lock order: handle lock then manager gate throughout manager and
  adapter paths; native status checks are handle-serialized and terminal checks
  are not repeated.
- Checked that terminal cleanup retry does not re-poll native and logging no
  longer retains the callback exception traceback.
- Checked cancellation does not overwrite business status on late success.
- No hardware/RDMA validation was run; CPU doubles do not substitute for that
  external acceptance test.

## Fix round 1: release retry race

### Review finding and fix

The review correctly found that `ResourceGuard.unpin()` can return while an
external `release_memory()` retry owns a callback.  The manager previously
treated that return as completion, released the slot, and forgot the transfer;
if the external callback then failed, a later terminal poll could not retry.

The first iteration used a pending-release indicator and moved
`TransferLifecycleManager.complete`'s `unpin` outside both the handle and
manager locks.  The atomic outcome that replaces that indicator is documented
in fix round 2 below.  `MooncakePVDTransferEngine.poll` records a native
terminal state under the handle lock, but calls lifecycle cleanup after
releasing that lock, so a blocking deregistration callback cannot hold the poll
lock.

### RED

Before this fix, the new deterministic event-gated test ran:

```powershell
& 'D:\code\sglang-V100-PVD\.venv\Scripts\python.exe' `
  'test\registered\disaggregation\run_pvd_cpu_tests.py' `
  'test\registered\disaggregation\test_pvd_mooncake_lifecycle.py::test_terminal_poll_retains_capacity_while_release_retry_is_running' -q
```

Expected failure observed:

```text
assert 0 == 1
```

The asserted value was `used_inflight` while a second, event-blocked
deregistration callback was running; the old manager had prematurely released
the slot and transfer record.

### GREEN and regression evidence

The same focused test passed after the fix:

```text
1 passed in 1.63s
```

It covers first deregistration failure, an event-gated external retry, a
concurrent terminal poll that must retain the record/capacity, retry failure,
and a final terminal-poll retry.  It verifies only one native status check.

Amended focused suite:

```powershell
& 'D:\code\sglang-V100-PVD\.venv\Scripts\python.exe' `
  'test\registered\disaggregation\run_pvd_cpu_tests.py' `
  'test\registered\disaggregation\test_pvd_transfer_lifecycle.py' `
  'test\registered\disaggregation\test_pvd_mooncake_lifecycle.py' `
  'test\registered\disaggregation\test_pvd_mooncake_metadata.py' -q
```

Result: `70 passed in 2.38s`.

Amended existing regression used the command recorded above and passed:
`86 passed in 4.37s`.  The exact Ruff scope above reported `All checks passed!`;
`git diff --check` passed with only CRLF-conversion advisories.

## Fix round 2: shared source owners

### Review finding and fix

The round-1 pending indicator conflated two different cases: a callback owned
by another thread is genuinely unresolved, while another transfer still
pinning the same guard is safe and must not retain this terminal transfer's
slot.  That leaked owner A's slot when A and B shared a registration, release
was requested, A completed first, and only B later released the guard.

`ResourceGuard.unpin` now returns an atomic `GuardUnpinOutcome` captured under
the guard lock: `OWNERS_REMAIN`, `RELEASE_NOT_REQUESTED`,
`RELEASE_IN_PROGRESS`, or `RELEASED`.  The manager retains the record only for
`RELEASE_IN_PROGRESS` (or an exception from a callback it started); it releases
the current transfer's capacity for the other outcomes.  This keeps the
event-gated external callback race from round 1 safe while allowing a terminal
owner to free its own slot as soon as another owner remains.  The callback is
still invoked outside manager and handle locks.

### RED

Before this change, the deterministic multi-owner test ran:

```powershell
& 'D:\code\sglang-V100-PVD\.venv\Scripts\python.exe' `
  'test\registered\disaggregation\run_pvd_cpu_tests.py' `
  'test\registered\disaggregation\test_pvd_mooncake_lifecycle.py::test_terminal_owner_releases_its_slot_while_another_owner_remains' -q
```

Expected failure observed:

```text
assert 2 == 1
```

The old code kept both slots after owner A completed, even though owner B was
the only remaining source pin.

### GREEN and regression evidence

The multi-owner test and the event-gated external-release-race test passed
together after the atomic outcome change:

```powershell
& 'D:\code\sglang-V100-PVD\.venv\Scripts\python.exe' `
  'test\registered\disaggregation\run_pvd_cpu_tests.py' `
  'test\registered\disaggregation\test_pvd_mooncake_lifecycle.py::test_terminal_owner_releases_its_slot_while_another_owner_remains' `
  'test\registered\disaggregation\test_pvd_mooncake_lifecycle.py::test_terminal_poll_retains_capacity_while_release_retry_is_running' -q
```

Result: `2 passed in 1.65s`.

Focused lifecycle/adapter/metadata suite (including
`test_pvd_transfer_lifecycle.py`) passed:

```powershell
& 'D:\code\sglang-V100-PVD\.venv\Scripts\python.exe' `
  'test\registered\disaggregation\run_pvd_cpu_tests.py' `
  'test\registered\disaggregation\test_pvd_transfer_lifecycle.py' `
  'test\registered\disaggregation\test_pvd_mooncake_lifecycle.py' `
  'test\registered\disaggregation\test_pvd_mooncake_metadata.py' -q
```

Result: `71 passed in 2.39s`.

Existing regression used the previously recorded four-suite command and
passed: `86 passed in 4.33s`.  The exact Ruff scope recorded above again
reported `All checks passed!`; `git diff --check` found no whitespace errors
and emitted only CRLF-conversion advisories.
