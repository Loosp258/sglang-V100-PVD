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
& 'D:\code\sglang-V100-PVD\.venv\Scripts\python.exe' -m ruff check --select E9,F821 ...
```

Result: `All checks passed!`  `git diff --check` also passed (only CRLF
advisories were emitted).

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
