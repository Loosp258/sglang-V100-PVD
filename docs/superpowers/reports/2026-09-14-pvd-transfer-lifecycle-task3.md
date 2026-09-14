# PVD lifecycle Task 3 verification

Date: 2026-09-14. Branch: `codex/pvd-transfer-lifecycle`.
Base: `0aef06296`. Scope: identity authorization and fence protocol only.

## Implemented

- Immutable, strictly parsed `WriteIdentity`: protocol, sender/receiver epochs,
  transfer ID, region ID, allocation generation, destination rank and Entry key.
- `WriteAuthorization` pins on creation and serializes begin/close. Closing is
  not terminal evidence. UNKNOWN, DRAINING and IN_FLIGHT cannot unpin; an
  authorization that began cannot subsequently claim NOT_SUBMITTED. Terminal
  replay is idempotent; failed local cleanup remains retryable. Resource owner
  tokens are object-specific, so duplicate identity objects cannot collapse pins.
- Coordinator stores identities from trusted shard reservation responses, not
  from fence callers. It snapshots destination metadata and rejects descriptor
  substitution, incomplete authorization sets and mismatched identities. Once
  lifecycle metadata is present, missing shard identities cannot fall back to
  legacy reservation semantics.
- Fence validates the exact destination rank set, including several D ranks
  mapped to one V shard. Every reply must carry the matching complete identity.
  Matching pending replies produce `fenced=false`, without business release.
  Missing, malformed, unreachable or stale replies never produce safe success.
- HTTP carries full identities on public and internal routes. The D-facing
  client independently rejects weak or mismatched successful HTTP replies.
- Valid fence closes subsequent retrieve, direct reserve and direct start
  entry points. Forged identities cannot close a known delivery. An unknown
  delivery is tombstoned against a late reserve, but cannot be declared safe.

## Deliberate integration boundary

This is NOT a deployable end-to-end lifecycle fix. Tasks 4-7 remain required.

The current VectorKVStore does not yet own native-terminal authorizations.
`LocalShardClient.fence_delivery()` therefore returns an identity-shaped
`fenced=false / transport_terminal_unverified` reply. It does not call the old
store cancellation/fence method: that method can free V pages without native
terminal proof. It must not be used to synthesize lifecycle-v1 success.

Production epoch generation, destination pin-before-publication, store-side
identity registries and allocator pins belong to Task 4 (V) and Tasks 5-6 (P/D).
One authoritative sender gate per identity remains a registry invariant; unique
resource-owner tokens are defensive accounting, not duplicate-write admission.
The protocol currently consumes identities from the configured trusted shard
endpoints. Process-epoch provenance must be enforced by those worker registries,
not by echoing caller fields. No new callback URLs or Mooncake changes were added.

Legacy reservation paths remain available for intermediate regression coverage
only when neither lifecycle metadata nor worker identities are present. They
cannot yield a successful lifecycle fence. Production Decode callers still need
the Task 6 identity plumbing; startup/admission capability enforcement is Task 7.
The D client checks against its supplied expected identities; Task 6 must obtain
those expectations from saved authorization, not from the fence reply itself.

`test_pvd3.py` now expects rejection of unproven legacy fences. Its two-rank
synchronous FakeTransferEngine teardown explicitly clears the test-only pending
clock to avoid entering the not-yet-migrated production close retry loop. This
does not test or claim safe real D cleanup. The formerly covered unreachable
shard case is retained in the new full-identity suite.

The recently discussed long-lived shared staging pool is not implemented here.

## Test evidence

All commands below run from:
`D:/code/sglang-V100-PVD/.worktrees/pvd-transfer-lifecycle`.
The shared Python runtime is used, but the runner imports worktree source.

### RED / GREEN

The inherited partial Task 3 implementation first passed 27 tests. Its original
RED output is not available in this report; it is not claimed as newly verified.

```powershell
& D:/code/sglang-V100-PVD/.venv/Scripts/python.exe test/registered/disaggregation/run_pvd_cpu_tests.py test/registered/disaggregation/test_pvd_transfer_authorization.py -q
```

After adding regressions and strengthening two existing tests, the same command
produced **14 failed, 25 passed**. Failures demonstrated collapsed resource pins,
lost cleanup retries, acceptance of untyped terminal strings, unsafe legacy
fence side effects, client acceptance of six malformed fence responses, forged
fence side effects, bypass through direct reserve/start, and mutable saved
destination metadata. After fixes: **39 passed**.

Additional concurrency, per-shard failure and HTTP request tests passed. A
second RED run exposed an all-missing authorization fallback and exception-only
handling of a matching pending reply: **2 failed, 53 passed**.

After fixes and moving the pending case from malformed-reply expectations into
its own positive pending test:

```powershell
& D:/code/sglang-V100-PVD/.venv/Scripts/python.exe test/registered/disaggregation/run_pvd_cpu_tests.py test/registered/disaggregation/test_pvd_transfer_authorization.py -q --tb=short
```

Result: **54 passed in 1.91s**.

### Combined CPU regression

```powershell
& D:/code/sglang-V100-PVD/.venv/Scripts/python.exe test/registered/disaggregation/run_pvd_cpu_tests.py test/registered/disaggregation/test_pvd_core.py test/registered/disaggregation/test_pvd3.py test/registered/disaggregation/test_pvd_rails.py test/registered/disaggregation/test_pvd_mooncake_metadata.py test/registered/disaggregation/test_pvd_transfer_lifecycle.py test/registered/disaggregation/test_pvd_mooncake_lifecycle.py test/registered/disaggregation/test_pvd_transfer_authorization.py -q --tb=short
```

Result: **192 passed in 4.77s**. Includes real aiohttp TestServer routes and
clients, real local PVD code and CPU tensors; native RDMA boundaries are fakes.

### Static checks

```powershell
& D:/code/sglang-V100-PVD/.venv/Scripts/python.exe -m ruff check --select E9,F401,F821,I python/sglang/srt/disaggregation/pvd/transfer_authorization.py python/sglang/srt/disaggregation/pvd/protocol.py python/sglang/srt/disaggregation/pvd/coordinator.py python/sglang/srt/disaggregation/pvd/client.py python/sglang/srt/disaggregation/pvd/control_server.py test/registered/disaggregation/test_pvd_transfer_authorization.py test/registered/disaggregation/test_pvd3.py
& D:/code/sglang-V100-PVD/.venv/Scripts/python.exe -m ruff format --check python/sglang/srt/disaggregation/pvd/transfer_authorization.py python/sglang/srt/disaggregation/pvd/protocol.py python/sglang/srt/disaggregation/pvd/coordinator.py python/sglang/srt/disaggregation/pvd/client.py python/sglang/srt/disaggregation/pvd/control_server.py test/registered/disaggregation/test_pvd_transfer_authorization.py test/registered/disaggregation/test_pvd3.py
git -c safe.directory=D:/code/sglang-V100-PVD/.worktrees/pvd-transfer-lifecycle diff --check
```

Results: **All checks passed**, **7 files already formatted**, no whitespace
errors. The Ruff selection includes the repository pre-commit F401/F821 gate,
plus syntax and import ordering. An earlier unscoped Ruff run reported 91
findings, mainly typing modernization/style rules in the selected files; those
rules are not the repository's selected Ruff gate. No full unscoped lint pass
or full pre-commit suite is claimed.

## Self-review and remaining work

Reviewed serialization types, begin/close races, callback lock boundaries,
terminal replay, complete rank aggregation, destination snapshots, stale
incarnations, legacy HTTP peers and direct late-start paths. No independent
subagent review was performed in this continuation. The above staged boundaries
remain explicit; no real GPU/RDMA execution or network-failure validation was
possible here. Next: Task 4, bind V source/target allocations and staging to the
authorization gates and native terminal state without waiting under store locks.
