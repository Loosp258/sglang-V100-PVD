# Rank runtime → model banks → Req output / rank 运行时与模型输出绑定

Updated / 更新：2026-09-22。

## Refresh-driver retirement concurrency / 刷新驱动异步回收

The next audit reproduced duplicate `remove()` calls entering the same close
twice and then raising `KeyError`, plus shutdown leaving later requests running
while awaiting the first drain. CPURefreshDriver now refuses overlapping
removal of the same lifecycle; after successful drain it checks the exact
registration object again before deleting it. Failed/cancelled removal retains
the old registration, blocks same-id replacement, and permits an explicit retry.
It never treats cancellation or a close exception as a remote-write fence.

Shutdown closes registration and terminates all registered lifecycles before
its first await. No later request may start a refresh while another request is
draining. Failed shutdown stays closed to new admission and retains the records
needed for retry. A shutdown refused by the live-forward precheck has not yet
entered shutdown. Successful close is idempotent; completed drivers cannot be
reopened. These are owner-thread asyncio rules, not GPU or distributed fences.

复现了重复 remove 导致二次清理/KeyError，以及关闭期间后续请求仍可刷新的问题。
现已拒绝同一生命周期的并发清理，await 后再次核验注册对象；失败/取消保留旧注册，
只有成功排空后才能注册同名新请求。关闭时先禁止新增、停止全部请求，再异步排空；
失败后不重新开放准入，可重试。五个新增测试覆盖并发、同名复用、异常、调用方取消
和关闭重试；仍不代表生产 Scheduler 或真实 RDMA 的清理验证。

Full regression after this follow-up: Windows **1740 passed / 14 skipped**,
WSL **1745 passed / 9 skipped**, with the same three CPU-platform warnings.
Ruff check/format pass. The tests reproduce the original duplicate-close and
shutdown-window defects rather than merely checking newly added flags.
The strict four-case real CPU dual-model matrix was rerun after this driver
change and passed, including slot/KV reuse in all three full-length scenarios.

## Request retirement and reuse / 请求回收与复用

Reproduced a result-scope hole: the CPU dispatcher can retire its lifecycle
permit while the rank result scope still owns its forward ticket and target
lease. `unregister_storage()` now refuses while that shared lease is held.
This is deliberately conservative for the CPU execution context; it is not
a replacement for a GPU event or a native transfer fence.

The real-model fixture now retires a cancelled request before creating its
successor. It reserves otherwise-free capacity through the real allocators,
drains refresh/Delivery ownership, removes the refresh registration, unbinds
storage, clears the mapping, and frees the owned rows/slot. The next request
must receive the same slot and a subset of those freed KV rows. Replaying the
old result and old successful cleanup must leave the successor's output,
mapping, KV tensors, binding and available capacity unchanged. The final
capacity check also ensures no double free. An ambiguous allocator-release
failure is not retried by the fixture. Fixture-owned V stores are closed here;
this is NOT a policy to delete a shared production Entry on every request end.

Report schema v3 requires all four reuse observations in the normal,
lost-RESUMED and cleanup scenarios. The partial-install scenario ends earlier
and does not claim to exercise reuse. The original result processor's streaming
and finish services remain fixture substitutes; production cleanup integration
is still outstanding.

已复现并修复 CPU permit 已清理、rank 结果作用域仍占用时允许解绑 slot 的漏洞。
实模测试通过真实分配器制造容量压力，要求后继请求实际复用旧 slot/KV rows；
检查旧结果回调和重复清理不会更改后继请求。不会手工修改空闲队列来伪造复用。
这一步完善 CPU 资源回收边界，不表示生产 Scheduler 回收或原生 RDMA 已接通。

Validation: the regression failed before the guard and passes after it. Thirteen
new tests, full Windows **1735 passed / 14 skipped**, WSL **1740 passed / 9
skipped** (three existing CPU-platform warnings). The four real-model scenarios
passed in WSL; the three full-length scenarios exercised actual resource reuse,
while partial installation retains its explicitly shorter evidence. Ruff and
diff whitespace checks pass. Skipped hardware tests remain unverified.

## Connected path / 已连接路径

The opt-in **CPU** validation path now connects rank control to the banks actually
read by `ModelRunner.forward` and to the existing authoritative Req result
processor. Previously these were separate validation paths. This is not a
production Scheduler switch and does not enable CUDA sparse mode.

1. `CPURuntimeInstallGroup` owns the exact banks/coordinator used by its runtime
   and local rank participants. Encoded messages pass through bounded in-process
   queues; only the runtime drives installation. Initial admission and Delivery
   installation ACK require every RESUMED, not merely every APPLIED.
2. `CPURankBatchDispatcher` reuses the CPU lifecycle, executor and shared target
   arbiter. It binds lifecycle incarnation, exact group object, request/Entry
   incarnation, committed count and installed epoch. There is one target lease
   and no second committed-token clock. All selected requests wait as a batch.
3. The model still consumes `CPUInstalledPromptView`. Its rank-bound group
   requires the exact active forward permit and checks actual bank payload
   operation id/target count against that permit's installed epoch. Real bank
   readers remain owned across the whole model forward.
4. `CPUScheduleBridge` enters the rank result scope after real synchronous
   executor completion. The original SGLang result processor remains the only
   writer of `Req.output_ids`; the bridge observes it into the lifecycle ledger.
   Rank tickets/target lease remain held until result processing exits.

控制、安装和模型读取使用同一个 coordinator、同一组 bank 和同一安装 epoch。
初始准入及 Delivery 安装确认均需全体 RESUMED。正式 token 仍由原 Req 结果处理器
写入，没有启用原生 speculative generation。新请求不重置旧请求时钟；提前预取、
边界等待和错过窗口时用真实前缀 Q 补查的语义保持不变。

## Ownership / 所有权

- Missing permits or changed group/incarnation/bank epochs refuse reads; no dense
  fallback. Cancellation, loss, bad control and timeout reject that request's row.
  A failed shared forward aborts all; unrelated live rows may otherwise commit.
- Early result callbacks retain both ownership layers. Stop/cancel never frees
  banks. Close refuses an active runtime ticket even after readers drain if
  result processing has not ended.
- Partial result-processor failure aborts without rolling back Req tokens.
  Direct CPU complete/fail callers must still guarantee execution has unwound;
  the bridge checks its executor marker. No CPU boolean is a GPU event/MR fence.

读者、forward 票据和结果提交作用域分别保护自己的生命周期。取消只关闭准入，
不能当成内存可提前回收的证据；结果处理异常不回滚已经写入或发出的正式 token。

## Executed evidence / 已执行证据

15 new focused cases cover exact bank/permit identity, single-writer observation,
early callbacks, cancellation/loss/timeout/bad messages/retraction, partial result
failure, refresh ordering, changed bindings, and dropped RESUMED with exact retry.
Unit-test constructor/bridge doubles do not count as real model execution.

Full regression: **Windows 1646 passed / 14 skipped**, **WSL 1651 passed /
9 skipped**, with three existing platform warnings. Ruff check/format and
`git diff --check` pass. Hardware-skipped cases remain unverified.

These gates **did execute actual CPU model forwards in WSL**:

```bash
PYTHONPATH=python python test/registered/disaggregation/run_pvd_draft_cpu_smoke.py --rank-runtime-decode
PYTHONPATH=python python test/registered/disaggregation/run_pvd_draft_cpu_smoke.py --rank-runtime-loop
```

The first uses a random tiny Llama target and fixed draft-fixture ids, actual
Req/ScheduleBatch/result processing, 11 batch forwards including injected failure,
and 21 attention comparisons (maximum absolute error about `2.38e-7`).

The second uses two independent random tiny Llamas, a shared toy tokenizer,
private draft pools, real target probe Q, HTTP V search/Delivery, rank installation,
actual target sparse attention and actual Req result processing:

- 21 attention comparisons; maximum absolute error about `3.58e-7`.
- Four sparse shard deliveries / 1600 payload bytes, no local pack callback.
- Installation before Delivery ACK; receive budgets/model pools restored.
- Independent request clocks, reordering, wait-all, retraction, length limit and
  injected real attention failure exercised.
- Two draft forwards for one predictive refresh; boundary fallback does not call
  draft. Draft preserves target state/RNG and restores its private pool capacity.

组合闭环已实际执行，不是拼接独立测试结果。但所有 rank/bank 位于同一进程，
目标模型为 **CPU TP1**，不是实际 TP2 collective。V 控制面使用 localhost HTTP；
载荷仍为 **fake in-process byte copy**。原独立 CPU 子进程控制测试仍是另一项证据。

## Remaining gates / 尚未完成

### Strict automated entrypoint / 严格自动验收入口

```bash
PYTHONPATH=python python test/registered/disaggregation/run_pvd_rank_model_acceptance.py --timeout-seconds 300
```

This entrypoint starts the real `--rank-runtime-loop` child with the current
Python interpreter, a fresh run id and this checkout on PYTHONPATH. It requires
one bounded framed report, zero exit status, the matching schema/run id, complete
rank/model/Req/draft/Delivery evidence, finite numerical errors and independent
request counts. A top-level `passed` without the required branch cannot pass.
Import failures, timeouts, nonzero exits, missing/duplicate frames/keys and
optimized Python (disabled asserts) are errors, never skips. Unknown smoke flags
are now refused rather than silently running the baseline only.

The strict entrypoint executed successfully in WSL against the actual dual-model
loop. 56 focused tests validate the report/CLI contracts; those unit tests use
synthetic reports and are not counted as model execution. This is a regression
gate for a trusted test fixture, not cryptographic attestation of an arbitrary
subprocess. Production GPU/RDMA claims must remain explicitly false.

After this step: full Windows **1702 passed / 14 skipped**, WSL **1707 passed /
9 skipped**; Ruff check/format pass. Hardware skips remain unverified.

入口使用当前解释器实际启动模型闭环，检查本次运行标识、完整证据和数值结果；
缺少环境、超时、报告不全、拼错开关或关闭断言均失败，不返回“跳过/通过”。
严格入口仍只验证 CPU TP1、本地 rank 控制和 fake payload copy。

No production Scheduler loop, real streaming/cache-release integration, CUDA
sparse attention/current-next banks, distributed model TP, native Mooncake sparse
execution, RDMA completion, V100S CAGRA or model-quality/performance gains are
established. No production capability flags were relaxed.

Next hardware-independent work: broader integration fault injection through the real result
processor (lost receipts, bank-swap failures and cleanup recovery). Hardware
gates remain pending, not silently enabled.

## Model-path fault acceptance and delayed-RESUMED fix / 实模故障与迟到回执修复

The strict CLI now accepts `--fault none|lost-resume|install|cleanup|all`:

```bash
PYTHONPATH=python python test/registered/disaggregation/run_pvd_rank_model_acceptance.py --fault all --timeout-seconds 300
```

`all` launches each case in a fresh child process with its own run id. The timeout
applies per child. One failure fails the matrix; no partial success report is
printed. Report schema v2 binds the expected fault mode and requires every
scenario-specific observation plus cleanup evidence. Unknown/unexecuted branches
cannot be substituted for a requested case.

- `lost-resume`: drops one real local RESUMED message at the first refresh. No
  new model forward/Req token or Delivery installation ACK may escape. Explicit
  delayed delivery of that exact reply must let the pending round finalize.
- `install`: raises **after** the second bank has actually swapped. The old
  request remains aborted, its output unchanged and its Delivery unacknowledged
  as installed; an unrelated request continues through the real result processor.
  This case deliberately ends early and does not claim the normal late-fallback
  or four-delivery checks ran.
- `cleanup`: after actual forward completion but before result commit, tries to
  close the cancelled member's group. The live ticket must prevent bank/budget
  release. The real result processor discards its row; explicit close after the
  result scope drains succeeds. Final fixture cleanup must restore budgets/pools.

Fault injection reproduced a real integration defect in **both**
`CPUDecodeLifecycle.try_install` and `CPURefreshDriver.progress`: after APPLIED,
the coordinator advances `next_boundary` (e.g. 4 → 8), even though the request
at token 4 may still await RESUMED. Both callers used that next boundary to
decide whether to finalize the old round, leaving the request stuck after the
missing reply arrived. The fix uses `CPUPrefetchRequest.pending_install_boundary`
from the exact ready epoch. This does **not** bypass RESUMED, advance the D clock,
reset deadlines or replace the in-flight query. Two regression cases exercise
manual lifecycle and automatic-driver finalization independently.

故障注入在单测及真实双模型路径中都复现了边界错误：APPLIED 后 coordinator 的
下一边界已是 8，而 D 在 token 4 等待本轮 RESUMED；此前生命周期与自动驱动都
因此不再尝试完成本轮。现改为以 ready epoch 的边界判断，收到全部 RESUMED 后
才能完成交付、清除本轮等待。丢包等待策略和正式输出时钟均未改变。

These remain CPU TP1 / local rank messages / fake payload transfers. An exception
after CPU bank swap is not native RDMA fault injection. No production capability
gate is relaxed and no GPU completion or remote MR cleanup is inferred.

Executed: the entire four-case strict matrix passed in WSL after the fix.
Normal/lost-resume/cleanup each performed 21 attention comparisons (max error
about 3.58e-7); partial-install performed 10 (about 2.38e-7). All case cleanup
checks passed. 20 new regression/report tests; full Windows **1722 passed /
14 skipped**, WSL **1727 passed / 9 skipped**. Ruff check/format pass.
