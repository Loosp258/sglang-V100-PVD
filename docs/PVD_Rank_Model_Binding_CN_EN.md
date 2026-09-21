# Rank runtime → model banks → Req output / rank 运行时与模型输出绑定

Updated / 更新：2026-09-22。

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
