# Owner-polled rank runtime / 协调线程推进的 rank 运行时

Latest integration / 最新接入：见 [rank/model/Req 绑定](PVD_Rank_Model_Binding_CN_EN.md)。
2026-09-22 已连接 opt-in CPU 模型执行路径；下文未接模型的描述是此前步骤的历史
边界。生产 Scheduler、GPU 和 RDMA 仍未完成，不能外推 CPU 验收结果。

## Implementation / 实现

`RankInstallRuntime` drives `RankInstallExchange` with three states of progress:
prepare/park → install/apply → resume/resumed. It never owns tensors, registered
memory, a model worker or a Scheduler. It does not register a production backend.

- Transport callbacks call `post(bytes, peer_rank, peer_epoch)` or `peer_lost`.
  They only enqueue bounded immutable bytes or latch failure under a lock.
  Rank membership and worker epochs are captured from trusted channel setup;
  old/foreign channel callbacks are refused without cancelling this incarnation.
- The owner calls `begin(..., timeout_seconds=...)` and bounded `progress()`.
  Only this thread parses messages, changes the coordinator, and emits commands.
  Both maximum pending event count and pending byte count are explicit parameters.
- One finite monotonic deadline covers the whole round through all RESUMED ACKs.
  Duplicates never extend it. Missing PREPARED, APPLIED or RESUMED fails the whole
  request at the deadline; it never promotes an incomplete round.
- Global dispatch must check **runtime.can_decode**, not the lower-level gates.
  Pending notifications block dispatch until processed. Exact last-completed
  round replays are ignored; replay history stays bounded to that one round.
- Overflow, malformed bound-channel messages, peer loss, send failure, clock
  failure and rank-reported failure close admission. A send exception may mean
  the command already escaped; it cannot be treated as "nothing happened".
- Failure signals `stop_peer(rank, request_identity, reason)` to **all** bound
  ranks, even those whose PREPARED was never received. Failed stop enqueue remains
  visible in `stop_notifications_pending` and retries on subsequent progress.
  Reentrant progression/stop callbacks cannot recursively advance the protocol.

Background callbacks must not mutate the exchange. The runtime exclusively owns
that exchange; bypassing it invalidates the admission contract. `send` and
`stop_peer` must enqueue nonblocking, request-scoped control notifications. A
successful stop callback is NOT a peer stop ACK or resource cleanup evidence.
`resource_cleanup_proven` intentionally remains false. GPU/MR owners must still
obtain their own completion/fence evidence before freeing anything.

该模块把 rank 协议推进接到有界回调队列：后台只入队，协调线程统一解析和推进。
一轮从准备到 RESUMED 使用固定截止时间，重复消息不续期。失联、溢出、坏消息、
发送结果不明或超时都会关闭请求准入，并通知全部绑定 rank，包括 PREPARED 回包
丢失的 rank。停止通知失败保留为待重试；通知成功不代表资源已释放。

## Integration boundary / 接入边界

This is an explicit nonblocking control adapter, not a background daemon. Its
owner must poll it in a scheduling loop and route messages to the correct request.
It now supplies the owner-local forward admission ticket described below, but
not target-worker ownership, a model executor, streaming or cache-release hooks.
Peer loss after dispatch cannot undo a running forward; production Scheduler
integration must apply the discard decision and preserve actual resource
ownership. No automatic command retransmission,
dynamic membership, failover or cross-restart recovery is added here. Direct
Exchange users can explicitly retry RESUME, while this runtime's first policy
is fixed-deadline failure for missing messages.

这不是已经接入正式 Scheduler 的服务循环。调用方还需负责轮询、按请求路由消息、
目标执行器/共享模型互斥、执行拒绝结果提交的决定以及真实资源回收。不能把票据当成 GPU event、
RDMA fence 或已经运行的 forward 的撤销证明。当前运行时丢失消息后采用固定超时
失败，不擅自引入重传、换 rank 或跨进程重启恢复策略。

## Validation / 验证

27 focused tests exercise producer threads, bounded polling, fixed deadlines in
all phases, lost resume ACK, capacity, malformed events, stale channels, send
failure after publication, callback reentry, stop retry and independent requests.

Five additional real spawn-process scenarios exercise this runtime against the
actual per-rank CPU bank participants (the earlier manual driver remains as an
independent reference):

```bash
python test/registered/disaggregation/run_pvd_rank_install_cpu_smoke.py --runtime --ranks 2
python test/registered/disaggregation/run_pvd_rank_install_cpu_smoke.py --runtime --fault install
python test/registered/disaggregation/run_pvd_rank_install_cpu_smoke.py --runtime --fault exit
python test/registered/disaggregation/run_pvd_rank_install_cpu_smoke.py --runtime --fault lost-resume
python test/registered/disaggregation/run_pvd_rank_install_cpu_smoke.py --runtime --fault lost-prepared
```

The fixture pumps real local pipe bytes between independent CPU processes;
runtime callbacks themselves only enqueue. Deadlines use an injected monotonic
clock to deterministically test failure, not to measure latency. Missing PREPARED
still stops both ranks; a lost RESUMED reply leaves global dispatch closed even
when that peer has locally reopened. Live rank bank budgets return to zero after
explicit CPU cleanup. A terminated rank's cleanup is not inferred from its exit.

实际子进程验证包含正常两轮、换 bank 后异常、进程退出、RESUMED 回执丢失和
PREPARED 回包丢失。真实 pipe 控制消息不等于网络数据面/RDMA；截止时间由测试
时钟注入，不是性能测量。所有存活 rank 显式关闭后预算归零，进程退出本身不作为
被终止 rank 的资源清理证据。GPU、真实模型 TP、CAGRA 和生产 Scheduler 仍未验收。

Full regression: **Windows 1590 passed / 14 skipped**, **WSL 1595 passed /
9 skipped**. All five runtime process scenarios executed in both environments.
New source/tests pass Ruff. Skipped hardware cases remain unverified.

## In-flight forward ticket / 在途 forward 票据

`begin_forward(committed_tokens)` returns an owner-local immutable permit bound
to the request identity, installed epoch, committed-token snapshot and unique
execution id. One runtime can hold only one such permit. Starting a fresh round
while a forward is owned is refused; a previously launched refresh can still
receive PREPARED/PARKED while the old forward runs. The runtime must not issue
INSTALL until the permit is retired, even if every rank already reports PARKED.

After **all real execution and readers have drained**, the owner calls
`finish_forward(permit, readers_drained=True, succeeded=...)`. Those flags are
assertions by the execution owner, not proof manufactured by the runtime. It
processes pending control while retaining the permit, rejects foreign/copied/
replayed completions, and returns whether the result may be accepted. Cancellation,
timeout, peer loss, bad pending events or execution failure reject late output.
Cancellation alone never clears the permit. Apply the decision synchronously
on the owner thread; this API does not append tokens or advance the formal D-token
count. The existing sampler/Req result processor must remain the sole writer.

The ticket owns **admission metadata only**. The caller must independently hold
target-worker execution ownership and all tensor/bank/MR reader leases until
their actual completion. No booleans or process-local ticket substitute for a
native event/fence. This step does not wire production Scheduler or attention.

在途票据绑定请求、已安装轮次和正式 token 计数快照。每个请求一次只允许一个
forward；已启动的刷新仍可收准备回执，但票据未退还时禁止发 INSTALL。取消、
超时或失联只关闭准入，不自动退还执行票据。只有执行方确认所有实际执行/读者
结束后，才可完成票据并获得“接受或丢弃输出”的决定。接口不写 token，不推进
正式输出计数，不能替代目标模型互斥、tensor 所有权、GPU event 或 MR fence。

13 new unit cases cover duplicate dispatch, exact completion ownership, reader
drain refusal, deferred INSTALL, queued failures and cancelled/late outputs.
All six runtime subprocess scenarios now hold real CPU bank reader scopes in
independent processes, prepare a refresh, drain the readers, and verify that the
host execution ticket still blocks INSTALL until explicitly finished. These
reader scopes model execution ownership; this test does not run a model forward.
The new scenario cancels while the ticket is in flight and verifies output refusal:

```bash
python test/registered/disaggregation/run_pvd_rank_install_cpu_smoke.py --runtime --fault forward-cancel
```

Full regression after this step: **Windows 1604 passed / 14 skipped**, **WSL
1609 passed / 9 skipped**. All ten manual/runtime process scenarios execute in
both environments. Ruff check/format pass on the changed Python files.

## Batch wait-all and result scope / 批量准入和结果提交作用域

`RankBatchDispatcher` composes the real rank runtimes with one explicit shared
`TargetExecutionArbiter`. It accepts a bounded, immutable tuple of
`RankBatchMember(runtime, committed_tokens)`. Membership must have unique request
ids/runtimes and the same trusted rank/worker-epoch bindings. Each request keeps
its own installed epoch, refresh deadline and committed-token snapshot. Joining
a batch neither starts a refresh nor resets another request's clock.

All selected requests must pass admission before any permit is acquired. A
missing initial installation, refresh boundary wait, missing RESUMED or queued
control event blocks the **entire selected batch**, not a silently filtered
subset. If a producer reports failure between preflight and permit acquisition,
only permits acquired by this not-yet-returned `begin()` are rolled back. No
forward has escaped at that point; this is not a cancellation/drain shortcut.

After the actual whole-batch execution and readers drain, the caller enters
`processing(ticket, readers_drained=True, succeeded=...)`. It validates all
permit ownership before changing any member. Invalid/copy/replayed completion
or a missing drain assertion keeps all tickets and the shared target lease.
The scope yields dispatch-order `RankBatchDecision(permit, accepted)` records:

- A failed shared forward rejects every result.
- With a successful forward, a failed/cancelled request rejects only its row;
  unrelated live requests may commit their normal sampled results.
- All tickets and the target lease remain owned through result processing.
  INSTALL cannot run during this scope, and single-request `finish_forward`
  cannot steal a batch-owned permit.
- The caller must use the existing authoritative result writer synchronously,
  without awaiting or pumping callbacks. Decisions are not reusable permits.
  Notifications after the decision snapshot apply at the next owner poll; no
  already committed token is rolled back. A result-processor exception aborts
  all members and retires the drained execution, never retries partial output.

该批量适配器实现整批 wait-all，失败时不偷偷筛选子 batch；每个请求保留自己的
刷新时钟。正式输出仍由原结果处理器写入，适配器只提供按原 batch 顺序排列的
接受/丢弃决定。执行和真实读者全部结束前不能完成票据；取消、超时、失联不能
提前释放整批执行锁。结果处理作用域结束前保留全部票据，防止 INSTALL 穿过
提交阶段。若结果处理器部分写入后异常，终止整批而不回滚已发出的 token。

The focused tests use actual runtimes/exchanges. An additional integration case
uses two requests with two real CPU banks each: the runtime controls the exact
same coordinator/epochs as its participants, real readers block PARKED, host
tickets block INSTALL after readers drain, and the result scope keeps it blocked
until exit. A refresh of request A leaves B's full-prompt bank unchanged; explicit
close returns all four bank budgets to zero. This is an in-process CPU control
and bank test, **not a model forward or a cross-process batch experiment**.

该接点尚未接入生产 Scheduler、正式 Req 结果处理器或 GPU bank。原有
`CPUBatchDispatcher` / `CPUScheduleBridge` 的模型验收路径仍独立存在，不能把两套
路径的通过记录拼接成“rank 控制已驱动模型执行”。下一步需要显式绑定同一
request/incarnation/Entry、安装 epoch 和实际执行 bank 后，验证结果处理器接入。
MR fence、原生传输、模型 TP 和 GPU 生命周期仍需要各自实现/验收。

Evidence for this step: **27 new tests**, full regression **Windows 1631 passed /
14 skipped**, **WSL 1636 passed / 9 skipped** (three existing platform warnings).
All ten previous independent-process scenarios still execute. Ruff check/format
and `git diff --check` pass. The new bank/batch case is in-process; skipped GPU
cases remain unverified. Refactoring also preserves single-forward stop callback
ordering: notification cannot reenter and complete the same execution twice.
