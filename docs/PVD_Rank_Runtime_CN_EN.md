# Owner-polled rank runtime / 协调线程推进的 rank 运行时

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
It supplies no model-forward execution lease, output discard policy, streaming
callback or cache-release hook. Peer loss after a dispatch cannot undo an already
running forward; production Scheduler integration must fail/discard that request
and preserve actual resource ownership. No automatic command retransmission,
dynamic membership, failover or cross-restart recovery is added here. Direct
Exchange users can explicitly retry RESUME, while this runtime's first policy
is fixed-deadline failure for missing messages.

这不是已经接入正式 Scheduler 的服务循环。调用方还需负责轮询、按请求路由消息、
正式 forward 票据、失败后丢弃结果以及真实资源回收。不能把该门控当成 GPU event、
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
