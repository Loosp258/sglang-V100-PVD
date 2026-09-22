# CUDA batch 正式结果绑定 / Authoritative CUDA batch result binding

## 已接通 / Implemented

`CUDAScheduleBridge` 显式绑定已经注册的 CUDA 请求、对应的 runtime group、
实际 pool owner、`ScheduleBatch` 和原来的 Decode 结果处理器。使用已有
`CUDARankBatchExecutor`，全 batch permits、目标锁和池 lease 贯穿 forward
完成到结果处理结束；不增加 sampler，也不向 Req 追加 token。

The explicit bridge binds registered CUDA requests, their exact runtime groups,
the actual pool owner, ScheduleBatch and the original Decode result processor.
It uses CUDARankBatchExecutor: all batch permits, the target lock and pool lease
cover forward completion through result processing. It adds no sampler and
never appends tokens to Req.

正常 `process_batch_result_decode` 入口仅在 batch 携带显式 CUDA bridge 时
进入新检查；无 bridge 时原服务行为不变。CPU bridge 与 CUDA bridge 不可混用。
首次结果回调必须来自 bridge.run 的实际 reader-drained scope，不能提前提交。
调用前检查完整成员顺序、Req 身份、slot、Prompt 和正式输出未变化；调用后检查
每个 Req 恰好新增其已经采样的一个 token。采样结果先转为不可变证据，不能通过
处理期间修改 `next_token_ids` 掩盖写入错误。

The normal result entrypoint changes behavior only for an explicit CUDA bridge;
unbound serving and the existing CPU bridge keep their paths. A result callback
must occur inside run's reader-drained scope, never before forward completion.
Membership/order, Req identity, slot, Prompt and output history are checked before
processing. Afterwards every Req must contain exactly its one sampled token.
Sampled IDs are frozen before processing so mutating the input result cannot
hide an incorrect authoritative write.

部分提交后处理失败时停止整个 batch，但不回滚已经正式追加的 token，不重试
这次结果。已完成 bridge 可由同一 executor/driver 的下一次 dispatch 替换，
支持连续复用同一个 ScheduleBatch；旧结果仍被当前 bridge 拒绝。未完成或失败
的 bridge 不能被覆盖来绕过所有权。

A partial-commit failure stops the entire batch without rolling back committed
tokens or replaying the result. A completed bridge may be replaced by the next
dispatch on the same executor/driver, allowing continuous ScheduleBatch reuse.
Old results remain refused; active or failed owners cannot be overwritten.

## 测试与限制 / Tests and limits

新增测试涵盖两个请求的实际 CPU attention 数学、runtime permits、pool pin、
真实源码结果入口，以及重放、早到、成员变化、异步 copy 拒绝、部分提交失败、
篡改采样证据和连续 batch 复用。WSL 额外执行真实 Req、ScheduleBatch 和
SchedulerBatchResultProcessor；Windows 缺少 serving 依赖时显式跳过该用例。

Tests exercise actual CPU attention math, runtime permits, pool pins and the
source result entrypoint, including replay, early results, membership changes,
unsupported async copies, partial commits, altered sampling evidence and batch
reuse. WSL additionally executes real Req, ScheduleBatch and
SchedulerBatchResultProcessor. Missing Windows serving dependencies produce an
explicit skip, not a claimed pass.

这仍是显式 TP1、普通 Decode、non-overlap 入口，不自动开启生产 sparse mode。
生产 startup factory、waiting-queue/初始接收挂接、原 allocator 的退还协调、
实际多进程 TP 和 native CAGRA 仍有实现任务。GPU forward、RDMA 与质量/性能
验收没有被 CPU/fake 测试替代。

This remains an explicit TP1, ordinary Decode, non-overlap entrypoint, not an
automatic serving sparse-mode switch. Startup factory, waiting-queue/initial
receive hooks, original allocator retirement coordination, real model TP and
native CAGRA still require implementation. CPU/fake tests are not GPU forward,
RDMA, retrieval-quality or performance acceptance.

最终证据 / Final evidence:

- Windows full regression: **2181 passed / 26 skipped**.
- WSL CUDA refresh/result plus existing CPU result-bridge coverage:
  **60 passed**, three existing CPU-platform warnings.
- Strict v5 real-model CPU acceptance: all four cases pass again after the
  result-entrypoint change (normal, delayed RESUMED, partial install, cleanup).
  This checks the existing CPU model path, not a CUDA model forward.
