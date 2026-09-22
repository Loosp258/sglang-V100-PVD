# CUDA 多请求 runtime batch / Multi-request CUDA runtime batch

`CUDARankBatchExecutor` 将现有 `RankBatchDispatcher` 绑定到同一 CUDA model
consumer，接收明确的 `(group, request slot, committed D tokens)` 成员。它不新增
token 时钟、sampler 或 Req 写入器；调用者传入现有同步 forward 和结果处理函数。

The explicit executor binds the existing rank batch dispatcher to one CUDA model
consumer using `(group, request slot, committed D tokens)` members. It introduces
no token clock, sampler or Req writer. Callers supply existing synchronous model
forward and authoritative result processing callbacks.

所有成员均 ready 才获取任何 permit，不移除迟到成员拼一个小 batch。成员可以
处于不同刷新轮次，但必须来自相同的已绑定 worker。重复请求/slot、错误 count、
async callback 或超出 batch 上限均拒绝。

All selected members must be ready before any permit is acquired; late members
are not silently dropped. Refresh rounds may differ, but bound worker identities
must agree. Duplicate requests/slots, invalid counts, async callbacks and oversized
batches are refused.

实际 GPU compute/Prompt readers 完成后才进入结果处理；整个结果处理期间仍持有
全部 runtime permits、共享 target 锁及模型分配器 lease。若任一 runtime 拒绝
结果，保守丢弃整个未提交 batch；不会回滚或重试结果函数此前已提交的 token。
任一完成屏障 UNKNOWN 保留全部 owner/permit/锁，close 不能伪装成功。

Result processing begins only after model/Prompt-reader completion. All runtime
permits, the shared target lock and allocator lease stay held through processing.
If any runtime refuses its result, the entire uncommitted batch is conservatively
discarded. Tokens already committed by a subsequently failing result callback are
not rolled back/replayed. UNKNOWN completion retains owners/permits/locks.

10 个 CPU 策略/实际 tensor 数学用例覆盖双请求 forward、wait-all、迟到 peer
故障、模型/结果处理失败、所有权保留及入口拒绝。CUDA driver/放置被替代，没有
真实 GPU 模型、TP2、RDMA 或生产 Scheduler 执行证据。此组件只处理一个 TP1
worker 上多个请求，不把多个请求冒充多个模型 TP ranks。

Ten CPU policy/real-tensor math tests cover two-request forward, wait-all, late
peer loss, model/result failures, ownership retention and admission refusals.
CUDA placement/driver are substituted: no real GPU model, TP2, RDMA or serving
evidence. Multiple requests on one TP1 worker are not multiple model TP ranks.
