# Draft 完成证明与隔离 / Draft completion and quarantine

## 已实现 / Implemented

`ModelExecutor.drain()` 是必须实现的接口，不再把缺失屏障当作成功。
实际 `DraftForwardAdapter` 支持 CPU 同步执行或明确编号的 CUDA device；
CUDA 使用指定设备的 `torch.cuda.synchronize(device)`。它只是本地计算完成
屏障，**不是 RDMA 远端 WRITE 完成证明**。

`ModelExecutor.drain()` is mandatory. The real adapter supports synchronous CPU
execution or an explicitly indexed CUDA device, fenced with device-wide
`torch.cuda.synchronize(device)`. This is **not** a remote RDMA WRITE fence.

正常和异常 forward 都保留 batch/已返回的 output，直到完成屏障成功。
屏障失败时保留这些引用并永久拒绝该适配器的后续 forward/drain；不自动重试。
健康路径不累积 batch。显式设备编号避免线程当前设备变化导致等待错误 GPU。

Both successful and failed forwards keep the batch and any returned output until
the fence succeeds. Failure retains these owners and permanently refuses further
forward/drain calls; there is no automatic retry. Healthy calls accumulate no
batches. Explicit device indices avoid fencing the wrong thread-current device.

分支释放顺序固定为：计算完成 → 清空私有请求映射 → 映射清空完成 → 归还 KV 行
和请求槽 → 分配器操作完成 → 才退还预算/并发名额。任何阶段失败都不可再次执行
部分 free。错误数量的 KV 分配也先记录所有已返回行，随后受同样的清理规则约束。

Branch retirement orders compute completion, private-map clearing, clear completion,
KV/request free, allocator completion, then budget/admission refund. No failed
partial free is retried. Even malformed allocation counts record all returned rows
before rejection, so cleanup retains ownership of every returned row.

失败 forward 在释放共享执行锁**之前**清理；清理失败在锁内隔离整个 provider。
已进入分支但尚未执行的 peer 同样不得再碰共享模型/分配器。隔离表保留实际 handle，
不只保留错误字符串；所有相关预算继续占用。当前无原地恢复 API，需停止并重建
独立 worker；不能通过修改内部状态或清零预算“恢复”。

A failed forward retires **before** releasing the shared execution lock. Cleanup
failure quarantines the entire provider under that lock; already-admitted peers
cannot execute on shared model/allocator state. The quarantine registry retains
actual handles, not just error strings, and keeps their reservations. No in-place
recovery API exists: stop/reconstruct the isolated worker, never reset flags or
accounting to pretend completion was proved.

## 验证边界 / Evidence boundary

`test_pvd_draft_completion.py` 用 CPU 故障注入验证三个释放屏障、共享 peer 拒绝、
引用保留、不重试、指定设备传递及异常分配清理；CUDA 调用被替代，**不证明 GPU
执行安全**。另以现有严格 CPU 实模矩阵检查正常执行、取消及回收没有退化。

The completion tests inject failures on CPU: all three retirement fences, peer
refusal, owner retention, no retry, exact device propagation and malformed-allocation
cleanup. CUDA calls are substituted, **not evidence of GPU execution safety**.
The existing strict real-model CPU matrix separately checks normal execution and
retirement regressions.

设备级同步是保守正确性基线，可能等待同卡其他任务；没有证明延迟隐藏或计算重叠。
CUDA query → HTTP 的有界快照、生产 Scheduler 组装和硬件验收仍未完成。

Device-wide synchronization is a conservative baseline and may wait for unrelated
work on the same GPU. No latency-hiding or overlap claim follows. Bounded CUDA-query
HTTP snapshots, production Scheduler assembly and hardware acceptance remain open.
