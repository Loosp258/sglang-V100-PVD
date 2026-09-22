# CUDA Decode 调度绑定 / CUDA Decode Scheduler binding

## 已接通的范围 / Implemented scope

`CUDADecodeSchedulerBinding` 将已经构建好的 `CUDARefreshDriver`、
`CUDARankBatchExecutor` 和实际 model-pool guard 显式绑定到 Scheduler。
普通非 overlap Decode 循环只在存在这一绑定时启用新分支；未绑定的生产服务
仍走原完整 Prompt 刷新路径。它不加载模型、不创建请求控制器、不替换 attention
backend，也不是一个可以直接启动预测检索的 CLI 开关。

The explicit binding connects an already assembled driver, executor and model-pool
guard to the ordinary non-overlap Decode loop. Unbound serving retains full-Prompt
refresh. It does not load models, create request controllers, replace attention
backends or provide an end-to-end predictive-retrieval launch switch.

绑定要求实际安装的 backend、consumer、请求池、KV 池、receiver manager、共享
target lock/arbiter 身份一致。当前仅接受 TP1/PP1/DP1/CP1、page size 1、关闭
CUDA graph/overlap/native speculation/Decode radix/offload/HiSparse。请求上限
不得超过 driver、executor 或 attention workspace 的容量。整个绑定生命周期
持有实际 model-pool guard；关闭后保留关闭标记，不能自动退回普通 attention。

The installed backend, consumer, request/KV pools, receiver manager and target
lock/arbiter must match exactly. Supported scope is TP1/PP1/DP1/CP1, page size 1,
with CUDA graphs, overlap, native speculation, Decode radix/offload and HiSparse
disabled. Scheduler capacity cannot exceed driver/executor/consumer bounds. A
pool guard is pinned for the binding lifetime; a closed marker prevents fallback.

## 队列和循环 / Queues and loop

1. 最终 waiting queue 中的新请求必须已完成完整 Prompt 导入、注册 retirement，
   并将实际 receiver session 交给 CUDA driver。缺少装配的新请求留在 waiting，
   不消耗本轮 batch admission 名额，不重置旧请求时钟或预取任务。
2. running batch 在 `update_running_batch` 分配下一 token KV **之前**检查所有
   请求的刷新边界；任一请求尚未可运行则 wait-all，不部分 forward。
3. forward 通过 `CUDAScheduleBridge`，仍调用原 `Scheduler.run_batch` 和原
   `Scheduler.process_batch_result`；不跳过外围 metrics/health，也不重复处理输出。
4. 驱动在 forward 之间轮询，暂停引擎时仍可推进清理。持有待清理请求时不进入
   原 idle 内存自检，避免将尚未排空的合法资源误报为泄漏。
5. 已停止请求在轮询可能删除记录前发送一次失败响应。KV 压力的首版明确策略为
   中止当前 batch 并异步排空，不调用原地 retraction；不是“自动恢复/重算”支持。
   稀疏控制器、完整接收器和 consumer lease 排空后，才归还原请求池资源。

New waiting requests need a completed full-Prompt import, retirement owner and
receiver-to-driver handoff. Unassembled arrivals remain waiting without spending
admission slots or resetting peers. A wait-all check runs before the next generated
KV allocation. The bridge preserves the original Scheduler forward and result
wrapper. Polling progresses between forwards and while paused; pending owners
suppress idle pool checks. Already-stopped requests are reported before retirement
can erase their records. Initial capacity policy aborts the current batch and drains
asynchronously; it does not implement in-place retraction or automatic recovery.

## 尚未解决的代码边界 / Remaining implementation boundaries

当前生产配置要求 D TP2/TP4，V 完整 KV 存储分片是 TP2；本绑定及实际 CUDA
consumer 目前仅 TP1。这不是硬件测试能补上的差异。`packed_transfer_slices`
当前仅表达一个 V shard 到一个 D rank；D TP1 接收 V TP2 需要多源 fan-in 的
descriptor、交付身份、完成证明和 head 映射，不能仅放宽 CLI 参数。

Standard serving currently requires D TP2/TP4 and V uses two storage shards, whereas
the actual CUDA consumer/binding is TP1. This is an implementation mismatch, not
merely missing hardware evidence. Full delivery currently maps one V shard to one
D rank. V TP2 to D TP1 requires multi-source descriptors, delivery identities,
completion proof and head mapping, not just a relaxed command-line check.

生产启动/每请求 admission 工厂、真实模型 TP、原生 CAGRA 仍待实现或接通。
测试中显式装配组件不代表这些工厂已经存在。新的 waiting admission 若没有工厂
提供 controller 将一直等待，因此当前不自动启用此绑定。

Production startup/request-admission factories, actual model TP and native CAGRA
remain implementation work. Test assembly does not imply a factory exists. Without
one, unassembled arrivals would remain waiting, so this binding is not auto-enabled.

## 验证边界 / Evidence boundary

新增 25 个 CPU 用例覆盖显式身份、拒绝不支持配置、batch 容量、新请求跳过、
分配前 wait-all、OOM 中止与排空、停止通知顺序、原循环轮询和结果路由、关闭标记。
其中执行当前 `decode.py` 的真实方法体，但 Scheduler 外围、模型和传输为 double。
WSL 定向 59 项通过，包含已有实际 Req/池/结果处理器测试；没有执行 GPU/RDMA。

Twenty-five CPU cases cover identities/refusals, capacity, admission, pre-allocation
wait-all, capacity abort/drain, stop notification, loop/result routing and closure.
They execute current Decode method bodies with Scheduler peripherals, model and
transport doubles. Fifty-nine focused WSL cases pass, including existing real
Req/pool/result-processor tests. No GPU or RDMA execution is claimed.

Windows 全量 **2265 passed / 29 skipped**；严格 v5 CPU 四场景再次全部通过，
完整场景含 21 次 attention 对照，最大误差约 3.58e-7。矩阵是已有 CPU reference
端到端回归，不是新 CUDA 绑定的 GPU 端到端证明。

Windows full regression: **2265 passed / 29 skipped**. All four strict v5 CPU
scenarios pass again; full scenarios include 21 attention comparisons at about
3.58e-7 maximum error. This is reference-path regression, not GPU end-to-end
evidence for the new CUDA binding.
