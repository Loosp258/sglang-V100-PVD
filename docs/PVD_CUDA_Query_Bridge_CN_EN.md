# CUDA 查询桥接 / CUDA query bridge

`CUDAPredictionPipeline` 显式组合 prediction-only SGLang draft 与
`CUDALlamaTargetProbe`；具体模型仍由调用者配置。两者和正式 target 执行必须共用
同一把可重入执行锁。scope 在主线程同步执行，不得跨 await；它保存/恢复 CPU
及明确编号的 target/draft CUDA RNG。此措施不隔离不遵守该锁的其他 RNG 使用者。
底层 probe 仍仅支持已声明的 TP1/PP1 Llama/torch_native 子集。

`CUDAPredictionPipeline` explicitly composes the prediction-only SGLang draft and
private CUDA target probe. Model selection stays configurable. Target execution
must honor the same reentrant lock. The main-thread synchronous scope saves and
restores CPU and declared target/draft CUDA RNG states; it never spans an await
and does not isolate unrelated users ignoring the lock. The probe's existing
TP1/PP1 Llama/torch_native capability restrictions remain.

`CUDAProbeSearchSession` 复用既有 Entry/请求/窗口/position/head/version 检查，
只复制所需 Q 行到 CPU，再发送既有 HTTP 检索协议。预留独立 copy budget 后才
启动预测，显式限制 head_dim，routes 和 positions 各不超过 64。逐路复制，
不建立整个层或所有 heads 的 GPU gather buffer。

The CUDA session reuses Entry/request/window/position/head/version validation.
It copies only requested Q rows to host and uses the existing HTTP search wire
format. An independent copy budget is reserved before prediction, with an explicit
head-dimension bound and at most 64 routes/positions each. Routes are copied in
sequence without a full-layer/all-head GPU gather buffer.

预算覆盖显式的 native-dtype host tensor 与 FP32 转换/有限值检查临时空间；
不是 Python 对象、JSON、HTTP runtime 或全部 CUDA workspace 的内存硬上限。
返回的不可变数值行另受上述协议数量限制，调用者负责不无限累积历史结果。

The budget covers explicit native-dtype host copies and FP32/finite-check temporary
storage. It is not a hard allocator cap on Python objects, JSON, HTTP internals or
all CUDA workspace. Returned immutable numeric rows are protocol-bounded; callers
must not retain unlimited historical results.

复制异常仍须完成屏障。屏障失败时 session 保留 source/destination 与 copy budget，
probe 保留 Q owner/预算/执行 lease。RNG 保存或恢复失败、draft 隔离也使 pipeline
保留共享锁并拒绝后续工作；不存在超时自动退款或清理重试。

A failed copy still needs a completion fence. Fence failure retains source,
destination and copy budget, and quarantines probe Q ownership/budget/execution
lease. RNG save/restore failure or draft quarantine also keeps the pipeline's
shared lock and refuses further execution. No timeout refund or cleanup retry
pretends UNKNOWN has completed.

验证：12 个新增 CPU 故障/契约用例，含实际 localhost HTTP + exact index 往返；
CUDA 放置、驱动调用及 GPU RNG 由替身覆盖，**不是 GPU forward、RDMA 或生产服务
验收**。CPU session 的 CPU-only 检查保留。CUDA request controller 与生产装配
是独立后续步骤，不能因为有此入口就声称服务已经开启预测检索。

Evidence: twelve CPU contract/failure cases include real localhost HTTP/exact-index
roundtrip. CUDA placement/driver/GPU RNG are substituted, **not GPU-forward, RDMA or
serving acceptance**. CPU-only session guards remain. CUDA request control and
production assembly are separate steps; this entry point does not activate serving.
