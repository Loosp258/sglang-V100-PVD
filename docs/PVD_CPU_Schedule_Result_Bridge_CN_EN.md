# CPU ScheduleBatch 结果接点 / Result bridge

## 已实现 / Implemented

`CPUScheduleBridge` 将实际 `Req` / `ScheduleBatch` 与受控 CPU batch 执行票据绑定。
正常 `SchedulerBatchResultProcessor.process_batch_result_decode` 是唯一正式输出写入点；
bridge 在原处理器写入后校验并更新自己的观察账本，不向 Req 再写一个 token，
不重新 argmax，不把 P 首 token 计入 D 时钟。未绑定 bridge 的路径仍调用原处理逻辑。

每次 dispatch 固定请求对象、顺序、前缀、输出快照与请求槽。异步结果错配、同 ID
不同实例、槽位变更、输出重复、结果缺行/非整数、重放全部拒绝。完成标记必须来自
同步 CPU executor 的成功返回；过早结果回调不得释放仍可能运行的执行 lease。
撤回/取消成员不追加输出；正常成员继续通过原处理器执行停止条件检查。
处理器半途异常时，全 batch 终止，不回滚已经正式提交的 Req token，也不续跑部分 KV。

`CPUScheduleBridge` binds actual Req/ScheduleBatch objects to one immutable CPU
dispatch. The ordinary result processor remains the sole Req output writer.
The bridge validates the complete authoritative result before advancing its
observer ledger. Cancellation/retraction skips that row; normal finish checks
remain in Req. Failure is terminal; neither committed output nor model writes
are rolled back. No second timeout poll can undo an already committed token.

## 证据 / Evidence

- 29 新增 CPU 契约测试；Windows 全量 1360 passed / 11 skipped。
- WSL 全量 1365 passed / 6 skipped。新增/相关 PVD 文件 Ruff 通过；原结果处理器
  有 49 个既有 lint 问题，本阶段未重写无关代码；其改动已格式化且 diff 检查通过。
- 严格真实 CPU Llama 验收包含普通无 bridge 基线路径、真实 Req、ScheduleBatch、
  原始 normalize/result 循环与 Req 停止条件。21 次 attention 对照，最大误差
  `2.384185791015625e-7`；old=9、new=2、third=0、length-limit=1 个正式 D token。
- 不同长度请求加入/重排、旧请求独立刷新边界 4/8、wait-all、Req 撤回、输出上限、
  真实前向失败均通过。第一次联调暴露 fixture 没提供 vocab_size 和 normalize 后
  的 SamplingParams；已按原 Req 接口补齐，没有绕开实际停止条件。
- 常规 metrics、streaming、完成后的 cache-release 回调在 fixture 中是 spy；
  真实 pool 由外围 driver 显式清理。这不是完整 Scheduler 服务验收。

```bash
PYTHONPATH=python python test/registered/disaggregation/run_pvd_draft_cpu_smoke.py \
  --probe --search --sparse-decode --controlled-decode --batch-decode --scheduled-decode
```

29 new contract tests plus strict real CPU inference, actual Req/ScheduleBatch
and the real result-processing method. Service callbacks (metrics/streaming/
finished-cache cleanup) are spies; the driver owns cleanup. There is no claim
that a full Scheduler event loop or production request allocator ran.

## 限制与后续 / Limits and next work

这是显式受控 CPU 接点，无公开 serving sparse 开关。仅现有 CPU FP32 TP1 Llama /
TorchNative 执行器允许，禁止 overlap/spec；还没接生产队列、TP、GPU attention、
源/目标 MR 授权/fence 或真实 CAGRA。首轮完整 KV 在本实验仍是本地 fixture 安装。
下一步移除闭环内固定 draft 候选的假设，连接独立真实 draft 运行器，同时继续收敛
请求级调度驱动。不得把这些 CPU 接点直接当作 V100S 可用的 serving 后端。

No production sparse flag, GPU/TP/RDMA support, CAGRA, real quality or latency
claim. Initial KV is locally installed in this experiment. Next: an independent
real draft runner in the closed loop and automatic request-level CPU scheduling,
without changing the original full-prompt serving default.
