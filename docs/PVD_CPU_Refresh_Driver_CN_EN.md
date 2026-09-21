# 请求级 CPU 刷新驱动 / Request-local CPU refresh driver

`CPURefreshDriver` 根据正式 D token 计数自动选择预取或边界补查、轮询结果，
并且仅在当前请求的精确边界安装。调用者在 forward 之间调用 `progress()`，
并让出 asyncio 执行机会；这不是后台线程，也不是已经接入生产 Scheduler 的事件循环。

## 明确策略 / Explicit policy

- 每请求独立时钟/epoch，加入新请求不会取消、重置或广播刷新其他请求。
- 配置的 lead 必须由 draft 预测长度覆盖；不够则注册失败，不能静默换早期 Q。
- 提前检索：使用即将到达的边界最后一个 token 位置的 target post-RoPE Q，
  覆盖已配置的 layer/Q heads；已有迟到任务继续等待，不用新查询替换。
- 第一次启动恰好在边界：使用实际正式前缀最后一个 token 的 target Q，不调用 draft。
- 多请求 capture 共用目标执行 lease，一次 poll 最多启动一个；lease 在 HTTP 等待前
  释放，因此多个请求的网络检索可以同时等待。
- 提前准备好的 bank 不提前切换。边界由本地 CPU group 的所有 rank 完成安装才推进。
  这里的 rank counts 是单进程镜像，绝不是实际 TP ACK 或 GPU/Mooncake fence。
- selected batch 仍由原 dispatcher 强制 wait-all；driver 不偷偷过滤 batch、
  不生成 token、不修改 Req 输出。取消/超时必须 drain 后才能删除注册与释放 bank。

Independent per-request clocks; no batch-entry refresh wave. One synchronous
capture launch per poll, concurrent HTTP waits after its lease releases. The
query is the last token position at the upcoming boundary; a missed window uses
the last actually committed prefix token. An existing late query is retained.
Install only at the exact boundary; initial full-Prompt admission is unchanged.
CPU mirrored rank counts are not distributed completion evidence.

## 验收 / Validation

完整回归：Windows **1373 passed / 11 skipped**；WSL **1378 passed / 6 skipped**。
本阶段新增/修改 Python 文件 Ruff check/format 通过，git diff 检查通过。

9 个新增测试使用真实本地 V HTTP + CPU bank；覆盖提前 ready、准确边界切换、
实际 Q 补查、注册新请求不扰动旧任务、多个请求 capture 串行/HTTP 并存、
拒绝坏配置、forward 未完成禁止 close、queued capture 超时取消后 drain。

组合严格验收：

```bash
PYTHONPATH=python python test/registered/disaggregation/run_pvd_draft_cpu_smoke.py \
  --probe --search --sparse-decode --controlled-decode --batch-decode \
  --scheduled-decode --real-draft-loop
```

真实双模型最后一条闭环启用此 driver；21 次 attention 对照误差上限约 `3.58e-7`，
draft 实际 2 次 forward，边界 4 预测/8 补查、真实 Req 撤回/停止条件、wait-all 与
故障终止均通过。测试刻意漏调下一轮预取窗口以验证 fallback；不是延迟 benchmark。

The strict real-draft loop now uses this driver to launch/install refreshes,
instead of hand-supplying query positions and rank counts. The fixture still
chooses when to poll and intentionally misses one window to test fallback.
No claim of background scheduling, TP/GPU/RDMA, actual CAGRA, quality or latency.

## 下一阶段 / Next gate

生产 Scheduler 的队列/资源释放/TP 分发还没接通；CPU bank 与执行器不允许直接在
GPU serving 中启用。后续需实现 GPU 稀疏 attention、请求级授权的 sparse 传输与
MR/ACK/fence、多 rank 激活，再与真实模型/硬件联调。CAGRA 仍只有预检工具，
没有可用生产后端。不能把剩余工作说成“只缺跑一次测试”。
