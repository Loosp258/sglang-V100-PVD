# PVD 最终等待队列异步首轮 KV / Asynchronous initial KV at the final waiting queue

日期 / Date: 2026-09-20

## 中文：目标与不可变约束

新请求只有到达 D 的最终 `scheduler.waiting_queue` 后，才发起首轮完整
Prompt KV 拉取。V 执行获授权的 RDMA WRITE，写入 D 已注册并 pin 的
请求级 staging；D 校验身份和原生完成、unpack 到已预分配的最终 KV 页、
完成 TP 一致性确认及 ACK 后，才把请求标为 RUNNABLE、接纳进运行 batch。

- prealloc/transfer 队列仍可交换元数据和等待 KV_STORED，但不得提前启动
  这次 V→D 首轮交付。P→V 上传不受此限制。
- D 发起“拉取”不等于 RDMA READ；数据方向仍为 V→D WRITE。
- 新请求等待网络或 ACK 时，调度循环必须返回，允许旧请求继续 Decode。
  这不是零开销保证：TP 控制通信、GPU 注册、同步和 unpack 仍占用调度时间。
- 新请求不强制旧请求刷新，不重置旧请求 M 时钟，不取消旧请求预取。
  运行中请求到期的同步刷新屏障暂时保留。
- 首轮只取完整 Prompt KV，不依赖 draft、query、CAGRA 或 INDEX_READY。
  正式 token 计数从 0 开始，不含 P 采样的首 token，接纳后不重复取 round 0。
- 容量由 prealloc 准入及共享 staging 字节预算限制。预算暂时不足时留队，
  无新到达、非 transfer 轮询轮次、存在 retracted 请求时也要继续推进或重试。
- 所有 TP ranks 先共同决定可拉取请求集合；不能各自按本地预算分支。
- 取消、HTTP 错误和超时均不是 RDMA 已停止的证据。已发布目标必须保留，
  直到原生终态或匹配的 fence 证明安全。迟到完成不能写回已回收的最终页。

## 中文：实施阶段与当前结果

1. **接入与门控（已有，保留）：** `BootstrapGate`、KV_STORED 通知、
   waiting queue 和 batch admission；通过 D 的 `--pvd-waiting-queue-bootstrap` 启用。
2. **首轮网络异步化（本次实现）：** `PVDDecodeRefresher.start_bootstrap/poll_bootstrap`
   在调度线程驱动共享交付步骤；rank 0 的控制线程处理 retrieve/poll/ACK，
   仅在 future 已完成后取结果。GPU 操作和 TP collective 不移到后台线程。
3. **延期与 TP 协调（本次实现）：** 每轮处理完整等待队列；按各 rank 的
   KV_STORED 和剩余预算选择共同的有序集合。每个 D worker group 同时推进
   一个 bootstrap wave；wave 的请求数受 prealloc 和 staging 字节限制。
4. **失败与回收（本次实现）：** 取消在安装前及 ACK 后均进行 TP 一致检查；
   失败请求走 session close/fence，完成回调不复活已移除请求。
   尚未发布 descriptor 的成功 prepare 可直接撤销 pin，无需等待不存在的远端交付。
5. **硬件验收（待完成）：** 在真实 V100S、Mooncake、RDMA 上记录传输时间、
   旧请求 TPOT、排队时延、首轮完成至接纳时间、吞吐和峰值显存。

共享步骤仍沿用保守的 wave 级失败处理：wave 内某个请求取消或协议校验失败，
可能使同 wave 其他新请求也失败；不包含已在运行 batch 的旧请求。
后续可细化逐请求失败隔离，但不得牺牲 TP 一致性和 fencing。

## 中文：验收与非目标

CPU 回归应覆盖：延迟 retrieve、延迟 ACK、TP2/TP4 一致推进、两种等待阶段取消、
源就绪/预算的 rank 差异、无新到达自动重试、not-runnable 不占 batch 名额、
一次性 bootstrap、round 0 不重复、周期时钟不变及原有生命周期测试。
CPU fake transport 只能验证逻辑与协议，不证明 GPUDirect 可见性或真实速度提升。

本次不实现：CAGRA 索引/搜索、目标模型 probe 的真实 Q 捕获、draft 的服务接线、
稀疏 KV/attention、M-r 预测预取及 active/next GPU 缓冲、周期刷新异步放行、
直接写最终 KV 页、RDMA READ 或新通信后端。完整 Prompt KV 仍必须能装入 D，
并额外预算请求级 staging；这不是显存压缩方案。

## English: objective and invariant contract

Start a newcomer's initial **complete Prompt KV** delivery only after it reaches
D's final `scheduler.waiting_queue`. V performs an authorized RDMA WRITE into
registered, pinned, request-owned D staging. D validates identity and native
completion, unpacks into already-preallocated final KV pages, agrees across TP
ranks, and finishes ACK before marking the request RUNNABLE and admitting it.

- Earlier queues may exchange metadata and wait for KV_STORED, but cannot start
  this initial V→D delivery. This restriction does not affect P→V uploads.
- D initiates the pull; V is the writer. No RDMA READ is introduced.
- Waiting for delivery or ACK returns control to the scheduler, allowing existing
  decoding to continue. TP control collectives, registration, GPU synchronization
  and unpack still cost scheduler time; zero interference is not promised.
- Newcomers never force old requests to refresh, reset their M clocks, or cancel
  their prefetches. The synchronous periodic due-refresh barrier remains.
- Bootstrap uses full Prompt KV, without draft/query/CAGRA/INDEX_READY. Its
  committed D-token count is zero, excludes P's first sample, and round 0 must
  not be fetched again after admission.
- Preallocation and the shared staging-byte budget bound capacity. Retry deferred
  requests even with no arrivals, on non-transfer-polling iterations, and while
  retracted requests remain. All TP ranks agree on the eligible ordered set first.
- Cancellation, HTTP errors and timeouts are not transport-terminal evidence.
  Published destinations stay alive until native terminal/fence proof. A late
  completion must never unpack into recycled final pages or resurrect a request.

## English: implementation phases and status

1. **Existing wiring, retained:** bootstrap gates, KV_STORED notification, final
   waiting queue, and admission. Enable with `--pvd-waiting-queue-bootstrap` on D.
2. **Implemented here:** `start_bootstrap/poll_bootstrap` drive the shared delivery
   steps on the scheduler thread. Rank 0's control loop runs retrieve/poll/ACK;
   future results are read only after completion. No GPU work or TP collective
   moves to a background thread.
3. **Implemented here:** progress the whole final waiting queue every scheduler
   pass; agree on source readiness and capacity across ranks before preparation.
   One bootstrap wave is active per D worker group, bounded by preallocation and
   staging bytes. This version does not promise fairness among pending waves.
4. **Implemented here:** TP-consistent cancellation checks before installation and
   after ACK; failed sessions close through existing fencing. Completed prepared
   pins whose descriptors have never left D can be dropped locally.
5. **Pending hardware acceptance:** real V100S/Mooncake/RDMA runs measuring transfer
   time, old-request TPOT, queue latency, ready-to-admission delay, throughput and
   peak memory. CPU fake-transport results do not establish hardware correctness
   or a speedup.

The shared protocol conservatively fails a bootstrap wave on a member's
cancellation or protocol failure; this can fail other newcomers in that wave,
but does not include already-running requests. Finer per-request failure
isolation is future work and must preserve TP agreement and fencing.

## English: acceptance and explicit non-goals

CPU coverage must include delayed retrieve and ACK, TP2/TP4 agreement,
cancellation during either wait, rank-local readiness/budget differences,
retry without arrivals, skipped admission without consuming batch slots,
exactly-once bootstrap, no repeated round 0, unchanged periodic clocks, and
the existing lifetime regressions.

Not implemented by this change: CAGRA serving, real target-probe Q capture,
draft serving integration, sparse KV/attention, M-r predictive prefetch and
active/next GPU buffers, asynchronous periodic release, direct-to-final-page
WRITE, RDMA READ, or a new transport backend. D must still fit the complete
Prompt KV plus request-owned staging; this is not a memory-reduction feature.

## 开启与验证 / Enable and validate

Append on D / 在 D 启动命令追加：

```bash
--pvd-waiting-queue-bootstrap
```

The flag remains off by default for compatibility. Turning it off preserves
the legacy in-batch synchronous initial refresh. / 为兼容保留默认关闭；关闭时
仍走原有 batch 内同步首轮刷新路径。

CPU suite (PowerShell, repository root):

```powershell
$pvdTestFiles = @(rg --files test/registered/disaggregation -g 'test_pvd*.py')
& .venv/Scripts/python.exe test/registered/disaggregation/run_pvd_cpu_tests.py @pvdTestFiles -q --tb=short
```

The full predictive pipeline remains governed by the
[English handoff](PVD_Predictive_KV_Prefetch_AI_Handoff_EN.md) and
[中文交接](PVD预测检索预取流水线_AI开发交接.md).

## 本地证据 / Local evidence

2026-09-20: **584 CPU tests passed** across the 18 PVD test modules, including
32 new parametrized/regression cases relative to the 552-test baseline.
Syntax compilation and `git diff --check` also passed.
这些结果验证 CPU 逻辑、假传输和线程模拟 TP 协调；不代表真实 GPU、原生 Mooncake、
RDMA、模型质量或性能验收已经通过。
