# PVD 当前实现与验收边界 / Current implementation and acceptance scope

Updated / 更新：2026-09-24。历史交接文档保留演进记录；本页集中说明当前边界。
Historical handoffs contain earlier states; this page consolidates the current scope.

## 2026-09-24 初始 Prompt fan-in 性能诊断 / Initial Prompt fan-in profiling

CloudLab 两张 V100S 上，V 独立检出 `e42d0a825` 与 `bbef59a1b` 分别运行
同一条 39-token Prompt / 8-token 输出请求，经 Gateway 返回 HTTP 200，
端到端分别约 36.2 秒、40.4 秒；两次均有 112 次 CAGRA 索引搜索。
每个 V rank 的初始完整 Prompt fan-in 都规划并提交 **2184 个** Mooncake
PUT，总计 1,118,208 字节，平均每 PUT 512 字节。第一轮每 rank 的提交
调用累计约 24–25 秒、fan-in 约 29 秒；第二轮提交调用累计约 29.8 秒、
fan-in 约 33.1 秒。第二轮 Mooncake 原生 `transfer_submit_write` 累计
约 29.3 秒/rank，而 CUDA 同步仅约 0.09/0.35 秒/rank。两 rank 均成功
完成，没有在途或 UNKNOWN 传输。这是瓶颈定位，不是受控性能对照。

Mooncake 0.3.13 的[原生 Python 包装源码](https://github.com/kvcache-ai/Mooncake/blob/v0.3.13/mooncake-integration/transfer_engine/transfer_engine_py.cpp)
显示：`transfer_check_status` 只查询 batch 的 task 0，不能作为批量
PUT 的完整完成证明；`get_batch_transfer_status` 才查询整个 batch，
但失败/超时路径会释放 native batch ID。后续批量优化必须把失败或
未知结果视为不确定，保留源和目标 MR，不能凭单任务完成或失败返回
就回收内存。

On two CloudLab V100S GPUs, isolated V checkouts `e42d0a825` and
`bbef59a1b` each served the same 39-token Prompt/eight-token output request
through Gateway with HTTP 200, taking about 36.2 and 40.4 seconds end to
end; each produced 112 CAGRA index searches. Per V rank, initial full
Prompt fan-in planned and submitted **2184 Mooncake PUTs** for 1,118,208
bytes (512 bytes per PUT on average). The first run spent about 24–25
seconds/rank in submit calls and 29 seconds in fan-in; the second spent about
29.8 seconds/rank in submit calls and 33.1 seconds in fan-in. In the second
run, native `transfer_submit_write` accounted for about 29.3 seconds/rank,
versus only 0.09/0.35 seconds/rank in CUDA synchronization. Both ranks
completed with no in-flight or UNKNOWN transfer. This locates a bottleneck;
it is not a controlled performance comparison.

The [Mooncake 0.3.13 Python binding source](https://github.com/kvcache-ai/Mooncake/blob/v0.3.13/mooncake-integration/transfer_engine/transfer_engine_py.cpp)
shows that `transfer_check_status` queries only task 0 of a batch and cannot
prove full-batch completion. `get_batch_transfer_status` checks the aggregate,
but its failure/timeout path frees the native batch ID. Any future batch
optimization must treat failure or uncertainty as non-terminal for MR
release, retaining both source and destination rather than releasing them
on a single-task result or a failed aggregate call.

## 2026-09-24 V 历史容量与 Coordinator 扫描 / V history bounds and coordinator scans

`413f8bf60`、`f76894d61` 分别给 V shard 的历史 Entry、Delivery 与旧版
缺席 fence 增加每 worker epoch 的容量上限；`e03350501` 使 Coordinator 的
每 ID 异步锁在最后一个操作结束后回收；`155d7ae4b` 给 Coordinator 的
历史 Entry、Delivery、Router admission 与未知检索 fence 增加上限。
默认上限分别为 Entry 8192、Delivery 65536、旧版缺席 fence 4096、
Router admission 8192、未知检索 fence 4096，均可由启动参数调整。
达到上限时拒绝**新**身份，保留旧身份的重试与 fence，不按 TTL 删除
重放证明。V shard 与 Coordinator 的上限各自独立，容量耗尽不是服务成功；
这只是有限内存内的失效保护，不是无限期持续接纳新请求的方案。

CloudLab V 从独立检出 `f76894d61` 启动后，两条真实 Gateway 请求均为
HTTP 200：第一条 36-token Prompt / 2-token 输出，第二条 41-token Prompt /
8-token 输出。第二条触发 **112 次** native CAGRA 索引搜索；两个 V rank
累计各完成并 ACK 3 次 Delivery，Mooncake 在途/UNKNOWN 均为 0。
TTL 后两 rank 均恢复 1024 空闲页，`live_entries=0`、
`pending_release_entries=0`，两个历史 Entry 均为 `released`。

V 随后从新隔离检出 `155d7ae4b` 启动，一条 41-token Prompt / 8-token
输出请求经 Gateway 返回 HTTP 200（约 39.7 秒），V 日志记录 112 次
索引搜索，两 rank 各完成并 ACK 2 次 Delivery。TTL 后两 rank 各恢复
1024 空闲页、无在途/UNKNOWN 传输，Coordinator admission 计数回到 0；
健康接口报告上限已生效。该在线检出不含随后 `a2c65c46d` 的活动扫描优化。
包含该优化的 `dc77fd605` 隔离检出随后也经 Gateway 完成一条 41-token
Prompt / 8-token 输出请求（HTTP 200，约 35.4 秒），记录 112 次索引搜索，
两 rank 各完成并 ACK 2 次 Delivery。300 秒 TTL 后，两 rank 各恢复
1024 空闲页，`live_entries=0`、`pending_release_entries=0`、Mooncake
在途/UNKNOWN 为 0；Coordinator 的 admission 与活动维护集合均归零。
这只验证该请求的功能及回收，不构成延迟改善的对照证据。
本地完整 PVD CPU 回归依次为 **2858、2859、2879、2881 passed**，
各有 23 skipped、21 subtests passed；最新结果对应 `a2c65c46d`。
这些少量请求不证明在接近容量上限、连续故障或长期负载下的表现。

`413f8bf60` and `f76894d61` add per-worker-epoch bounds for V-shard
historical Entry, Delivery and legacy absent-fence records. `e03350501`
lets per-ID coordinator operation locks die after their last user exits.
`155d7ae4b` bounds coordinator historical Entry and Delivery records, Router
admissions, and unknown retrieval fences. Defaults are 8192 Entries, 65536
Deliveries, 4096 legacy absent fences, 8192 admissions and 4096 unknown
retrieval fences; launch flags can adjust them. Capacity refuses **new**
identities while preserving retries and fences for existing identities. No
TTL eviction of replay proof occurs. Shard and coordinator bounds are
independent. This is fail-closed memory containment, not unbounded continued
admission; capacity exhaustion is not a successful-service state.

On an isolated `f76894d61` V checkout, two real Gateway requests returned
HTTP 200: a 36-token Prompt/two-token output and a 41-token Prompt/eight-token
output. The second generated **112 native CAGRA index searches**. Each V
rank cumulatively completed and ACKed three Deliveries, with zero in-flight
or UNKNOWN Mooncake work. After TTL, both ranks returned to 1024 free pages,
`live_entries=0` and `pending_release_entries=0`; both historical Entries
were `released`.

The next isolated V checkout, `155d7ae4b`, served a real 41-token Prompt /
eight-token output Gateway request with HTTP 200 in about 39.7 seconds.
V logged 112 index searches; each rank completed and ACKed two Deliveries.
After TTL both ranks had 1024 free pages and no in-flight/UNKNOWN transfer,
while coordinator admissions returned to zero. Health exposed the configured
bounds. This live checkout does not contain the later `a2c65c46d` active-scan
optimization.
An isolated `dc77fd605` checkout including that optimization subsequently
completed a 41-token Prompt/eight-token output Gateway request (HTTP 200,
about 35.4 seconds), with 112 index searches and two completed, ACKed
Deliveries per rank. After the 300-second TTL, both ranks returned to 1024
free pages, with no live Entry, pending release, in-flight or UNKNOWN
Mooncake work. Coordinator admissions and active maintenance sets returned
to zero. This verifies one request and its reclamation, not a controlled
latency improvement. Local full PVD CPU regressions across these steps reported
**2858, 2859, 2879 and 2881 passed**, each with 23 skipped and 21 subtests
passed; the latest result is for `a2c65c46d`. A few requests do not
establish near-capacity, repeated-failure or long-run behavior.

## 2026-09-24 V 释放队列在线验证 / Live V release-queue gate

提交 `93e2d7373` 将 V 分配释放进度限制在待释放集合，不再每个维护周期重扫
所有历史 Entry；在分配回调真正结束前仍保留 pool/MR pin。V 节点从新的隔离
worktree 启动两张 V100S 和原生 CAGRA，P、D、Gateway 沿用上一轮服务。
一个真实 35-token Prompt、8-token 输出请求经 Gateway 返回 HTTP 200，
耗时约 40.3 秒。两 V rank 各完成并 ACK 两次 Delivery，Mooncake 在途/
UNKNOWN 为 0。请求后两 rank 各有 989 空闲页，300 秒 TTL 后均恢复
1024 页；历史 Entry 为 `released`，`pending_release_entries=0`，reaper
仍健康。完整本地 CPU 回归 **2840 passed / 23 skipped**。这些结果验证
单请求兼容和一次 TTL 回收，不证明长期负载、原生轮询频率或性能收益。

随后提交 `b52e24e5d` 让索引构建与到期扫描只遍历仍有资源活动的 Entry，
历史记录仍保留以拒绝重放。完整 CPU 回归 **2842 passed / 23 skipped**，
新增的索引扫描专项测试另通过；随后包含此提交的 `f76894d61`
已在上文所述 CloudLab 在线请求中验证。
两个提交都没有解决历史 Entry/Delivery 元数据永久保留的主机内存上界问题。

Commit `93e2d7373` progresses only pending V allocation releases instead of
rescanning every historical Entry on each maintenance tick; the pool/MR pin
remains until the allocation callback itself finishes. V ran native CAGRA on
two V100S GPUs from a new isolated worktree while P, D and Gateway retained
their prior services. One real 35-token Prompt/eight-token output Gateway
request returned HTTP 200 in about 40.3 seconds. Both V ranks completed and
ACKed two Deliveries, with zero in-flight or UNKNOWN Mooncake work. Each had
989 free pages immediately after the request and 1024 after the 300-second
TTL. The historical Entry was `released`, `pending_release_entries=0`, and
the reaper remained healthy. The full local CPU suite had **2840 passed / 23
skipped**. This establishes one-request compatibility and one TTL reclaim,
not sustained load, native poll counts or a latency benefit.

Commit `b52e24e5d` also limits index-build and expiration scans to entries
whose resources remain active, keeping historical records for replay refusal.
Its full CPU regression had **2842 passed / 23 skipped**, with the final
index-scan focused case rerun separately. The subsequent `f76894d61`
checkout containing this commit passed the live CloudLab requests above.
Neither scan change itself bounds the host memory retained by historical
Entry/Delivery metadata.

## 2026-09-24 V 终态轮询优化在线验证 / Live V terminal-polling gate

V 从新隔离检出 `4b12da0c4` 启动两张 V100S 的原生 CAGRA worker group；
D 使用修复后的 `6537b8f02` 和 90 秒配置，P/Gateway 不变。一个真实
29-token Prompt、8-token 输出请求经 Gateway 返回 HTTP 200（约 33.1 秒）。
V 日志记录 112 次分组索引搜索；两 rank 各有 2 次 Delivery 完成并 ACK。
coordinator/reaper 健康、reaper 失败 0、待取消 0、Mooncake 在途/UNKNOWN/
隔离为 0。请求后两 rank 各有 995 空闲页，Entry 尚在 TTL 内；本次重启后的
TTL 回收与长时轮询成本仍待复查。该请求验证新 V 生命周期代码的功能兼容，
不证明终态原生 poll 次数或端到端延迟改善；后两者需要专门计数/对照实验。

V launched native CAGRA on two V100S GPUs from a new isolated `4b12da0c4`
checkout; D used fixed code `6537b8f02` and the 90-second profile while
P/Gateway were unchanged. One real 29-token Prompt/eight-token output Gateway
request returned HTTP 200 (about 33.1 seconds). V logged 112 grouped index
searches, and both ranks completed and ACKed two Deliveries each. The
coordinator and reaper were healthy, with zero reaper failures, pending
cancellations, in-flight or UNKNOWN Mooncake operations, or quarantine. Each
rank had 995 free pages immediately afterwards because the Entry remained
inside its TTL. TTL reclamation after this restart and long-run polling cost
need later checks. This is a functional compatibility gate for the updated V
lifecycle, not proof that native poll calls or end-to-end latency improved;
those require dedicated counters and controlled comparisons.

## 2026-09-24 分组检索在线验证 / Live grouped-search gate

D 节点从隔离检出 `b91642561` 运行真实 Qwen2.5-7B-Instruct 和独立
Qwen2.5-0.5B draft；P、V、Gateway 仍在各自隔离检出中运行，V 使用两张
V100S、原生 cuVS CAGRA、单条 `mlx5_0` rail。一个 27-token Prompt、
8-token 输出的 Gateway 请求返回 HTTP 200。V 的索引搜索总数由 784 增至
896，即本次仅 **112 次**；原逐 Q-head 路径同类请求为 784 次。
两 V rank 的 Delivery 累计各为 6 次完成、6 次 ACK；本次之后 coordinator
健康，reaper 失败 0，Mooncake 在途和 UNKNOWN 均为 0。Entry 尚在 300 秒
TTL 内，故当时每 rank 有 997 空闲页而非 1024；随后复查两 rank 各恢复
**1024 空闲页**，reaper 仍 healthy 且失败 0。
本次端到端约 **71.1 秒**，不能据 HTTP 次数减少推断延迟改善。

第一次使用样例配置中的 15 秒 `request_timeout_seconds` 时，真实请求在
CUDA 批次结果提交门禁处因 `rank installation round timed out` 被拒绝，
旧代码把该请求级拒绝上抛导致 D scheduler 退出。诊断提交 `64e62dddd`
让异常带上拒绝阶段与原因；验证配置提交 `b91642561` 将此样例窗口提高到
90 秒后上述请求通过。90 秒是此 V100S 验证配置，不是普适超时承诺；
请求级拒绝不应杀死整个 D 服务；该边界已在 `6537b8f02` 修复并做了下面的
受控验证。

The isolated D checkout `b91642561` ran real Qwen2.5-7B-Instruct with an
independent Qwen2.5-0.5B draft; P, V and Gateway stayed in their isolated
checkouts. V used two V100S GPUs, native cuVS CAGRA and the single active
`mlx5_0` rail. A 27-token Prompt/eight-token output Gateway request returned
HTTP 200. V's cumulative index searches rose from 784 to 896: **112 searches**
for this grouped request versus 784 for a comparable per-Q-head request.
Each V rank reported six completed and six ACKed Deliveries cumulatively;
the coordinator and reaper were healthy, with zero reaper failures, in-flight
Mooncake operations or UNKNOWN operations. The Entry was still inside its
300-second TTL, so each rank had 997 rather than 1024 free pages at that
snapshot. A later check found **1024 free pages** on both ranks, with a
healthy reaper and zero failures. The request took about
**71.1 seconds** end to end, so fewer HTTP calls are not evidence of lower
latency.

With the earlier 15-second sample `request_timeout_seconds`, a real request
was rejected at the CUDA batch result gate because the rank-installation
round timed out; the request-local refusal propagated out of the D scheduler
and terminated that process. Diagnostic commit `64e62dddd` exposed the phase
and reason; the V100S validation-profile commit `b91642561` raised the sample
window to 90 seconds, after which the request above passed. This is an
experiment-specific bound, not a universal timeout guarantee. Handling a
request-local refusal without killing D was subsequently implemented in
`6537b8f02` and tested as described below.

## 2026-09-24 请求级超时隔离 / Request-local timeout isolation

提交 `6537b8f02` 只在 CUDA forward 已排空、结果回调尚未开始、所有请求
已停止且正式 `Req` 输出未改动时，将 rank-installation 结果拒绝转为该 batch
的请求级终止。隔离/UNKNOWN、部分提交或清理不确定仍上抛，不能静默放行。
完整 PVD CPU 回归为 **2837 passed、23 skipped、21 subtests passed**。
CloudLab D 从新隔离检出运行该提交，并故意指向旧检出的 15 秒配置：
一个真实 Gateway 请求使 D 自身返回 HTTP 503；Gateway 重试后熔断，
但 D 进程 `409865` 仍在、健康接口 HTTP 200，日志中 scheduler 致命异常
计数为 0。此实验只证明一次请求级失败不会杀死 D，不证明连续故障、
资源长期压力或在 15 秒配置下成功生成。

Commit `6537b8f02` converts a rank-installation result refusal to a
request-local batch abort only after the CUDA forward has drained, result
processing has not begun, every request has stopped, and authoritative Req
outputs remain unchanged. UNKNOWN completion, partial commitment and
uncertain cleanup still fail closed. The full PVD CPU suite reported
**2837 passed, 23 skipped and 21 subtests passed**. The isolated CloudLab D
checkout used this commit with the prior 15-second config as deliberate fault
injection: D itself returned HTTP 503 to a real Gateway request; Gateway then
opened its circuit on retry, but D PID `409865` remained alive, `/health`
returned 200 and the D log contained zero fatal Scheduler exceptions. This
proves one request-local failure did not kill D, not sustained failure
handling, resource-pressure safety or successful generation at 15 seconds.

## 2026-09-24 V 清理恢复在线验证 / Live V cleanup-recovery gate

在本地提交 `52ed8ebf6`、`97cc2f556`、`8adf2ef2b`、`922de7098`
后，完整 PVD CPU 回归为 **2806 passed、23 skipped、21 subtests passed**。
V 节点从新的隔离 worktree `922de7098` 重启；旧检出保留。
在该节点原有 conda 环境中（未安装 pytest），直接执行三个 Entry
生命周期故障注入函数和五个后台清理函数，均通过。随后原 Gateway/P/D 服务
向新 V 发出一个真实 Qwen2.5-7B 29-token Prompt/8-token 输出请求：
HTTP 200，两个 V rank 各完成并 ACK 两次 Delivery；健康接口显示
`maintenance_reaper=healthy`、66 轮中 0 失败，待确认取消数为 0，
Mooncake staging/inflight/UNKNOWN 为 0。请求结束时 Entry 仍在 300 秒
TTL 内；随后再次查询，两个 shard 的 Entry 均为 `released`、各恢复
**1024 空闲页**，reaper 仍 healthy 且累计失败 0。这只证明一次 TTL
回收，不是长期或高负载验收。

After local commits `52ed8ebf6`, `97cc2f556`, `8adf2ef2b` and
`922de7098`, the complete CPU PVD suite reported **2806 passed,
23 skipped and 21 subtests passed**. V was restarted from a new isolated
`922de7098` worktree; the old checkout remains intact. Its conda environment
lacks pytest, so three Entry lifecycle and five reaper fault-injection test
functions were executed directly and passed. The existing Gateway/P/D
services then sent a real Qwen2.5-7B 29-token Prompt/eight-token output
request through the new V: HTTP 200, two Deliveries completed and ACKed
per V rank. Health reported `maintenance_reaper=healthy`, zero failures
in 66 rounds, zero pending cancellations, and zero Mooncake staging,
inflight or UNKNOWN work. The Entry initially remained within its
300-second TTL; a later health check showed `released` on both shards,
**1024 free pages** per shard, and a still-healthy reaper with zero
failures. This proves one TTL cleanup, not long-run or high-load behavior.

上述旧验收时，终态 Entry/Delivery 与 fence 元数据无界保留且每轮重扫。
后续提交已经增加每 worker epoch 的容量上限、弱生命周期操作锁以及活动
维护集合；详见本页开头。历史记录仍保留用于拒绝迟到重试，触及上限时拒绝
新 ID。若要求无需重启地无限期接纳新 ID，仍需明确的 epoch/序号水位或
可强制执行的重试期限，不能简单按 TTL 删除旧身份。
At the time of the older acceptance above, terminal Entry/Delivery and fence
metadata accumulated without a bound and maintenance rescanned history.
Subsequent commits added per-worker-epoch limits, weak-lifetime operation
locks and active maintenance sets as described at the top of this page.
Historical identities remain for late-retry refusal; new IDs are refused at
capacity. Indefinite admission without restart still needs an explicit
epoch/sequence watermark or an enforceable retry horizon. Blind TTL eviction
is unsafe.

## 2026-09-24 生产请求链路首轮验收 / First live serving acceptance

在授权的 CloudLab 隔离检出中，P=`clgpu020`/TP1、V=`clgpu021`/两张
V100S、D=`clgpu019`/TP1，使用 `mlx5_0` 单 rail 和真实
Qwen2.5-7B-Instruct（D 独立 draft 为 Qwen2.5-0.5B-Instruct）。
Gateway `/generate` 连续两次返回 HTTP 200，各生成 8 token；
`--pvd-kv-refresh-interval 4` 的预测检索和稀疏交付路径已实际执行。
V 日志记录了每次 784 次索引搜索；两次请求后，V 每 rank 的
P→V 字节数大于零、4 次 Delivery 均完成并 ACK，V→D 字节数大于零，
Mooncake 未留在途事务或隔离状态。D 本地验证提交为 `2095dd60a`；
P/V/Gateway 使用隔离检出的 `fbd77287b`（本轮修复只涉及 D）。
本地 PVD CPU 回归为 **2797 passed、23 skipped、21 subtests passed**。

On authorized isolated CloudLab checkouts, P=`clgpu020`/TP1,
V=`clgpu021`/two V100S ranks and D=`clgpu019`/TP1 ran on the single
active `mlx5_0` rail with real Qwen2.5-7B-Instruct and an independent
Qwen2.5-0.5B-Instruct draft on D. Two consecutive Gateway `/generate`
requests returned HTTP 200 and eight tokens each; the four-token predictive
refresh and sparse-delivery path executed. V logged 784 index searches per
request; after both requests, each rank had nonzero P→V and V→D bytes,
four completed/ACKed Deliveries, and no in-flight or quarantined Mooncake
transfers. D ran local commit `2095dd60a`; the isolated P/V/Gateway checkouts
ran `fbd77287b` because this round's fixes affected D only. Local PVD CPU
regression: **2797 passed, 23 skipped, 21 subtests passed**.

This is an **exact-index, synchronous CUDA-packing baseline**, not native
CAGRA, a recall/quality distribution, proof of RDMA/GPU overlap, or a
throughput/latency improvement claim. The tested D union cap was 32 for a
26–27-token prompt with per-Q-head Top-4; smaller caps fail closed rather
than truncate. `mlx5_1` remains physically DOWN, so dual-rail has not been
validated. Earlier paragraphs below describe historical milestones and may
say that production integration was not yet installed; this newer live test
supersedes those historical status statements.

这是**精确索引、同步 CUDA 打包**基线，不是原生 CAGRA、召回/质量分布、
RDMA 与 GPU 重叠证明或性能提升证明。实际 D 并集上限为 32，
Prompt 为 26–27 token、每 Q head Top-4；容量不足时拒绝而不截断。
`mlx5_1` 仍为物理 DOWN，双 rail 未验收。以下旧段落保留历史里程碑，
其中“生产接线尚未完成”的旧状态已由本节真实请求验收取代。

同日随后完成两次**原生 cuVS CAGRA**的真实 Gateway 8-token 请求：
26–27-token Prompt 在 `intermediate_degree=16` 下确实触发 224 次原生
建图与 1568 次 HTTP 索引搜索；两 V rank 的四次 Delivery 均完成/ACK，
Mooncake 预算排空。详见 [CAGRA 真实请求验收](PVD_CAGRA_Acceptance_CN_EN.md)。
这取代上文“仅 exact”的功能边界，但不证明检索质量或性能收益。

Later the same day, two more real eight-token Gateway requests passed with
**native cuVS CAGRA**. Their 26–27-token prompts exceeded the configured
16-row fallback threshold, triggering 224 native builds and 1568 HTTP
searches. All four Deliveries per V rank completed/ACKed and Mooncake
budgets drained. See [native-CAGRA live acceptance](PVD_CAGRA_Acceptance_CN_EN.md).
This supersedes the exact-only functional boundary above, but does not
establish retrieval quality or a performance gain.

另有两个同时入 batch 的真实 CAGRA 请求各生成 8 token；并发暴露并修复
Qwen QKV 非连续视图的 CUDA 打包问题（D `e197e928c`）。
这只证明两请求功能正确；更高并发、吞吐、尾延迟与资源长期压力尚未验收。
Two simultaneous native-CAGRA requests also generated eight tokens each
after the batched-QKV packing fix in D `e197e928c`. This establishes only
two-request functional correctness; higher concurrency, throughput, tail
latency and sustained resource pressure remain unverified.

同一组原生 CAGRA 服务又通过一个实际 **91-token Prompt / 8-token 输出**
的 Gateway 请求（HTTP 200，约 42.3 秒）；结束时 V 两 rank 各有 1024
空闲页、无 Mooncake 在途或隔离事务。D 的测试工具检出为 `b9ba454e6`，
运行中的 D 服务仍为 `e197e928c`。这是有限的较长 Prompt 功能验证，
不是 1024-token 生产 Gateway 容量或性能结论。详见
[CAGRA 验收边界](PVD_CAGRA_Acceptance_CN_EN.md)。

The same native-CAGRA services also passed one actual **91-token Prompt /
eight-token output** Gateway request (HTTP 200, about 42.3 seconds). Both
V ranks ended with 1024 free pages and no Mooncake in-flight or quarantined
work. The D smoke utility checkout was `b9ba454e6`, while the running D
service remained `e197e928c`. This is a bounded longer-Prompt functional
gate, not a 1024-token production Gateway capacity or performance result.
See the [CAGRA acceptance boundary](PVD_CAGRA_Acceptance_CN_EN.md).

2026-09-24 CloudLab D 节点（`clgpu019`）使用**新建、干净的隔离检出**验证了本地
`ecf94fb07`：CUDA 路由组装、收到 Prompt 后准入、refresh driver 的 38 项
聚焦测试通过；进一步运行 `test_pvd_cuda_*.py`，**445 项通过**。该检出来自
经用户授权传输的增量 Git bundle；原有实验工作区未被覆盖。
这些测试未执行真实模型 forward、Mooncake WRITE 或生产 Scheduler 自动准入。

On 2026-09-24, a **new clean isolated checkout** on CloudLab D (`clgpu019`)
validated local commit `ecf94fb07`: 38 focused CUDA routed-assembly,
received-Prompt admission and refresh-driver tests passed; the broader
`test_pvd_cuda_*.py` run passed **445 tests**. The checkout used a user-authorized
incremental Git bundle; the existing experimental workspace was untouched.
These tests did not exercise real model forwards, Mooncake WRITE or automatic
production Scheduler admission.

实验性 CUDA Scheduler binding 现在可在**显式提供启动侧准备回调**时消费已发现的
V 路由：在最终 waiting queue 上重新核验完整 Prompt receipt，按实际 Prompt
长度收紧检索上限，先构建待导入工作集，再执行 provisional 准入事务。注册前
失败只清理未认领资源；driver 已认领后由 driver 排空或隔离。尚无生产启动
代码安装该回调，因此默认服务行为仍不变。

The experimental CUDA Scheduler binding can now consume a discovered V route
when startup **explicitly supplies a preparation callback**. It revalidates
the full-Prompt receipt in the final waiting queue, clamps retrieval bounds
to the actual Prompt length, constructs a pending working set, and invokes
provisional admission. Pre-registration failures retire only unclaimed
resources; claimed requests drain or quarantine through the driver. No
production startup installs this callback yet, so default serving is unchanged.

CUDA 请求准入已有只读 preflight 和显式的预构建资源接管事务：核对完整 Prompt
receipt、同一 Req 与 Gateway 所选 V 路由；随后按 provisional driver 注册、
Req/KV 延迟释放绑定、初始 Prompt 导入、receiver claim 的顺序执行。失败走有序
关闭或 UNKNOWN 隔离。**生产 Scheduler 尚未创建并调用这个事务**，因此
`--pvd-predictive-retrieval-config` 仍不启用预测检索。

CUDA admission now has a read-only preflight and an explicit transaction over
prepared resources: provisional driver registration, deferred Req/KV release
attachment, full-Prompt import, then receiver claim. Failure follows ordered
drain or UNKNOWN quarantine. The production Scheduler **does not yet construct
or invoke this transaction**, so predictive serving is still inactive.

Decode 侧现在可用 `--pvd-predictive-retrieval-config` **只校验配置**：
显式指定 V 向量空间、每路 Top-K、同 KV-head 并集上限、工作集与临时字节预算，
并要求当前 CUDA TP1/full-KV fan-in/native-attention 环境及独立 draft 配置。
它**不会**创建预测/检索 Scheduler binding，生产路径仍为完整 Prompt KV 刷新；
通过参数检查不等于 PVD 预测已启用。

Decode now accepts `--pvd-predictive-retrieval-config` for **configuration
validation only**: explicit V vector-space identity, per-query Top-K,
same-KV-head union bound, bank/scratch budgets, the current CUDA TP1/full-KV
fan-in/native-attention envelope and independent draft settings. It **does
not** construct a predictive Scheduler binding; serving still uses full-Prompt
KV refresh. Passing argument validation is not predictive activation.

CloudLab D 隔离工作区现备有固定 revision 的独立 Qwen2.5-0.5B-Instruct
实验 draft，且与目标 Qwen2.5-7B 的实际 `VocabularySignature` 相同。
其独立 smoke 已通过；进一步在**同一 D GPU** 同时加载目标 7B 与 draft
0.5B，两步预测驱动 28 层目标 post-RoPE Q 捕获，私有池和预算归还、目标
canary/RNG 保持不变。尚未接入生产 Scheduler。见
[draft 模型记录](PVD_Qwen_Draft_CloudLab_CN_EN.md)。

The isolated CloudLab D worktree now has a pinned independent
Qwen2.5-0.5B-Instruct experimental draft with an exact target-matching
`VocabularySignature`. A newer **same-GPU dual-model** gate also passed:
two draft predictions drove target post-RoPE Q capture in all 28 layers,
with private pools/budgets refunded and target canaries/RNG unchanged.
Production Scheduler integration is still missing. See the
[draft checkpoint record](PVD_Qwen_Draft_CloudLab_CN_EN.md).

最新隔离三节点验收在两路稀疏交付任务 pending 时运行第 4 次真实
Qwen2.5-7B Decode 前向，再等待写入终态、于正式计数 4 安装新 bank，
第 5 次前向成功消费它。最新门槛等待两个 V rank 的 `start_delivery` 响应，
强于旧版 D 本地发布标志；但仍无网络/计算重叠的时间测量或延迟收益证明。详见
[CAGRA 验收边界](PVD_CAGRA_Acceptance_CN_EN.md)。

The latest isolated three-node gate ran a fourth real Qwen2.5-7B Decode
forward while two sparse-delivery tasks were pending, awaited terminal write
proofs, installed at committed count four, and consumed the new bank in a
fifth forward. The latest gate waits for both V `start_delivery` responses,
rather than relying on D-local publication. It validates source-start and
safe-installation order, not measured RDMA/GPU time overlap or latency hiding; see the
[CAGRA acceptance gate](PVD_CAGRA_Acceptance_CN_EN.md).

最新 112 组真实 Qwen GQA 原生 CAGRA 对精确并集的召回：初始查询最低
0.97727、平均 0.99943；位置 1027 的正式前缀刷新查询最低 0.98、
平均 0.99982。仅一个 Prompt/一次刷新，不能外推到其他负载；见
[CAGRA 验收边界](PVD_CAGRA_Acceptance_CN_EN.md)。

The latest native-CAGRA versus exact-union result covers all 112 real Qwen
groups: initial minimum/mean recall 0.97727/0.99943, actual-prefix refresh
minimum/mean 0.98/0.99982. This is one prompt and one refresh only; see the
[CAGRA acceptance gate](PVD_CAGRA_Acceptance_CN_EN.md).

最新 CloudLab 真实 Qwen2.5-7B 验证已经移除 first-token 测试种子：P
对真实 Prefill logits 贪心取样，作为选定 Entry 元数据提交；D 从 Coordinator
查询该 Entry 并核对身份，本次使用同一 token ID **198** 完成完整 Prompt
bank 前向、正式计数 4 的稀疏刷新及其后续前向。旧版文档中 token `42`
仅描述此前离线验证。见 [CAGRA 验收边界](PVD_CAGRA_Acceptance_CN_EN.md)。
独立 draft 预测、流水线重叠和生产 Scheduler 自动装配仍未验收。

The latest real-Qwen2.5-7B CloudLab gate no longer seeds D with a synthetic
first token. P greedily sampled real Prefill logits and stored the token in
the selected Entry; D validated that Entry through Coordinator `select` and
used the same token ID **198** for the full Prompt-bank forward and generated-
token sparse refresh. Earlier references to seed `42` are historical.
Independent draft prediction, pipeline overlap and production Scheduler
activation remain unvalidated.

## 先前验收阶段 / Earlier acceptance stages

2026-09-24 CloudLab 新增真实 token 驱动刷新验收：D 在完整 bank 上连续执行
4 次 Qwen2.5-7B 目标前向并保留生成 K/V；位置 1027 的正式前缀目标 Q
用于补查 V，正式计数 4 安装稀疏 bank，第 5 次前向已实际消费它。初始
token 42 是测试种子，不是 P 的真实采样输出；网络交付未与计算重叠，
独立 draft 预测和生产 Scheduler 仍未验收。详见
[CAGRA 验收边界](PVD_CAGRA_Acceptance_CN_EN.md)。

CloudLab now validates a generated-token-driven refresh: D ran four real
Qwen2.5-7B target forwards over the full bank, kept generated KV locally,
captured actual-prefix Q at position 1027, installed the sparse bank at
committed count four and consumed it in a fifth target forward. Token 42 was
a test seed, not P's sampled first token. No network/compute overlap,
independent draft prediction or production Scheduler activation is claimed.

GQA 硬件验证已从单一代表 Q head 扩展为每个 KV head 对应的全部 7 个
Q heads：每路 Top-10、同组去重并集、每组最多 70 token；真实
Qwen2.5-7B 的 112 组并集实测 17–65 token，模拟边界 4 的远端 K/V
已安装并逐字节核对。刷新后模型前向仍未执行。详见
[CAGRA 验收边界](PVD_CAGRA_Acceptance_CN_EN.md)。

The hardware gate now uses all seven GQA query heads per KV head: Top-10
per query, deduplicated within-group union, maximum 70 tokens per group.
Across real Qwen2.5-7B's 112 groups the observed unions held 17–65 tokens;
D installed their remote K/V at synthetic boundary four and checked every
value. A subsequent model forward over this sparse bank remains untested.

真实 Qwen2.5-7B 的 D GPU0 现已完成一次**目标模型 Decode 前向**，实际消费
从两个 V 分片 RDMA 安装的 28 层完整 Prompt 工作集；相对原生 dense 前向
最大 logits 绝对误差 0.01611328125，贪心输出 token 相同。详见
[CAGRA 验收边界](PVD_CAGRA_Acceptance_CN_EN.md)。这不包括真实 token
驱动的边界 4 稀疏刷新后续前向，也不意味着生产 Scheduler 已启用。

A real Qwen2.5-7B target Decode forward on D GPU0 now consumed the full
28-layer Prompt bank installed by RDMA from two V shards. Its maximum logit
difference from native dense Decode was 0.01611328125 and greedy top-1 token
matched. See the [CAGRA acceptance gate](PVD_CAGRA_Acceptance_CN_EN.md).
The boundary-four sparse refresh has not been driven by generated tokens or
consumed by a later forward; production Scheduler activation remains open.

2026-09-24 新增 CloudLab 真实 Qwen2.5-7B 三节点离线验收：P 实际 Prefill
上传完整 1024-token Prompt KV；V 两 GPU 各建 56 个原生 CAGRA 索引；
D 对全部 112 层/head 组接收并安装完整初始工作集及一次模拟边界 4 的
Top-10 稀疏工作集。全部安装值与 D 独立模型前向逐字节相同，资源归还。
详见 [CAGRA 验收边界](PVD_CAGRA_Acceptance_CN_EN.md)。这尚未执行
真实 token 生成、目标模型 attention 消费该远端 bank、完整 GQA Q-head
并集或生产 Scheduler 流水线。

On 2026-09-24 an isolated three-node CloudLab real-Qwen2.5-7B gate passed:
P uploaded actual 1024-token Prompt KV, two V GPUs built 56 native CAGRA
indexes each, and D received and installed both a complete initial bank and
a Top-10 sparse bank at synthetic boundary four across all 112 layer/head
groups. Installed values matched an independent D model forward bit-for-bit;
owners and budgets drained. See the [CAGRA acceptance gate](PVD_CAGRA_Acceptance_CN_EN.md).
Real generated-token attention over this remote bank, full GQA Q-head union
and production Scheduler/pipeline integration remain unvalidated.

## 结论 / Bottom line

CPU 参考路径已实际跑通独立小模型 → 目标模型 post-RoPE Q → V 精确检索 →
稀疏交付 → D 工作集安装/attention → 原 Req 输出处理 → 自动回收。
**这不代表生产 PVD 已开启预测检索，也不能证明所有非硬件实现缺口已清零。**
当前生产路径仍是完整 Prompt KV 刷新；`--pvd-draft-*` 记录配置，不自动装配
生产预测检索 Scheduler。启动日志和参数帮助现在明确提示这一点。

The real CPU reference executes independent draft prediction, target post-RoPE Q,
exact V search, sparse delivery/install/attention, original Req output processing
and automatic retirement. **This is not production predictive retrieval and is
not proof that every hardware-independent implementation gap is closed.** Serving
still uses full-Prompt refresh. Draft configuration does not instantiate the
production prediction pipeline; startup now warns explicitly instead of implying
activation.

## 最新增量 / Latest increment

### 真实 Qwen P→V→D 检索及回写 / Real Qwen P-to-V-to-D search and WRITE

三节点已用同一真实 Qwen2.5-7B FP16 checkpoint 将 P 的完整 Prompt KV 上传
V 双 rank；D 产生位置 1024 的真实 post-RoPE Q，在第 0 层从两个 V shard
各检索一个 KV head。两路 Top-10 都与精确点积 10/10 重合，各收到 5120 字节
稀疏 K/V，并与 D 独立模型 forward 的对应值逐字节一致。Entry/目的 MR 正常释放。
这仍只覆盖**一个 prompt、一层、两个 head**；完整 D 工作集、生成期间刷新、
生产 Scheduler 与流水线时延尚未连通。详见
[CAGRA 验收](PVD_CAGRA_Acceptance_CN_EN.md)。

With the same real FP16 Qwen2.5-7B checkpoint across three nodes, P uploaded
complete Prompt KV to both V ranks; D produced a real post-RoPE Q at position
1024 and searched one KV head from each V shard at layer 0. Both native
Top-10 sets had 10/10 overlap with exact dot products. Each 5120-byte sparse
K/V payload matched D's independent model forward bit-for-bit. The Entry and
receive MRs were safely released. This is **one prompt, one layer, two heads**;
full D bank installation, generated-token refresh, production Scheduler and
pipeline latency remain open.

### 真实 Qwen Prompt KV 跨节点 P→V / Real Qwen cross-node P-to-V

node-0 已用真实 Qwen2.5-7B FP16 权重 Prefill 1024 tokens，按 V TP2
存储布局拆成两个完整 K/V shard，经 Mooncake/RDMA `mlx5_0` 上传 node-1。
V0/V1 各建成 56 个可搜索 CAGRA 索引。Entry 留存待 D 实验；尚未证明
真实目标 Q 经过该跨节点服务检索并交付给 D。详见
[CAGRA 验收](PVD_CAGRA_Acceptance_CN_EN.md)。

Node-0 used the real FP16 Qwen2.5-7B checkpoint to prefill 1024 tokens,
split complete Prompt K/V into two V TP2 storage shards, and uploaded both
over Mooncake/RDMA `mlx5_0` to node-1. Each V rank built 56 searchable
CAGRA indexes. The Entry remains for the D gate; real target Q has not yet
searched this cross-node Entry or delivered its selected KV to D.

### 真实 Qwen K/Q 检索对照 / Real Qwen K/Q retrieval check

node-1 V100S GPU0 已运行真实 Qwen2.5-7B FP16 checkpoint：1024-token
Prompt 的第 0 层/KV head 0 的 K 与目标 probe 在位置 1024 的 post-RoPE Q
进入原生 CAGRA，单次 Top-10 与精确 GPU 点积完全重合，分数误差最大
0.000244140625；见 [CAGRA 验收](PVD_CAGRA_Acceptance_CN_EN.md)。这是
**单 query、单层、单 head** 的实测，不是整体 recall 证明；真实模型 K/Q
尚未进入跨节点 V→D 安装或生产 Scheduler。

Node-1 V100S GPU0 ran the real FP16 Qwen2.5-7B checkpoint. Native CAGRA
searched 1024 actual layer-0/KV-head-0 Prompt K rows using a matching
post-RoPE target Q at position 1024; one Top-10 query had 10/10 overlap with
exact GPU dot products and 0.000244140625 maximum score error. This is one
query, layer and head, not a general recall guarantee. These real model K/Q
have not yet passed through cross-node D installation or production scheduling.

### D 节点真实 checkpoint 的离线稀疏 attention / Real-checkpoint offline D attention

在 node-2 的 V100S GPU0 上，用已有本地 FP16
`Qwen2.5-7B-Instruct` checkpoint 运行
`run_pvd_cuda_model_smoke.py --architecture qwen2 --dtype float16`，结果
`passed`、5 次真实模型 forward、140 次逐层独立 dense-SDPA 数值对照，
最大绝对误差 0.0038767；完整 Prompt 初始化和一次稀疏刷新、allocator 释放
均通过。这项验收在 D 节点独立运行，**不是**把上一项跨节点 RDMA 收到的 KV
送进该模型，也不证明目标 Q 检索、生产服务或性能。

The existing FP16 Qwen2.5-7B-Instruct checkpoint passed the offline sparse
model smoke on node-2 D GPU0: five real forwards, 140 layer-wise independent
dense-SDPA checks (maximum absolute error 0.0038767), full-Prompt bootstrap,
one sparse refresh and allocator retirement. This is a separate D-node test;
the model did **not** consume the KV delivered by the preceding cross-node
RDMA gate. Target-Q retrieval, production serving and performance remain open.

### 三节点 TP1 D GPU 工作集安装 / Three-node TP1 D GPU bank install

沿用 P→V 合成上传及 V 双 rank 原生 CAGRA 建图，node-2 D GPU0 已通过
Mooncake/RDMA 从两个 V rank 聚合完整 Prompt KV 并安装于边界 0，然后把
稀疏选中 KV 安装于边界 4。两轮四个 layer/head 组逐字节一致，安装协议
完成 RESUMED、Delivery 完成 ACK，D 预算归零、V Entry 释放。见
[CAGRA 验收](PVD_CAGRA_Acceptance_CN_EN.md)。**边界 4 是合成协议推进，
没有真实 D 模型生成四个 token；真实 Q、attention 输出、流水线时延和生产
Scheduler 仍未验收。**

Building on native P-to-V upload and two-rank CAGRA indexes, node-2 D GPU0
now aggregates full Prompt KV from both V ranks and installs it at boundary
0, then installs a sparse selection at boundary 4. All four layer/head groups
were bit-exact in both rounds, with RESUMED and Delivery ACK complete, D
budgets zero, and the V Entry released. Boundary 4 is synthetic protocol
progression, not four real target-model Decode tokens. Real Q, attention
output, latency overlap and production Scheduler remain unvalidated.

### V→D 原生稀疏 KV 接收 / Native V-to-D sparse KV receive

三节点单 rail `mlx5_0` 合成验收已将 P→V 完整 Prompt KV、V 双 rank 原生
CAGRA 检索和 V→D GPU RDMA 稀疏交付串起。D GPU0 在远端 terminal-success、
fence 与字节数确认和本地 CUDA 同步之后，比对两 V rank、每 rank 两层的 K/V
均逐字节一致（各 512 字节）。D 关闭目的 MR 后预算归零；Entry 释放后 V 的
索引记录均清空、预算只保留根预留。**尚未验证 D 工作集安装、模型 attention、
真实目标 Q recall、投机流水线或生产 Scheduler**。详见
[CAGRA 验收](PVD_CAGRA_Acceptance_CN_EN.md)。

A three-node, single-rail `mlx5_0` synthetic gate now connects full Prompt KV
upload, two-rank native CAGRA search and V-to-D GPU RDMA sparse Delivery.
After exact remote success/fence/extent proof and local CUDA ordering, D GPU0
verified bit-exact K/V for two layers on each V rank (512 bytes/rank). D's
receive budget returned to zero; releasing the Entry cleared both V indexes,
leaving only the root reservations. Decode working-set installation, model
attention, real target-Q recall, speculative overlap and production Scheduler
remain unvalidated.

### P→V 原生上传、建图与 HTTP 检索 / Native P-to-V upload/index/search

node-0 P GPU0 到 node-1 两张 V100S 的合成完整 Prompt KV，经单 rail
`mlx5_0` Mooncake/RDMA 完成双 shard commit；V0/V1 各建两个原生 CAGRA 图，
合成 K 行 HTTP 查询各 4/4 self-hit，释放 Entry 后只保留共享根预算。实验先
暴露并修复 CPU HTTP query 与 CUDA 后端的设备不匹配：索引管理器现于身份
校验和搜索预算预留后搬运 query。node-2 索引定向 **226 passed / 3 skipped**。
没有真实目标 Q、D 稀疏接收、生产 Scheduler 或吞吐验收，详见
[CAGRA 验收](PVD_CAGRA_Acceptance_CN_EN.md)。

Synthetic complete Prompt KV was uploaded from node-0 P GPU0 to both node-1
V100S ranks over Mooncake/RDMA `mlx5_0`. Both shards committed; each V rank
built two native CAGRA graphs and returned 4/4 self-hits for synthetic K-row
HTTP queries. Entry release left only the shared root reservation. This run
exposed and fixed CPU HTTP Q versus CUDA CAGRA placement: the manager now
places Q after identity validation and search-budget admission. Focused
node-2 tests: **226 passed / 3 skipped**. Real target Q, D sparse delivery,
production Scheduler and throughput remain unvalidated.

### 原生 CAGRA 服务启动入口 / Native CAGRA service entry point

CloudLab V 节点的 cuVS 25.02 候选环境中，普通 `python -m sglang...pvd.server`
先经 SGLang 包初始化加载 torch，再导入 cuVS，复现 `libcuvs_c.so` 动态库
错误。仅交换服务内部 Mooncake/索引构造顺序无效，因此没有保留该改动。
新增顶层 `python -m pvd_cagra_server`，保证先导入 cuVS。限时真实服务测试中，
V0/V1 与 coordinator 全部健康，Mooncake 0.3.13.post1 在单 rail `mlx5_0`
完成 GPU 注册和本地传输预检；两张 V100S 的 CAGRA-auto 索引快照各显示
671088640 bytes 根预算预留。未产生 Entry、未跑真实目标 Q 检索或 D 端服务。
此启动/健康/预算检查现已写成自清理 `run_pvd_cagra_v_service_gpu.py`；node-1
运行 `passed`，服务退出码 0，隔离端口已释放。

In the isolated cuVS 25.02 candidate, normal `python -m sglang...pvd.server`
loads torch through SGLang's package initializer before cuVS and reproduces
the `libcuvs_c.so` loader failure. Reordering Mooncake and index construction
inside the server was insufficient and was reverted. The new top-level
`python -m pvd_cagra_server` imports cuVS first. In a bounded service run,
V0/V1 and the coordinator were healthy; Mooncake 0.3.13.post1 registered GPU
memory and passed local transfer preflight on single-rail `mlx5_0`. Both
V100S index snapshots showed one 671088640-byte root reservation. No Entry,
real target-Q search or D-side service was exercised.
The check is now repeatable through `run_pvd_cagra_v_service_gpu.py`; its
node-1 run passed, exited with code 0, and released the isolated ports.

### V 索引预算接入共享 CAGRA 上限 / Shared CAGRA cap in V admission

V 可显式传 `--prompt-index-cagra-global-native-bytes N`（仅 `cagra` / `cagra-auto`）。
`N` 必须覆盖一个图的 native cap 且不超过 V rank 的总索引预算。后端必须实际
持有 RMM 根 limiter；管理器在任何 Entry 构建前对 `N` 一次性预留，并在管理器
存续期间保持预留。图内 native 分配仍受每图子 limiter 和共享根 limiter 双重
约束；向量副本、短 Prompt 精确索引及其他 scratch 另行计费。Entry 释放不会
错误退还全局预留。未设置此参数时仍沿用每图终生预留。CloudLab node-2
定向索引回归 **219 passed / 3 skipped**。node-1 两张 V100S/cuVS 25.02 的新增
组件门控把真实 V CLI/工厂、Prompt pack/extract、管理器和四个原生图组合运行：两个合成 Entry
各两图，共享 640 MiB 根限额，预算仅预留一次；释放 Entry 后图内分配归零，
根预留仍保持。它尚不是生产 V 服务验收；也不保证给定预算能容纳 56 图或
多个真实 Entry，未测真实目标 Q 的召回。

V can opt into `--prompt-index-cagra-global-native-bytes N` for `cagra` or
`cagra-auto`. The root must cover one graph cap and fit the rank's total index
budget. A real native RMM root limiter is required; the manager reserves `N`
once before any Entry build and holds it for its lifetime. Child and root
limiters bound native allocations; vector copies, short-prompt exact indexes
and other scratch are charged separately. Closing an Entry does not refund the
root. Without the flag, the per-graph lifetime reservation remains. Focused
node-2 regressions: **219 passed / 3 skipped**. A new node-1 dual-V100S/cuVS 25.02
component gate combined the real V CLI/factory, Prompt packing/extraction, the manager and four
native graphs from two synthetic Entries. It charged the 640 MiB root once;
native graph allocations returned to zero after Entry close while the root
reservation remained. This is not a production V-service run or evidence of
56-graph/real multi-Entry capacity or real-target-Q recall.

### 跨图 CAGRA 原生限额能力 / Shared CAGRA native-limit capability

node-1 V100S/cuVS 25.02 隔离探针验证：两个原生图的子 RMM limiter 可共用
640 MiB 父级上限；两笔各 360 MiB 的 C API 分配中第一笔通过，第二笔被
父级拒绝，图销毁后计数归零。CPU 契约与原 CAGRA 模式 **41 passed / 2 skipped**。
此段仅记录 allocator 能力；上文新增了可选生产预算接线，但尚未完成真实 V
进程验收。见
[CAGRA 验收](PVD_CAGRA_Acceptance_CN_EN.md)。

An isolated node-1 V100S/cuVS 25.02 probe proved two native graphs can share
a 640 MiB parent RMM limiter: the first 360 MiB cuVS C-API request succeeded,
the second was rejected at the parent, and root allocation returned to zero
after disposal. CPU/CAGRA regressions passed **41 / 2 skipped**. This is an
allocator capability; the optional serving reservation described above was
added later and still needs a real V-service acceptance run.

### 短 Prompt 的显式 CAGRA 自动模式 / Explicit CAGRA-auto short fallback

V 可选 `--prompt-index-backend cagra-auto`：短 Prompt 用同设备精确检索，长
Prompt 用原生 CAGRA；纯 `cagra` 保持拒绝短 Prompt。node-0 V100S 索引相关
**213 passed / 3 skipped**，node-1 cuVS 25.02 上 16-row 精确索引与
1024-row 原生索引同时 build/search/dispose 通过。预算按实际分支计，native
UNKNOWN 会阻止继续服务。仍无真实目标 Q recall、56 图/多 Entry 容量与生产
Scheduler 的验证，详见 [CAGRA 验收](PVD_CAGRA_Acceptance_CN_EN.md)。

V can explicitly select `cagra-auto`: short prompts use exact search on the
same GPU, larger ones use native CAGRA; pure `cagra` keeps its refusal.
Node-0 V100S index regressions passed **213 / 3 skipped**, and node-1 cuVS
25.02 ran short exact and long native indexes concurrently through
build/search/dispose. Actual-path budgets and sticky native UNKNOWN are
enforced. Real target-Q recall, 56-graph/multi-Entry capacity and production
Scheduler integration remain open.

### 合成 Prompt-KV 的三节点原生接力 / Synthetic Prompt-KV native relay

原生接力工具新增可选 `--payload-kind packed-kv`：用项目 packer 打包合成
2-layer/2-head FP16 Prompt KV，P→V→D 复用已注册 GPU 区，D 用项目 unpacker
验证四个 K/V 分量及最终页的 padding 保护。CloudLab V100S 上 GPU 0、GPU 1
各自的 P/V/D 均报告 `passed`，D 两次都报告解包成功；Linux 契约 9/9 通过。
这不等于真实模型产生的 KV、TP2 collective、检索或生产 Scheduler 验收。

The relay tool now optionally packs synthetic two-layer/two-head FP16 Prompt
KV through the project's packer, relays it through V's registered GPU buffer,
and checks the D unpacker and final-page padding. Separate V100S GPU-0 and
GPU-1 P/V/D runs all passed; D verified unpack in both. Linux contract tests
passed 9/9. Model-generated KV, TP2 collective, retrieval and production
Scheduler serving are still unproven by this sample.

### Draft 分支准入失败清理 / Draft branch admission-failure cleanup

`SGLangDraftProvider.branch()` 创建 handle 后，如果 scratch 大小计算、类型校验或
预算预留失败，现在先在共享执行锁下调用该 handle 的 `release()`，再退还准入槽位；
若清理失败则保留 handle 并隔离整个 provider，拒绝再次复用可能仍活着的私有资源。
scratch 大小不再把布尔/浮点数强制转成整数。CloudLab node-0 隔离 worktree 的
draft 测试组为 **254 passed**。此项只修复分支生命周期，不会装配生产 Scheduler。

After `SGLangDraftProvider.branch()` opens a handle, failure to size, validate
or reserve scratch now releases that handle under the shared execution lock
before returning admission. Failed cleanup retains the handle and quarantines
the provider. Boolean/floating scratch declarations are rejected rather than
coerced. The draft regression group passed **254 tests** in an isolated
CloudLab node-0 worktree. Production Scheduler activation remains open.

### V100S 上的真实 CUDA draft 执行 / Real CUDA draft execution on V100S

独立 `run_pvd_cuda_draft_smoke.py` 复用真实 SGLang ModelRunner/CUDA
attention，随机 tiny Llama 和 Qwen2 分别执行 2 次完整前缀重算及 1 次
EXTEND+DECODE draft 分支；两个预测 token 都与重算一致，释放后私有
request/KV 池容量恢复。CloudLab node-0 V100S、FP16、`torch_native`：
Llama 已知保留张量 2,725,712 字节，Qwen2 为 2,727,760 字节。
这是随机小模型的执行/回收证据，**不是**选定 draft checkpoint、
生产 Scheduler 装配、显存峰值、预测质量或隐藏网络时延的证据。

The standalone `run_pvd_cuda_draft_smoke.py` runs real SGLang ModelRunner
CUDA forwards with random tiny Llama and Qwen2. For each, both predicted
tokens matched independent full-prefix recomputation, and private request/KV
pool capacity was restored after release. On CloudLab node-0 V100S with FP16
and `torch_native`, known retained-tensor bytes were 2,725,712 and 2,727,760
respectively. This establishes execution/cleanup for random tiny models,
**not** a chosen draft checkpoint, production Scheduler assembly, peak VRAM,
prediction quality or network-latency hiding.

### draft 设备绑定 / Draft device binding

`build_prediction_only_worker()` 在加载权重前核对 `--pvd-draft-device`
与 worker 的 `gpu_id`：显式值必须是相同编号的 CUDA 设备；未提供时以
`gpu_id` 生成 `cuda:N`。CPU draft 仍通过独立 CPU smoke 路径测试，
不伪装成 GPU `TpModelWorker`。此校验只避免错误设备加载，未启动生产预测。

Before loading weights, `build_prediction_only_worker()` now requires an
explicit draft device to be an indexed CUDA device matching the worker's
`gpu_id`; when omitted it derives `cuda:N` from that id. The separate CPU
draft smoke remains CPU-only. This prevents misleading placement, but does
not activate production prediction.

### draft 私有 KV 池比例 / Private draft KV-pool fraction

独立 draft 加载配置现在要求显式 `--pvd-draft-mem-fraction-static`（0 到 1
之间），并将私有 request 槽数设为允许的并发分支数；不再继承目标模型的
`mem_fraction_static` 和 `max_running_requests`。CLI 可以先记录模型路径而
暂不提供该比例，但实际构造 worker 时会拒绝启动。此比例仍不是加载峰值显存
保证，须与独立的常驻字节预算及设备峰值实测配合使用；生产预测尚未自动启用。

Loading a private draft now requires an explicit
`--pvd-draft-mem-fraction-static` in (0, 1), and its private request-slot
limit follows admitted concurrent branches. It cannot inherit the target's
static-memory fraction or request count. CLI configuration may record a draft
path without the fraction, but worker construction then fails closed. This is
not a peak-memory guarantee and does not activate production prediction.

### 已加载 draft 的保留张量计量 / Loaded-draft retained-tensor accounting

新增 `measure_draft_retained_tensors()`：对已加载模型权重/缓冲区、私有
request-to-token 映射、K/V 池及 allocator 索引张量按底层 storage 去重计量，
未知布局拒绝计量；CPU 实际 draft smoke 改为复用它。这个数值是**已知张量的
占用下界**，不包括加载峰值、CUDA allocator 缓存和 backend 工作空间；仍未
自动装配生产 draft，也不构成显存硬上限。配置预算必须另留余量，并在 V100S
上测量峰值。

`measure_draft_retained_tensors()` now deduplicates underlying storage for
loaded model weights/buffers, private request mapping, K/V pools and allocator
index tensors; unknown layouts are refused. The real CPU draft smoke uses the
same counter. This is a **known-tensor accounting floor**, excluding loading
peak, CUDA allocator cache and backend workspace. It does not activate a
production draft or enforce a hard VRAM limit; device measurements and margin
are still required.

### Draft 常驻显存预算参数 / Draft persistent-memory budget flag

新增可选 `--pvd-draft-persistent-budget-bytes`，与每分支 scratch 预算分离，
只接受正整数，不能在未配置 draft 模型时孤立出现；默认仍不猜测预算。
启动校验/原 speculative 禁令相关定向回归 **85 passed**。该参数当前只被
记录和校验，**尚未**把 draft worker 装配进生产 Scheduler，也不会因此开始
分配或限制常驻显存；实际装配前仍必须测量模型权重/私有池并与此上限核对。

The optional `--pvd-draft-persistent-budget-bytes` is distinct from per-branch
scratch, accepts only positive integers and cannot appear without a draft
model. No capacity default is guessed. Focused startup/speculation regressions
passed **85 tests**. This is configuration validation only: it does **not**
instantiate a production draft worker or yet charge its retained weights and
private pools. A serving factory must measure those bytes before admission.

### 真实 7B 权重的稀疏 Decode 数值验证 / Real-7B sparse Decode numerical check

node-0 V100S 在独立临时 worktree 中加载现有 FP16
`Qwen2.5-7B-Instruct` checkpoint，用 TP1、`torch_native` 运行离线稀疏模型
smoke：5 次真实 Decode forward，28 层共 **140** 次逐层独立 dense SDPA
对照，最大绝对误差约 **0.00388**；初始完整 Prompt、一次稀疏刷新和
allocator 退还均通过。该 smoke 的 Prompt 很短、检索选择由测试提供；
**没有经过 V 上的 CAGRA/精确检索、Mooncake 稀疏 RDMA、生产 Scheduler、
TP2 或性能测试**。

In an isolated node-0 V100S worktree, the existing FP16 Qwen2.5-7B-Instruct
checkpoint passed an offline TP1 `torch_native` sparse model smoke: five real
Decode forwards, **140** layer-wise comparisons with independent dense SDPA
across 28 layers, about **0.00388** maximum absolute error, one sparse refresh
and allocator retirement. Its Prompt is short and its selection is supplied
by the test. It does **not** exercise V search/CAGRA, native sparse Mooncake
RDMA, production Scheduler, TP2 or latency.

### Qwen2 CUDA 稀疏 attention 基线 / Qwen2 CUDA sparse-attention baseline

显式 `make_cuda_sparse_backend` 工厂现接纳精确的非量化
`Qwen2ForCausalLM`（仍限 TP1/PP1、`torch_native`、page 1、无图/overlap/
原生 speculative）；离线稀疏模型 smoke 可选择 `--architecture qwen2`。
CloudLab node-0 V100S 上随机 tiny Qwen2 真实执行 5 次 Decode forward、
10 次逐层独立 dense SDPA 对照，FP16 最大绝对误差约 **0.00188**；
初始完整 Prompt 与一次稀疏刷新、请求 allocator 退还均通过。tiny Llama
原路径复跑通过（最大误差约 0.00194）。这不证明 7B 稀疏 forward、生产
Scheduler、Mooncake 稀疏 RDMA 或性能。

The explicit sparse-backend factory now admits exact unquantized
`Qwen2ForCausalLM` under the same TP1/PP1, `torch_native`, page-1,
no-graph/overlap/native-speculation policy. On node-0 V100S, random tiny
Qwen2 completed five real Decode forwards and ten layer-wise comparisons with
an independent dense-SDPA oracle (FP16 max absolute error about **0.00188**),
including an initial full Prompt, one sparse refresh and allocator retirement.
Tiny Llama passed again (max error about 0.00194). This does not prove a 7B
sparse forward, production Scheduler, sparse Mooncake RDMA or performance.

### 真实 Qwen2.5-7B checkpoint 的 TP1 Q probe / TP1 Q probe on real Qwen2.5-7B weights

独立 CUDA smoke 新增严格本地 `--model-path` 模式：不下载、不重写权重，
按层使用 QKV 投影输出和 attention 输入构造独立 pre-/post-RoPE oracle；
大型权重只取每个 tensor 的有界 canary，不复制整个 15GB checkpoint。
CloudLab node-0 V100S 上加载现有 `Qwen2.5-7B-Instruct` 的 FP16 权重并
完成 TP1、`torch_native` probe；两层目标 Q 的最大绝对误差 **0**，
正式前缀补查一致，KV canary/映射/CUDA RNG 与预算检查通过。
原 tiny Llama 回归也再次通过。首次大模型尝试因测试 oracle 错按共享 RoPE
模块计数而失败；改为按层 hook 后重测成功。**尚非 TP2、V 检索、Mooncake
稀疏交付、生产 Scheduler 或端到端性能验证。**

The standalone CUDA smoke now has a strict local `--model-path` mode. Its
independent per-layer QKV/attention hooks compare pre- and post-RoPE Q without
duplicating the 15 GB checkpoint; only bounded canaries are kept per tensor.
Node-0 V100S loaded the existing FP16 Qwen2.5-7B-Instruct weights and passed a
TP1 `torch_native` probe: two target-Q layers matched with **0 maximum absolute
error**, committed-prefix fallback matched, and KV canaries/mapping/CUDA RNG
and budget checks passed. Tiny Llama passed again. The first checkpoint run
failed because the test oracle counted a possibly shared RoPE module instead
of individual layers; per-layer hooks fixed that test defect. **This is not
TP2, V search, sparse Mooncake delivery, production Scheduler or latency
validation.**

### V100S tiny Qwen2 真实 CUDA Q probe / Real CUDA tiny-Qwen2 Q probe

`run_pvd_cuda_probe_smoke.py` 现在可选 `--architecture qwen2`，同时保留默认
Llama。node-0 的独立 CloudLab 验证 worktree（Tesla V100S、PyTorch
`2.9.1+cu128`）上，两种随机 tiny 模型的真实 CUDA forward/probe 均返回
`passed`；两层 post-RoPE Q 对独立 hook oracle 的最大绝对误差均为 **0**，
正式前缀补查一致，目标权重/KV 池映射/CUDA RNG 未改变，probe 预算归还。
这不是 Qwen2.5-7B checkpoint、TP2、RDMA、CAGRA 或生产 Scheduler 的验证。

The standalone CUDA probe smoke now accepts `--architecture qwen2` while
retaining Llama as its default. On node-0 V100S with PyTorch `2.9.1+cu128`,
both random tiny-model forwards passed: two layers of post-RoPE Q matched an
independent hook oracle with **0 maximum absolute error**, committed-prefix
fallback matched, target state/CUDA RNG stayed unchanged, and the probe budget
was refunded. This does not validate the Qwen2.5-7B checkpoint, TP2, RDMA,
CAGRA or production Scheduler admission.

### Qwen2 目标 Q 捕获入口 / Qwen2 target-Q capture seam

Qwen2 注意力现可在 RoPE 之后、调用 attention 之前把目标 Q 交给 batch-owned
PVD collector；新增严格匹配 `Qwen2ForCausalLM` 的 CPU/CUDA probe 类型，继续
使用私有 KV 池、共享目标权重与原有锁/预算策略。19 个定向测试验证调用顺序、
post-RoPE 数值与 probe 类型契约。**尚未运行真实 Qwen2.5-7B 模型 forward、
TP2 或 GPU 端到端检索**；当前 CUDA probe 仍只支持 TP1、`torch_native`。

Qwen2 attention can now pass post-RoPE target Q to a batch-owned PVD
collector before attention. Exact `Qwen2ForCausalLM` CPU/CUDA probe types
reuse private KV pools, target weights and the existing lock/budget policy.
Nineteen focused tests cover ordering, post-RoPE values and type contracts.
**No real Qwen2.5-7B forward, TP2 or GPU end-to-end retrieval has run**;
the CUDA probe still requires TP1 and `torch_native`.

### D 请求工作集身份前置核对 / D request-bank identity precheck

已选 V 的 CUDA 请求工厂在建立 HTTP 客户端前，现需显式核对本地 Req ID、
bank/coordinator 的三元身份及 D KV layout 指纹；只检查 Entry transfer ID 不足以
防止同一 Entry 的不同 D 请求误用工作集。双 V 分片装配回归 **10 passed**。
这仍是显式请求工厂，不会自动开启生产预测检索。

Before creating V HTTP clients, the selected-V CUDA request factory now
checks the local Req ID, bank/coordinator identity and D KV layout fingerprint.
An Entry transfer ID alone cannot distinguish two D requests reusing that
Entry. The two-shard assembly regression passed 10 tests. This remains an
explicit factory, not production predictive-serving activation.

### 已选 V 路由一次性领取 / One-shot selected-V route claim

CUDA waiting queue 的 `ready_for()` 现在只交付一次路由绑定；已领取但仍在 waiting
的请求保留 tombstone，不会在下一轮轮询重复查询 V 或生成第二个控制器。只有该
Req 离开 waiting 后，队列才清除记录。准入工厂仍须在领取后成功注册并绑定接收
会话；失败时应终止请求，不能尝试用同一绑定重放。

The CUDA waiting queue now delivers a selected route binding only once. A
claimed request remains represented while waiting, so the next poll cannot
issue a second V lookup or construct a duplicate controller. The record is
removed when that Req leaves waiting. The serving admission factory still
needs to register and claim the receiver; a failed claim is not replayable.

### 已选 V 查询完成语义 / Selected-V lookup completion semantics

D 的路由发现现在向 CUDA waiting queue 暴露不可取消的完成 Future。它只在控制
协程真正返回或抛错后变为完成；取消 `run_coroutine_threadsafe` 的代理 Future
不能提前释放并发名额或允许相同 rid 的后续查询。真实控制线程与路由队列定向
回归通过。这只修复控制面生命周期，不表示生产预测请求已经自动准入。

D route discovery now exposes a non-cancellable completion Future to the CUDA
waiting queue. It becomes done only after the control coroutine has returned
or raised; cancellation of the `run_coroutine_threadsafe` proxy cannot free
lookup capacity or admit a same-rid successor early. Focused real-control-loop
and queue tests pass. This is a control-plane lifetime fix, not automatic
predictive request admission.

### 接收凭据驱动的 TP1 CUDA Prompt group 工厂 / Receipt-derived Prompt group

新增 `plan_received_prompt_bank` 与 `create_received_prompt_group`：前者从真实
`PVDDecodeSession.require_initial_prompt()` 凭据和 D 模型池逐层 K/V 元数据推导
完整 Prompt 工作集身份、所有 layer/KV-head、dtype、device、请求 epoch 和预算上限；
错误 TP、空页、混合 dtype、错误设备或过大的并集都在构造前拒绝。后者仅在 CUDA
placement 下构造局部 bank/group/importer，**不**自动注册接收 MR 或让请求
进入 Decode batch。CPU 元数据与工厂参数测试通过。随后在 node-0 V100S
`torch 2.9.1+cu128` 的独立临时克隆中，用真实 CUDA tensor 完成 group 构造与
`install_received()` Prompt 导入：模型池 K/V 导入前后逐值一致，staging 与 bank
预算按预期释放。三文件相关回归 **26 passed**。接收完成凭据来自测试夹具，
不等于原生 Mooncake WRITE/ACK 或生产 Scheduler 自动请求装配已验收。

The new Prompt-group factory derives TP1 bank metadata from the receiver's
actual completion receipt and model KV pool rather than caller JSON. It rejects
wrong TP/layout/pages/dtype/device and oversized unions before construction.
Its creation phase builds a local CUDA bank/group/importer but does not
register an MR or admit Decode. Node-0 V100S then passed actual CUDA group
construction and `install_received()` on real device tensors: source model KV
was unchanged and staging/bank reservations were released. The completion
receipt was minted by a fixture, not by native Mooncake transport. Automatic
serving assembly and a native receive-to-import proof remain open.

### CUDA waiting queue 的非阻塞已选 V 路由发现 / Nonblocking selected-V lookup

显式安装的 CUDA Scheduler binding 现在可以附带有界 `CUDARouteDiscoveryQueue`：
只对完整 Prompt 已安装、仍在 waiting queue 的请求异步查询 Router 选中的 V；
owner 线程轮询完成结果，错误按请求停止，其他请求不被队首阻塞。换请求身份、
取消或退出时不发布迟到回复；废弃的 HTTP Future 不用 `cancelled/done`
冒充底层协程已排空，而是保留占用直到实际结果返回。发现结果只供后续工厂使用，
**不会**直接让请求进入 Decode batch，也不自动装配 draft、probe 或稀疏银行。

An explicitly installed CUDA Scheduler binding can now poll a bounded,
nonblocking selected-V discovery queue for waiting requests whose initial
Prompt is installed. A stale or failed lookup cannot publish a route or block
unrelated waiting requests. Abandoned HTTP calls retain their bounded slot
until they settle; a cancelled Future is not treated as coroutine completion.
The result is only an input to the still-missing production request factory:
it does not activate predictive serving or admit a request to Decode.

### D 侧已选路由绑定 / D-side selected-route binding

D 的异步路由发现现在返回仅供本地使用的 `PVDSelectedRouteBinding`，将结果绑定到
同一个 manager、Req 对象、rid、Entry key、Gateway 选中的 V group 与 delivery ID。
请求工厂拒绝裸 `PVDSelectedShardRoutes`，也拒绝发现后换请求、换 V group 或换
delivery ID；检查发生在分配稀疏接收目标之前。这是生产准入的安全前置步骤，
**尚未**自动构造 draft/target pipeline 或启用预测检索。WSL 相关路由/工厂
回归通过。先前以登录目录为基准误判 CloudLab 环境缺失：实际旧检出和隔离
Python 位于 `/mnt/sglang-data/yiliu124-node-0-sglang-pvd/`；旧检出落后当前分支
且有本地修改，所以本轮使用 `/tmp` 的独立临时克隆验证，没有覆盖旧检出。
这一非阻塞路由步骤本身仍没有 RDMA 验证。

Route discovery now returns a local `PVDSelectedRouteBinding` tied to the same
manager, Req object, rid, Entry key, Gateway-selected V group and delivery ID.
The D request factory refuses a bare route reply or a changed request/group
before allocating a sparse destination. This is an admission prerequisite,
not automatic draft/probe construction or predictive-serving activation.
Focused WSL route/factory tests pass; no GPU execution was performed in this step.

### 隔离 cuVS 25.02 CAGRA 探针 / Isolated cuVS 25.02 CAGRA probe

node-1 V100S 的独立候选环境现已用 cuVS 25.02 跑通真实 CAGRA build/search，
并跑通本项目 `CagraIndexBackend` 的 RMM 限额、build/search/dispose。
4096×128 合成向量、512 MiB 每图上限通过；64/128/256 MiB 上限失败。
完整证据与 56 图 × 512 MiB 的显存预算风险见
[CAGRA 验收记录](PVD_CAGRA_Acceptance_CN_EN.md)。生产 V 仍没有启用 CAGRA；
真实目标 Q recall、多个 Entry 和服务端生命周期尚未验收。

The isolated cuVS 25.02 environment on node-1 V100S now passes both the
native CAGRA smoke and PVD adapter's bounded build/search/dispose probe.
The candidate is **not** enabled in the running V service, and real target-Q
recall and multi-Entry memory pressure remain open.

### 三节点原生 PVD 全 Prompt 通路 / Three-node native full-Prompt PVD path

2026-09-23 在三台 CloudLab V100S 上使用**独立验证 worktree** 和固定
`mooncake-transfer-engine==0.3.13.post1` 完成 node-0 P → node-1 V → node-2 D
单 rail (`mlx5_0,mlx5_0`) 验证。Mooncake 原生 GPU WRITE 的本地与跨节点
4 KiB 样本均逐字节校验；随后同一 Qwen2.5-7B-Instruct TP2 模型的 Gateway
请求成功生成 4 token 和 20 token。P 使用 `torch_native`，D 使用
`flash_attn_v100`；两个并发的 20-token 请求也通过。每个 20-token 请求跨过默认
16-token 刷新边界，V 两 rank 的 delivery 和 ACK 各增加两次，且 Mooncake
`unknown_transfers=0`、`used_inflight=0`、`quarantined=false`。这证明当前
**完整 Prompt KV** 传输/刷新路径的一组真实配置可运行，不证明 CAGRA、draft
预测、稀疏传输/attention、双 rail 或正式性能。原始检出与 Conda 环境未改动；
这些本地提交尚未推送 GitHub。

The isolated three-node V100S validation now runs native Mooncake GPU writes
on the single active rail. The same Qwen2.5-7B-Instruct TP2 model completed
Gateway requests of 4 and 20 output tokens, including two concurrent 20-token
requests. Each long request produced one initial and one refresh delivery per
V rank, both ACKed, with no unknown or quarantined transfer. Prefill used
`torch_native`; Decode used `flash_attn_v100`. This validates one real
**full-Prompt KV** configuration, not CAGRA, draft prediction, sparse transfer
or attention, dual rail, or representative performance. Original checkouts
and Conda environments were left untouched; new commits remain local.

首轮 P 使用 `flash_attn_v100` 启动成功，但真实 prefill 因节点 CUDA 13
`nvcc` 不支持 `sm_70` 而失败。通用 PD warmup 与 `/health` 原本会发送缺少
Gateway ID 的假 PVD 请求；现已跳过该 warmup，`/health` 直接反映 worker 状态，
`/health_generate` 返回 503。Gateway HCA 校验不再硬编码设备名，4 个相关
Rust 测试通过。P 崩溃后留在 V 上的两个取消 Entry 因缺少 sender terminal
证明仍占页，这是防止旧 RDMA WRITE 触及重用显存的保守机制；TTL 不能代替证明，
需确认旧 P 已停止，再在维护窗口重启 V 回收。

The first Prefill attempt with `flash_attn_v100` failed on the actual model
forward because this environment's CUDA 13 `nvcc` rejects `sm_70`; a successful
server startup did not prove that backend works. Generic PD startup warmup and
`/health` previously sent identity-free synthetic PVD requests; PVD now skips
that warmup and reports `/health` from worker status, while
`/health_generate` returns 503 rather than claiming to generate. Gateway HCA
validation now accepts valid configured names such as `mlx5_2,mlx5_3` instead
of only `mlx5_0,mlx5_1`; four related Rust tests passed. A P crash after V
reservation left two cancelled Entry shards pinned without sender-terminal
proof. This is intentional fail-closed ownership; TTL does not authorize MR
reuse or release. Operators must fence the old P and restart V in a maintenance
window to reclaim such abandoned reservations.

### 首次 CloudLab V100S CUDA 组件验收 / First CloudLab V100S CUDA component acceptance

2026-09-23 在 `node-0` 的独立验证 worktree、Tesla V100S-PCIE-32GB（SM70）、
PyTorch `2.9.1+cu128` 上运行 `run_pvd_cuda_acceptance.py --expected-gpu V100S`。
首次运行发现测试本身的 rail 配置不一致：store 使用 `mlx5_test`，Entry manifest
使用 `mlx5_0`。测试改为使用 `shard.rail` 后，严格验收 **9/9 passed，0 skipped**。
覆盖真实 CUDA 接收排序、稀疏打包生命周期、工作集切换和 attention 数值检查；
V payload 仍是 fake transport。三台节点均有 2×V100S，只有 `mlx5_0` 为 ACTIVE；
当时三台原 Conda 环境均未安装 `mooncake-transfer-engine`。因此该次验收没有原生 Mooncake/RDMA、
CAGRA、真实模型 forward、性能或生产 Scheduler 验收。原实验检出的
`scripts/install_v100.sh` 本地修改保留；验证在独立 worktree 完成。

On 2026-09-23 the strict component runner executed on node-0 with a real
Tesla V100S-PCIE-32GB (SM70) and PyTorch `2.9.1+cu128`. Its initial failure
was a test-only rail mismatch (`mlx5_test` store versus `mlx5_0` manifest).
Using the manifest's `shard.rail` yielded **9/9 passed, 0 skipped**. This
exercises real CUDA ordering, sparse packing lifetime, bank switching and
attention math, but V payload transport is fake. All three nodes have two
V100S GPUs and only `mlx5_0` ACTIVE; none of the original Conda environments
had `mooncake-transfer-engine` installed at that time. That run did not validate native Mooncake/RDMA, CAGRA, real-model
forward, performance and production Scheduler remain unvalidated. The existing
checkout's local install-script edits were preserved by using a separate
validation worktree.

后续 node-1、node-2 的原检出 `scripts/smoke_v100.sh` 均通过，确认这两台的
SM70 内核、FlashInfer、Marlin/TurboMind 与 NCCL 2.27.5。node-0 最新验证
worktree 的离线 tiny-Llama 首次被 SM70 预热阻断：`torch_native` backend
没有 `get_cuda_graph_seq_len_fill_value()`，而 `ModelRunner` 仍运行了
FlashInfer/TileLang 专用预热。只对 `torch_native` 跳过该预热后，真实 CUDA
目标 Q probe 通过（post-RoPE oracle 最大误差 0），稀疏模型 forward 通过
（5 次 forward、10 次逐层 oracle 检查，FP16 最大绝对误差约 0.00194）。
这些离线 tiny-Llama 测试没有运行 draft 小模型或原生网络传输；独立 worktree
未包含旧检出位置的 Marlin MoE 扩展，因此不能凭该 worktree 的 Llama 结果
宣称其 MoE 扩展也已验证。

The original checkouts on node-1 and node-2 passed `scripts/smoke_v100.sh`,
including SM70 kernels, FlashInfer, Marlin/TurboMind and NCCL 2.27.5. The
latest node-0 validation worktree initially failed its offline tiny-Llama
probe because the SM70 FlashInfer/TileLang warmup invoked an unsupported
CUDA-graph metadata method on `torch_native`. After skipping only that
irrelevant warmup, the real-CUDA target Q probe passed (zero maximum
post-RoPE oracle error), and sparse model forward passed (five forwards,
ten per-layer oracle checks, about 0.00194 maximum FP16 absolute error).
Neither offline test ran the draft model or native network transport. The
independent worktree does not carry the old checkout's Marlin MoE extension,
so its tiny-Llama result does not validate MoE there.

### 原生单节点预检 / Native single-node preflight

随后只在 node-0 的隔离依赖目录安装精确版本
`mooncake-transfer-engine==0.3.13.post1`，原 Conda 环境和原检出均未修改。
`run_pvd_native_local_preflight.py` 在全新进程里、导入 Mooncake 前设置
`MC_DISABLE_METACACHE=1`。使用 `10.0.1.1`、`mlx5_0` 分别对 GPU 0 和 GPU 1
运行严格预检，均返回 `passed`：HCA ACTIVE、CUDA MR 注册、原生异步本机
GPU→GPU PUT、终态与字节校验成功；结束时没有遗留注册区或传输句柄。
这是 **single-rail debug** 的本机 loopback 结果，不能据此宣称跨节点
RoCE/GPUDirect、P→V→D 链路或生产 PVD 已验证。

The exact `mooncake-transfer-engine==0.3.13.post1` was installed only into
an isolated dependency directory on node-0; the original Conda environment
and checkout were untouched. In fresh processes,
`run_pvd_native_local_preflight.py` set `MC_DISABLE_METACACHE=1` before
Mooncake import and passed strict preflight on both GPU 0 and GPU 1 through
`10.0.1.1` / `mlx5_0`. Active HCA, CUDA MR registration, native asynchronous
local GPU-to-GPU PUT, terminal status and byte equality were observed, with
no live registrations or transfer handles at exit. This is a **single-rail
debug loopback** result, not proof of cross-node RoCE/GPUDirect, the P→V→D
path, or production PVD serving.

### 原生跨节点小样本 / Native cross-node sample

`run_pvd_native_cross_node.py` 在 node-1 V100S GPU 0/1 上分别注册 4 KiB 目标区，
node-0 对应 GPU 0/1 经 `mlx5_0` 原生 Mooncake WRITE 写入，两个 rank 均得到
终态成功、远端 GPU 字节一致和双端 MR/传输句柄清零。私网地址为
node-0 `10.0.1.1`、node-1 `10.0.1.2`；TCP 仅用于传递 descriptor 与完成确认，
payload 没有走 TCP。隔离依赖目录在 node-1 也安装了精确 Mooncake 版本，
原 Conda 环境保持不变。脚本拒绝将 `FAILED` 状态直接当作安全释放证明：
还要核对 native transport state 为终态或未提交。此结果验证了 P→V 方向的
单 rail 小样本，不证明零拷贝 GPUDirect 性能、V→D、完整 KV 生命周期、
生产 Scheduler 或 CAGRA。

The one-shot `run_pvd_native_cross_node.py` passed for both matching GPU pairs
from node-0 (`10.0.1.1`) to node-1 (`10.0.1.2`) over `mlx5_0`: each native
Mooncake WRITE reached terminal success, the destination V100S GPU matched
all 4 KiB, and both sides ended with zero registered MRs/transfers. TCP carried
only the descriptor and completion acknowledgement, not payload bytes. The
exact Mooncake version was installed into node-1's isolated dependency target;
its original Conda environment was unchanged. The validation runner additionally
requires a proven terminal or not-submitted transport state before releasing
an MR; a `FAILED` status alone is insufficient. This establishes only a small
single-rail P→V sample, not zero-copy GPUDirect performance, V→D, the complete
KV lifecycle, production Scheduler, or CAGRA.

### V→D 原生跨节点小样本 / Native V-to-D sample

node-2 (`10.0.1.3`) 也只在隔离依赖目录安装了精确 Mooncake 版本并通过
GPU 0 的本机预检。随后对 node-1 V→node-2 D 的 GPU 0、GPU 1 各运行
4 KiB 原生 WRITE：两次均在发送端达到安全终态，D 端 GPU 字节一致，
双端 MR/传输句柄清零。这样 P→V 和 V→D 两段的两个 rank 均有独立的
单 rail 原生小样本证据；**没有**把同一请求的 KV 经 V 连续转发至 D，
也没有运行完整服务、真实模型 KV、并发压力或性能基准。

Node-2 (`10.0.1.3`) received the exact Mooncake version only in its isolated
dependency target and passed GPU 0 local preflight. Native 4 KiB WRITEs from
node-1 V to node-2 D then passed independently for GPU 0 and GPU 1: safe
terminal status, destination-GPU byte equality, and no live MRs/transfers on
either side. Both P→V and V→D legs now have independent single-rail samples
for both ranks. **No** same-request KV was relayed continuously through V to D;
full serving, real-model KV, concurrency stress, and performance remain open.

新增独立的[同请求 P→V→D GPU 缓冲区接力验收工具](PVD_Native_Relay_CN_EN.md)：
V 对同一 GPU 注册区先接收、后作为源发送；未知 WRITE 完成时保留 MR。
8 个 CPU 契约用例已通过。CloudLab 三节点在提交 `db13c5b9c` 上分别对 GPU 0、
GPU 1 跑通 4 KiB 同请求连续接力，六份 P/V/D 报告均为 `passed`。这不是同时
TP2、真实 Prompt KV、并发压力、稀疏交付或生产服务验收。随后又将 GPU0/GPU1
两套进程同时运行在 `mlx5_0` 上，独立端口的六份报告仍全部 `passed`；这只证明
双 GPU 原生会话并存，不是模型 TP2。

The standalone [same-request GPU-buffer relay gate](PVD_Native_Relay_CN_EN.md)
reuses one V registration as the receive target and then the send source,
retaining the MR on unknown completion. Eight CPU contract tests pass.
At commit `db13c5b9c`, both separate GPU-0 and GPU-1 three-node 4 KiB relay
runs passed on CloudLab, with P/V/D reporting success. This does not establish
simultaneous TP2, real Prompt KV, concurrent pressure, sparse Delivery or
production serving. A later run coexisted both GPU sessions on `mlx5_0` with
distinct ports and again passed all six reports; it was not TP2 model execution.

RDMA 预检现等待异步 PUT 的终态（最多 5 秒），不再将首次 `PENDING` 当作
链路失败。超时、轮询异常或未知状态一律拒绝继续启动，并保留源/目标 MR，
连同原生 engine 一起由进程级隔离表保有，避免启动栈回退时丢失 owner 或让
尚在飞行的写使用已注销 rkey。此本机 loopback 预检仍不能证明跨节点
链路与真实 V100S GPUDirect；后者必须在实验机器单独验收。
本步 Windows 全量 **2506 passed / 31 skipped**，WSL 隔离定向 **5 passed**。

The RDMA preflight now waits up to five seconds for an asynchronous PUT's
terminal status instead of treating the first `PENDING` as failure. Timeout,
poll error or an unknown status fails startup and retains both registered
regions and their native engine in a process-lifetime quarantine, including
if startup unwinds. An in-flight write must not target an unregistered MR. This
local loopback check still does not prove cross-node connectivity or V100S
GPUDirect; those require native acceptance on the experiment machines.
This step passed **2506/31 skipped** in the full Windows CPU suite and
**5 passed** in focused WSL quarantine tests.

新增 [CUDA 路由请求工厂](PVD_CUDA_Routed_Request_Factory_CN_EN.md)：把已选 V
的双 shard 路由、真实 D 组件和显式预算组装成一个请求级检索/多源 Delivery；
请求排空后才关闭它拥有的 HTTP 客户端。它仍需生产启动/准入逻辑提供真实组件，
不能凭此认为预测检索已自动启用。
新增 D 接收端显式多 rail 组合层：每个 V 源 rank 可以映射到 D 上独立的
同 rail adapter；注册和注销由同一 adapter 拥有。单 rail engine 仍拒绝
混用 V rails。原生 engine 构造函数现可逐 HCA 建立 session 并严格预检；
worker 可在显式配置时调用，GPU/RDMA 验收尚未完成。
组合层的注册表现在可从 Scheduler 构造线程交接至接收控制线程使用，并锁定
注册/注销操作；这不放松接收 Registry 本身的单线程 owner 约束。
Decode TP1 的 `--pvd-d-receive-rails` 已在启动参数与 PVD manager 接线：
仅完整 KV fan-in 场景可用，要求不重复且包含 D compute rail，启动时逐 HCA
建立并预检原生 session；同一配置还会在 Scheduler owner 线程建立
`sparse_receive_registry`，与现有传输共享预算，请求使用前不注册目标。
它仍不启用生产 Scheduler 稀疏检索。
双 rail 请求工厂现为每个 V 源绑定原生 D rail session ID；无 session ID 的
测试 adapter 需显式 `d_endpoints`，错误的原生 endpoint 在分配前拒绝。
D manager 现可异步发现 Gateway 已选 V group 的 typed shard 路由，先核对
Entry 和本地已预检的 rail 覆盖；生产预测请求准入尚未自动消费这个结果。
D manager 现还可用自己的 CUDA Registry、compute layout、D rail/session
装配已选 V 的请求，拒绝外来 Registry/Entry；真实 pipeline、工作集和生产
请求准入仍待接线。
此请求装配接线 Windows 全量 **2522 passed / 31 skipped**，WSL 定向 **7 passed**。
此路由发现接线 Windows 全量 **2521 passed / 31 skipped**，WSL 定向 **31 passed**。
此 endpoint 绑定回归 Windows 全量 **2518 passed / 31 skipped**、WSL 定向
**13 passed**。
此 Registry 接线 Windows 全量 **2518 passed / 31 skipped**，WSL 定向 **39 passed**。
此启动接线 Windows 全量 **2517 passed / 31 skipped**，WSL 定向 **35 passed**。
此线程交接修复 Windows 全量 **2515 passed / 31 skipped**，WSL 定向 **4 passed**。
此工厂增量 Windows 全量 **2514 passed / 31 skipped**、WSL 定向 **11 passed**。
本步 Windows 全量 **2501 passed / 31 skipped**，WSL 多 rail/工厂/双源定向
**16 passed**（最终单 rail 绕过回归测试随后补充）。真实 GPU/Mooncake/RDMA
仍未执行。

The CUDA routed-request factory assembles selected two-shard V routes with
explicit D resources into one search/fan-in controller and owns HTTP clients
until request drain. Production startup/admission still has to provide the
real components; this is not automatic predictive serving activation.
An explicit D multi-rail receive composite maps each V source rank to an
independently owned, matching D rail adapter. The single-rail engine still
refuses mixed-rail V sources. A native factory can now initialize one session
and strict local preflight per HCA. The worker calls it only with explicit
configuration; GPU/RDMA acceptance remains open.
Its adapter registry now supports handoff from Scheduler construction to the
receive control thread with locked registration/unregistration. The receive
Registry itself remains single-owner-thread only.
Decode TP1 now has `--pvd-d-receive-rails` wired into worker startup for the
full-KV fan-in configuration: distinct HCAs including the D compute rail are
initialized and strictly preflighted. The same opt-in creates a CUDA receive
registry on the Scheduler owner thread with the existing transfer budget; no
destination is registered before request use. Sparse retrieval still is not
activated in the serving Scheduler.
The dual-rail request factory now binds each V source to its D rail's native
session ID. Test adapters without session IDs need explicit `d_endpoints`, and
a mismatched native endpoint is refused before request allocation.
The D manager can now discover typed routes for the Gateway-selected V group
asynchronously and check Entry plus preflighted HCA coverage. Production
predictive admission does not yet consume this result automatically.
The D manager can also assemble a selected-V request with its own CUDA
Registry, compute layout and D rail/session, refusing foreign Registry/Entry
bindings. A real pipeline, working set and serving admission are still needed.
This request-assembly wiring passed **2522/31 skipped** in the full Windows CPU
suite and **7 passed** in focused WSL tests.
The route-discovery wiring passed **2521/31 skipped** in the full Windows CPU
suite and **31 passed** in focused WSL tests.
This endpoint-binding regression passed **2518/31 skipped** in the full Windows
CPU suite and **13 passed** in focused WSL tests.
The registry wiring passed **2518/31 skipped** in the full Windows CPU suite
and **39 passed** in focused WSL tests.
This startup wiring passed **2517/31 skipped** in the full Windows CPU suite
and **35 passed** in focused WSL tests.
The handoff fix passed **2515/31 skipped** in the full Windows CPU suite and
**4 passed** in focused WSL tests.
The factory increment passed **2514/31 skipped** in the full Windows CPU
suite and **11 passed** in focused WSL tests.
This increment passed **2501/31 skipped** in the full Windows CPU suite and
**16 passed** in focused WSL tests (followed by a single-rail bypass regression
test). Native GPU/Mooncake/RDMA remains unrun.

已选 V group 的 coordinator 新增 [Entry 级 shard 路由发现](PVD_Selected_Shard_Routes_CN_EN.md)：
只对 STORED Entry 返回双 shard 的显式 URL、当前 sender epoch、rail 和 manifest，
并核对活跃 shard 状态。HTTP 客户端按请求 key 验证。生产请求工厂仍未接入此接口，
不因此宣称预测检索自动启用。
本步 Windows 全量 **2495 passed / 31 skipped**，WSL 路由与 PVD3 定向
**58 passed**；均无原生 GPU/RDMA 证据。

The selected V coordinator now exposes Entry-scoped shard route discovery:
explicit shard URLs, live sender epochs/rails and the stored manifest, with
fail-closed live-shard and client-side key checks. This feeds a future D
request factory; production predictive serving is still not auto-enabled.
This increment passed **2495/31 skipped** in the full Windows CPU suite and
**58 passed** in the focused WSL run; neither is native GPU/RDMA evidence.

双源请求现将检索与 Delivery 的所选 V 路由绑定：构造时要求每个 D layer/KV head
对应的全部 Q heads 不重不漏、Entry/向量空间/post-RoPE/范围一致；刷新前要求
传入同一对象。错误路由在开启刷新 epoch、注册目标和网络请求之前即拒绝。
Windows 全量 **2492 passed / 31 skipped**（增加精确 KV-head 数校验前），随后
Windows/WSL 最新定向各 **47 passed**；仍需生产 Scheduler 工厂装配。

The routed two-source request now binds search to the Delivery's exact selected
V group. Q-head coverage and Entry/model/encoding/scope are checked during
construction; each refresh requires the same routed client before opening its
epoch, registering destinations or issuing HTTP. The full Windows run passed
before the final exact KV-head count assertion; the latest focused Windows and
WSL runs each passed 47 cases. Production Scheduler assembly remains open.

新增 [稀疏多源请求交付](PVD_Sparse_FanIn_CN_EN.md)：`CUDAPrefetchRequest`
现在可驱动 D TP1 的双 V 源：逐源检索、注册和完成证明，有界合并到 D 暂存区，
提交完整工作集；安装后各源独立 ACK，UNKNOWN 保留 MR 和预算。CPU 策略
组合测试覆盖 draft、目标 Q、真实双 V store/HTTP 与 fake byte copy。生产
Scheduler 尚未自动装配此请求，原生多 HCA/GPU/RDMA 仍未实测。

`CUDAPrefetchRequest` can now drive two selected V sources through search,
registered delivery, independent completion proofs, bounded aggregation into
one D bank, installation and per-source ACK. The composed CPU policy test runs
draft/target-Q and two real V store/HTTP servers with fake byte copying.
Production Scheduler assembly and native multi-HCA/GPU/RDMA acceptance remain open.

本步验证 / This increment: Windows 全量 **2491 passed / 31 skipped**（新增取消
测试前）、随后 fan-in 定向 **10 passed**；WSL 相关定向 **23 passed**。
取消期间两源未完成 WRITE 时仍保留 MR/预算，直到完成证明与显式 close。
The full Windows run preceded the additional cancellation test; its targeted
run and WSL targeted regression passed. These remain CPU policy checks, not
native transport evidence.

以下为前序步骤的当时边界 / Earlier steps retain their historical scope.

多 V 检索路由已区分 D compute rank 与 V storage rank：同一 D 的不同全局 KV
heads 可以查询不同 V 源，每源独立固定版本；完整 D selection 可拆为各源 wire
manifest，分别保留 storage/compute layout 身份。新增 30 项 CPU 测试。前 22 项
加入后的 Windows 全量 **2473 passed / 31 skipped**；随后 8 项分源 plan 用例
加入后定向 **30 passed**。共享查询路径修改后严格 v5 CPU 实模四场景再次通过。
见 [多源检索说明](PVD_Sparse_Source_Routing_CN_EN.md)。

Multi-V search now separates D compute ranks from V storage shards, pins versions
per source, and splits complete D selections into source manifests while retaining
both storage/compute layout identities. Thirty new CPU cases; Windows full after
the first 22: **2473 passed / 31 skipped**; focused file after eight plan cases:
**30 passed**. All four strict v5 real-model CPU cases passed after the shared query
change. This is logical routing, not sparse multi-source delivery activation.

发现的下一项实际阻断：旧 sparse Delivery 仍一 D bank 对一 V sender；CUDA stage
仅接受单个受 guard 保护的连续源 buffer。还需接多源接收/完成/有界聚合/安装后 ACK，
之后才能自动装配生产预测 Scheduler。不能只因为完整 Prompt fan-in 已接通就
声称稀疏刷新支持 V TP2 → D TP1。

Next concrete blocker: legacy sparse Delivery assumes one V sender per D bank,
and CUDA staging requires one guarded contiguous source. Owned multi-source
receives, all-source completion, bounded aggregation and post-install ACK still
need implementation before automatic predictive Scheduler assembly. Full-Prompt
fan-in does not by itself enable V TP2 → D TP1 sparse refresh.

以下为前序步骤的当时边界 / Earlier steps retain their historical scope.

V 已新增可显式选择的原生 CAGRA 后端：`--prompt-index-backend cagra`，需独立
索引总预算与每 layer/KV-head native cap。工厂传入实际 V device；原生 build/
search workspace 计入整个索引生命周期的 cap，显式输出 buffer 和失败 fence，
UNKNOWN 保留 owner/预算。通过加载的 cuVS C 库检查 RMM registry 共享与超限拒绝。
默认仍为 CPU exact；参数、限制与设备命令见
[CAGRA 后端说明](PVD_CAGRA_Backend_CN_EN.md)。

Native CAGRA is now an explicit V backend choice with separate total-index budget
and per-layer/head native cap. The factory uses the actual V device; native
workspace stays within the lifetime cap, output buffers are explicit, and failed
completion proof retains owners/budgets. The loaded cuVS C library must prove
RMM-registry sharing and over-cap rejection. Default remains CPU exact.

新增 **34 项 CPU 测试**，含真实 store/manager 预算生命周期和库调用边界 doubles；
**2 项原生 cuVS/CUDA 用例未执行**。Windows 全量 **2451 passed / 31 skipped**，
WSL 索引相关 **213 passed / 10 skipped**。这不是原生执行、召回或 GPU/RDMA 证据。
生产预测 Scheduler 自动装配、多 rail/通用拓扑仍需实现；当前 CAGRA 配置对短于
intermediate degree 的 prompt 明确拒绝建索引，完整 Prompt 交付不受影响。

Thirty-four new CPU cases cover actual store/manager budget lifetimes and native
call boundaries with doubles; two native cuVS/CUDA cases are unexecuted. Windows
full: **2451 passed / 31 skipped**; WSL index-related: **213 passed / 10 skipped**.
No native execution, recall, GPU or RDMA evidence is claimed. Production predictive
Scheduler assembly and multi-rail/general topology remain code work. The current
CAGRA configuration explicitly refuses index builds with rows no greater than the
intermediate degree; independent full-Prompt delivery remains available.

以下保留前序步骤的当时边界 / Earlier steps below retain their historical scope.

完整 Prompt fan-in 已接入现有 `PVDKVReceiver`、Decode session 和异步 waiting
queue 驱动。D 配置 `--pvd-full-kv-fanin-max-slices`、
`--pvd-full-kv-fanin-response-bytes`，并开启 waiting bootstrap 后启用。
该路径允许 D TP1/TP2/TP4；V 仍是两个 storage ranks。每个 D rank 控制自身 MR，
所有 ranks 先同意网络成功，再解包/本地 fence，再同意安装，最后 ACK 和生成
原初始 receipt。周期刷新复用注册并更新 generation，保留生成 KV。

Full-Prompt fan-in now uses the existing PVD receiver/session and asynchronous
waiting-queue driver. Explicit Decode slice/response bounds plus waiting bootstrap
activate it for D TP1/TP2/TP4 with V storage TP2. Each D rank controls its MR;
all ranks agree on network completion, import/local fence, installation, then ACK
and the original initial-Prompt receipt. Periodic refresh reuses the registration
with a new generation and preserves generated KV. Required V sources must match
the destination rail; multi-HCA receive registration is still pending.

新增 8 项实际 store/HTTP/线程控制/解包集成测试，含真实 waiting gate；之后新增
5 项参数准入测试，定向共 **13 passed**。Windows 全量在前 8 项加入后为
**2412 passed / 29 skipped**；WSL 相关 **106 passed**。公共 Decode 路径修改后
严格 v5 CPU 实模四场景再次通过。GPU/RDMA 未执行。预测检索 Scheduler 自动
装配、原生 CAGRA、多 rail 和更通用拓扑仍需代码实现；不能据完整 Prompt 接入
宣称预测流水线已经开启。

Eight actual store/HTTP/thread-control/import integration cases include the real
waiting gate; five additional configuration tests bring the focused file to **13
passed**. Windows full after the eight integrations: **2412 passed / 29 skipped**;
WSL related: **106 passed**. All four strict v5 CPU model scenarios passed after
the shared Decode change. GPU/RDMA remain unexecuted. Predictive Scheduler assembly,
native CAGRA, multiple rails and broader topology remain implementation work.

D 显式 fan-in 会话也已接通 HTTP：首次 RPC 前固定完整源 epoch，验证完整响应
及逐 writer 凭据，响应丢失/协程取消后转入 fence；关闭 HTTP 不等于排空。
MR 网络 pin 与本地读取 pin 分离，本地回收失败后禁止重新发布。23 个新 CPU
用例包含真实双 V store → coordinator HTTP → D MR 的字节核对。
最新 Windows 全量 **2404 passed / 29 skipped**，WSL fan-in **110 passed**。
`ack_after_install()` 是明确的调用方义务，不是自动证明已完成 CUDA/所有 rank
安装；尚未接现有 `PVDDecodeSession`/waiting queue 或自动 Scheduler 工厂。

The explicit D fan-in session now drives bounded HTTP, pre-pins every source epoch,
validates the entire response and per-writer proofs, and fences after lost replies
or coroutine cancellation. Closing HTTP is not draining. Network and local-reader
pins remain distinct; failed local retirement cannot republish the destination.
Twenty-three new CPU cases include real two-store/coordinator HTTP/D-MR byte checks.
Latest Windows **2404 passed / 29 skipped**; WSL fan-in **110 passed**.
`ack_after_install()` is a caller obligation, not CUDA/all-rank installation proof.
Existing Decode-session/waiting-queue and automatic Scheduler assembly remain open.

全局 fan-in 协调与 HTTP 已接通：`--full-kv-fanin-max-records` 与两项 shard
限制共同显式启用。协调器在任何发布前固定全部 V epoch/写入身份，按所有 writer
的确切凭据聚合完成；丢失响应、取消和超时保持 Entry 引用直到排空。终止记录
保留且受容量限制，不做可能放过迟到请求的隐式淘汰。P 尚未完成时发出的 start
会记住，源就绪后继续执行。新 20 项 CPU 测试；Windows 全量 **2381 passed /
29 skipped**，WSL fan-in 定向 **87 passed**。D 自动准入、生产工厂、原生
CAGRA、多 rail/真实 TP 拓扑和 GPU/RDMA 验收仍未完成。

Global fan-in orchestration/HTTP is now opt-in through `--full-kv-fanin-max-records`
plus both shard bounds. All V epochs/identities are pinned before publication;
exact all-writer proofs govern completion. Lost replies, cancellation and deadlines
retain Entry ownership until drain. Terminal records remain bounded tombstones,
without unsafe implicit eviction. Start requests survive waiting for P completion.
Twenty new CPU cases; Windows **2381 passed / 29 skipped**, WSL focused **87 passed**.
Automatic D admission, the serving factory, native CAGRA, multi-rail/real-TP topology
and GPU/RDMA acceptance remain unfinished.

以下条目保留各步骤当时的验证边界；其中“尚未接通”描述的是该步骤当时的状态。
The following entries preserve evidence at each earlier step; their "not yet wired"
statements describe that historical step, not the global coordinator increment above.

V store 和 shard HTTP 现可 opt-in 完整 KV fan-in：真实 Entry 授权、双参数
容量限制、统一 start/poll/ACK/cancel、超时/关停排空和 absent-writer tombstone
已接通。14 个 CPU 用例覆盖实际 store/allocator 与 localhost HTTP。尚待全局
coordinator 聚合和 D 自动准入；不能解读为端到端预测检索已上线。

Opt-in V store/shard HTTP fan-in now uses real Entry authorization, explicit bounds,
shared delivery APIs, timeout/shutdown draining and absent-writer tombstones.
Fourteen CPU cases exercise real stores/allocators and localhost HTTP. Global
coordinator aggregation and automatic D admission remain; end-to-end predictive
serving is not enabled.

完整 KV fan-in 新增严格 wire plan 校验和有界 V writer：从实际源区间直接提交
offset PUT，保留原生句柄并防止取消/重复启动提前释放或重放。21 个 CPU 用例；
本步骤尚未接 V store/HTTP，不改变默认生产服务或 TP/rail 支持范围。

Full-KV fan-in adds strict wire-plan validation and a bounded V writer using direct
offset PUTs, retained native handles and one-shot authorization. Twenty-one CPU
cases cover the new executor. This step does not yet activate V store/HTTP or
change default serving, TP or rail support.

[完整 KV fan-in 字节规划](PVD_Full_KV_FanIn_CN_EN.md) 已新增全局 head 交集及
各 source/destination 相对偏移计算。29 个 CPU 用例使用真实 packer 验证跨 V
shard 重建；旧 wire 接口仍拒绝 fan-in，多 writer 的身份/完成/fence 协议尚待接入。

Full-KV fan-in byte planning now computes global-head intersections and both
source/destination offsets. Twenty-nine CPU cases reconstruct actual packer bytes.
Legacy wire APIs still refuse fan-in; multi-writer identity/completion/fencing
integration remains implementation work.

上述 fan-in 又增加了显式共享 MR 生命周期组件：按 V writer 验证身份和计划
指纹，全部终止后方可释放，全部成功后才 network-ready。32 个 CPU 测试覆盖
部分写入、取消、UNKNOWN 和两路 fake writer；尚未接 coordinator/HTTP 或旧
接收器，不能解读为生产多源交付已开启。

An explicit shared-MR lifetime component now validates V-writer identities and plan
fingerprints, requires all-terminal closure and all-success network readiness.
Thirty-two CPU cases cover partial writes, cancellation, UNKNOWN and two fake
writers. Coordinator/HTTP and legacy receiver integration are still pending;
production multi-source delivery is not enabled.

[CUDA Scheduler 显式绑定](PVD_CUDA_Scheduler_Binding_CN_EN.md) 已接普通 Decode
循环、waiting admission、分配前 wait-all 和原 Scheduler 结果包装器。未绑定时
仍走旧路径。新增 25 个 CPU 用例；WSL 定向 59 项通过。KV 压力当前选择中止 batch
并排空，不支持原地 retraction。生产工厂和拓扑仍未完成：实际 CUDA consumer
仅 TP1，而当前 CLI 的 D 是 TP2/TP4、V storage 是 TP2；不得只放宽参数冒充兼容。

An explicit CUDA binding now reaches the normal Decode loop, waiting admission,
pre-allocation wait-all and original Scheduler result wrapper. Unbound serving is
unchanged. Twenty-five new CPU cases and 59 focused WSL cases pass. Capacity
pressure aborts/drains the batch rather than retracting in place. Factories and
topology remain code gaps: the actual consumer is TP1, while serving config uses
D TP2/TP4 and V TP2. Relaxing flags alone would not make these compatible.

已逐步测试、提交并推送：V opt-in CUDA packing (`6992c78e2`)、D CUDA banks
(`23f9e5337`)、CUDA rank participant (`ff8e653f2`)。本页所在提交还增加有界显式
scratch 的独立 CUDA attention 消费基线。默认生产服务未切换为预测稀疏模式。
Tested incremental components include opt-in V CUDA packing, D banks, rank
agreement and standalone tiled attention. Serving remains on its original path.

最新 Windows 全量：**2240 passed / 29 skipped**；CUDA Delivery 定向 **7 passed**。
新增 draft 完成屏障、UNKNOWN 实际 owner 保留、整个共享 provider 隔离及错误分配
清理；14 个新增 CPU 故障用例通过，其中首批 5 个在修复前失败。
详见 [Draft 完成与隔离 / Draft completion](PVD_Draft_Completion_CN_EN.md)。
Latest full Windows regression is 2240/29; CUDA Delivery policy tests pass 7 cases.
Draft retirement now fences work/map clearing/allocator updates, retains actual
owners on UNKNOWN, quarantines the shared provider and cleans up malformed
allocations. Fourteen new CPU cases pass; the first five failed before the fix.
Device-wide fencing is a conservative baseline, not latency/overlap evidence.

最终代码另通过 WSL draft 定向 **140 passed**（3 条已有 CPU 平台警告），以及
严格 v5 四场景真实 CPU 模型矩阵。仍为 TP1 CPU、fake payload，不是 GPU/RDMA。
Final-source WSL draft regression passes 140 cases (three existing CPU-platform
warnings), and all four strict v5 real-model CPU cases pass again. This remains
TP1 CPU with fake payload transport, not GPU/RDMA evidence.

[CUDA 查询桥接](PVD_CUDA_Query_Bridge_CN_EN.md) 新增独立 copy budget、目标/draft
RNG scope、共享执行锁及 UNKNOWN owner 保留。12 个 CPU 策略/HTTP 用例通过；
生产 CUDA request/controller 与 Scheduler 装配尚未因此完成。
The explicit CUDA query bridge adds copy admission, serialized target/draft RNG
scope and UNKNOWN owner retention. Twelve CPU policy/HTTP cases pass; this alone
does not complete production Scheduler assembly.

[CUDA 每请求控制器](PVD_CUDA_Request_CN_EN.md) 随后已接通上述桥接、逐 shard 搜索、
GQA union、Delivery 和 runtime 安装；6 个新增 CPU 策略用例通过，共享控制器
改造后的严格 v5 四场景实模 CPU 矩阵也再次通过。完整初始 Prompt 的生产引导、
多请求 CUDA batch/队列、真实 TP 和服务启动工厂仍待接入。
The CUDA per-request controller now connects that bridge, shard search, bounded
union, Delivery and runtime installation. Six new CPU policy cases and all four
strict v5 real-model CPU cases pass after the shared-controller refactor. Serving
full-Prompt bootstrap, multi-request CUDA batch/queues, real TP and startup
factory assembly remain separate implementation work.

[CUDA 多请求 runtime batch](PVD_CUDA_Rank_Batch_CN_EN.md) 已绑定 wait-all、
model consumer、结果处理期间的全部 permits/池 lease/target 锁。10 个新增
CPU 用例通过，WSL batch/runtime/model 定向 59 项通过；生产队列/工厂尚未
调用此组件，实际模型 TP 也不由多个请求的测试替代。
The CUDA multi-request executor now binds wait-all admission and model consumption
to permits, allocator leases and target locking through result processing. Ten
new CPU cases and 59 focused WSL batch/runtime/model cases pass. Serving queues
and factories do not yet invoke it; request batching is not actual model TP.

[无索引完整 Prompt 引导](PVD_CUDA_Prompt_Bootstrap_CN_EN.md) 已增加有界 staging、
实际请求映射导入及初始 runtime 安装。10 个 CPU 用例通过；全量之后加强的
UNKNOWN 不重试规则另经 21 个定向用例复验。严格 CUDA smoke 已接此入口但未执行。
调用者仍需先证明原完整 Prompt 接收/unpack 已完成，生产接收器自动挂接尚未完成。
Index-independent full-Prompt bootstrap now imports actual request-mapped pool
rows through budgeted staging and initial runtime installation. Ten CPU cases
pass; the final sticky-UNKNOWN refinement passes 21 focused cases after the full
suite. The CUDA smoke is wired but unexecuted. Serving must still provide proven
full-receive/unpack completion and invoke the importer at the correct boundary.

[V 索引完成与隔离](PVD_Index_Completion_CN_EN.md) 已增加 build/search/dispose
完成契约；UNKNOWN 保留实际源 pin、reader 和预算，停止新操作且不自动重试。
7 个新增 CPU 用例、WSL 定向 108 passed / 1 skipped；尚未实现原生 CAGRA backend。
The index lifecycle now fences build/search/dispose and retains real source pins,
readers and reservations on sticky UNKNOWN. Seven new CPU cases and focused WSL
108/1 pass. Native CAGRA integration is still separate implementation work.

[CUDA 同线程刷新驱动](PVD_CUDA_Refresh_Driver_CN_EN.md) 已增加同步 owner-loop polling，
直接读取正式 Req 的 token 数而不追加输出；新请求不重置旧时钟。WSL 组合 33 项通过，
包括真实 Req 字段与 HTTP；CUDA placement/payload 仍为 CPU/fake 替代。
生产工厂、waiting-queue hook、结果处理绑定及原始 allocator 退还仍须接入。
The owner-thread CUDA refresh driver now polls asynchronous work from a synchronous
loop and observes authoritative Req counts without writing tokens. New requests
do not reset old clocks. Thirty-three focused WSL cases pass, including real Req
fields and HTTP; placement/payload remain CPU/fake. Serving factory/queue hooks,
result binding and original allocator retirement still require integration.

[CUDA 正式结果桥接](PVD_CUDA_Result_Bridge_CN_EN.md) 随后已为显式 CUDA batch
接通原结果处理器：只观察正式 token 写入，不重采样；结果重放和 partial commit
失败都禁止重试。可复用已完成的 ScheduleBatch，未完成 owner 不可覆盖。
生产 factory/队列仍未自动调用此入口，原 allocator 退还与实际 TP 仍待接入。
The explicit CUDA result bridge now invokes the original result processor,
observing authoritative writes without resampling. It refuses replay and retries
after partial commit, and permits completed ScheduleBatch reuse without replacing
active owners. Serving factory/queue activation, original allocator retirement
and actual TP integration remain separate work.

[CUDA 原始请求池回收](PVD_CUDA_Request_Retirement_CN_EN.md) 已接公共 cache 回调和
driver close：allocator 读取、映射清零、host slot 归还分别排序；UNKNOWN 禁止
两个池继续分配。工厂仍须为每个显式 CUDA 请求安装 owner，尚未自动全服务启用。
Explicit CUDA request retirement now connects the common cache callback to driver
close, ordering allocator reads, map clearing and host-slot reuse. UNKNOWN poisons
both pools. The production factory must still install each owner; serving-wide
activation is not implied.

完整 Prompt 接收器现只在最终 ACK/rank agreement 成功后生成 session 绑定凭据，
`install_received` 验证真实 Req/池/映射后导入 CUDA bank。padding 行不得导入；
UNKNOWN 保留 arbiter 并 poison 两个原池。新增 24 个接收交接 CPU 用例通过，
WSL 接收/引导/真实池回收/旧 PVD 定向 **103 passed**。控制器装配及刷新所有权
切换仍须由后续接点完成，默认服务未启用预测检索。
The full receiver now mints session-bound evidence only after final ACK/rank
agreement; install_received checks the live Req/pools/map before importing the
CUDA bank. Padding is refused and UNKNOWN retains the arbiter and poisons both
pools. Twenty-four new CPU handoff cases pass, with 103 focused WSL receiver,
bootstrap, real-pool retirement and legacy PVD cases passing. Controller assembly
and refresh-ownership transfer still remain; default serving is unchanged.

随后增加 [接收 session 所有权交接](PVD_CUDA_Receiver_Ownership_CN_EN.md)：
显式导入/注册/release owner 全部一致后才可 claim；旧全量刷新不重复拉取，
结束时先排空稀疏 controller，再关闭原 session，最后回收请求池。12 个新增
CPU 用例、WSL 定向 66 项和严格 v5 真实 CPU 四场景矩阵通过。默认生产启动
工厂/队列仍未自动调用，尚不代表完整服务激活。
The subsequent receiver-ownership handoff requires matching import, registration
and release ownership. Legacy refresh stops pulling claimed requests. Sparse
controller drain precedes source-session close and original pool retirement.
Twelve new CPU cases, 66 focused WSL cases and all four strict v5 real-model CPU
cases pass. Default factory/queue activation remains unimplemented.

结果桥接进一步支持原 Scheduler 外层处理，保留负载/指标/健康回调，并在外层
返回后复核正式输出与 batch 成员；7 个新增 CPU 用例通过，WSL 结果组合 21 项
通过。实际运行生产 Scheduler 队列/工厂仍是后续实现任务。
Result bridging now supports the original outer Scheduler handler, preserving
load/metrics/health callbacks and rechecking output/membership afterwards. Seven
new CPU cases and 21 focused WSL result cases pass. Actual serving queue/factory
activation remains implementation work.

原生 retraction 现明确拒绝带 CUDA release owner 的 Req：防止 offload 后在延迟
释放前清零 KV 长度，或复用旧 tombstone。新增 5 个用例，Windows 4 passed /
1 skipped，WSL 真实 Req/回收/结果定向 47 项通过。异步 OOM 恢复/新 incarnation
准入仍未实现，不能将防护性拒绝当作该功能完成。
Native retraction now refuses CUDA-owned Req objects before offload/reset can
destroy the deferred-release ledger or reuse a tombstone. Five new cases yield
4 passes/1 skip on Windows; 47 focused WSL real-Req/retirement/result cases pass.
Async OOM recovery/new-incarnation admission remains unimplemented; protective
refusal is not implementation of that feature.

这些 CUDA 路径已有代码和 CPU 策略/数学验证，**尚无 CUDA 执行证据**。生产模型池、
接收可见性、真实 rank transport 和原生 CAGRA 仍有接入任务，不能写成“仅缺硬件测试”。
CUDA implementations have CPU policy/math coverage, not CUDA execution evidence.
Model-pool integration, receive visibility, real rank transport and native CAGRA
still require implementation as well as hardware validation.

## 较早完成记录 / Earlier completion record

| Step | Commit | Result / 结果 |
|---|---|---|
| 1 | `851cd3262` | 有界 owner 回收驱动、普通 Decode 轮询、pending owner 阻止 idle 误报/休眠 / bounded retirement and idle protection |
| 2 | `19fcc8985` | 实模自动回收、一次性映射清零、严格 v5 证据 / real-model automatic cleanup and strict evidence |
| 3 | `21c0575e8` | 单请求回收入口失败隔离、关闭先停止全部生命周期 / isolate release-intent failure without blocking peers |
| 4 | `7d8af4369` | 修复 draft 常驻预算漏计/绕过、限制诊断历史 / persistent accounting and bounded diagnostics |
| 5 | `3bcb18983` | 分支私有请求 owner，释放与执行共用锁 / branch-local request metadata and serialized cleanup |
| 6 | 本页所在提交 / this commit | 启动能力提示、严格整数校验、集中状态说明 / truthful startup status, bounds and consolidated scope |

提交存在不等于远端已更新。推送结果以 Git 命令成功回执和远端 hash 为准。
A local commit is not proof of upload. Confirm successful push and the remote hash.

## 可复验的证据 / Reproducible evidence

在具备本项目 CPU 依赖的 Linux/WSL 环境，从仓库根目录运行：
With the project's CPU dependencies installed, run from the repository root:

本轮最终全量回归：Windows **1833 passed / 15 skipped**，WSL **1839 passed /
9 skipped**。最后一项启动配置/帮助修改另有 195 个定向用例通过（其中 10 个新增）。
跳过的检查不计为通过；完整四场景实模矩阵在分支池修复后已通过，最后的配置
提示修改不改变模型执行路径。
Final full regression: Windows **1833 passed / 15 skipped**, WSL **1839 passed /
9 skipped**. Startup/configuration changes pass 195 focused cases (10 new).
Skipped tests are unverified. The four-case real-model matrix passed after the
branch-pool fix; the final configuration/help change does not alter model execution.

```bash
PYTHONPATH=python python test/registered/disaggregation/run_pvd_rank_model_acceptance.py --fault all --timeout-seconds 300
```

- 严格 v5 四场景：正常、迟到 RESUMED、部分安装失败、清理重试。
  Strict v5 covers normal, delayed RESUMED, partial install and cleanup recovery.
- CPU FP32、TP1 实际模型；两个随机 tiny Llama，共用测试 tokenizer；本地 rank
  控制与 HTTP，payload 是 fake byte copy，不是 RDMA。
  Actual models are CPU FP32/TP1, two random tiny Llamas and a toy tokenizer;
  local rank control/HTTP, fake payload copies, not RDMA.
- 完整场景 21 次 attention 对照，最大误差约 3.58e-7；4 次 HTTP 分片交付、1600
  字节。部分安装失败提前结束，不冒充完成后续正常路径。
  Full-length cases perform 21 attention checks, about 3.58e-7 max error and four
  HTTP deliveries / 1600 bytes. Partial install deliberately exits earlier.
- 实际 CPU Req/分配器、ChunkCache、结束回调和等待队列取消方法已执行；并未启动
  完整生产 Scheduler 服务，辅助服务仍有 fixture doubles。
  Actual CPU pools, Req, ChunkCache and finish/waiting-abort methods are exercised;
  this is not a complete production Scheduler process.

## 仍待完成：区分代码接入与设备验收 / Remaining implementation versus hardware gates

| Area / 部分 | Remaining / 尚缺 |
|---|---|
| Production Scheduler | 预测/检索/稀疏安装的完整服务装配、真实队列与多进程协同；CPU hook 不等于此项完成 / full serving activation and queue/process integration |
| Memory-pressure retraction | CUDA 延迟回收不允许原位 reset；尚需压力准入/退出或排空后的新 incarnation 恢复 / no in-place reset; pressure policy or drained new-incarnation resume still required |
| GPU sparse attention | 已有显式 TP1 模型池/backend 工厂；尚未装配生产 Scheduler，仍缺 stream/event 优化与真实 GPU forward 验证 / explicit TP1 model-pool/backend factory exists; serving assembly and GPU acceptance remain |
| Native sparse Delivery | 接到 Mooncake 原生 submit/poll/fence/ACK，并验证取消/错误时仍有 WRITE 的内存保护 / native transport integration and in-flight WRITE safety |
| Real model TP | 实际 TP ranks 的安装、恢复与失败协同；本地逻辑 rank 镜像不能代替 / actual distributed model-rank integration |
| V CAGRA | 原生 backend 与单图 V100S 合成探针已通过；仍缺服务端多图/多 Entry 预算、真实目标 Q recall 与模型质量对照 / native adapter and one synthetic V100S graph pass; serving multi-graph budgets and real-query recall/quality remain |
| Performance | 有代表性的目标/draft 模型、数据、V100S/RDMA 实验；测预取窗口、等待、TPOT、尾延迟、吞吐和显存 / representative model/hardware benchmarks |

以上不应缩写为“代码已全部完成，只需上机器测试”：仍有生产实现任务。
设备可用后需要边实现边验证；不能删除 CPU-only 检查或把 fake transport 换名后
宣称完成。具体模型保持用户自定义，不因本地无硬件而锁死 checkpoint/version。

These are not merely tests of already-finished production code. Hardware-facing
implementation remains. Implement and validate together when the target stack
is available; do not remove CPU-only guards or relabel fake transport as native.
Model/checkpoint choice remains configurable.

增量实现：[V CUDA 稀疏打包基线](PVD_CUDA_Sparse_Packing_CN_EN.md) 已增加默认关闭的
显式服务开关、GPU 最终 staging 和完成后提交策略。同步失败保留全部租约/预算。
这只补 V 端 source packing 接点，不代表 D GPU 稀疏路径或原生 RDMA 已通过。
The opt-in [V CUDA packing baseline](PVD_CUDA_Sparse_Packing_CN_EN.md) now owns final
GPU staging and synchronizes before submission, retaining owners on uncertainty.
It does not activate or validate D GPU sparse execution or native RDMA.

D 端增加独立的 [CUDA current/next 工作集](PVD_CUDA_Working_Set_CN_EN.md)：
显式设备/预算、源范围 guard、复制与 reader 完成后释放，以及 UNKNOWN 隔离。
这不是生产 GPU attention/接收器接入；CPU 安装器仍拒绝 CUDA bank。
D now has a separate [CUDA bank implementation](PVD_CUDA_Working_Set_CN_EN.md)
with bounded copies and fail-closed completion ownership. Production attention,
receive visibility and distributed installation are not activated by this change.

进一步增加 [CUDA 逐 rank 安装 participant](PVD_CUDA_Rank_Install_CN_EN.md)，
将 bank 接入 PREPARED/PARKED/APPLIED/RESUMED 协议；全组 ACK 门控保留。
这是逻辑协议接点，不是实际 TP launcher、GPU attention 或 RDMA 接收集成。
The [CUDA rank participant](PVD_CUDA_Rank_Install_CN_EN.md) connects bank completion
to the existing rank agreement. Actual model-rank transport and serving assembly
remain distinct implementation/acceptance work.

独立 [CUDA 稀疏 attention 基线](PVD_CUDA_Sparse_Attention_CN_EN.md) 已实现按块在线
softmax、固定大小显式 scratch、participant reader 和输入/output guard。
没有注册为生产 backend；native library workspace、模型池绑定和性能仍未完成。
The standalone [CUDA attention baseline](PVD_CUDA_Sparse_Attention_CN_EN.md)
adds fixed explicit scratch and guarded tiled consumption without concatenating
full context. It is not a production backend or a total device-memory bound.
增量支持直接按非连续生成 KV 池行读取，未选中槽位不读；另修复后续 reader 排空
失败时提前 unpin generated/output 的漏洞。CPU 数值/故障回归通过，GPU 用例未执行。
Mapped generated-pool rows now feed fixed tiles directly. A reproduced late-reader
drain failure no longer unpins generated/output ownership. GPU cases remain unrun.

[CUDA 模型池/backend 适配](PVD_CUDA_Model_Attention_CN_EN.md) 增加整次 forward 的
allocator lease、目标锁、batch 映射预检查、output 预算和异常隔离。显式工厂与
严格五步真实 CUDA 模型检查已写好；38 个 CPU 用例通过，GPU 检查仍为 blocked。
The explicit model adapter adds whole-forward pool ownership and a strict real-GPU
smoke. CPU tests cover its policies/math, not actual CUDA execution or serving.
随后补充的元数据读取早期失败回归在 Windows/WSL 均通过（adapter 定向 38 项）。
The subsequent early metadata-failure regression passed on Windows and WSL
(38 adapter cases), retaining inputs when device completion is unknown.

[CUDA 本地 runtime](PVD_CUDA_Runtime_CN_EN.md) 进一步绑定接收、安装状态机和
model forward permit；输出只有在 runtime 接受后才能提交，UNKNOWN 保留 permit。
当前明确仅 TP1、每次单请求 forward，不代表真实 TP2 或生产多请求 batch 已接通。
The local CUDA runtime binds receive/install to a model execution permit and
post-drain result admission. TP1/single-request scope only; not real TP2 or
production multi-request batch assembly.

HTTP sparse Delivery sink 已接入该 CUDA runtime，统一目的地、远端完成、staging、
安装后 ACK 和回收。目标 dtype 来自 bank，等待中超时不会被当作 native fence。
The CUDA HTTP sink now drives destination publication, successful receive,
runtime staging, post-install ACK and retirement. Timeout never proves native
completion; dtype is read from the destination bank. This is not production activation.

设备可用后运行 [CUDA 组件严格验收](PVD_CUDA_Component_Acceptance_CN_EN.md)。入口要求
9 个明确 CUDA 用例全部执行成功，无设备/skip/缺测不会成为通过；无 GPU 的本地环境
仍是 blocked。CloudLab `clgpu020` 已在提交 `82de6a548` 上以 V100S-PCIE-32GB
通过 9/9 项，但不证明 RDMA、模型 forward、CAGRA、性能或生产服务。
The strict CUDA component gate refuses missing/skipped evidence. Its local CPU-only
result remains blocked. CloudLab `clgpu020` has now run all 9/9 fixed CUDA
component cases successfully on a V100S-PCIE-32GB at commit `82de6a548`; this
does not certify RDMA, model forward, CAGRA, performance or production serving.

[CUDA sparse receive](PVD_CUDA_Sparse_Receive_CN_EN.md) 已将私有 GPU destination、
注册前 SYNC_MEMOPS、精确远端成功证明、CUDA 排序和 bank staging/RESUMED ACK 接通。
可使用 Mooncake adapter；生产工厂装配与真实 NIC/GPU 验证仍未完成。
The explicit CUDA receive component now connects registration ordering and remote
proofs to bank staging and all-rank ACK. Production factory wiring and native
NIC/GPU acceptance remain separate tasks, not completed by CPU policy tests.

[CUDA target-Q probe](PVD_CUDA_Target_Probe_CN_EN.md) 已增加显式 CUDA placement、
私有池、与正式目标共用的 execution lock 和失败排空/隔离；共享核心的构造失败
回收路径也已补齐。仍限 Llama/TP1/torch_native，尚未由生产 Scheduler 构造。
The CUDA probe component and strict real-GPU smoke are implemented. Local evidence
is CPU policy coverage plus the re-run real CPU model matrix, not GPU execution.

CAGRA 版本核对及预检方法见 [兼容性说明](PVD_CAGRA_Compatibility_CN_EN.md)。
预检 v3 记录实际导入的版本和模块路径，可选版本断言不等于强制锁版本。
cuVS 历史版本和当前版本的架构要求不同，尚未据此宣布 V100S 实测通过。
See the [CAGRA compatibility note](PVD_CAGRA_Compatibility_CN_EN.md): probe v3
records imported identity and optional version assertions without imposing a pin.
Release-specific architecture requirements are not V100S execution evidence.

### 索引退预算顺序 / Index refund ordering

新增 CPU storage 弱引用回归，复现并修复关闭、迟到构建、部分构建失败及关闭中
检索退出时先退预算后释放 tensor 的窗口。现在先清掉 manager 的引用，再退预算；
失败后已结束的 Python traceback frame 也清理 tensor locals，保留异常和调用位置。
这不构成 CUDA stream/native handle 已完成的证明，未来 CAGRA 仍需原生生命周期。

Real CPU storage weak-reference regressions exposed refunds preceding tensor release
on close, late/partial builds and searches leaving a closed record. Manager references
and finished exception-frame tensor locals are now dropped before refund; exception
types and traceback locations remain. This is not a CUDA/native completion fence.
Native CAGRA lifecycle integration is still required.

本次增量验证 / Incremental validation: Windows 全量 **1849 passed / 15 skipped**；
WSL 索引、回收与 CAGRA 预检定向 **121 passed / 1 skipped**。6 个新回收测试检查
实际 storage；CAGRA 实际 build/search 未执行。
Six new retirement cases observe actual CPU storage; native CAGRA was not executed.

### 检索后端结果边界 / Retrieval backend result boundary

`select()` 不再将后端结果直接 flatten/zip 后转换：先检查返回结构、query/top-k、
精确形状、行号整数类型、score 浮点类型、设备、范围、有限性及每条 query 内无重复。
不同 query 选择同一 token 仍取最高分并去重，GQA 语义不变。错误不能因 zip 截断、
`int(0.5)` 或 NaN 比较而成为看似有效的工作集。29 个新增 CPU 契约测试；其中
16 个在修复前失败。`l2` 的实际 score 约定以 -5 而非 -25 的数值例固定。

`select()` now validates result structure, query/top-k, exact shape, integer rows,
floating scores, device, range, finiteness and per-query uniqueness before mapping.
Cross-query deduplication/best-score union is unchanged. Truncated zip results,
coerced fractional IDs and NaNs cannot silently become a working set. Twenty-nine
new CPU contract cases include sixteen that failed before the fix; an explicit
-5 versus -25 example fixes negative-Euclidean `l2` semantics.

检索契约修改后 Windows 全量 **1878 passed / 15 skipped**，WSL 定向 **203 passed /
6 skipped**；严格 v5 四场景实模 CPU 矩阵再次全部通过。完整场景仍为 21 次
attention 对照，最大误差约 3.58e-7。传输仍为 fake byte copy；这些结果不是原生
CAGRA、生产 Scheduler、GPU 或 RDMA 验收。
After the contract change, Windows full regression is **1878 passed / 15 skipped**
and WSL focused regression is **203 passed / 6 skipped**. All four strict v5
real-model CPU cases passed again (21 attention checks in full cases, about
3.58e-7 max error). Payload transport remains fake; no native CAGRA, full serving,
GPU or RDMA claim follows.

WSL 全量补验 / WSL full regression: **1884 passed / 9 skipped**，3 条已有 CPU
平台警告 / three existing CPU-platform warnings.

### 索引发布边界 / Index publication boundary

构建返回值也新增验证：必须是 `BuiltIndex`，行数/维度须是精确整数并与该 head
输入一致，vector space 和 metric 必须与本次请求一致。此前错误 space/metric
可以被发布为 READY；错误对象或浮点 count 还可能在发布阶段抛异常，留下
BUILDING 状态及预算。9 个用例先复现再修复，覆盖第二个 head 才出错时清理
已建部分、保留完整 KV 交付能力，以及后续正确重试。校验不声称能证明不透明
原生句柄里的向量内容正确。

Build results must be `BuiltIndex` objects with exact integer shape and the requested
vector space/metric before publication. Previously wrong identities could become
READY, while a wrong object or float count could strand BUILDING state and budget
during publication. Nine reproduced cases cover partial-build disposal, continued
full-KV deliverability and a later successful retry. Metadata checks do not prove
the contents of opaque native handles.

发布校验完成后 / After publication validation: Windows 全量 **1887 passed /
15 skipped**；WSL 构建/回收/索引定向 **95 passed / 1 skipped**。四场景 CPU
矩阵在前一步结果契约修改后通过；本步骤只新增构建返回值检查，没有另称已重跑
矩阵。仍未执行 GPU/cuVS/RDMA。
The four-case CPU matrix passed at the preceding result-contract step; this final
build-metadata guard does not claim a separate matrix rerun. GPU/cuVS/RDMA remain
unexecuted.

## 固定设计约束 / Invariants to preserve

### 原生 MR 注册失败隔离 / Native MR registration failure isolation

Mooncake 原生 `register_memory()` 抛异常或返回非零时，PVD 不能证明 GPU MR
没有生效，因此将 engine 和 CUDA buffer 保留至进程结束，标记共享 engine
不健康，并拒绝新的注册及 PUT。原生注册成功、但 descriptor/guard/发布失败时，
先尝试原生注销；只有注销成功才放开 buffer。注销失败同样隔离并保留 buffer。
健康检查即使无法读取 session ID 也会返回不健康状态。这是保守的失效保护，
不是实际 Mooncake/GPU/RDMA 的错误注入验收。

If native `register_memory()` raises or returns nonzero, PVD cannot prove the GPU
MR was never installed. It retains the engine and CUDA buffer for process life,
marks the shared engine unhealthy, and refuses new registrations and PUTs. If
descriptor/guard/publication fails after a successful native registration, the
buffer is released only after native unregister succeeds; failed rollback is
quarantined the same way. Health remains reportable when session-ID lookup fails.
This is conservative fail-closed behavior, not GPU/RDMA fault-injection evidence.

另外，原生 submit/poll 进入不可追踪状态时，`health()` 现在会显示底层
`TransferLifecycleManager` 的隔离状态及原因，新 MR 注册在调用 Mooncake 前即被拒绝。
已有未知 WRITE 的源 buffer 和预算继续保留，不能因健康报告而被释放。
If native submit/poll becomes untrackable, `health()` now reports the lifecycle
quarantine and its reason. A new MR is refused before calling Mooncake; the
source buffer and budget of an unknown WRITE remain retained.
原生注销失败或抛异常时也会把对应 CUDA buffer 与 engine 保留在进程级隔离表，
避免 adapter 被回收后悬挂 MR；只在该 MR 后续成功注销时删除它的隔离项。
若同一 engine 还有别的未确认 MR，仍保持不健康。重试注销仍允许，
新注册与 PUT 在故障恢复前拒绝。
Native unregister failure or exception likewise retains its CUDA buffer and
engine at process scope, even if the adapter becomes unreachable. Only a later
successful unregister clears that MR's quarantine; another uncertain MR keeps
the shared engine unhealthy. Retrying release remains allowed, while new
registrations and PUTs are refused until recovery.

- 新请求不重置旧请求的时钟或预取。刷新按每个请求正式提交的 D token 计数。
  New requests never reset existing clocks/prefetch; count committed D tokens only.
- draft 只预测检索位置；目标模型输出是唯一正式输出，不启用原生 speculative
  generation。迟到查询等待；错过窗口首次补查用正式前缀的目标 Q。
  Draft predictions never become output; missed-window fallback uses actual-prefix Q.
- GQA 在同一 layer/KV head 内取 token 并集并去重，有显式上限；不跨组排名。
  Bound the union per layer/KV head; never merge unrelated groups' scores.
- 首次完整 Prompt 到位后方可执行；生成 KV 保留 D；Entry 可被多次 Delivery 复用。
  Initial Prompt must be installed before Decode; generated KV stays on D; Entries are reusable.
- 取消、超时、UNKNOWN 都不是原生完成证明；未排空不复用资源、不退预算。
  Cancellation, timeout and UNKNOWN are not native fences; ownership outlives the operation.
