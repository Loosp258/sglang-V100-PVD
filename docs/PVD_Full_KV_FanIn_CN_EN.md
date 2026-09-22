# 完整 Prompt KV 多源合并 / Full-Prompt KV fan-in

## Decode 服务接入 / Decode serving integration

当前已由 `PVDKVReceiver` 选择 `PVDDecodeFanInSession`，并由原
`PVDDecodeRefresher` 的 bootstrap/periodic 驱动执行。以下是需添加到既有
D 命令的参数，具体容量按 prompt 长度、KV components 和源数选择：

```text
--pvd-waiting-queue-bootstrap
--pvd-full-kv-fanin-max-slices <positive-plan-bound>
--pvd-full-kv-fanin-response-bytes <positive-HTTP-byte-bound>
```

V 同时启用本文的三项 fan-in bounds。仅该显式路径允许 D TP1（TP2/TP4 亦可）；
V storage 仍 TP2。D TP1 合并两个 V source 时，两源必须能使用 D descriptor
指定的同一 rail。双 rail 跨源目标注册尚未实现，预检不符会拒绝交付。
`/v1/fanin/preflight` 只返回全局能力和各源 rank/epoch/rail/ready/bounds，
避免把不断增大的 Entry 列表传给每个请求。

Add the three Decode flags above to the existing launch command and enable the
three V bounds documented below. This explicit path supports D TP1/TP2/TP4,
with V storage still TP2. A TP1 destination combining both sources currently
requires a matching rail for every writer. Multi-HCA destination registration
is pending and incompatible rails fail preflight. Compact preflight returns
only capabilities and per-source rank/epoch/rail/readiness/bounds.

Scheduler 负责 prepare、TP 对齐、解包与本地 fence；控制线程创建/持有
fan-in receiver 和异步 HTTP 会话。每个 D rank 都有自己的 Future，原
bootstrap 驱动共同确认所有 Future 完成。解包前确认全部 ranks 成功，安装后
再次对齐才 ACK，ACK 后对齐才放行 waiting gate、生成原始 Prompt receipt。
控制结果使用本地对象身份绑定，外部 JSON 不能伪造解包凭据。

The Scheduler owns preparation, TP agreement, import and local fences. The control
thread owns the receiver and HTTP session. Every D rank has its own Future, and the
existing bootstrap driver waits for agreement across all of them. Network agreement
precedes import; installation agreement precedes ACK; ACK agreement precedes the
waiting gate and original initial-Prompt receipt. A local object identity binds the
network result to its session before unpack. Whole-operation timeout is 300 seconds;
expiration keeps ownership for fencing instead of imposing a fixed poll-count limit
that rejects long prompts. Registration persists across refresh generations.

测试：8 项实际 V store/HTTP/线程模拟 TP 集成（D1/2/4、原 waiting gate、部分
解包失败、取消、rail 拒绝、周期复用及生成 KV 保留）；5 项参数测试。
Windows 全量在前 8 项加入后 **2412/29 skipped**；定向最终 **13 passed**；
WSL 相关 **106 passed**；严格 v5 CPU 四场景再次通过。GPU、真实 TP collectives
和 RDMA 未执行。以下内容保留此前步骤的历史集成边界。

Eight actual-store/HTTP integration tests simulate TP with thread barriers; five
configuration cases complete the focused suite. Full Windows after integration:
**2412 passed / 29 skipped**; final focused: **13 passed**; WSL related: **106
passed**. All four strict v5 CPU model cases passed again. No GPU, native TP
collectives or RDMA execution. Older sections below preserve earlier boundaries.

## D 会话与下一接入点 / D session and next integration point

`FullKVFanInDelivery` 包装一个尚未发布的真实 `FullKVFanInReceiver`；构造时
用 caller 从可信 V 预检固定的完整 `source_epochs` 生成身份，再一次性发布。
`FanInHTTPClient` 要求显式 RPC 超时和响应字节上限，不隐藏重试或跟随重定向。
即使第一次 reserve 已在远端处理但回复丢失，D 仍持有原身份，可 fence 所有源。

The session wraps an unpublished real receiver, derives the full identity set from
caller-pinned trusted V preflight epochs, then publishes once. HTTP has explicit
timeout/response-size bounds and no hidden retry or redirect. A lost initial reserve
reply cannot lose the identities needed to fence every possible writer.

正常顺序：`reserve()` → `start()` → `poll()` 到 `ready` → caller 解包、完成
本地 fence 及所有 D ranks 安装确认 → `ack_after_install()` → `close()`。
`ready` 只表示网络成功；本地读取者自行持有同一 MR guard 的额外 pin。
失败顺序：`cancel()`/RPC 异常 → `poll()` 或 `fence()` 持续取得原身份凭据 →
`close()`。缺任一 writer 的确切终止凭据，close 拒绝且不释放 MR。HTTP client
close、逻辑 timeout、协程 cancellation 均不替代 writer fence。异步操作仍在
进行时不能 close；MR release callback 失败后由原 guard 重试本地清理，禁止
该会话再次发布。调用方必须保存尚未排空的会话，不可丢弃引用冒充取消。

Normal sequence: reserve, start, poll until network-ready; caller imports, locally
fences and obtains all-D-rank installation agreement; then ACK and close. Local
readers need separate guard pins. On error/cancel, poll/fence with the original
identities until all exact terminal proofs arrive. Missing proof prevents close.
Neither closing HTTP nor a logical timeout/coroutine cancellation proves native
closure. An outstanding RPC prevents close. If MR cleanup fails, the original
guard retries local cleanup; the session cannot republish. Retain undrained sessions.

23 个新 CPU 用例覆盖实际 localhost HTTP、双 V 字节合并、慢 writer、丢失
reserve/start/ACK 回复、取消协程、伪造/不完整回复、独立本地 reader pin 和
MR 清理失败。Windows 全量 **2404 passed / 29 skipped**，WSL fan-in 定向
**110 passed**。GPU/RDMA 未执行；严格 CPU v5 矩阵在前一协调层步骤通过，本
步骤不另称重跑。

Twenty-three new CPU cases cover actual localhost HTTP, two-source byte reconstruction,
slow writers, lost reserve/start/ACK replies, cancelled coroutines, bad envelopes,
separate local-reader pins and MR cleanup failure. Windows full: **2404 passed /
29 skipped**; WSL focused fan-in: **110 passed**. No GPU/RDMA run; the strict CPU v5
matrix passed at the preceding coordinator step, not separately rerun here.

下一代码任务：将这个会话接入 `PVDDecodeSession` 的原生控制线程/调度线程边界，
复用既有 waiting 准入、staging budget、解包/本地 fence、全部 D ranks ACK 与
初始 receipt；不能只因 network-ready 就进 batch。当前仍是显式组件，未替换
原始 session，未开放 TP1 服务或多 rail 接收，也没有自动开启预测检索。

Next code task: integrate with the existing Decode session's control/scheduler-thread
boundary, waiting admission, staging budget, import/local fence, all-D-rank ACK and
initial receipt. Network-ready alone cannot admit a request. This is still explicit,
not legacy-session replacement, TP1 serving/multi-rail activation or automatic
predictive retrieval. Native CAGRA and serving factory work also remain.

## 全局协调接口 / Global coordinator API

显式启用：同时给 V 配置 `--full-kv-fanin-max-slices`、
`--full-kv-fanin-max-inflight` 和 `--full-kv-fanin-max-records`。最后一项限制
包括终止 tombstone 的保留记录总数，达到上限拒绝新交付，不默默丢弃迟到请求屏障。
缺省关闭；两种 V launcher 均传递该配置。现有 delivery/retrieve 不变。

Opt in with all three bounds above. `max-records` caps retained records including
terminal tombstones; exhaustion refuses new deliveries without unsafe eviction.
Both launchers forward the option; legacy delivery/retrieve behavior is unchanged.

- `POST /v1/fanin/reserve`：`manifest` 和完整 `source_epochs: {"0": "...", ...}`。
- `POST /v1/fanin/start|poll|ack`：`delivery_id` 和精确整数 `destination_rank`。
- `POST /v1/fanin/fence`：同 reserve，包括最初固定的所有 V epoch。

One group names **one D destination**, not an all-D-rank installation decision.
Reserve/fence require the full original V epoch map; start/poll/ack require the
delivery ID and exact integer D rank. The response contains the plan fingerprint,
complete `write_identities` and available exact terminal `writer_proofs`, both keyed
by V rank, plus state and `entry_count_held`. `fenced` means network closure, not
successful delivery. D must independently validate every proof before MR retirement
and perform unpack/local fence/all-D-rank install agreement before serving output.

coordinator 在向任一 V 发布前记录身份集合，V 在 reserve 锁内检查期望 epoch。
预留响应丢失后对所有源 fence；源缺失先建立 tombstone。V 重启或身份不匹配
不能换一个 epoch 冒充旧 writer 已完成。只有所有源 ACK，或各源确认 fence，才
退父 Entry 的 delivery 计数。取消中的未知写入保持引用；后台继续排空。

Identities are persisted before any V sees the destination. V atomically checks
the expected epoch at reservation. Lost reserve responses fence every potential
writer, including absent-writer tombstones. Restart/epoch mismatch cannot be healed
by substituting a new identity. The parent Entry count survives until all ACKs or
confirmed fences; unknown/cancelling work remains owned and progresses in the reaper.

验证：20 个新增 CPU 用例覆盖实际 coordinator、store、LocalShardClient 和
localhost HTTP；Windows 全量 **2381/29 skipped**，WSL fan-in **87 passed**。
这不是 GPU/RDMA 证据。D 自动接入、多 HCA 目标注册、拓扑放开和 CAGRA 仍未完成。

Validation: 20 new CPU cases use actual coordinator/stores/local clients and
localhost HTTP; full Windows **2381 passed / 29 skipped**, WSL fan-in **87 passed**.
No GPU/RDMA evidence. Automatic D integration, multi-HCA destination registration,
topology activation and native CAGRA remain unfinished. Sections below record earlier
steps and their then-current integration boundaries.

公共协调层修改后，严格 v5 CPU 实模四场景矩阵再次全部通过；fake payload，
完整场景 21 次 attention 对照，最大误差约 3.58e-7，非 GPU/RDMA 验收。
After the shared coordinator changes, all four strict v5 CPU model scenarios passed
again (fake payload, 21 attention checks in full cases, max error about 3.58e-7).

## 字节映射已实现 / Byte planning implemented

`sharding.source_shard_intersections` 根据全局 KV head 区间计算一个 D rank 与
各 V shard 的交集。每个交集同时保留 V 内 head offset、D 内 head offset 和
head count，不要求 V/D 的 TP 数互相整除，只要求各自均分总 KV heads。

`packed_fanin_transfer_slices` 返回按 V rank 分组的相对 byte offset/length，
遵循现有 `kv_packer` 的 component/page/token 顺序，包含末页 padding。
例如总共 12 heads、V TP3、D TP2：D1 的 heads 6–11 来自 V1 的本地 heads 2–3
以及 V2 的本地 heads 0–3；后者写入 D1 的本地 head offset 2。

The head-intersection planner partitions each D rank's global KV-head interval
across V shards, retaining both source and destination offsets. V/D TP sizes need
not divide each other, but each must evenly partition total KV heads. The packed
fan-in planner returns relative byte ranges grouped by V rank in existing
component/page/token order, including final-page padding. With 12 heads, V TP3
and D TP2, D1 combines V1 local heads 2–3 and V2 local heads 0–3; the latter starts
at D1 local head offset 2.

旧 `source_rank_and_head_offset` 与 `packed_transfer_slices` 共用此算法，但
**仍拒绝需要多个 V shard 的交付**。没有更改 coordinator、descriptor、
WriteIdentity、fence、CLI TP 限制或真实模型消费能力。不能将 planner 的支持
矩阵解读为当前可部署的 PVD 拓扑。

Legacy one-source APIs share this planner but **still reject multi-source delivery**.
Coordinator, descriptors, WriteIdentity, fences, CLI TP limits and model-consumer
capabilities are unchanged. Planner coverage is not a deployable topology matrix.

## 下一阶段必须满足 / Required before wire activation

### V store 与 shard HTTP 已接入 / V store and shard HTTP integrated

V launcher 增加两个必须同时提供的 opt-in 参数：
`--full-kv-fanin-max-slices`（一个 D 计划中全部 writer 的 slice 总上限）和
`--full-kv-fanin-max-inflight`（每个 V writer 的在途 PUT 上限）。不猜测默认值；
不提供时新的 reserve 路由拒绝请求。engine 原有总 transfer budget 继续生效。

新增 `/internal/v1/fanin/reserve` 和 `/internal/v1/fanin/fence`，Local/HTTP shard
client 均有对应方法；start/poll/ACK/cancel 复用现有 delivery 接口。V store 从
真实 Entry 取得 key/layout/源 allocation，而非相信调用者对源内存的声明。
同 ID/同计划重试返回同一 writer；不同计划或旧协议不能替换已有授权。

Entry 的 active delivery 计数、超时 reaper、cancel/close 及原 allocator 回收
均已接入。UNKNOWN 隔离 V store，已取消但未完成的 PUT 继续保留 allocation。
不存在的 writer 只有在同锁内写入有界 tombstone、阻止迟到 reserve 后，才返回
NOT_SUBMITTED fence。网络完成与 source 本地 cleanup 仍独立处理。

The V launcher exposes two opt-in bounds: all slices in one D plan, and outstanding
PUTs per V writer. Both are required; existing engine-wide transfer budgets still
apply. New internal shard reserve/fence routes and Local/HTTP client methods use
the real Entry allocation, while start/poll/ACK/cancel reuse delivery APIs. Exact
retries return one writer; changed plans or legacy protocol cannot replace it.
Entry counts, TTL reaping, cancel/close and allocator retirement now drive fan-in
writers. UNKNOWN isolates the store. Absent-writer fences require a bounded
tombstone under the reservation lock before reporting NOT_SUBMITTED.

14 个新增 CPU 测试覆盖实际 V store/allocator、localhost HTTP、双 writer 重建、
部分完成、超时/关停、重复 reserve、Entry 复用、拒绝陈旧 fence 和 launcher 参数。
**这一步尚未接全局 coordinator 聚合和 D 的自动 receiver/admission**；默认
生产仍不是预测检索路径，也没有放宽 TP 或跨 rail 限制。

Fourteen CPU cases cover real stores/allocators, localhost HTTP, two writers,
partial completion, TTL/shutdown, retries, Entry reuse, stale-fence rejection and
launcher arguments. **Global coordinator aggregation and automatic D admission
are still pending.** Default serving and TP/cross-rail restrictions are unchanged.

本步骤 Windows 全量 **2361 passed / 29 skipped**，WSL 定向 **165 passed**；
严格 v5 CPU 实模四场景再次全部通过。原生 GPU/Mooncake 多源路径未执行。
This step passes **2361 / 29 skipped** on Windows and **165** focused WSL cases;
all four strict v5 real-model CPU scenarios pass again. Native GPU/Mooncake
multi-source execution remains unverified.

### 发送端执行器已实现 / Sender executor implemented

`validate_fanin_plan` 严格解析完整协议、重算 hash 和布局派生范围，拒绝已重新计算
hash 的越界/重叠计划以及类型强制转换。`FullKVFanInWriter` 将计划绑定到 V
实际 Entry key/layout、rank、allocation guard 和注册源区间；使用有界在途 PUT
逐段直接写入 D，不分配 repacking GPU buffer。授权仅开始一次，重复 start/poll
不会重放写入；所有句柄完成后才回复 fence 并退还源 allocation pin。

提交中取消不能越过未返回的句柄；部分提交抛异常、native 状态 UNKNOWN、重复
句柄或不可信 byte count 都保留源区间并停止新提交。取消前尚未提交可证明
NOT_SUBMITTED；取消后只看逻辑 CANCELLED 不能回收。当前仍要求 source/D descriptor
使用兼容 rail，尚未增加跨 HCA 的接收注册方案。

The parser recomputes the hash and layout-derived ranges and rejects rehashed
invalid ranges or type coercions. The writer binds the plan to the authoritative V
Entry/layout/rank/allocation guard and registered source slice. Bounded PUTs write
directly to D without GPU repacking. Start is one-shot; repeated progress never
replays writes. Cancellation cannot fence an unreturned submission. Ambiguous
submits, UNKNOWN, repeated handles and untrusted byte counts retain the source and
stop admission. Logical CANCELLED is not terminal evidence. Source/destination
rail compatibility is still required; multi-HCA receiver registration is not added.

21 个新增 CPU 用例包含真实 fake byte writes、两路 writer 与 D receipt 联测、
在途上限和提交/取消线程竞态。尚未以本步骤声称已接生产 V store/HTTP。
Twenty-one new CPU cases include actual fake byte writes, two-writer receiver
composition, bounded in-flight work and threaded submit/cancel races. Production
V store/HTTP activation is not implied by this executor step.

发送端步骤：Windows 全量 **2347 passed / 29 skipped**；WSL fan-in/授权
定向 **136 passed**。未执行 GPU 或原生 RDMA。
Sender-step evidence: Windows **2347 passed / 29 skipped**; WSL focused **136
passed**. No GPU or native RDMA execution.

### 接收端多 writer 生命周期已实现 / Receiver-side lifetime implemented

`FullKVFanInReceiver` 现在将实际 `RegisteredMemory` 及其 guard 固定为一组
接收资源，在发布 descriptor 前 pin。它按 **V rank** 保存完整身份集合；
原 `WriteIdentity.shard_rank` 仍表示 D rank，子交付名包含 `:d<D>:v<V>`，
不将多个 writer 覆盖进同一个 D-rank 字典项。计划指纹绑定 Entry、layout、
descriptor、token count 和完整相对偏移；caller 必须显式提供 `max_slices`，
超出上限在物化计划和 pin 前拒绝。

发布失败或取消后，缺少任一 writer 的终止证明都不能关闭 MR。完成证明必须包含
确切身份、计划指纹、writer rank、终止状态、`fenced=true` 和合法字节数；
只有全部成功且覆盖预期字节才报告 network-ready。NOT_SUBMITTED/FAILED 可以
证明可回收但不能变成可安装结果。重复相同证明幂等，矛盾证明拒绝。网络 pin
完成后，本地 unpack/install 读取者仍必须自行持有 guard pin 并完成本地 fence。

The receiver pins the actual registered MR before descriptor publication and keeps
the complete authorization set by **V rank**, while `WriteIdentity.shard_rank`
retains its D-rank meaning. Source-qualified subdelivery IDs and a plan fingerprint
bind the destination, layouts, Entry, token count and offsets. An explicit
`max_slices` bounds plan materialization before pinning. Failed publication or
cancellation cannot close the MR without every writer's exact terminal fence proof.
Only full successful byte coverage from all writers makes it network-ready.
NOT_SUBMITTED/FAILED may permit retirement, never successful install. Identical
proofs are idempotent; conflicting proofs are refused. Local import readers still
need their own guard pins and local completion fences.

**当前仍是显式组件，没有接入 coordinator/HTTP 或替换旧 receiver。**
32 个新增 CPU 用例中，一个联测使用真实 `WriteAuthorization` 和两路
`FakeTransferEngine` 写入同一个 CPU MR；其余发送端证明为受控 fixture。
这些不是原生 RDMA、远端可信证明或恶意 peer 的硬件隔离验证。实际发送端还必须
强制执行指纹中的范围授权，并在禁止迟到重试后才回复 fence。

**This remains an explicit component, not coordinator/HTTP or legacy receiver
activation.** Of 32 new CPU cases, one combines real `WriteAuthorization` objects
with two fake-engine writers to one CPU MR; other sender proofs are controlled
fixtures. This is not native RDMA evidence, authentication validation or hardware
isolation against malicious peers. The sender must still enforce planned ranges
and prohibit delayed retry before reporting a fence.

共享 MR 生命周期步骤验证：Windows 全量 **2326 passed / 29 skipped**，WSL
fan-in/原授权定向 **115 passed**。没有以这些 CPU 结果替代 GPU、真实多进程
传输或生产自动装配验收。

Shared-MR lifecycle step: **2326 passed / 29 skipped** on Windows full regression;
**115 passed** in WSL fan-in/existing authorization tests. These CPU results do not
replace GPU, real multi-process transport or automatic serving-assembly acceptance.

- 一次 D 接收内存的授权必须区分每个 V writer，且将 writer 身份绑定到本次
  receiver incarnation、entry generation、source rank、destination rank 和子交付。
- 每个 writer 只能写入其计划的目标子范围，不能用“共享同一个 MR”代替软件授权。
- 所有计划内 writer 的完成证明到齐后才可 unpack/安装；部分成功不能判为已交付。
- 取消/超时/部分提交异常后，必须收集所有可能已提交 writer 的终止证明。单个
  writer 完成不允许释放共享接收 MR，也不允许复用目标地址。
- 多源完成后仍遵守原所有 D ranks 的安装/ACK 协议。重试不能复用旧 generation
  身份，或遗漏已提交但返回失败的 writer。

Before activation, authorize each V writer against receiver incarnation, Entry
generation, source/destination rank and subdelivery; restrict writes to its planned
ranges. Unpack/install requires completion from every expected writer. Cancellation,
timeouts and partial submission failures must fence every potentially submitted
writer before releasing or reusing the shared MR. Preserve all-D-rank install/ACK
agreement and generation-safe retry. A shared MR is not per-writer authorization.

## 验证 / Validation

29 个新增 CPU 用例，使用真实 `pack_full_prompt_kv` 字节数据，对 V2→D1、
V3→D2、V4→D3、V2→D4、V1→D3、V3→D3 的所有 D ranks、page size 1/2、
非连续乱序 pages、多个 K/V layers 重建并逐字节对照。另检查每字节恰写一次、
源/目标范围不越界、非整数参数和不兼容布局拒绝。没有执行原生多源 RDMA。

Twenty-nine new CPU cases use actual packed KV bytes across six V/D partitions,
all destination ranks, page sizes 1/2, reordered pages and multiple K/V layers.
They check byte-exact reconstruction, exactly-once coverage, range bounds and
invalid input/layout refusal. No native multi-writer RDMA has been exercised.

本步骤 Windows 全量 **2294 passed / 29 skipped**，WSL 定向 **98 passed**。
严格 v5 CPU 四场景已在前一步 Scheduler 修改后通过，本次只改 byte planner 与
布局校验，不声称重新运行该矩阵。

This step passes **2294 / 29 skipped** on the full Windows suite and **98** focused
WSL cases. The strict v5 CPU matrix passed after the preceding Scheduler change;
this byte-planner/layout-validation step does not claim a separate matrix rerun.
