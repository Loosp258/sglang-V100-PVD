# PVD 当前实现与验收边界 / Current implementation and acceptance scope

Updated / 更新：2026-09-23。历史交接文档保留演进记录；本页集中说明当前边界。
Historical handoffs contain earlier states; this page consolidates the current scope.

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

新增 [CUDA 路由请求工厂](PVD_CUDA_Routed_Request_Factory_CN_EN.md)：把已选 V
的双 shard 路由、真实 D 组件和显式预算组装成一个请求级检索/多源 Delivery；
请求排空后才关闭它拥有的 HTTP 客户端。它仍需生产启动/准入逻辑提供真实组件，
不能凭此认为预测检索已自动启用。
当前 D TP1 Mooncake 每 rank 只绑定一个 rail。工厂对混用 V rails 的请求在注册前
拒绝；只有显式同 rail 的 V0/V1 → D0 可以装配。双 rail 的 V TP2 → D TP1
仍有 D 多 HCA 代码和原生验收缺口。
本步 Windows 全量 **2497 passed / 31 skipped**，WSL 工厂/双源定向
**12 passed**。真实 GPU/Mooncake/RDMA 仍未执行。

The CUDA routed-request factory assembles selected two-shard V routes with
explicit D resources into one search/fan-in controller and owns HTTP clients
until request drain. Production startup/admission still has to provide the
real components; this is not automatic predictive serving activation.
Current D TP1 Mooncake uses one rail per rank. The factory fails closed for
mixed-rail V sources; only explicit same-rail V0/V1 → D0 is assembled. Native
dual-rail V TP2 → D TP1 remains a code and hardware acceptance gap.
This increment passed **2497/31 skipped** in the full Windows CPU suite and
**12 passed** in focused WSL tests; native GPU/Mooncake/RDMA remains unrun.

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
| V CAGRA | 真实 cuVS/CAGRA backend、目标软件栈验证、真实目标 Q 的 recall 与模型质量对照 / real backend integration and target-query recall/quality |
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
9 个明确 CUDA 用例全部执行成功，无设备/skip/缺测不会成为通过；本地仍是 blocked。
The strict CUDA component gate refuses missing/skipped evidence. Its local CPU-only
result remains blocked, and a future component pass will not certify RDMA/serving.

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
