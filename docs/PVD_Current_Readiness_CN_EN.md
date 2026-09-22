# PVD 当前实现与验收边界 / Current implementation and acceptance scope

Updated / 更新：2026-09-22。历史交接文档保留演进记录；本页集中说明当前边界。
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

## 本轮完成 / Completed in this run

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
| GPU sparse attention | GPU dtype/layout、current/next banks、stream/event 生命周期接入与逐层数值验证 / GPU backend and stream-safe bank integration plus numerical checks |
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
