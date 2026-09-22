# 完整 Prompt 初始引导 / Index-independent CUDA Prompt bootstrap

`CUDAPromptBootstrap` 从 D 的真实模型池读取完整 Prompt，并在 boundary=0 安装
到请求自己的 CUDA bank/runtime。它读取实际 `req_to_token` 映射，支持非连续
物理行；保持绝对 Prompt token IDs，不覆盖模型池或生成 token 的 KV。

The importer reads complete Prompt KV from D's actual model pools into its own
CUDA bank/runtime at boundary zero, using the request's real `req_to_token` map.
Noncontiguous physical rows preserve absolute Prompt IDs. Model pools and
generated-token KV are never overwritten by the import.

它不需要 V 检索、CAGRA 或 index READY。内部版本标记 `full-prompt:no-index`
只标识初始完整内容，不得当作检索索引版本向 V 查询。后续稀疏刷新仍验证 V 返回
的真实 index/mapping version。

No V retrieval, CAGRA or index READY is required. The internal
`full-prompt:no-index` marker labels initial contents; it is not a version to
submit to V search. Later sparse refresh still pins actual V index/mapping versions.

前提：调用者必须先完成既有完整 Prompt 接收器的身份校验、远端 native 完成证明、
unpack 及设备排序，然后提供精确的 request/incarnation/Entry/layout/count 和
与实际分配器回收绑定的 `CUDAModelPools` guard。`CUDAPromptPoolSource` 是源声明，
**不是 RDMA fence**，不能用它绕过前述步骤。

Precondition: existing full-Prompt receive identity validation, remote native
completion, unpack and device ordering must finish first. Supply exact
request/incarnation/Entry/layout/count and an allocator-backed model-pool guard.
The source declaration is **not an RDMA fence** and cannot replace those steps.

预留 staging budget 后 pin 源池，逐行复制完整 KV，排空后 staging/install/resume。
同一共享 target 锁覆盖整个过程。staging 与 bank 的预算分别计算（引导有真实
峰值显存代价）；成功排空并移除所有视图/guard 后才能退还 staging 预算。
任一次屏障 UNKNOWN 保留源、staging、预算和锁，不自动再试一次来“证明”可以退款。

Reserve staging, pin source pools, copy rows, fence, then stage/install/resume under
the shared target lock. Staging and bank copies have separate charges and real
peak-memory cost. Staging refunds follow completion and removal of owned views/
guards. Any UNKNOWN fence retains source/staging/budget/lock without automatic retry.

CPU 策略/数学测试覆盖无索引引导、非连续映射、错误身份/行/类型、预算拒绝、
安装和清理 UNKNOWN。严格 CUDA 模型 smoke 已改用此初始引导，但本地没有执行
GPU；生产接收器/Scheduler 的自动调用仍是接入任务，不因独立入口存在就算完成。

CPU policy/math tests cover no-index bootstrap, noncontiguous maps, bad identity/
rows/types, admission and UNKNOWN install/cleanup. The strict CUDA model smoke now
uses this importer but has not run on hardware. Automatic serving receiver/
Scheduler invocation remains an integration task, not implied by this component.
