# Probe → Search 基础衔接 / Probe-to-search foundation

## 中文：2026-09-21 工作区状态

基于 `pvd-disaggregation` / `3b1b4e75b`，本轮未提交、未推送。
目标模型架构、具体模型及 draft 部署仍未确定；没有替用户选择或硬编码它们。
本轮只完成模型无关的接口、生命周期和 CPU 隔离验证，不改变现有 P/V/D 服务流程。

### 已实现

- `prediction.py` 增加可覆盖的 draft/probe `branch()` 生命周期，以及
  `PredictionPipeline.query_branch()`。调用者在作用域内校验并复制 Q；适配器
  用 `finally` 清理临时状态。测试使用一个先预留预算再分配的 scratch probe，
  在退出时污染其张量，确认后续搜索读取独立的主机数值快照。
- `QueryVectors` 补充 `request_id`、`prefix_version`、`positional_encoding`。
  原接口测试允许省略，新搜索衔接层严格要求匹配。Q 明确采用
  `[positions, query_heads, head_dim]`，层、全局 Q head 和 KV head 不混用。
- `probe_search.py` 中的 `ProbeSearchSession` 绑定请求 incarnation、独立操作 ID、
  Entry、不可变前缀快照、目标刷新边界和采集位置。明确处理关闭、Entry 替换、
  前缀失效、窗口过期、取消及失败；未完成操作不允许被新窗口悄悄覆盖。
- 用显式 `QueryHeadMapping` 校验每条路线。不同 Q head 分别搜索，即使属于同一
  KV head 也不擅自合并结果；不同层也不做全局 Top-K。
- 一次窗口内多次搜索以第一条响应的 index/mapping 版本钉住后续请求；
  中间发生重建时丢弃整个结果，不发布部分成功。未知版本在第一次请求仍保持未知。
- 通过真实本地 HTTP 接入现有 V store / 精确索引。`take_selection()` 只消费一次
  逻辑选择，不安装 KV、不标记 PrefetchClock READY、不授权 RDMA、不推进时钟。

### 接口与位置语义

调用顺序为：

```text
snapshot_committed(...)
  → session.begin(prefix, target_tokens=..., query_positions=(...))
  → session.prepare(window, pipeline, routes=(...), head_mapping=...)
  → await session.search(prepared, selected_shard_client)
  → session.take_selection(window)
```

`target_tokens` 是 D 正式生成 token 的计数（不含 P 的首 token），不是序列位置。
`query_positions` 是预测续写中的绝对序列位置，调用者必须明确给出，本模块不猜测
“最后一个 Q”或具体窗口的建模策略。例如已提交前缀长度为 20、D 计数为 12、目标边界
为 16，则预测位置是 20、21、22、23；是否采集其中某个或多个，必须显式选择并在真实
模型适配时验证，不能把目标边界 16 误当成 Q 的位置。

正常提交 token 到边界不会使预测快照失效，允许预测近似；但前缀被替换必须显式
`invalidate()`，Entry 改变必须 `replace_entry()`。若 committed 计数回退，需要新的
session incarnation，不能在原 session 偷偷倒退时钟。新请求使用自己的 session，
不会影响其他请求。到期结果没有就绪时，本模块不决定正式 batch 的等待策略。

取出的选择仍然携带窗口身份。未来传输和安装必须在各自的边界再次验证请求存活、
Entry、索引版本和目标窗口；这里的一次验证不能替代后续的生命周期校验。

### 限制和证据

- **仅 CPU foundation**：拒绝声明为 CUDA 的 draft 配置以及 CUDA Q。配置为 CPU
  不代表能强制任意第三方适配器遵守，真实适配器仍需审计。
- 作用域使用 Torch CPU `fork_rng` 和 inference mode，涵盖作用域进入/退出及异常。
  它同步运行在单一 owner 线程，不能与未协调的全局 RNG 使用者并行；不承诺 CUDA、
  Python/NumPy RNG 或真实模型共享可变状态的隔离。没有在 Scheduler 上运行。
- 适配器应在 `branch()` 内预留后分配并清理 append-KV/scratch；默认钩子不拥有资源。
  测试证明作用域清理、预算归还及主机快照独立性，不证明原生/GPU 资源排空，
  也不是进程物理内存峰值的测量。主机 query/JSON 和真实模型 workspace 的完整
  预算方案尚未接入此独立层。
- 每 session 只允许一个操作，每次 1..64 条明确路线，1..64 个指定 Q 位置；
  HTTP 的 top_k、超时和回复大小限制沿用现有客户端。没有自动重试。
- `prepare()` 同步执行；检索才异步。**没有证明计算与 Decode 重叠，也没有隐藏网络延迟的性能证据。**
- Fake probe 的编码标签仅测试契约，不意味着真的执行了 RoPE 或目标模型。

本轮新增 **50 项测试**；全量 **967 passed, 6 skipped**，运行环境为本地 Windows
`.venv`、`torch 2.14.0+cpu`。跳过项需要 CUDA。没有真实模型、GPU、RDMA 或 V100S 验收。

```powershell
$pvdTestFiles = @(rg --files test/registered/disaggregation -g 'test_pvd*.py')
& .venv/Scripts/python.exe test/registered/disaggregation/run_pvd_cpu_tests.py @pvdTestFiles -q --tb=short
```

### 下一阶段

在选定一种目标模型架构后，实现隔离的真实 probe 适配器，验证 Q 与参考执行的一致性、
位置语义、临时内存预算及正式输出不受影响。模型名称、路径、revision、draft 和设备
继续作为配置，而不是固定版本。未选定前可以继续做边界/资源测试，但不能用 fake
代替真实模型完成声明。

仍未完成：真实 draft/probe 服务接线、CAGRA、稀疏 KV 交付/安装/attention、M-r 调度、
GPU 双缓冲、跨 shard 路由/合并、硬件性能与质量验收。完整初始 Prompt KV 的等待队列
拉取保持不变，新请求仍不触发旧请求更新。

## English: implementation and handoff contract

This uncommitted change builds on `3b1b4e75b` on `pvd-disaggregation`. No architecture,
model or deployment device was chosen for the user. Production serving is unchanged.

### Implemented

- Scoped draft/probe resource hooks and `PredictionPipeline.query_branch()`. Adapters
  own their temporary state; consumers materialize independent host queries before the
  scopes exit. A budgeted fake poisons its scratch on exit to test ownership.
- Explicit Q metadata: request id, prefix version and positional encoding, required by
  the new bridge. Layout is `[positions, query_heads, head_dim]`; Q heads are mapped to
  global KV heads through an explicit `QueryHeadMapping`.
- `ProbeSearchSession` binds request incarnation, fresh operation id, Entry, immutable
  prefix, target committed-token boundary and explicitly selected absolute Q positions.
  Cancellation, replacement, invalidation, expiry and failure discard stale results.
- Separate searches and results per layer/Q head, with no implicit cross-head/layer merge.
  After the first response, subsequent routes pin its index/mapping versions. A rebuild
  fails the entire operation rather than publishing a mixed-version partial selection.
- CPU fake probe -> real local HTTP -> V store/exact index integration. A result is a
  one-shot logical selection, **not** KV_READY, a transport grant or an installation.

### Semantics that must not drift

`target_tokens` counts committed D tokens excluding P's first token. `query_positions`
are absolute positions in the predicted continuation. The caller chooses them explicitly;
the bridge does not invent which predicted Q should serve the next attention window.
For example, prefix length 20 and D count 12 imply predicted positions 20..23 for four
predictions, even when the target committed-token boundary is 16.

Normal progress toward that boundary does not invalidate an approximate prediction.
Explicitly invalidate prefix replacement; replace Entry identity when it changes. A
committed-count retraction needs a fresh session incarnation. A newcomer cannot alter
another session. Later transfer/installation must revalidate the attached lifetime,
Entry, versions and window, even after this layer validated a response.

### Validation and limitations

**967 passed, 6 skipped**, up from 917, with 50 new CPU tests. Local Windows `.venv`,
`torch 2.14.0+cpu`; no actual model, GPU, RDMA or V100S validation.

The bridge is explicitly CPU-only. The synchronous branch runs on one owner thread,
restoring Torch CPU RNG and disabling gradients, including scope entry/exit and failure.
This is not isolation from concurrent uncoordinated global RNG consumers, CUDA RNG,
Python/NumPy RNG, or adapters retaining mutable external model/request state. No model
execution runs on the production scheduler. Fake encoding labels do not prove real RoPE.

Adapters must reserve before allocating and release temporary state in their scope.
Tests establish hook cleanup, budget refund and independent query snapshots, not native
GPU drainage or physical peak memory. Full query/JSON and real-model workspace budgeting
is still a serving-integration task. The synchronous preparation and asynchronous search
do not yet demonstrate overlap with Decode or hidden network latency.

Next: once an architecture is chosen, implement its isolated real probe and compare Q
against a reference execution at identical positions, then measure memory/latency and
prove committed-output isolation. Keep concrete model names, paths, revisions and draft
selection configurable. CAGRA, sparse delivery/installation/attention, M-r scheduler
wiring, GPU buffers and multi-shard policy remain unfinished. Preserve waiting-queue
full-Prompt-KV bootstrap and independent per-request clocks.
