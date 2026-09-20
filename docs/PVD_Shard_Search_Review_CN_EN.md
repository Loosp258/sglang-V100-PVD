# PVD 单 shard 检索复核 / Single-shard search review

## 中文：本轮结果（2026-09-20）

本轮基于 `pvd-disaggregation`，HEAD 为 `f83dc115e`，在既有未提交修改上继续工作。
保留了已有索引预算、并发关闭、调用者身份、分块检索、独立构建版本和设备别名修复。
本轮未 commit、未 push；以下状态描述的是工作区，不代表 GitHub 已有这些改动。

### 修复与推进

1. **精确 L2 参考检索的数值错误。** 对向量 `[10000, 10000]`、
   `[10001, 10000]`，以第二个向量为 query，范数展开会把两个距离都算成零，
   错选第一个向量。改为逐 query、逐块先相减再平方求和，保持临时内存有界；
   移除不再使用的范数缓存并调整内存声明。三个分块尺寸均验证正确排名与分数。
2. **HTTP 身份字段不再强制转换。** `layer` / `kv_head` 的小数、布尔值、
   字符串被拒绝，不再被 `int(...)` 悄悄转换。拒绝布尔、非有限和超 float32
   范围的 query 元素，测试检查具体拒绝原因，而非只断言 400。
3. **正常背压不再变成内部服务错误。** 可重试的索引未就绪返回
   `400 / index_not_ready`（保留原 400 状态）；检索预算不足返回
   `507 / index_capacity`。重试次数耗尽的构建失败不标为可重试。
4. **新增独立的单 shard 搜索客户端** `PVDShardSearchClient`。客户端接收明确的
   shard URL、调用者身份和可信请求范围，发送合成/未来 probe query，返回逻辑
   token/page 选择。已有本地 HTTP → V store → exact backend 的往返测试。
   这不是正式 D Scheduler 接线，不产生 probe，不安装 KV，不改变输出。

### 新客户端接口约束

- 每次请求带 `search_protocol=pvd.search.v1` 和独立 `search_id`，回复必须匹配。
  V 兼容不带这两个字段的原请求；新客户端要求相关性字段，不会静默降级。
- `SearchRequestIdentity` 来自调用者。版本钉未知时不伪造，已知时必须发送并核对；
  `validated` 必须准确反映本次实际请求的比较项。元数据声明不是模型来源的密码学证明。
- `SearchScope(prompt_tokens, page_size, head_dim, metric)` 必须来自调用者可信布局，
  不从 V 回复反推。验证 token 范围、去重、page 映射、分数和结果数量。
- 请求最多 64 个 query，`top_k <= min(512, prompt_tokens)`；拒绝非法输入后才允许
  建立 session。默认请求超时 30 秒，回复上限 2 MiB，均可配置；不跟随重定向。
- 不自动重试；调用者决定等待/失败策略。取消操作传播取消，不产生可安装结果。
  网络取消不保证 V 的后台 CPU 检索立刻停止；已有 reader/budget 生命周期仍负责释放。
- 外部 session 不由客户端关闭；客户端关闭后不可再次搜索。
- 不选 V group、不跨 shard 合并、不获取 RDMA 地址、不提交 Delivery。
  未来必须继续核对请求生命周期和目标窗口，不能仅凭 search_id 安装 KV。

### 验证

Windows 本地 `.venv`，Python + `torch 2.14.0+cpu`。本轮基线 **865 passed, 6 skipped**，
加入 52 项测试后的全量结果 **917 passed, 6 skipped**。测试含本地真实 HTTP，但
query 为合成向量；没有真实 draft/probe、GPU、RDMA、V100S 或质量/性能验收。
分块检索测试继续测量张量存储峰值，不能将其等同于进程 RSS 或原生库全部工作空间。

```powershell
$pvdTestFiles = @(rg --files test/registered/disaggregation -g 'test_pvd*.py')
& .venv/Scripts/python.exe test/registered/disaggregation/run_pvd_cpu_tests.py @pvdTestFiles -q --tb=short
```

### 接手者下一步

先阅读中英主交接文档和当前代码。下一阶段是确定目标模型的独立 probe 适配边界，
产生正确位置、post-RoPE、目标模型空间的真实 Q，接入这里的单 shard 客户端。
模型名称、路径、revision、draft 部署设备仍由配置决定，不硬编码用户必须使用的模型。
架构相关适配若需要用户选择，应提问；无硬件时继续隔离性、过期回复及生命周期测试，
不要把 fake probe 当成真实模型已完成。CAGRA/cuVS 兼容性、稀疏 KV 传输/安装与
attention、M-r 服务接线、多 shard 路由/合并策略及 GPU 验收仍未完成。

必须保留：等待队列触发完整初始 KV 拉取；新请求不触发旧请求更新或重置时钟；
每请求独立 M 周期；预测只用于预取；已提交 token 和生成 KV 由 D 掌握。

## English: results and continuation contract

This work continues the existing dirty tree on `pvd-disaggregation`, HEAD `f83dc115e`.
Earlier budget, concurrency, caller-identity, chunking, build-version and device-alias
changes were preserved. **Nothing was committed or pushed this turn.**

### Fixes

- Reproduced cancellation in float32 squared-norm L2: query `[10001, 10000]` against
  `[10000, 10000]` and itself returned two zero distances and the wrong first neighbour.
  L2 now subtracts before squaring, one query and one row block at a time. Unused
  retained norms were removed and footprint declarations adjusted. Chunked peak-storage
  regression tests still pass; the declaration is not a bound on total process RSS.
- HTTP no longer coerces layer/head identity via `int()`. Booleans, strings and fractional
  ids are rejected. Query values must be finite, numeric, and float32-representable.
- Normal search capacity refusal is `507 / index_capacity`, not a 500. Retryable pending
  indexes use `400 / index_not_ready`; exhausted build failures are not retryable.

### Completed bounded next step

`PVDShardSearchClient` is an independent, explicitly addressed single-shard HTTP client.
Local integration tests exercise the actual V store, extraction, exact backend and HTTP
route using synthetic queries. **No production Decode scheduler calls this client yet.**

The client requires caller-owned `SearchRequestIdentity` and trusted `SearchScope`.
Each call has a fresh `search_id` under `pvd.search.v1`. It validates response correlation,
Entry/space/encoding/layer/head/metric, optional version pins, exact validation claims,
logical token bounds and uniqueness, page mapping and finite scores. Unknown pins stay
absent. Metadata validation does not prove the actual model origin of numeric queries.

Requests have at most 64 queries and `top_k <= min(512, prompt_tokens)`. Configurable
defaults are 30 seconds and 2 MiB maximum reply size. No redirects or automatic retries;
only the defined status/code pairs indicate a retryable refusal. Cancellation propagates,
but does not promise to stop a server-side background search immediately. External HTTP
sessions remain caller-owned; a closed client cannot be reused.

Legacy V requests without correlation fields remain supported; the new client requires
those fields and never silently downgrades. It chooses no V group, merges no shards,
returns no memory addresses, fetches/installs no KV and changes no committed Decode state.
Future installation must also validate request lifetime and target-window generation.

### Evidence and next work

Baseline: **865 passed, 6 skipped**. Current full CPU suite: **917 passed, 6 skipped**,
including 52 new tests. Runtime: local Windows `.venv`, `torch 2.14.0+cpu`. No real
model, CUDA, RDMA, V100S, retrieval-quality or latency validation was performed.

Next, define the architecture-specific isolated target-model probe adapter and produce
real position-correct post-RoPE Q before connecting this client to serving. Keep target
and draft selection configurable; ask about architecture-dependent choices rather than
inventing them. In the absence of hardware, continue isolation, stale-result and lifetime
tests without claiming that fake probes validate actual inference.

CAGRA/cuVS compatibility, sparse KV transfer/installation/attention, M-r serving wiring,
multi-shard routing/merging and hardware acceptance remain open. Preserve waiting-queue
full-KV bootstrap, independent per-request M clocks, no newcomer-triggered refresh of
existing requests, predictive-only draft tokens, and D-local generated KV.
