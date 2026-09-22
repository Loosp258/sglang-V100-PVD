# Sparse source routing / 稀疏检索源路由

## Implemented / 已实现

`RoutedShardSearchClient` separates D's compute rank from V's storage rank.
It uses the trusted storage/compute layouts' global KV-head intersections,
then routes each `(layer, KV head)` query to a caller-supplied shard HTTP client.
The Router's selected V group is unchanged; this class does not load-balance
or discover another V. Explicit endpoints must cover exactly the required V
sources and cannot be changed after construction.

`RoutedShardSearchClient` 区分 D compute rank 与 V storage rank，通过可信
storage/compute layout 的全局 KV-head 交集，把每个 layer/KV head 查询发送到
调用方提供的 V shard HTTP client。不改变 Router 选中的 V group、不重新负载
均衡、不猜节点。端点集合必须恰好覆盖所需 V 源，构造后不能换端点。

For V TP2 with four KV heads, examples are:
V TP2、总 KV heads 为 4 的示例：

| D configuration / D 配置 | Required source / 所需 V 源 |
| --- | --- |
| TP1 rank0, heads 0–3 | V0 heads 0–1 + V1 heads 2–3 |
| TP2 rank0 / rank1 | V0 / V1 |
| TP4 rank0 / rank1 / rank2 / rank3 | V0 / V0 / V1 / V1 |

`ProbeSearchSession` pins one index/mapping version pair **per V source** in a
window. Alternating V0 → V1 → V0 cannot reset V0's pin; V1 is not forced to have
V0's version. A changed version within either source invalidates the entire
window. Legacy single-shard clients retain their original single-version rule.
The existing GQA union stays per layer/KV head, retains that source's versions,
and never ranks scores across heads/sources.

同一 probe window 按 V 源分别固定 index/mapping version。V0 → V1 → V0
不会重置 V0 的版本，也不要求 V1 与 V0 版本相同；任一源在 window 中换版本，
整个 window 失效。旧单 shard 路径规则不变。GQA 仍按同一 layer/KV head
做有界去重并集，保留该源版本，不跨 head/source 排分。

`partition_specs()` validates a complete D-bank selection and one refresh epoch,
then produces one `SourceSparseSelection` per V source. Each carries:

`partition_specs()` 校验完整 D bank 和单一刷新 epoch，再为每个 V 源生成：

- A normal `SparseDeliveryManifest` with **V storage** layout fingerprint and
  that source's original index/mapping versions.
  普通 sparse wire manifest，使用 V storage layout 与该源原版本。
- Matching immutable D-local specs with **D compute** layout fingerprint.
  对应的不可变 D-local specs，保留 D compute layout。

The only translation is the explicit layout fingerprint; global layer/head,
token IDs, Entry, request incarnation, operation and index versions are retained.
Incomplete/duplicate groups, generated-token references, another Entry/layout,
mixed epochs, and mixed versions within one V source are rejected.

仅明确转换 layout fingerprint；全局 layer/head、token IDs、Entry、request
incarnation、operation 和版本均保留。不完整/重复组、生成 token、错误 Entry/
layout、混 epoch、单 V 源内混版本全部拒绝。

## Remaining integration / 尚待集成

This does **not** make existing sparse Delivery multi-source. Its current route
still expects one V sender per D bank, and the CUDA stage path expects one guarded
contiguous source buffer. Next implement owned per-source receive records,
all-source completion/visibility proof, bounded merge or multi-buffer ownership,
one complete D-bank stage, and ACK only after installation. A logical route/plan
is not a destination grant, a WRITE fence, an install receipt or permission to
advance the token clock. Production Scheduler assembly must wait for that path.

本步骤尚未让旧 sparse Delivery 变成多源：其路由仍是一 D bank 对一 V sender，
CUDA stage 仍要求单个受 guard 保护的连续 buffer。下一步需实现分源接收记录、
全源完成/可见性证明、有界合并或多 buffer 所有权、完整 D bank stage，以及安装
后的 ACK。逻辑路由/plan 不是写授权、WRITE fence、安装凭据或推进 token clock
的许可；生产 Scheduler 自动装配仍需等这条路径接通。

The router borrows shard clients. Closing it refuses new queries but does not
close shared HTTP clients or cancel active windows. The external owner must drain
all windows before closing those clients. Search results still contain only logical
IDs, not addresses or rkeys.

路由器借用 shard clients；关闭只拒绝新查询，不关闭共享 client 或取消窗口。
外部 owner 在关闭共享 clients 前必须排空所有 window。检索仍只返回逻辑 IDs，
不传地址或 rkey。

## Evidence / 证据

Thirty new CPU tests cover head ownership for D TP1/TP2/TP4, refusal before HTTP,
real two-store/two-HTTP-index search with independent source versions, GQA union,
source manifest splitting and layout identity preservation. They use synthetic
queries; no CUDA/native CAGRA/RDMA or real-model retrieval-quality claim follows.

新增 30 项 CPU 测试覆盖 D TP1/2/4 head 归属、HTTP 前拒绝、真实双 store/HTTP
索引独立版本查询、GQA 并集、分源 manifest 和两种 layout 身份。query 为合成值，
不构成 CUDA/原生 CAGRA/RDMA 或真实模型召回证据。
