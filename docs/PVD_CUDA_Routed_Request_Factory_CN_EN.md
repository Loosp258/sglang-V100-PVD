# Routed CUDA request factory / CUDA 路由请求工厂

`assemble_routed_cuda_request()` consumes the typed shard-route reply from
the Gateway-selected V coordinator. The caller must also supply the already
installed D TP1 bank, receive registry, actual prediction/target-Q pipeline,
compute layout and explicit budgets/endpoint/HCA. It constructs separate
search and Delivery clients for each selected V shard, then one routed search
client, one fan-in Delivery and one `CUDARoutedPrefetchRequest`. It generates
all Q-head routes from the supplied GQA mapping rather than accepting a
partial or duplicate caller list. Selected Entry, model space, layout, layer
coverage, KV-head count, sender epochs and rails are checked before use.

The factory supports two explicit receive policies. With the existing
single-rail Mooncake D adapter, both selected V shards must use the same
rail. With `RailMappedReceiveEngine`, the caller may provide `d_rails` mapping
each V source rank to an independently owned D receive adapter for that rail.
The composite dispatches each destination registration and unregister to its
exact owner. `create_native_receive_group()` can construct an additional
Mooncake session per selected D HCA, reuse the already initialized D adapter,
require Linux/CUDA/RDMA and ACTIVE ports, and run strict local GPUDirect
preflight on every adapter. The serving Scheduler does not yet invoke that
factory; a mixed-rail request with only a single-rail D engine is refused.
True native V TP2 → D TP1 still needs serving startup wiring and GPU/RDMA
validation; changing only a descriptor label is unsafe.

For Decode TP1 full-KV fan-in, `--pvd-d-receive-rails mlx5_2,mlx5_3`
now initializes that receive group at worker startup, reusing the D compute
rail adapter. The option requires the compute rail in its distinct HCA list.
The group is exposed on `PVDKVManager.sparse_receive_engine`; it does not yet
construct the predictive CUDA receive registry or switch the serving Scheduler
from full-Prompt refresh to sparse retrieval.

`assemble_routed_cuda_request()` 使用 Gateway 已选 V coordinator 返回的类型化
shard 路由。调用方仍须提供已安装的 D TP1 工作集、接收注册表、真实 draft/目标 Q
pipeline、compute layout 以及显式预算、端点和 HCA。工厂为各 V shard 建立独立
检索与 Delivery 客户端，再组装路由检索、多源交付和一个
`CUDARoutedPrefetchRequest`。Q-head 路由按 GQA 映射生成，不接受缺失或重复
的外来列表。Entry、模型向量空间、layout、层覆盖、KV-head 数、sender epoch 和
rail 均须匹配。

工厂现在有两种显式接收策略：现有单 rail Mooncake D adapter 要求两个 V 源
都使用同一 rail；`RailMappedReceiveEngine` 则允许调用方通过 `d_rails` 将
每个 V 源 rank 映射到 D 上独立拥有的同 rail 接收 adapter。组合层将注册和
注销交还对应 owner。`create_native_receive_group()` 可复用已初始化的 D
adapter、为额外 HCA 创建独立 Mooncake session，并逐 rail 核查 Linux/CUDA/RDMA、
ACTIVE 端口和严格的本地 GPUDirect 预检。生产 Scheduler 尚未调用此工厂；
仅有单 rail D engine 时，混用 V rails 仍会在注册前拒绝。真正双 rail 的
V TP2 → D TP1 还需服务启动接线和 GPU/RDMA 验收，不能只修改 descriptor 标签。

Decode TP1 全量 KV fan-in 现在可用 `--pvd-d-receive-rails mlx5_2,mlx5_3`
在 worker 启动时建立该接收组，复用 D compute rail 的 adapter；HCA 列表必须
不重复且包含 compute rail。组合层挂在 `PVDKVManager.sparse_receive_engine`；
它尚未建立预测 CUDA 接收 registry，也不会把生产 Scheduler 从完整 Prompt KV
刷新切换到稀疏检索。

The returned `clients` mapping goes directly to `CUDARefreshDriver.register`.
The request owns all HTTP clients. Its synchronous `close()` is prohibited;
`aclose()` first drains remote destinations and the D bank, then closes search
and Delivery sessions. An unresolved native owner prevents those control
clients from closing, so fence/cleanup remains possible. The existing driver
already calls `controller.aclose()` during request retirement.

返回的 `clients` 映射可交给 `CUDARefreshDriver.register`。HTTP 客户端由请求
拥有；禁止同步 `close()`，必须 `aclose()`：先排空远端写目标和 D 工作集，再
关闭检索/Delivery 会话。原生 owner 未确认完成时不会关闭控制客户端，保留
后续 fence/清理能力。现有 driver 在请求退休时已调用 `controller.aclose()`。

This factory does not load the draft model, install the attention backend,
obtain the selected routes on the manager's control loop, attach the full
Prompt receiver or turn on predictive serving in the production Scheduler.
Those startup/admission pieces and native CUDA/Mooncake/V100S tests remain.

该工厂不加载 draft 模型、不安装 attention backend、不在 manager 控制线程
获取路由、不接完整 Prompt receiver，也不自动打开生产 Scheduler 预测检索。
这些启动/准入接线与真实 CUDA/Mooncake/V100S 测试仍待完成。
