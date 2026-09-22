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

The current Mooncake D adapter is bound to one rail per D rank. For D TP1
receiving two V shards, this factory accepts only a selected V group whose
source rails both equal the D rail (explicit single-rail mode). A mixed
`mlx5_0`/`mlx5_1` V group is refused before destination registration. Fake
transport can copy those bytes, but real V1 rejects a destination on the
wrong rail and D's single-rail adapter cannot register on the other rail.
True dual-rail V TP2 → D TP1 needs a multi-HCA D adapter and native validation;
changing only a descriptor label is unsafe.

`assemble_routed_cuda_request()` 使用 Gateway 已选 V coordinator 返回的类型化
shard 路由。调用方仍须提供已安装的 D TP1 工作集、接收注册表、真实 draft/目标 Q
pipeline、compute layout 以及显式预算、端点和 HCA。工厂为各 V shard 建立独立
检索与 Delivery 客户端，再组装路由检索、多源交付和一个
`CUDARoutedPrefetchRequest`。Q-head 路由按 GQA 映射生成，不接受缺失或重复
的外来列表。Entry、模型向量空间、layout、层覆盖、KV-head 数、sender epoch 和
rail 均须匹配。

当前 Mooncake D adapter 每个 D rank 只绑定一个 rail。因此 D TP1 同时接收双 V
shard 时，本工厂只接受两个 V 源与 D rail 一致的显式单 rail 配置。混用
`mlx5_0`/`mlx5_1` 会在注册前拒绝。fake transport 可以拷贝字节，但真实 V1
会拒绝错误 rail 的目标，D 的单 rail adapter 也无法在另一个 rail 注册。
真正双 rail 的 V TP2 → D TP1 需要 D 多 HCA adapter 和原生验证，不能只改
descriptor 标签。

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
