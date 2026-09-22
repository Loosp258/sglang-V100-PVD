# Selected V shard routes / 已选 V 分片路由

## Contract / 契约

The Gateway still selects the V group. D sends the selected Entry key to that
group's coordinator with `POST /v1/entries/routes`. The coordinator does not
choose another group: it returns the stored Entry manifest and exactly the two
V shard HTTP URLs, current sender epochs and rails. It refuses an Entry that is
not STORED, missing or duplicate advertised URLs, or a shard whose live
rank/world-size/rail/ready state differs from the manifest. It rechecks the
Entry after collecting shard health. The client validates the complete reply
against the requested key before any route can be used.

Gateway 仍选择 V group。D 使用该 group 的 coordinator，带选定 Entry key 调用
`POST /v1/entries/routes`。coordinator 不重新选组，只返回 STORED Entry 的
manifest、两个 V shard 的 HTTP URL、当前 sender epoch 和 rail。Entry 未 STORED、
URL 缺失或重复、shard 的 rank/world-size/rail/ready 与 manifest 不符都拒绝。
读取 shard 状态后还会复查 Entry；客户端使用前再按原 key 校验完整响应。

The V launcher explicitly advertises each local shard URL using
`--advertise-host` and the shard port. Legacy split-process mode uses its
configured rank-1 shard URL. An RDMA endpoint, request Host header or inferred
GPU index is never silently treated as a shard HTTP URL. D pins the returned
sender epochs and URLs in its request-local routed search/Delivery objects.
If a V shard restarts, it must rebuild that request rather than reuse stale
write authorization.

V 启动器通过 `--advertise-host` 和 shard 端口显式发布本地 URL；旧双进程模式
沿用配置的 rank-1 shard URL。不会把 RDMA endpoint、请求 Host header 或 GPU
序号默认为 HTTP URL。D 应把返回的 epoch 和 URL 固定到请求级检索/Delivery；
V shard 重启后必须重建请求，不能复用旧写授权。

## Boundary / 边界

This endpoint supplies trusted route data for the future D admission factory.
It does not instantiate a draft model, target probe, CUDA bank or Scheduler
binding. The production path remains full-Prompt refresh until that assembly
exists. CPU HTTP tests prove the schema and refusal policy only, not V100S,
native Mooncake or multi-HCA operation.

该接口仅提供后续 D admission 工厂所需的可信路由数据；它不会构造 draft 模型、
目标 Q probe、CUDA bank 或 Scheduler 绑定。在装配完成之前生产路径仍为完整
Prompt 刷新。CPU HTTP 测试只验证协议及拒绝策略，不代表 V100S、原生 Mooncake
或多 HCA 验收。
