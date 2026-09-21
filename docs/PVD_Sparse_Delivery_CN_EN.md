# 稀疏交付生产路径推进 / Sparse delivery integration

## Step 1：载荷清单与源索引 lease / Manifest and source-index lease

`SparseDeliveryManifest` 是有版本、有上限的 wire 契约，固定 request/incarnation/
operation/boundary、Entry、index/mapping version、布局 fingerprint 和每个
layer/KV-head 的绝对 token ids。数据按清单顺序连接各组 `[K/V, token, dim]`，
不跨层合并、无隐含 padding、无远端地址。D 必须保留自己原先发出的清单做对照。
清单指纹只绑定内容，不是认证凭证、MR 授权或传输完成证明。

`PromptIndexManager.pin_selection()` 对当前真实 index/mapping 做版本与 token/head
校验，并在打包期间持有旧 record 的 reader。并发 close/rebuild 不会提前退还旧
index budget，也不会由旧 lease 释放新 index 的预算。源 Entry 页由另外的 allocation
guard 持有，不能用 index lease 替代。副本打包完成后即可归还 index lease。

The immutable, versioned wire manifest describes paired K/V bytes and logical
coordinates, not a remote-write permission. It has strict fields, bounded group
and token-reference counts, exact dtype/dimension/byte extent and a content
fingerprint. The receiver must compare against its own requested manifest.
The index lease validates current versions and mapping membership, keeps the
record charged across close/rebuild, and ends after packing. Entry ownership,
destination authorization and native completion remain separate requirements.

21 new tests cover wire refusal, byte offsets and pairing, actual search-result
versions, stale/missing versions, foreign heads/tokens, and concurrent record
retirement/rebuild accounting. CPU contract evidence only; no RDMA claim.

## Step 2：V 的异步稀疏发送 / V-side asynchronous sparse send

目的 descriptor 携带 `pvd_sparse_delivery` 时，现有 VectorKVStore 的
reserve/start/poll/fence/ACK 使用该清单：校验真实 Entry 布局/源 shard/有效 token、
精确 destination 长度、必需的 lifecycle-v1 元数据与 staging budget。
无该字段时完整 Prompt 路径保持原行为。稀疏路径暂显式拒绝 GPU packing，
不会静默把 V 的 GPU pool 搬到 CPU。

源 Entry allocation 由原 write authorization pin；打包阶段额外 pin 当前 index，
逐 group 打包到有预算的 staging，不复制整个 Prompt。staging 注册后由原 Delivery
终态推进保留/释放，业务取消、TTL 或超时本身不能释放。短字节 SUCCESS 判失败。
注册抛异常且结果不明确时，整个 store 隔离并保留 tensor/budget；unregister 失败时
预算保留，之后 progress 重试成功才退还。此保守 quarantine 不是可恢复性保证。

Fourteen new tests use the real store/index/packing code with a delayed transfer
engine. They verify exact paired bytes, Entry reuse, cancellation followed by
late terminal success/failure, premature ACK refusal, stale selection, invalid
destination, capacity failure, short successful writes, unregister retry and
unknown-registration quarantine. Full Windows suite: 1408 passed / 11 skipped.
No native RDMA/GPU evidence is claimed.

## 接下来 / Next

V 已接原 reserve/start/poll/fence/ACK 并保留整个异步传输期间的 staging；
下一步接 D 的独立 destination 生命周期、安装与 ACK，以及真实本地 HTTP 交付测试。
完整 Prompt bootstrap 不依赖检索索引；新增稀疏路径必须显式启用，不能改变默认行为。
