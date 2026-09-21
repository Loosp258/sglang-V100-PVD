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

## 接下来 / Next

将此契约接到 VectorKVStore 的已有 reserve/start/poll/fence/ACK，保留整个异步
传输期间的 packed staging；然后接 D 的独立 destination 生命周期、安装与 ACK。
完整 Prompt bootstrap 不依赖检索索引；新增稀疏路径必须显式启用，不能改变默认行为。
