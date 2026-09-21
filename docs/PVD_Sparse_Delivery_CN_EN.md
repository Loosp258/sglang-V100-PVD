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
直接将各 group 的配对 K/V 写到有预算的最终 staging，不复制整个 Prompt。staging 注册后由原 Delivery
终态推进保留/释放，业务取消、TTL 或超时本身不能释放。短字节 SUCCESS 判失败。
注册抛异常且结果不明确时，整个 store 隔离并保留 tensor/budget；unregister 失败时
预算保留，之后 progress 重试成功才退还。此保守 quarantine 不是可恢复性保证。

Fourteen new tests use the real store/index/packing code with a delayed transfer
engine. They verify exact paired bytes, Entry reuse, cancellation followed by
late terminal success/failure, premature ACK refusal, stale selection, invalid
destination, capacity failure, short successful writes, unregister retry and
unknown-registration quarantine. Full Windows suite: 1408 passed / 11 skipped.
No native RDMA/GPU evidence is claimed.

## Step 4 / 第四步：Request-local HTTP sparse delivery integration

The controlled `CPUPrefetchRequest` can now select an explicit `CPUSparseDelivery`
sink instead of a local packing callback; configuring both is refused. After one
shared draft/probe and the rank searches, each selected V shard packs and writes
only the selected paired K/V into an owned D destination. All receiving ranks
must stage before boundary installation. Completion is latched synchronously;
remote ACK/cleanup run asynchronously with visible errors and explicit retries.
New requests do not reset or cancel any existing request's clock or retrieval.

`CPURefreshDriver` and `CPUDecodeLifecycle` accept this path and asynchronously
drain it on close. Per-request scopes prevent a shared receive registry from
closing another request's buffers, including registrations whose creation raised.
The original local packing path remains an explicit reference option. Initial
full-Prompt admission remains independent of retrieval and unchanged.

CPU 请求级流水线已可通过真实 shard HTTP Delivery 获取稀疏 KV，不再必须在 D
测试流程里直接读取 V 的内存。本地打包与远端 Delivery 二选一；跨 rank 等待、
边界安装、新请求不干扰旧请求的时钟均保留。安装后的 ACK 异步执行，失败显式报告
并可重试；关闭时按请求范围处理资源，不能释放其他请求的目标缓冲区。

Tests use exact CPU search, localhost HTTP and a fake byte-copy transport. They
verify two rounds, delayed delivery at a boundary, new admission independence,
stale source indexes, ACK loss/retry, scoped cleanup and the automatic refresh
driver. This does not claim a production Scheduler, CUDA or native RDMA path.

## Step 3 / 第三步：D-owned sparse receive lifecycle

`SparseReceiveRegistry` now owns CPU receive allocations before registration or
descriptor publication. It charges bytes and one in-flight slot, reconstructs
the complete expected write identity from its own descriptor plus the selected
V incarnation, and rejects mismatched manifests, generations, byte counts and
terminal proofs. V Delivery replies expose a non-mutating `write_fence` snapshot;
reading that proof does not cancel a successfully delivered operation.

CPU FP32 payloads are copied into `CPUInstallGroup` only after successful,
exact-length fenced delivery. ACK requires the exact all-rank installation
receipt, not merely receipt or staging of the bytes. Lost ACKs can be retried
without reinstalling. Cancellation, a lost reserve/start response, and unknown
registration retain the destination and its budget until actual fence evidence
exists. Unregister failure retains storage and charges for a later retry.

新增 D 端接收注册表：注册/发布之前就建立资源所有者并计费。只有匹配完整身份、
manifest、字节数和写入终态证明后，才允许读取。CPU FP32 安装使用真实工作集副本；
所有 rank 安装完成后才能 ACK。取消、回包丢失、注册结果不明均不能提前释放目标内存。
V 回包中的终态快照仅用于观察，不会像取消型 fence RPC 一样关闭正常交付的业务流程。

Validation uses real localhost HTTP routes and the real V store/index/packer,
with a controlled delayed byte-copy engine. It tests late writes after cancel,
foreign completion identities, short/invalid success, lost replies, ACK retry,
and cleanup failure. This is **not Mooncake, RDMA, GPU or production Scheduler
acceptance**. Step 4 above connects the controlled request loop to this receiver;
CPU installation deliberately refuses FP16/BF16 rather than allocating an
unaccounted conversion. An unknown reservation cannot be freed on absence alone.
Step 7 below now supplies an explicit closed gate for the known-Entry/current-V
case; unavailable, stale or capacity-refused proofs still retain the receiver.

真实本地 HTTP 已覆盖正常交付与故障路径，但未证明 GPU/RDMA 正确性或性能。
CPU 请求级预取流水线接入见 Step 4；生产运行时集成仍需单独推进。

## Step 5 / 第五步：Real models consume delivered bytes

```bash
PYTHONPATH=python python test/registered/disaggregation/run_pvd_draft_cpu_smoke.py \
  --probe --search --sparse-decode --controlled-decode \
  --batch-decode --scheduled-decode --wire-sparse-loop
```

Strict WSL CPU execution passed. The independent smaller SGLang draft runs two
forwards and predicts `(13, 13)`; the target model generates its own committed
tokens. Real post-RoPE Q drives exact V search, four HTTP-controlled sparse
Deliveries (two boundaries × two shards), and **1600 selected K/V bytes**. The
Decode attention consumes the installed received bytes, with no local packing
callback. Twenty-one independent attention comparisons have maximum absolute
error `3.5762786865234375e-7`. Receive charges and private pool capacity recover.

Actual Req/ScheduleBatch result processing, independent request clocks, new
admission/reordering/retraction, wait-all, actual-forward failure and length
limits remain covered. A first query at the boundary uses committed-prefix Q
without calling the draft. Counts are old=9, new=2, third=0, length-limit=1.

两个真实随机小模型的 CPU 闭环已通过：独立 draft → 目标 Q → V 检索 → 4 次
HTTP 控制的稀疏交付 → 安装 → Decode 实际读取 → 原 Req 提交。共交付 1600 bytes，
21 次 attention 对照最大误差约 3.58e-7，接收预算归零。这里不是“只返回了 token
编号”：配对 K/V 的实际字节走过 V Delivery 与 D receiver，再被模型使用。

**Evidence boundary:** HTTP control is real localhost networking, but payload
transfer is still a fake in-process byte copy, not network data transfer or
Mooncake/RDMA. Models are randomly initialized tiny Llamas with a toy tokenizer.
No output-quality, throughput, GPU memory saving or network-hiding claim follows.
Full suite at the Step 4 gate: Windows 1436 passed / 11 skipped; WSL 1441 passed /
6 skipped. Step 5 additionally executes the strict real-model command above.

## Step 6 / 第六步：Direct copies into owned staging

`copy_sparse_kv_into` validates the complete manifest, source layout and versions,
all selected token/head/layer coordinates, destination device/extent/alignment,
and distinct backing storage before writing. It copies paired rows directly into
the final owned buffer: no per-group payload allocation, gather or device move.
V uses it under the existing Entry/index leases and registration lifetime guard.
A transfer budget fitting only the final payload now suffices for this packing
step. This is a memory-accounting improvement, not a throughput measurement.

The primitive has an explicit CUDA opt-in and uses the caller's current stream.
Returning is NOT completion proof. Callers must retain both allocations and
leases and drain queued work on success AND exceptions before reuse/release.
Production V CUDA sparse packing is still refused; neither registration nor
native stream/transport ownership is supplied by this copy function. A runtime
copy failure can partially write the destination; that generation is unusable.

直接打包现在省去逐组临时副本，只需要最终载荷的 staging 预算；所有布局、版本、
有效 token、归属、地址重叠和对齐检查在写入前完成。运行时拷贝异常可能留下部分
结果，不能当作交付成功。新增的 CUDA 入口仅是显式底层原语，返回不表示 GPU 完成；
调用方必须在成功和异常时都保留资源直到真正完成。默认服务仍拒绝 GPU sparse。

Validation: 25 new CPU tests cover FP16/BF16/FP32, two shards, exact bytes,
pre-write refusal, aliasing, partial-copy failure and real store Delivery with
exactly one final-buffer budget. Three real CUDA stream/event tests are present
but skipped in this CPU environment. Full Windows: 1476 passed / 14 skipped;
WSL: 1481 passed / 9 skipped. Skips are not hardware acceptance.
The strict `--wire-sparse-loop` real CPU two-model command also passes after
this change: four Deliveries, 1600 bytes, 21 attention comparisons, maximum
error `3.5762786865234375e-7`, and restored receive budget. Existing legacy
typing/style warnings in `vector_store.py` are not broadly rewritten here.

## Step 7 / 第七步：Fence a reserve request that has not arrived

Reproduced the recovery gap through real localhost HTTP before fixing it: D
published a destination, reserve did not reach V, and receiver close returned
`unknown write authorization`, permanently retaining its buffer and budget.

For a known Entry in the current V worker epoch, `fence_write` now checks for an
absent Delivery and installs a full-identity tombstone under the SAME store lock
used by reserve/start. It then returns an identity-complete fence. A delayed
reserve or start with that Entry/Delivery id is refused, even if it changes the
destination. An existing Delivery always follows its original authorization and
native completion path; in-flight/UNKNOWN writes are never treated as absent.

This relies on Entry/Delivery history being retained for the worker epoch (as
the current store does). Future record pruning MUST retain equivalent write
history or revoke this proof. An unknown Entry or old worker epoch is refused.
The proof closes a write gate; it is NOT successful delivery, installation or ACK.

Tombstones retain exact identity until the worker epoch ends: no TTL or LRU
eviction. `VectorKVStore(max_absent_write_fences=4096)` bounds the number of new
absence proofs; snapshots expose usage and limit. This is a constructor option,
not a new launcher flag. At capacity, old identical proofs remain retryable but
new proofs are refused and D retains memory. This step does not bound all
historical Entry/Delivery records, nor implement cross-restart recovery.

已补充“reserve 尚未抵达 V”的取消恢复：V 在 reserve/start 共用的锁内确认尚无
Delivery，并先保存完整身份的永久关闭标记，才允许 D 根据返回证明释放缓冲。
迟到的 reserve/start 因此无法向已释放地址发起写入。已提交或 UNKNOWN 的传输
仍走原有完成证明，绝不按“未提交”回收。关闭标记不代表 KV 已交付或已安装。

同一 V epoch 内关闭标记不淘汰；默认最多 4096 个，达到上限拒绝新的缺失证明，
不删除旧标记。未知 Entry、旧 V epoch、证明回包丢失或容量不足时 D 继续持有缓冲。
该安全性依赖当前 Entry/Delivery 历史保留规则，未来做历史清理不能直接删除依据。

Seventeen new tests cover late reserve/start, exact-identity retries, stale V,
unknown Entry, capacity, TTL, lost fence replies, retained D budget on refusal,
both threaded reserve/fence orders and existing UNKNOWN writes. Tests use real
HTTP and controlled in-process transport; no RDMA/GPU evidence is implied.
Full regression after this step: Windows **1493 passed / 14 skipped**, WSL
**1498 passed / 9 skipped**. New test files pass Ruff; existing broad-exception
and legacy-typing/style findings in `vector_store.py` remain outside this change.

## Remaining production gates / 尚未完成

- GPU sparse packing, destination visibility/current-next banks, and sparse
  attention kernels (CPU FP32 is explicitly enforced today).
- Native Mooncake submit/poll, CUDA stream ownership, actual TP rank activation,
  and production Scheduler admission/cleanup integration.
- V100S-compatible CAGRA backend execution, real-Q recall/quality and budgets.
- Model-loading ownership, draft-prefix reuse and measured latency hiding.

完整 Prompt 默认服务不变。初始完整 Prompt 不依赖检索索引；不得把 CPU 工作集或
fake transport 注册为生产 GPU 后端，也不得把以上验证称为最终目标完成。
