# Sparse V→D fan-in / 稀疏刷新多源聚合

## Current component / 当前组件

`CUDASparseFanInStage` binds one D TP1 working-set bank to all V source
destinations described by `RoutedShardSearchClient.partition_specs()`. Each V
source owns its own registered destination, exact manifest, endpoint, sender
epoch and terminal delivery proof. The stage refuses missing or unready sources,
wrong Entry/layout/operation identity, a changed endpoint or reused records.

`CUDASparseFanInStage` 将一个 D TP1 工作集绑定到分源计划中的全部 V 目标。
每个 V 源独立拥有注册目标、manifest、端点、sender epoch 和最终交付凭据。
缺源、未完成、Entry/layout/operation 不一致、端点变化或记录重用均拒绝。

After all V sources have proved terminal success, the component orders each
remote GPU write before local reads. It reserves an explicit aggregate budget,
copies the source buffers into one D-owned contiguous byte buffer, fences the
copy, rewrites only the layout identity from storage to compute, and stages the
complete D bank once. The bank's existing CUDA staging fence and rank-install
runtime supply the returned local receipt. This receipt only becomes ACKable
after the installation runtime confirms resume. ACK remains one per V source.

全部 V 源证明成功后，逐源完成远端 WRITE 可见性排序。先保留有界聚合预算，再把
各源数据拷入 D 独占的连续暂存 buffer，等待拷贝完成，仅将 storage layout
身份换成 compute layout，一次性提交完整 D bank。现有 CUDA bank fence 和
rank-install runtime 产生本地 receipt；只有确认安装与恢复后，每个 V 源才能
分别 ACK。

Capacity refusal occurs before claiming records or copying. An uncertain device
completion retains every source MR, the aggregate bytes and their reservations;
the records are quarantined. Python cancellation, HTTP completion or a receipt
object alone cannot trigger release. This component holds at most one aggregate
per call. The full Prompt path and generated D-token KV remain independent.

容量不足发生在认领源记录、拷贝之前，可在容量释放后重试。设备完成未知则保留
全部源 MR、聚合 buffer 和预算，并隔离记录。Python 取消、HTTP 已返回或
仅有 receipt 不能触发回收。完整 Prompt 和 D 新生成 token KV 不受此组件管理。

## Integration still needed / 尚待接通

The present `CUDASparseDelivery` still drives one source per D rank. Its request
controller must be extended to publish and poll every source record, invoke this
stage, and then drive all ACK/close tasks. The production Scheduler factory and
real multi-HCA CUDA/RDMA acceptance follow. This component alone is not a
production predictive serving switch.

当前 `CUDASparseDelivery` 仍按单源调用。还需让请求控制器发布、轮询全部源记录，
调用本聚合组件，并驱动所有 ACK/close；之后再接生产 Scheduler 工厂，并做真实
多 HCA CUDA/RDMA 验证。本组件尚不会自动开启生产预测检索。

## Verification / 验证

CPU policy tests use two real V stores, two localhost shard HTTP servers, exact
Prompt indexes and delayed fake byte transfers. They exercise waiting for both
terminal proofs, exact KV bytes in the D bank, installation before either ACK,
two-source cleanup, refusal before allocation, budget retry and UNKNOWN retention.
They substitute CPU tensors for the CUDA operations; no GPU/RDMA claim follows.

CPU 策略测试使用真实双 V store、双 localhost HTTP、精确索引和延迟 fake 字节
传输，核对双源成功、D 工作集字节、安装后 ACK、双源回收、分配前拒绝、预算重试
和 UNKNOWN 保留。CUDA 操作以 CPU tensor 替代，尚无 GPU/RDMA 实测证据。
