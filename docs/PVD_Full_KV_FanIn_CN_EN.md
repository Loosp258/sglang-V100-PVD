# 完整 Prompt KV 多源合并 / Full-Prompt KV fan-in

## 字节映射已实现 / Byte planning implemented

`sharding.source_shard_intersections` 根据全局 KV head 区间计算一个 D rank 与
各 V shard 的交集。每个交集同时保留 V 内 head offset、D 内 head offset 和
head count，不要求 V/D 的 TP 数互相整除，只要求各自均分总 KV heads。

`packed_fanin_transfer_slices` 返回按 V rank 分组的相对 byte offset/length，
遵循现有 `kv_packer` 的 component/page/token 顺序，包含末页 padding。
例如总共 12 heads、V TP3、D TP2：D1 的 heads 6–11 来自 V1 的本地 heads 2–3
以及 V2 的本地 heads 0–3；后者写入 D1 的本地 head offset 2。

The head-intersection planner partitions each D rank's global KV-head interval
across V shards, retaining both source and destination offsets. V/D TP sizes need
not divide each other, but each must evenly partition total KV heads. The packed
fan-in planner returns relative byte ranges grouped by V rank in existing
component/page/token order, including final-page padding. With 12 heads, V TP3
and D TP2, D1 combines V1 local heads 2–3 and V2 local heads 0–3; the latter starts
at D1 local head offset 2.

旧 `source_rank_and_head_offset` 与 `packed_transfer_slices` 共用此算法，但
**仍拒绝需要多个 V shard 的交付**。没有更改 coordinator、descriptor、
WriteIdentity、fence、CLI TP 限制或真实模型消费能力。不能将 planner 的支持
矩阵解读为当前可部署的 PVD 拓扑。

Legacy one-source APIs share this planner but **still reject multi-source delivery**.
Coordinator, descriptors, WriteIdentity, fences, CLI TP limits and model-consumer
capabilities are unchanged. Planner coverage is not a deployable topology matrix.

## 下一阶段必须满足 / Required before wire activation

- 一次 D 接收内存的授权必须区分每个 V writer，且将 writer 身份绑定到本次
  receiver incarnation、entry generation、source rank、destination rank 和子交付。
- 每个 writer 只能写入其计划的目标子范围，不能用“共享同一个 MR”代替软件授权。
- 所有计划内 writer 的完成证明到齐后才可 unpack/安装；部分成功不能判为已交付。
- 取消/超时/部分提交异常后，必须收集所有可能已提交 writer 的终止证明。单个
  writer 完成不允许释放共享接收 MR，也不允许复用目标地址。
- 多源完成后仍遵守原所有 D ranks 的安装/ACK 协议。重试不能复用旧 generation
  身份，或遗漏已提交但返回失败的 writer。

Before activation, authorize each V writer against receiver incarnation, Entry
generation, source/destination rank and subdelivery; restrict writes to its planned
ranges. Unpack/install requires completion from every expected writer. Cancellation,
timeouts and partial submission failures must fence every potentially submitted
writer before releasing or reusing the shared MR. Preserve all-D-rank install/ACK
agreement and generation-safe retry. A shared MR is not per-writer authorization.

## 验证 / Validation

29 个新增 CPU 用例，使用真实 `pack_full_prompt_kv` 字节数据，对 V2→D1、
V3→D2、V4→D3、V2→D4、V1→D3、V3→D3 的所有 D ranks、page size 1/2、
非连续乱序 pages、多个 K/V layers 重建并逐字节对照。另检查每字节恰写一次、
源/目标范围不越界、非整数参数和不兼容布局拒绝。没有执行原生多源 RDMA。

Twenty-nine new CPU cases use actual packed KV bytes across six V/D partitions,
all destination ranks, page sizes 1/2, reordered pages and multiple K/V layers.
They check byte-exact reconstruction, exactly-once coverage, range bounds and
invalid input/layout refusal. No native multi-writer RDMA has been exercised.

本步骤 Windows 全量 **2294 passed / 29 skipped**，WSL 定向 **98 passed**。
严格 v5 CPU 四场景已在前一步 Scheduler 修改后通过，本次只改 byte planner 与
布局校验，不声称重新运行该矩阵。

This step passes **2294 / 29 skipped** on the full Windows suite and **98** focused
WSL cases. The strict v5 CPU matrix passed after the preceding Scheduler change;
this byte-planner/layout-validation step does not claim a separate matrix rerun.
