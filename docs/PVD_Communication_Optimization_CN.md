# PVD 通信与通算融合设计及实验状态

## 实测依据（2026-09-25）

三机 Qwen2.5-7B-Instruct、P TP1 / V 两个 GPU shard / D TP1、
`mlx5_0` 单 rail 的约 1938-token 请求，完整 Prompt fan-in 由 V 两个
rank 各写约 55.6 MB。每 rank 的现有计划包含约 108,528 个小切片，
分为 14 个 Mooncake 原生 batch。V 日志中的原生 batch submit 累计
约 54.65 s，整个 writer 约 55.01 s。这个数据说明首先应减少
请求数和布局重排成本；不能仅凭它断言 RDMA 链路带宽或 Mooncake
底层实现本身是根因。该请求虽完成 8/8 token，端到端仍耗时
242.79 s（显式启用后台搜索 I/O）或 285.69 s（新默认路径、另一次
运行）。约 515-token 输入的 2→4 并发总吞吐基本不变。
这些数据都是单轮冷 Entry，不构成统计性加速结论。

现有 `packed_fanin_transfer_slices` 为每个 component、V shard、
token 生成切片。V 源数据按本 shard 的 head 连续，D 目标的完整
head 布局却在 token 内交错；因此只在当前地址列表上合并邻接
slice，不能解决 V TP2 → D TP1 的主要碎片化。

## 第一优先级：Rank-packed fan-in

维持 Mooncake 作为可靠的 RDMA transport，先改变交给它的**传输
形状**，而不是直接 fork/修改 Mooncake。新协议必须协商版本；
旧 `FULL_KV_FANIN_PROTOCOL` 和现有接收端在未显式协商时保持不变。

1. D 为一个 Delivery 预留、注册并 pin 一个带有 V-rank 分区的
   receive staging；每一分区的 offset、长度、layout fingerprint、
   generation 与 receiver epoch 写入不可变计划。预算先于 GPU
   分配；如果重排需要第二份 canonical 缓冲区，也必须单独计费。
2. 对当前 V TP2 → D TP1，V 的完整 shard 已是连续源字节，
   可由每个 V rank 向自己的连续 D 分区提交一次或少量有界
   Mooncake WRITE，无须先在 V 再复制一遍。一般异构 TP 的子
   head 交集才需要 V-side pack；该 pack 缓冲区必须有独立所有者、
   预算及 CUDA 完成 fence。
3. 只有全部 V writer 的 Mooncake terminal-success 证明到齐，
   D 才在本机将 rank-packed staging 重排成
   `unpack_full_prompt_kv` 所需的 canonical 布局。重排完成并
   完成 CUDA 可见性 fence 后才安装工作集和 ACK。失败、取消、
   超时或 UNKNOWN 时保持 MR 和所有 staging 被 pin；不得凭
   HTTP 超时释放、复用或降级到另一个可能重叠写入的目的地。
4. 保持 Entry / EntryShard / Delivery 分离，单个 Entry 可复用。
   每个 Delivery 的 receive slot、writer set、plan fingerprint、
   epoch 和 ACK 独立；V/D TP 数量不同依旧按 head 交集规划。

首版不需要新增 Mooncake API。传输层接口仍是
`submit_batch_put`/status，变化是 108k 个约 512-byte slice
变为按 V rank 的少量大块。生产启用前必须有字节级 TP1/TP2/TP4
往返测试、计划篡改拒绝测试、提交失败/UNKNOWN/晚到写入的资源
生命周期测试，以及在 CloudLab 上同配置 A/B 实测。性能目标
优先检查原生 submit 总时间和 p95，而不是只看发送字节数。

当前代码已有**显式 opt-in** 的 v2 wire：D 设置
`--pvd-full-kv-fanin-rank-packed`，旧 v1 仍为默认。v2 要求完整 V
shard、V coordinator 与每个必需 shard 公告 v2 能力，以及 V 原生
batch PUT。D 为注册接收区和独立 canonical 重排缓冲区分别预留预算；
只在全部 writer terminal-success 后做本地重排和原有 unpack/ACK。
CPU 的真实类 V↔D 测试已验证字节重建、重复刷新、预算归还、
旧 V 拒绝与 v1 兼容。下面记录首轮 CloudLab A/B；v2 仍为
实验性 opt-in，不能凭单轮结果建议默认开启。

### 2026-09-25 三机首轮同代码 A/B

在隔离端口 P=`clgpu020:30002`、V=`clgpu021:9100/9300/9301`、
D=`clgpu019:30003`、Gateway=`clgpu021:8001` 上实测。P 使用已有
TP1 服务；V 的两个 GPU shard 和 D TP1 均使用提交 `55d364a15`，
Qwen2.5-7B-Instruct、`mlx5_0` 单 rail、`cagra-auto`、相同 D
attention/刷新配置。只切换 D 的
`--pvd-full-kv-fanin-rank-packed`；V 均提供 v1/v2 能力。客户端均以
相同句子重复 213 次、输出 8 token；请求 ID 的 tokenization 使实际
Prompt 分别为 1938/1937 token。两轮都是单请求、冷 Entry，顺序为
v2 后 v1；不是多轮、随机化或并发统计实验。

| 指标 | v2 rank-packed | v1 对照 |
| --- | ---: | ---: |
| 完整生成 | 8/8 | 8/8 |
| TTFT | 8.33 s | 95.53 s |
| 总耗时 | 158.30 s | 239.67 s |
| 每个 V rank 写入字节 | 55,566,336 | 55,537,664 |
| 每个 V rank 计划切片 | 1 | 108,472 |
| 每个 V rank Mooncake submit | 1 | 14 |
| 每个 V rank writer 生命周期 | 约 0.044 s | 约 55.13 s |

两轮 V writer 均为 `terminal_success`；之后两 rank 的
`used_inflight=0`、`unknown_transfers=0`、`quarantined=false`。
v2 运行 ID `7c46db11d88d`，v1 运行 ID `e1c023d6ec34`。
这些数据证明当前 TP2→TP1 场景中的切片/提交瓶颈显著下降，
但单轮端到端差值不能作为稳定加速比；v2 仍耗时 158 s，
长上下文生成的 D target forward/稀疏刷新前注意力是下一瓶颈。
更大的 Prompt、并发、重复轮次、数值一致性和故障恢复仍须验收。

恢复 v2 后另做了单轮 4 客户端、各 55 次句子重复的排队测试
（`run_id=e6509bfc8daf`）：实际每请求 513-token Prompt，4×8 token
全部完成，总墙钟 181.17 s、合计 0.177 token/s、p95 TTFT 47.23 s。
两 V rank 随后仍为零 in-flight、零 UNKNOWN、未隔离。该 D 实例的
最终 token 容量只允许同时运行 1 个请求，所以这里验证的是并发到达时
的排队、Delivery 生命周期和释放，不是 4 路同时 Decode 的扩展性；
也不能把它与历史 v1 单轮的 0.142 token/s 当作严格同条件加速比。

## 专用集合通信：Select–Pack–FanIn–Install

这是 PVD **应用层集合操作**，不是把 `alltoall`、
`allgather` 或 `allreduce` 换个名字。参与者是 Router 选出的
一个 P worker group、一个 V worker group 和一个 D worker group；
它们不要求 GPU 数量相同。P→V 初始化完整 Prompt KV；后续
每个 D Delivery 独立触发一次有界的稀疏刷新：

```text
D: (Entry, Delivery, boundary, generation, Q[rank,layer,Q-head], credits)
    -> V shards: 本地索引搜索 -> 同一 (layer,KV-head) 内 Q-head
       token 并集/去重/限额 -> gather/pack K,V
    -> Mooncake: 每 V rank 向 D 的独立 staging 分区 WRITE
    -> D: 等待所有 writer terminal + 本地 CUDA fence
       -> 按逻辑 token/page ID 验证、重排 -> 边界原子安装 -> ACK
```

首次完整 Prompt fan-in 是该操作的 dense 特例；周期刷新是
sparse 特例。一个新请求加入 batch **不**重置旧请求的 m-token
刷新时钟。D 用正式生成的 token 决定下一轮；预测分支仅提前
产生搜索 query，绝不提交预测 token。提前搜索、打包和传输可与
当前 bank 的 D 计算并行；到边界仍未就绪时 D 全部等待。
为避免覆盖，使用有界 credits + 双缓冲或环形 slot；slot
只有 ACK 和所有写入 fence 后才能复用。不同 layer/KV head
的分数不作无声明的全局 Top-K。必须验证版本、选择 ID、
head ownership、bank generation 和精确 writer 集合。

## 类 MegaMoE 的通算融合：借鉴原则，不移植内核

MegaMoE 的可借鉴点是合并 dispatch、计算和 combine 之间的
中间写回，并让通信与计算流水重叠；其具体内核、指令和
多 GPU 内存假设不能直接移植到 V100S（SM70）或跨节点 RDMA。
本项目可分两处实现 V100S 专用 CUDA/Triton 路径：

* V：融合 CAGRA 结果的 logical ID 映射、同 KV head 去重/限额
  和 K/V gather+pack，避免每个 Q head 的 GPU→CPU 往返及
  大量小地址列表。先保持 exact backend 作召回基准。
* D：把 rank-packed staging 的本地重排、稀疏 bank 构建和
  一 token GQA attention 尽量批量化/融合。当前 `online`
  基线是逐 Q head、逐 chunk 的 Python/Torch 循环，长 Prompt
  的第一次 target forward 约 35.6 s；不能把 Mooncake 优化
  误当作消除此计算瓶颈。内核必须保持 post-RoPE Q/K、共享
  KV-head 工作集、生成 token 的 D 本地 KV、数值精度和
  bank read lease/CUDA fence。

## 实施与验收顺序

1. 固定 A/B 负载：同模型、prompt 长度、输出长度、冷/热 Entry、
   1/2/4 并发和相同 D attention 配置；分开记录 pack、native
   submit、transport status、D scatter、CAGRA、target forward、
   TTFT/TPOT/p95、显存峰值和 UNKNOWN。
2. rank-packed dense fan-in 的计划器、字节级测试和显式协议协商
   已实现，独立 sidecar 首轮真实 RDMA A/B 和 v2 四客户端排队测试
   如上；下一步增加重复轮次、严格同条件并发 A/B 与真正多 running
   request 的容量，验证正确性和尾延迟。绝不在一次不确定写入后自动
   fallback。
3. 扩展同一协议为 sparse Select–Pack–FanIn–Install，保持现有
   m-token 边界和全部等待语义；做故障注入和 1/2/4 并发验收。
4. 最后做 V 与 D 的融合 kernel，分别证明真实 query 的召回、
   目标输出质量、V100S 数值正确性与端到端性能。双 rail 只有
   在 `mlx5_1` 物理链路恢复并完成 GPUDirect 预检后才加入。

本文件是设计与验收合同。v2 是实验性 opt-in，虽已通过首轮真实
GPU/RDMA A/B，尚未通过多轮及真正并行 Decode 的性能验收；稀疏集合通信和融合
kernel 仍是后续工作。
