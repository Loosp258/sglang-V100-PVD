# 逐层交付继续优化：本地验证

2026-10-05，`codex/pvd-oasiskv`；按用户指示每步本地 commit，不推送。
当前没有 GPU，所有性能收益待真实 CUDA/Mooncake/双物理 rank 的完整路径对照。
保留原快图、逐层 Q、候选预算及 actual-only 写回，新选项默认关闭。

## 1. 二进制 Q 与融合检索交付

D 配置 `binary_queries=true,fused_search_delivery=true`；嵌套搜索使用原始
little-endian float32 字节。冻结搜索元数据与精确 Q 字节、缓存/驻留快照
绑定摘要，V 在写入前核对与 allocation-only 接收授权匹配。原 JSON/base64
接口继续可用；二进制融合 endpoint 为 `search-deliver-binary`。

本地 gate：**126 passed，1 Linux CUDA skipped**。真实 CPU HTTP、两个逻辑
rank、miss/mixed/hit、D 实际 `_select_and_fetch` 原选择与 KV 字节一致；
冻结输入修改、摘要变化、短/多余载荷、NaN 和错误嵌套元数据被拒绝。
原有 failed/unknown native 与丢响应 fence 测试继续通过；native PUT 是明确
标注的 delayed CPU test engine，不能当作 RDMA 验证。既有 asyncio_mode warning。

日志与源文件 normalized-LF hash：
`artifacts/delivery_followup_20261005/step1/gate03/{unit.txt,status.json}`。
未进行真实 CUDA、线上 D wait/TPOT 或输出质量实验，不能累加历史阶段收益。

## 2. 融合交付复用 D 接收槽

在融合模式下允许 `reuse_receive_slots=true`，兼容第 1 步二进制 Q。
allocation-only 明确绑定原选择 scope，既有池检查 rank/Entry/dtype/宽度/上界；
逻辑记录只收取 inflight charge，物理容量在池中持续计费。每次 lease 使用
独立 generation，实际 KV 是同一授权的精确前缀。原 GPUDirect ordering、
native 字节证明、安装后 ACK 及 UNKNOWN 保留没有省略。全缓存命中复用同一
物理槽，仍用 absent-write fence 确认不会再有 writer 后返还。

本地 gate：**87 passed，1 Linux CUDA skipped**。真实 HTTP 与 FP16 CPU
字节、明确 CUDA policy doubles；miss→hit→miss 三次仅一次物理 MR 注册，
结束只注销原 physical owner。旧请求重放不新增 writer；失响应且 native 尚未完成时
不返槽，原池注册/注销未知状态与关闭测试通过。日志及 hash 位于
`artifacts/delivery_followup_20261005/step2/gate03/`。真实 GPU/RDMA 与组合
D wait/TPOT 待测；旧独立接收池实验的注册收益不能当作当前延迟收益。

## 3. 精确缓存快照压缩

融合配置增加默认关闭的 `compact_cache_snapshots=true`。少于等于 8 个 ID
保留原列表；其余按实际原始字节大小选 uint16 稀疏整数或 little-endian
bitset，随后 base64 包装。生产者不再将较大的集合变成 Python 整数列表。
不使用跨消息增量状态；每个授权仍绑定完整、冻结的 head 缓存集合。

2159-token Prompt 的完整 bitmap 原始载荷为 **270 B/head**（另有 base64
和明确编码元数据）；稀疏载荷为 **2 B/ID**。这是载荷大小，非延迟实测。
接收端检查精确长度、顺序、重复、越界、高位 padding 与规范 base64。

本地 gate：**107 passed，1 Linux CUDA skipped**。1/7/8/9/2159/32768-token
边界、密集/稀疏集合、错误编码以及真实 HTTP 二进制/JSON 混合 miss 的
原选择和 wire manifest 完全等价；D 的实际两逻辑 rank 路径继续通过。
日志与 hash：`artifacts/delivery_followup_20261005/step3/gate01/`。
GPU/RDMA、线上质量、缓存集合规模下的 D wait/TPOT 收益待测。

## 4. 有界 pinned scratch 与 bank 完成事件

新增默认关闭的 `reuse_pinned_scratch`、`event_bank_ready`；后者要求前者。
每个活跃 job 独占 Q float32、四 head 的 H2D host rows 及 rank D2H host
字节缓冲。capacity32、两个槽持续计费上限 **225280 B（220 KiB）**，
顺序 job 可复用；同一线程在实际 event/stream completion 后才返还。
本地完成未知时隔离全部槽、tensor 与预算，close 不猜测完成。

event 模式发布带 completion 的独立 bank，前台已有 `wait_event`；发布
表示可以排队 GPU 依赖，不表示 GPU copy 已完成。原 worker 仍等待实际完成
并拥有清理责任，未声称释放 worker 或提升任务吞吐。新安装显式等待 resident
bank 的事件，防止早期发布的 bank 被另一个 stream 提前读取。远程 PUT
终态字节检查和 D 上 GPUDirect receive ordering 保持原策略。

本地 gate：**93 passed，2 actual CUDA skipped**。固定 host prefix 的真实
CPU HTTP/FP16 bytes、显式 CPU CUDA policy 下的完整 job，在阻塞 completion
时仍保留 lease/预算、可消费带事件 bank；随后两个 job 只分配一个 scratch
槽。event/stream 错误保留 UNKNOWN，原 request/pipeline/cleanup gate 通过。
日志与 hash：`artifacts/delivery_followup_20261005/step4/gate02/`。
真实 pinned CUDA copy 测试已加入但未执行；native/RDMA/输出及完整 TPOT 待测。
CPU consumer_wait 可能转移到 GPU 事件依赖，不能用 callback 提前 READY 的
时间单独宣称 Decode 加速。
