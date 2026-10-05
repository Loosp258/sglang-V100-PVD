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
