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
