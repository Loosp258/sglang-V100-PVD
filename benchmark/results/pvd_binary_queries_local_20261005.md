# 逐层 Q 的直接二进制传输

顺序优化第 4 步，D 配置 `binary_queries=true`，默认关闭，独立对照
`d-binary-q`。保留原每层立即发布 Q、候选数量及查询/version 校验。

原系统已有可选 base64 Q；本实验用一个有界 binary envelope，包含版本
magic、metadata 长度、原 search identities 和连续 little-endian float32。
正常路径直接使用已完成 pinned-host Q 的 ndarray，不生成 Python float 列表，
不经过 base64。每 rank 每层 14×128 Q cells 为 7168 原始字节，另加 metadata。

129 CPU 测试通过。实际本地 HTTP 的新旧两 head 查询及 pinned/unpinned
结果完全一致；精确 f32 bytes、独立所有权、负零、finite、shape/总大小、
截断/尾随/编码混用均已覆盖。原 JSON/base64 路径保持兼容。
后续补充单条 binary search 分派及独立实验配置，最终 gate 见
`artifacts/ordered_delivery_20261005/step4/gate02/`。

没有 GPU/native/完整 Decode 耗时测量。此修改减少 Q 序列化工作，不能据此
断言总检索或 D 等待已经降低；还需 ABBA latency/输出对照。
