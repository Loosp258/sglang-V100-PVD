# 融合逐层检索和缺失 KV 交付

顺序优化第 5 步，D 配置 `fused_search_delivery=true`，默认关闭。
独立 ABBA 对照 `d-fused-search-delivery` 保留原 Q 编码、Top4/capacity32/
max_new16、快 CAGRA、两 worker 及逐层发布。没有启用前四步的实验选项。

## 协议与路径

1. D 冻结这一层两 head 的 resident/cache snapshot 与 Q；分配注册最多
   32768 字节的物理接收区，先建立完整 WriteIdentity。allocation-only metadata
   明确绑定 scope/Q digest，没有伪装成已确定的 sparse manifest。
2. 一个 HTTP 请求在 V 执行原搜索、相同 score 排序和 `select_resident`，
   为 CPU cache miss 构造精确 manifest，然后调用原 reserve/start。
   写入仅覆盖物理 MR 的有界前缀，不增加候选或实际 KV 字节。
3. D 独立校验搜索回复身份/版本/预算，重算选择和 miss；只有 manifest、
   destination、exact native terminal bytes 都通过才读 KV。
   物理 RegisteredMemory 保持原对象；prefix descriptor 不用于 MR 注销。
4. 响应丢失时，预先已知的 WriteIdentity 走原 fence。不存在的 writer 被
   tombstone，尚在搜索中的 late reserve 必须拒绝；未知 native writer 保留
   接收区与预算，不能以 HTTP timeout 推断终态。ACK/清理仍在原路径中。

有 miss 时，搜索＋reserve＋start 的前置 RPC 从 3 次变成 1 次；poll/ACK 和
native/GPU 完成证明仍保留。无 miss 时不 PUT，但仍需注册并 fence/退休
预授权区，因此这种情况可能增加开销，必须在实际负载中单独统计。

## 本地验证

阶段 gate：120 passed、1 Linux CUDA skipped。两逻辑 rank 的实际 CPU index、
本地 HTTP、真实 packing/copy 和显式 delayed fake PUT 验证通过，覆盖：

- 全 miss/部分 miss/无 miss，逐 token K/V oracle 及 exact wire bytes。
- D 的两 shard `_select_and_fetch`，JSON/base64 两种旧 Q 编码均可；下一轮
  cache hit 不再 PUT，预算归零。
- 同 RPC 重放只 submit 一次、变更 scope 拒绝、重复 start 拒绝。
- 响应丢失、搜索期间 fence、错误 manifest/selection/digest、短终态字节和
  native failure；未完成 writer 保留，不安装错误 KV。

初次 gate 修复了 V shard manifest 的 Prompt 长度取法；第二次修复 CPU fixture
缺少显式 V transfer budget。原始记录/source hashes 位于
`artifacts/ordered_delivery_20261005/step5/`。最终跨步骤 gate 另见总报告。

实际 Linux CUDA、Mooncake、双物理 rank、峰值预算、完整 Decode 输出与
D wait/TPOT 尚未测。没有宣称 RPC 数下降已成为线上速度收益。
