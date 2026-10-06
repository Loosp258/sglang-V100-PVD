# 五项逐层固定开销优化：本地证据

工作树 `sglang-V100-PVD-oasiskv`，分支 `codex/pvd-oasiskv`。
按用户要求每步本地 commit，不 push。无 GPU，CPU 验证不是 RDMA/CUDA/TPOT 证据。
快图、V/CAGRA、Q、TopK、工作集和实际 token 写回语义保持原配置。

## 1. V 结果对象与单次最终编码

- 默认关闭 V 环境变量 `PVD_TYPED_BATCH_RESULTS=1`。
- 共用经过原校验的 batch 数据处理器；融合路径直接消费结果对象，省掉内部
  JSON 编码/解析；普通 batch 和融合响应最终只编码一次，并发送检查过的字节。
- 保留 item/row/token/space/version 校验，最终响应上限 2 MiB，拒绝非有限 JSON。
- `step1/gate01`：155 passed、2 CUDA skipped，一个既有 asyncio_mode warning。
  真实两 rank CPU HTTP/channel 测试验证选择、manifest、FP16 字节、预算与退休；
  计数证明融合路径只有一次最终响应编码，包含大小边界与坏值拒绝。
- 原始日志、命令及归一化源码 hash：`artifacts/layer_overheads_20261006/step1/gate01/`。
- 尚未测量 GPU/native 搜索、D 等待和 TPOT，不据编码次数推断端到端降幅。
