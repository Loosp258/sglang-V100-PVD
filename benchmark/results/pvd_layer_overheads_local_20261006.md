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

## 2. 两 rank owned ACK/close 并行

- 默认关闭 D 配置 `parallel_owned_cleanup`，要求 `ready_before_cleanup`。
- 最多两个不同 rank 的 record，同一 owner 线程并行 await；每个 record 的锁、
  ACK/close 顺序和精确身份不变。一边失败仍尝试另一边，全部完成后报告失败。
- 外部 cancellation 不取消真实清理；join 两边以后再传播取消，保留 UNKNOWN。
- `step2/gate01`：51 passed，一个既有 warning。确定性阻塞测试证明两边 ACK
  在释放前都进入；覆盖 lost ACK、取消、配置边界与完整双 V 异步交付回归。
- 无 GPU/TPOT 实测；此项减少清理槽占用的潜在时间，不把 ACK 算成 D 前景等待。
- 输出：`artifacts/layer_overheads_20261006/step2/gate01/`。

## 3. 持久通道 ACK/fence

- 默认关闭 D 配置 `channel_cleanup`，要求 binary control channel。
- 同一请求/rank 通道承载完整 WriteIdentity 的 ACK/fence；服务端绑定请求、
  incarnation、Entry 与 heads，只接受本通道已声明的原始 writer 身份。
- 搜索/交付仍最多 2 项；清理独立最多 2 项，总在途最多 4，未退休身份最多 4。
  每个 record 的本地安装/ACK/native terminal 校验沿用原协议。零 miss 证明
  当场移除已 fenced 身份，长串命中不积累通道 writer。
- 通道断线、坏回复或丢证明不代表安全；fence 使用原 HTTP 完整身份恢复。
- `step3/gate02`：97 passed、1 CUDA skipped。真实两 rank CPU 通道各连续
  六次 miss/hit/zero-miss proof/ACK/fence；1 条连接、精确退休，错误 generation
  拒绝，断线仍 HTTP fence 恢复。阻塞网络证明查询/清理各 2 项独立有界。
- gate01 的断线计数断言错误（断线后的 fence 正确走 HTTP，不新增 channel
  请求），已修正测试，原失败日志保留。无 GPU/RDMA/TPOT 实测。
- 输出：`artifacts/layer_overheads_20261006/step3/gate02/`。
