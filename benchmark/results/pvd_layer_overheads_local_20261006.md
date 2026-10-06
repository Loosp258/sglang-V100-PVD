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

## 4. 本地 CUDA 完成异步等待

- 默认关闭 D 配置 `async_cuda_completion`，要求 bounded async layer jobs。
- Q D2H、接收 KV D2H、bank H2D 记录 local event，在原 owner 上 query/yield，
  让另一任务的网络处理继续；CUDA/inference context 不跨 await。
- 原 GPUDirect ordering 仍保留；异常恢复仍用原保守 stream fence。异步 KV
  copy 持有 record 锁与全部 MR/host/view owners，成功以后才写 cache valid。
- 两个 rank 的 pinned receive scratch 分区：capacity32/两 lease 新增 32 KiB
  物理最大容量，计入原 transfer budget；完成证明前不复用或退费。
- `step4/gate02`：70 passed、2 real-CUDA skipped。CPU policy 验证事件等待
  能让出循环、取消仍 join、cache 未完成不安装、unknown 保留 scratch/budget；
  双 V 完整异步 job 在 pinned/event 两种模式保持精确 FP16 字节与退休。
- gate01 新测试误用 fixture.prepare，修正为既有 prepare helper；原日志保留。
- `query()` 的完成语义依据 [PyTorch Event 文档](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.Event.html)。
  新 GPU gate 已写但无 GPU 跳过；没有 RDMA/GPU/TPOT 资格结论。0.5 ms 轮询
  间隔也需真实 GPU 对照其唤醒延迟，不宣称它一定比同步更快。
- 输出：`artifacts/layer_overheads_20261006/step4/gate02/`。

## 5. 精确缓存版本与增量

- 默认关闭 D 配置 `cache_delta_snapshots`，要求 compact snapshots 与 owned
  binary channel。逐行 CPU copy 成功后更新 journal/bitmap，不扫描完整 validity。
  完整 canonical 快照按版本缓存，摘要/选择与旧 compact snapshot 完全一致。
- 首次发送完整集合；后续发送自最后确认版本以来的新 ID、目标版本和精确
  bitmap digest。V 在查询结果中确认快照，独立于 native delivery ACK。
- V 只保存当前请求/rank 的 28x2 bitmaps。重复累积增量可幂等处理；旧增量或
  缺基准要求 full resync，不猜缓存命中。完整旧快照可用于原查询但不倒退最新状态。
- resync 响应带原始 WriteIdentity 的 absent-write fence；D 验证并关闭原 MR
  lease，创建新身份/generation 后仅重试一次完整冻结快照。坏或丢证明不重用身份。
- D journal/bitmap/canonical metadata 计入已有 request scratch；V persistent
  metadata 和两查询临时集合由同一 lifecycle transfer budget 有界预留，join
  通道工作后退费。32K Prompt 的 D bound=9,748,480 bytes，V/rank/channel
  host bound=8,839,168 bytes；有全局 admission，不是无限 32 通道叠加。
- `step5/gate01`：122 passed、1 CUDA skipped；`step5/gate02`：57 passed。
  第二次覆盖真实通道 full resync、旧身份 tombstone、新 generation、零新增 PUT
  和组合双 V pipeline；后续补齐保守 memory bounds，以最终 gate 为当前源资格。
- 全部五项组合 `final/gate01`：**567 passed、4 real-CUDA skipped**，一个既有
  asyncio_mode warning；156 个 PVD Python 模块 AST parse 通过。分步数字有重叠，
  不相加。测试前后归一化源码 hash 一致；最终 commit blob 对照另存 evidence_commit。
- 完整组合包含 pinned/event/no-pinned、local async completion on/off，两 V
  真实 CPU channel、阻塞前两层 ACK 时后两层精确 bank、随后全命中、8 MR 退休
  和预算回到零；CAGRA/native/CUDA 使用明确 CPU policy doubles，不是模型质量证据。
- 密集且不变的两 head 缓存控制字段字节（包含整个 delta 描述，不含 Q/其他身份）：

  | Prompt | 原完整快照 | 增量 |
  |---|---:|---:|
  | 2159 | 822 B | 378 B |
  | 8192 | 2838 B | 378 B |
  | 32768 | 11030 B | 383 B |

- 初次或频繁增长的增量不保证更小。新 GPU/RDMA/完整质量、D wait/TPOT、负载与
  故障 gates 尚未测量，全部默认关闭；保留过去无收益实验，不相加历史收益。
- 日志/命令/源码：`artifacts/layer_overheads_20261006/`；字节与 AST 证据
  为 `diagnostics.json`。最终 runner 的初次参数提取错误在正式 pytest 启动前
  已修正，没有从那次启动声称任何测试结果。
