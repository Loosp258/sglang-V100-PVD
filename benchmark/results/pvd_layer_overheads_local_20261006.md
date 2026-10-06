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
