# V 端 CUDA 稀疏打包基线 / V-side CUDA sparse packing baseline

这是实验性实现，默认关闭，尚未经过真实 GPU/RDMA 验证。
This is experimental implementation, off by default, without GPU/RDMA acceptance.

## 实现 / Implementation

V 启动参数 `--experimental-cuda-sparse-packing` 允许显式稀疏 Delivery 从 CUDA
Entry 中打包选中的配对 K/V。单进程双 shard 与独立 rank 启动均传递该开关。
还必须配置 Mooncake、`--prompt-index-vector-space`、独立 index budget 以及
原有 transfer budget；不能与 CPU/fake launcher 模式混用。模型、HCA、设备编号
仍由用户配置。完整 Prompt 交付行为不变。

The V launcher option enables selected paired K/V packing from CUDA Entries for
explicit sparse Deliveries. Both group and per-rank launchers pass it through.
Mooncake, an explicit vector space and separate index/transfer budgets are required;
CPU/fake launcher combinations are refused. Model/HCA/device choices remain
configurable. Full-Prompt Delivery is unchanged.

顺序 / Ordering:

1. 先占用 staging 预算，再在源 shard 的同一 GPU 分配最终 byte buffer；不分配
   逐组 payload、不复制整个 Prompt 到 host。
   Reserve bytes before allocating final staging on the source GPU. No per-group
   payload or full-Prompt host copy is introduced.
2. 保留 Entry、staging、index/mapping 租约后执行原有逐行 copy。
   Hold Entry, staging and index/mapping ownership through row copies.
3. 成功和部分拷贝失败都在租约内同步源设备；完成后才退出索引租约、注册 staging。
   Synchronize the source device on success AND partial-copy error before ending
   the index lease. Registration happens only after successful packing.
4. start_delivery 仍在提交前检查取消，再通过已有 TransferEngine 提交；只有匹配
   原生终态后释放传输资源。已提交的取消不等于释放。
   Recheck cancellation before the existing engine submit. Matching terminal
   evidence, not cancellation, controls transport resource release.

若同步抛异常，Delivery 进入 UNKNOWN，worker 隔离，Entry/staging/index 租约和预算
均保留。外层之后偶然同步成功也不会自动撤销隔离或推断可释放。没有 force-free。
注册结果不明仍按既有规则隔离。该设计优先建立安全基线，不追求这一阶段的重叠。

A synchronization failure quarantines the Delivery/worker and retains all owners
and charges. An incidental later outer synchronization does not repair the unknown
operation. Registration uncertainty retains the previous quarantine policy. There
is no force-free. This is a correctness baseline, not a latency-overlap optimization.

## 可验证与不可声称 / Evidence boundaries

- CPU doubles 覆盖顺序、设备传参、半途失败、取消、同步失败、隔离及两个 rank 的
  启动配置；它们不是 CUDA kernel 的执行证据。
  CPU doubles test ordering, device selection, failure/cancel/quarantine and launcher
  wiring; they do not execute CUDA kernels.
- 新增两个真实 CUDA store 测试（成功字节对照、排队拷贝后异常），无设备则跳过。
  即使这些测试通过，其 payload engine 仍是 fake，不能称为 RDMA 验证。
  Two real CUDA store tests cover bytes and post-copy failure, and skip without a
  CUDA runtime. Their fake engine does not establish RDMA even if CUDA passes.
- 默认 exact index 的向量镜像仍在 CPU。本开关不是 CAGRA，也不启动 D 的预测循环、
  GPU 接收可见性、稀疏 attention 或真实多 rank 安装。
  The default exact index remains host resident. This flag does not enable CAGRA,
  D prediction, GPU receive visibility, sparse attention or distributed installation.
- 当前使用设备同步，会阻塞提交线程并可能等待同 GPU 的其他任务；不声称隐藏通信。
  Device-wide synchronization blocks submission and can wait for unrelated work on
  that GPU. No network-hiding or performance claim follows.

本步验证 / Validation: Windows 全量 **1898 passed / 17 skipped**；WSL 打包、
Delivery、重打包及 index 定向 **135 passed / 6 skipped**。新增 11 项 CPU policy
测试通过，新增 2 项 CUDA store 测试因无 CUDA runtime 跳过。没有运行 Mooncake。
Eleven new CPU policy cases pass; two new CUDA cases skip. No native Mooncake run.
