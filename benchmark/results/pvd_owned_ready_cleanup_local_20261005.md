# D READY 与接收清理分离

顺序优化第 2 步，配置 `ready_before_cleanup=true`，默认关闭。
原来的两 worker 在 native 完整证明、CPU 独立副本及 bank 本地完成后先
发布 READY，再由同一线程 ACK/close/退休 registry 和 HTTP clients。
worker 数不变，最多保留 56 个请求级任务；清理不完成不会释放 worker 或接收区。

## 已执行

- 64 个 CPU 测试通过，包含真实线程排队、慢清理、取消、容量、超时、错误
  传播及原接收/请求/流水线回归。初次 runner 缺少 stdout.isatty，修复后通过。
- 显式 CPU CUDA policy 的完整 job 测试确认：bank 内容与安装结果一致，
  READY 在阻塞 ACK 前可消费，完成/ACK/退休发生在同一线程，close 等待清理。
  此测试不证明实际 GPU 完成顺序或 native 性能。
- ACK 异常继续尝试所有接收记录的 fence/close；UNKNOWN 保留原 registry，
  清理错误锁住后续 admission，并由 request close 报告。公开 Future 的取消
  不取消拥有 native owners 的内部任务。
- 已准备 `d-owned-cleanup` 独立 ABBA 对照，仅改变上述配置。

原始日志/source hashes：`artifacts/ordered_delivery_20261005/step2/gate03/`。
无 GPU/native/full-path 测量。慢 ACK 仍占 worker，可能延迟后续层，因此
不能把提前 READY 直接称为 TPOT 改善，恢复资源后必须测整体等待及 cleanup。
