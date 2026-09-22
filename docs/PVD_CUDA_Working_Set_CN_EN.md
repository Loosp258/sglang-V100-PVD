# D CUDA current/next 工作集 / D CUDA current/next banks

## 范围 / Scope

`CUDASparseWorkingSet` 增加独立、显式设备的 GPU Prompt 工作集实现。
它不是 `CPUSparseWorkingSet` 子类，因此不能意外绕过 CPU 安装器的限制。
当前没有接入生产 Scheduler、GPU attention 或 Mooncake 接收器，默认服务行为不变。

`CUDASparseWorkingSet` implements separate, explicitly placed GPU Prompt banks.
It is not a `CPUSparseWorkingSet` subtype; CPU installers continue to reject it.
Production Scheduler, attention and Mooncake receive integration remain separate
implementation tasks. Serving defaults do not change.

## 接口与所有权 / Interface and ownership

- 构造必须提供 indexed CUDA device、浮点 KV dtype 和独立预算；不降级到 CPU。
  Constructor requires an indexed CUDA device, floating KV dtype and explicit
  budget. There is no CPU fallback. Accepted storage dtypes do not certify model
  or V100S kernel support (in particular BF16).
- `stage(payloads, source_guard=...)` 要求完整覆盖全部 payload 的连续源 tensor。
  注册内存 guard 也可用，但其 buffer 必须是该 tensor。先 pin，再检查和复制。
  The source guard must cover every byte, not merely share a storage allocation.
  Source registration/receive budgeting remains the receiver's responsibility.
- 调用方必须在 stage 前证明远端 WRITE 结束且 GPU 可见；普通 Python 返回值或
  CUDA device synchronize **不替代 GPUDirect RDMA 可见性协议**。
  Receive completion and GPU visibility are caller preconditions, not inferred
  by this bank. A local CUDA synchronization is not a GPUDirect visibility proof.
- 为 current 和 next 的每份副本分别预留预算，复制完成后才发布 next。
  First install is complete Prompt at boundary zero. Refresh preserves original
  positions and bounded per-layer/KV-head unions. Generated KV is never owned here.
- `read()` 必须包住所有消费 bank 的 CUDA 操作提交；退出时同步指定设备，才撤销
  reader。不得将 tensor alias 留到作用域外使用。当前有 reader 时 install/close 拒绝。
  Readers encompass all consuming kernel submissions; no alias may escape the
scope. Device synchronization precedes lease release and old-bank retirement.
- 复制失败也必须排空设备再回收。同步失败隔离整个 bank，保留源 guard、部分副本、
  reader（如适用）及预算；没有 force-free。源释放失败也停止后续使用。
  Unknown completion quarantines owners and charges. A source release callback
  failure also quarantines the bank. `source_guard_held` reports a retained guard;
  after an unpin callback failure this does not claim its pin is still registered.

`CPUInstallCandidate` 在共享核心中仅是本地 staged identity 名称，不能当成 CUDA
完成证明或跨 rank ACK。CUDA bank 目前只有单 owner 线程上的本地安装操作。
The shared core's candidate type is metadata only, not a device fence or TP ACK.

## 验证边界 / Validation boundary

18 个新 CPU 策略用例覆盖预算、读取、异常排空、隔离、源范围和 CPU 安装器拒绝。
策略 fixture 显式使用 CPU 存储和同步回调，不伪装为真实 CUDA 测试。
另有 FP16/FP32 真 CUDA 用例，覆盖非默认 stream 上复制、reader 和切换；本地 CPU
环境跳过。尚无 GPU attention 数值、真实 RDMA、TP ranks 或吞吐证据。

Eighteen CPU ownership-policy cases use explicit CPU storage and completion
callbacks. Two separate real-CUDA FP16/FP32 cases exercise a non-default stream,
copy ownership and switching; they skip without CUDA. Neither category validates
production attention, RDMA, model TP or throughput. This synchronous baseline
prioritizes safe lifetime; it does not claim transfer/compute overlap.

本步回归：Windows 全量 1916 passed / 19 skipped；WSL 定向 77 passed /
4 skipped；严格 v5 四场景真实 CPU 模型矩阵再次通过，完整场景 21 次 attention
对照、最大误差约 3.58e-7。传输仍为 fake byte copy。
Step regression: Windows full 1916/19, WSL focused 77/4; all four strict v5
real-model CPU cases passed again. Full cases perform 21 attention comparisons
(about 3.58e-7 maximum error), still with fake payload copies, not RDMA.

```bash
PYTHONPATH=python python test/registered/disaggregation/run_pvd_cpu_tests.py \
  test/registered/disaggregation/test_pvd_cuda_working_set.py -q
```
