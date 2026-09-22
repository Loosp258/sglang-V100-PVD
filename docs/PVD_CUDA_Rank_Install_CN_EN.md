# CUDA bank 逐 rank 安装 / Per-rank CUDA bank installation

`CUDARankInstallParticipant` 将 `CUDASparseWorkingSet` 接入现有逻辑 rank 消息协议。
CPU 与 CUDA participant 共享同一个安装状态机，但保留互斥的 bank 类型检查；
生产 CPU receiver/runtime 装配没有改为接收 CUDA，也没有启动真实 GPU TP。

The CUDA participant binds CUDA-bank ownership to the existing logical rank
protocol. CPU and CUDA share the state machine, not a permissive device gate.
Production CPU registries/runtime assembly still reject CUDA objects. This is
an implementation component, not a production TP launcher or GPU attention hook.

## 顺序 / Ordering

1. 调用方先证明远端接收完成及 GPU 可见，提供源 guard。`stage` 完成 GPU 副本并
   保留 next 预算后才返回 PREPARED。
   The caller proves receive completion/visibility first; staging pins the source
   and drains local copy work before emitting PREPARED. This is not an RDMA fence.
2. 旧 bank 的 reader 必须完成后才可 PARKED；边界与 request/Entry/incarnation 必须
   精确匹配。进入 PARKED 后不再接受旧 forward。
   PARKED requires drained readers at the exact request-local boundary.
3. 协调器收齐所有 PREPARED/PARKED 后发 INSTALL。本地切换成功返回 APPLIED，但此时
   **仍不能执行 Decode**。部分切换异常将 participant 置为 terminal。
   A local APPLIED is not permission to decode; partial-install failure is terminal.
4. 协调器收齐 APPLIED 才发 RESUME；participant 收到匹配 RESUME 后回复 RESUMED。
   全组执行调用方必须使用 `RankInstallExchange.can_decode()` 或现有 runtime permit，
   等全部 RESUMED，不能只检查某一个 participant。
   The group execution owner must gate on all RESUMED receipts, not one peer's
   local read gate. Coordinator notifications must use a trusted bound channel.

重复 INSTALL/RESUME 不重复切换/递增时钟。旧进程 epoch、旧 staged identity、
错误 rank、旧 operation 和提前 RESUME 均拒绝。CUDA bank quarantine 会阻止所有
命令（包括重复 ACK 路径），但 snapshot/stop 仍可用于观测和停止请求。
Unknown CUDA completion retains bank/source/reader charges rather than emitting
a usable receipt. Cancellation and logical failure never replace device draining.

## 验证与后续接入 / Evidence and remaining assembly

14 个新 CPU 策略用例使用 CPU tensor 与显式同步回调，包括两轮、多 logical ranks、
迟到 ACK、取消、部分切换和 quarantine。另有真实 CUDA 单 rank 两轮测试；无 CUDA
时跳过。logical ranks 不等于实际 GPU TP；没有网络故障或原生 RDMA 验收结论。

Fourteen CPU policy tests cover round progression, missing receipts, cancellation,
partial swap and quarantine. One separate real-CUDA local two-round test skips
without CUDA. Production transport wiring, CUDA receive visibility, model TP and
attention integration still require implementation and hardware validation.

本步 / This step: Windows 全量 / full **1930 passed / 20 skipped**；WSL
rank 协议定向 / focused rank-protocol regression **102 passed / 1 skipped**。
四场景真实 CPU 模型矩阵在前一步 bank 修改后通过；本步未另称已重跑该矩阵。
The real-model CPU matrix passed at the preceding bank step, not a separate
matrix rerun for this participant addition. No CUDA/RDMA execution is claimed.

```bash
PYTHONPATH=python python test/registered/disaggregation/run_pvd_cpu_tests.py \
  test/registered/disaggregation/test_pvd_cuda_rank_install.py -q
```
