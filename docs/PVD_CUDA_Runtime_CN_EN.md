# CUDA 本地安装与执行 runtime / Local install and execution runtime

## 已实现 / Implemented

`CUDARuntimeInstallGroup` 把一个 CUDA bank、participant、安装 exchange 和有界
owner-polled command queue 绑定为同一个请求 owner。仅允许 **TP1 单 rank**；
不是多 GPU/多进程 collective，也不是生产 Scheduler 的自动装配。

The group binds one CUDA bank, participant, exchange and bounded owner-polled
command queue to one request. It explicitly supports TP1 only. It is not a
multi-process collective or automatic production Scheduler activation.

- `stage(..., source_guard=...)`：已确认本地 CUDA 可见性的源，复制后发布 PREPARED。
  Explicit local sources require guarded ownership and prior visibility proof.
- `stage_received(record, epoch)`：精确 remote success、CUDA ordering、bank copy
  和 PREPARED 接到同一 runtime；仅接受 CUDA receive record，不放宽 CPU 接收路径。
  Remote proof, local ordering, bank copy and preparation share the same runtime.
- `try_install`：所有 prepared/parked 后发 INSTALL，APPLIED 后发 RESUME；只有完整
  RESUMED 才允许解码。超时/取消停止本地 reads，不自动释放 bank/MR。
  Completion requires the existing PREPARED/PARKED/APPLIED/RESUMED protocol;
  timeout/cancellation stops reads but is not a memory-completion fence.
- `model_forward`：先申请 runtime permit，再进入 CUDA consumer 的整个 forward
  租约。scope 退出完成设备排空后，runtime 再判断是否接受结果；**只在成功退出后
  提交 token**。UNKNOWN 保留 permit，close 不能强制回收。
  Acquire a runtime permit before the consumer scope; accept output only after
  successful scope exit and runtime validation. UNKNOWN keeps the permit.

目前 model scope 每次绑定一个请求；多请求 wait-all 批次、真实 TP rank transport
与生产调度工厂尚未接入。原 CPU group 继续执行 CPU-only 检查；CUDA group 不是
CPUInstallGroup 的子类，不借用 CPU 生命周期承诺。

The model scope currently binds one request per forward. Multi-request wait-all
batch assembly, real model-TP transport and production factories remain work.
CUDA groups do not inherit CPUInstallGroup or weaken its CPU-only checks.

## 验证 / Validation

CPU 策略测试使用真实 tensor/协议/数学、替换 CUDA placement 与 fence，检查两轮
安装、边界等待、精确 receipt、runtime permit、取消、超时及 UNKNOWN 保留。
这不是 CUDA 执行证据。

CPU tests exercise real protocol/math with explicit placement/fence substitutes.
They establish policies, not actual CUDA execution.

严格 GPU 模型检查已改用该 runtime，不再在脚本中手工发送 INSTALL/RESUME：
The strict GPU model smoke now uses the runtime instead of manually sending
installation commands:

```bash
PYTHONPATH=python python test/registered/disaggregation/run_pvd_cuda_model_smoke.py --dtype float16
```

本地无 CUDA，检查只能返回 blocked；不证明 GPU/RDMA/CAGRA/服务性能。
Locally it remains blocked and certifies none of GPU/RDMA/CAGRA/serving performance.
