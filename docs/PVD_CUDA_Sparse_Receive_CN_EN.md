# CUDA 稀疏接收与安装 / CUDA sparse receive and install

## 已接通的链路 / Connected path

`CUDASparseReceiveRegistry` 复用原 HTTP Delivery 和精确 `WriteIdentity` 校验，
新增私有 CUDA destination 分配、注册前 SYNC_MEMOPS、远端成功后的 CUDA 排序、
有 guard 的 bank 复制和全组 RESUMED 后 ACK。可直接使用 Mooncake PVD adapter；
没有通过删除原 CPU 接收器的类型检查来启用 CUDA。

The explicit CUDA registry connects the existing HTTP Delivery protocol to private
device destinations and the CUDA rank participant. It accepts the Mooncake adapter
without replacing native submit/poll/fence semantics. This is not yet constructed
by the production predictive Scheduler factory; model/queue assembly remains.

## 两种完成证明不可混淆 / Two distinct ordering requirements

1. 接收区发布前，Linux CUDA driver 设置并回读 `CU_POINTER_ATTRIBUTE_SYNC_MEMOPS`。
2. V 回传的成功必须通过 Entry、worker epoch、receiver epoch、Delivery、region、
   generation、rank、manifest、确切字节数和 terminal-success/fence 校验。
3. 完成上述远端证明后，D owner CPU 线程发起该设备的 CUDA synchronization，
   随后才调用 bank copy。接收区从未暴露给并行 kernel；当前 attention 只读旧 bank。
4. bank source guard 保持 MR/接收内存；bank 拷贝排空后才撤销该 pin。新的 bank
   完成全 rank 安装、收到全部 RESUMED 后才能 ACK Delivery。

Before publication the driver enables/verifies SYNC_MEMOPS. Exact remote terminal
proof precedes CPU-initiated CUDA ordering and bank copying. A CUDA synchronization
alone never establishes completion of an outstanding NIC WRITE. The private receive
allocation is not concurrently read by a kernel. Delivery ACK additionally requires
the exact all-rank resume receipt; APPLIED alone is insufficient.

依据：[NVIDIA CUDA 11.4 GPUDirect RDMA 同步与内存排序](https://docs.nvidia.com/cuda/archive/11.4.0/gpudirect-rdma/index.html#sync-memory-ordering)，
[CUDA pointer attribute ABI](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-driver-api/group__CUDA__TYPES.html)。
These documented ordering rules guide the implementation; they do not certify this
deployment's NIC/GPU topology, peer-memory driver or actual RDMA completion.

## 回收 / Retirement

取消/超时不释放仍可能被写入的 destination。远端完成未知、注册失败或 CUDA 排序/
copy 完成未知均保留 owner、MR 与预算。注销失败可以重试；不会更换 region 后复用
旧地址来绕过错误。guard 回调先完成注销并清掉 registry 的 tensor 引用；只有 guard
自身也已清掉 RegisteredMemory 后，后续 close 才退接收预算，避免先退预算后掉引用。

Cancellation is not a fence. Unknown states retain owners and charges. Native
unregister failures remain retryable. Retirement is two-phase: unregister/drop
registry references, then observe the source guard empty before refunding capacity.
If a source pin remains, close returns false and the owner must continue progress.

## 证据范围 / Evidence scope

新增 CPU 测试使用真实 localhost HTTP、真实 CPU tensor、延迟 fake WRITE 和显式
CUDA 排序策略替身，覆盖 ACK、晚到写入、错误身份、GPU 完成未知、注销重试、退预算
顺序。ctypes ABI 测试验证高于 32 位的指针及 set/get 错误，不加载真实驱动。
另有 Linux CUDA 原生属性设置/回读测试（无设备时跳过），已加入严格组件验收。

CPU tests exercise real HTTP and ownership policy, not GPU visibility or RDMA.
The separately gated real-driver test validates SYNC_MEMOPS on CUDA storage using
a **local CUDA producer**, not a NIC. Native RDMA and target hardware validation
remain mandatory. Unsupported driver/allocator/platform behavior refuses the path
rather than falling back to CPU memory or an assumed visibility guarantee.

本步验证 / Step validation: Windows 全量 **1997 passed / 24 skipped**；WSL
接收/协议/验收入口定向 **110 passed / 1 skipped**。新增 23 个 CPU 用例；一个
新增真实 driver 用例因没有 Linux CUDA 设备而跳过。未执行原生 RDMA。
