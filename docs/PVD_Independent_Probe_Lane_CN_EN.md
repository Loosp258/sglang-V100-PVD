# PVD 独立预测执行通道 / Independent prediction lane

状态：设计与接口阶段；**未接入 serving，不是性能成果**。

## 为什么需要 / Why

2026-09-27 的三机 V100S 负载表明：私有 Prompt KV 初始化后，
D 在每次刷新仍同步占用 Scheduler 约 0.44–0.46 s，其中 draft
约 0.26 s、目标模型 Q probe 约 0.16–0.17 s。输出 128 token、
M16 的七轮刷新为约 9.54–9.75 s，完整 KV 为约 6.17–6.36 s。
当前同进程 `ModelRunner`、`ForwardContext` 和 CUDA RNG/执行锁
不允许把这段工作简单扔进另一个 Python 线程。

The measured problem is serialized D compute, not just a V kernel or an
RDMA call. A separate thread over the current target runner is unsafe.

## 第一版边界 / First-version boundary

- D GPU1 保留唯一正式 Decode/Scheduler/采样输出所有权。D GPU0
  运行**独立进程**，加载相同目标模型的独立权重副本及已选的小模型，
  持有私有 KV、CUDA context、RNG 和预算。不能把正式 Req、池、
  `ModelRunner` 对象传给它。GPU0 可用性与模型容量必须启动前检查；
  不满足时保持当前同步路径。
- Sidecar 尽早对已入队且身份已确定的请求建立独立 Prompt 前缀。
  第一版可重算；后续可在验证 P2P 拷贝、布局和完成语义后使用
  D GPU1 的已接收 Prompt KV。不能假定不同 GPU 的指针或 rkey
  可以直接复用。正式前缀变更时仅增量扩展已确认 token，预测
  token 的临时 KV 在每轮末释放。
- Scheduler 在请求自己的 `boundary - lead` 时生成不可变 ticket，
  含请求 incarnation、Entry/Delivery、目标模型权重和 tokenizer
  身份、Prompt/输出 token 快照及摘要、prefix version、boundary、
  query positions、层/Q-head 范围、post-RoPE 空间、nonce、deadline
  和预算上限。新请求加入 batch 不重置旧请求的时钟。
- 同机通信优先权限受限的 Unix domain socket；对端进程启动身份
  与模型身份在连接时固定。回复只含完整、有限、CPU FP32 Q 行
  及原 ticket 身份，不含 MR 地址、rkey 或 KV 数据。D 本地逐项
  验证后才将行送入既有 V search / Delivery / install 流程。
- 一个请求至多一个待处理轮次。取消、retraction、Entry 版本变更、
  超时或 sidecar 重启会使旧回复不可安装；sidecar 资源必须完成
  CUDA 栅栏后才能复用。迟到到边界时，D 暂停并以当前正式前缀
  的目标 Q 补查，不使用预测 Q 冒充正式 Q。
- 需要独立预算：GPU0 权重与私有 KV、每请求前缀、预测 scratch、
  回复 host 字节、sidecar 请求数及 IPC 队列长度。容量不足要有
  明确 backpressure；绝不能悄悄使正式 Decode 超过预算。

## 验收门槛 / Acceptance gates

1. 确定 Qwen2.5-7B 和 draft checkpoint 的精确文件身份；启动
   实测 GPU0 显存预算与单请求 prefix 容量。不同模型或 tokenizer
   不得被一个字符串标签伪装成相同向量空间。
2. 实现并测试 ticket/reply 验证、权限受限 IPC、最大帧和超时、
   重启/取消/重复/错序拒绝，以及 close 时 CUDA 资源栅栏。
3. 在 V100S 实测独立进程 target-Q 对当前同步 target-Q 的数值
   一致性、检索选择一致性和六事实输出哈希；保留近似检索本身
   与完整 KV 输出可能不同的事实。
4. 同输入、同配置、独立暖态重复对照 20/128 token、1/2/4
   客户端，报告 request median/p95、wall、stall、GPU 利用率、
   峰值显存与 V/RDMA 时间。只有输出/安全门槛通过且端到端
   稳定超过完整 KV，才能宣称最终性能目标。

No sidecar process, IPC transport, asynchronous Scheduler handoff or
compute overlap exists merely because the protocol types are present.

当前接口检查点 / Current protocol checkpoint: `probe_lane_protocol.py`
签发有 token 摘要、模型/词表 SHA256、nonce、deadline、层/head 范围
及最大回复字节的 ticket；回复验证拒绝错序、错模型、缺层、错
post-RoPE、非有限/错 dtype/错位置的 Q，并复制为 D 自有 CPU
tensor。V100S 环境的纯协议测试 **18 passed**，Ruff E/F/I 和
格式检查通过。它尚未实现认证、IPC、模型执行或 Scheduler 接入。

线格式检查点 / Wire checkpoint: `probe_lane_wire.py` 使用标准库
JSON 元数据与定长二进制 FP32 Q 行，不新增 `msgpack` 依赖。
Ticket 帧限制 256 KiB；回复原始 Q 最多 16 MiB，并限制元数据
开销。解码拒绝重复 JSON key、非有限常量、超长/截断帧、错模型、
错 nonce、缺层和错 Q 长度；数据验证后复制到 D 所有的 tensor。
纯协议与编解码测试合计 **36 passed**，Ruff E/F/I 和格式通过。
仍无 socket 连接、认证或实际独立模型执行。

同机传输检查点 / Local transport checkpoint: `probe_lane_unix.py` 已提供
权限 0700 的专用目录、0600 socket、Linux `SO_PEERCRED` 同 UID 与
指定 PID 校验、有限帧、deadline、单 handler 串行执行、有界连接数
及按 inode 校验后的关闭清理。假预测器的 Unix 往返与协议测试
合计 **45 passed**；Ruff E/F/I 和格式通过。其中使用独立
`spawn` 进程验证了真实 `SO_PEERCRED` 往返，以及 sidecar 重启
后旧 PID 客户端拒绝新进程；不是只在同进程模拟。客户端在读取回复前
必须以显式 `TransferBudget` 预留 host Q 字节和一个并发槽，
`async with request(...)` 的作用域结束时归还，包括超时与调用方
异常；调用方必须在作用域内转换/消费 Q，不能在退出后继续持有。
这还不是 GPU0 模型显存预算，也不能代替服务端队列预算。PID 绑定是
本机进程身份校验，不是远端认证；目前仍未启动 GPU0 模型进程，
也未把 D Scheduler 请求送入此通道。
