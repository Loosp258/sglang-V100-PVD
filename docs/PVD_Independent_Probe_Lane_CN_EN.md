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
合计 **46 passed**；Ruff E/F/I 和格式通过。服务端记录已验证
ticket nonce 至其 deadline，拒绝重复提交，记录数设上限且过期回收；
容量满时明确拒绝，绝不二次运行预测。使用独立
`spawn` 进程验证了真实 `SO_PEERCRED` 往返，以及 sidecar 重启
后旧 PID 客户端拒绝新进程；不是只在同进程模拟。客户端在读取回复前
必须以显式 `TransferBudget` 预留 host Q 字节和一个并发槽，
`async with request(...)` 的作用域结束时归还，包括超时与调用方
异常；调用方必须在作用域内转换/消费 Q，不能在退出后继续持有。
这还不是 GPU0 模型显存预算，也不能代替服务端队列预算。PID 绑定是
本机进程身份校验，不是远端认证；目前仍未启动 GPU0 模型进程，
也未把 D Scheduler 请求送入此通道。

检索接入检查点 / Search-entry checkpoint: `ProbeSearchSession.prepare_from_lane`
现在可在回复预算作用域内接收经过 Unix 协议验证的 CPU Q，重验
window、deadline、Entry、模型空间、RoPE、layer/Q-head/GQA 和维度，
然后进入原有 `search()` / `take_selection()` 逻辑检索 V。使用假
sidecar 与真实 V HTTP/exact index 的测试确认本地 draft/target probe
没有被调用；连同既有 probe-search 回归共 **69 passed**。这仍不包含
GPU0 模型执行、Scheduler 异步派发或与正式 Decode 的计算重叠。

服务端预算检查点 / Sidecar reply budget: Unix 服务端也要求显式
`TransferBudget`；接受 ticket 后、调用模型 handler 前按允许的最大
Q 回复预留 host 字节与一个并发槽，写完/拒绝/异常时归还。容量不足
不调用 handler，也不消耗 nonce。协议、Unix 与 V 检索相关测试合计
**50 passed**；目前仍是假模型 handler。

真实模型 handler 检查点 / Real-model handler gate: `probe_lane_model.py`
现在可由独立进程的主线程执行私有 `CUDAPredictionPipeline`，仅将 ticket
指定的 post-RoPE Q 位置/层/head 拷到 CPU，并在退出预测分支前做 CUDA
完成栅栏。2026-09-27 在新的 CloudLab D 节点 GPU0，以本地
Qwen2.5-7B-Instruct + Qwen2.5-0.5B-Instruct 运行单独的 dual-model
smoke：28 层 Q 与直接 probe 逐值一致，私有池/预算归还，峰值 allocated
约 16.33 GB。这个阶段尚未经 Unix socket 运行真实模型进程，更未
验证 serving、RDMA 或性能收益。

跨进程真实模型检查点 / Real-model Unix gate: 同一 D GPU0 的独立
model-owner 进程加载 7B+0.5B，另一个 `spawn` 的 CPU-only 进程通过
受限 Unix socket 传 ticket、收 Q；全部 28 层 FP32 Q 的 SHA256 与
直接目标 probe 的 Q 逐层一致。客户端和服务端 host 回复预算归还，
目标/draft 私有池与 CUDA 预算也归还；进程退出后没有保留实验服务。
`run_pvd_qwen_dual_draft_gpu.py` 报告
`real_model_q_cross_process_unix_matched=true`。
`probe_lane_identity.py` 随后改为流式哈希本地 `config.json`、所有
`.safetensors`、可选 index 和 tokenizer 工件，拒绝链接成员或读取时
改变的文件。D 节点目标权重/配置共 15,231,300,303 字节，摘要
`5725e17b4bd31a7d1f723215ebc2402028658615a1b9fcab0d11f4445dedd5f2`；
tokenizer 摘要
`fc79977ab8ac1b4fedc4b83e5eb86414438030dd7b2c31c701310efb58f9c7b8`。
实验 ticket、server 和 handler 均使用这组真实内容摘要，哈希约
10.7 s（启动期成本，不计入推理延迟）。仍未接入 Decode Scheduler、
V 检索/RDMA 或同条件性能比较。

路由签发检查点 / Routed ticket issuance: `probe_lane_routing.py` 从 D
已验证的 V search routes 派生 ticket，而非让 sidecar 自行选 V 或让
上层手填层/head。只接受完整的 layer × 连续 Q-head 矩形覆盖，逐条
校验 Entry、目标向量空间、post-RoPE、GQA→KV-head 映射与维度，
并从实际 query positions 计算回复字节上限。纯逻辑测试通过；
Scheduler 尚未调用此函数。
