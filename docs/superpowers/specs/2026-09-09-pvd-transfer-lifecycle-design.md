# PVD 传输生命周期与安全回收设计

日期：2026-09-09。状态：用户已确认采用首版方案；本文不是已实现功能说明。

基线：`44ce4198908828b64be1bc460a590e38cf1f0471`。保留已实现的 fresh metadata 策略；本设计解决另一条独立风险链：业务失败后，原生传输尚未结束，源内存或目标内存却被释放或复用。

## 1. 已验证事实及证据边界

- 当前四组 PVD CPU 测试共 86 项通过。它们不证明真实 RDMA 超时后的安全性。
- 当前 Mooncake adapter 使用同步 WRITE，把负返回值记录为 FAILED；`abort()` 只修改 Python 状态，没有原生 drain。
- 当前 V 的 `fence_delivery()` 可以在 Delivery 业务失败后返回 `fenced=true`，没有证明原生传输已经结束。
- 在真实 VectorKVStore 上替换传输调用边界的 CPU 故障探针中，模拟“超时返回失败，但写入仍在途”：active delivery 计数变为 0、fence 返回 true，而模拟在途数仍为 1。该探针证明协议对这种返回语义缺少保护，不证明实验现场的错误一定由此造成。
- 风险包括 P 发送 staging、V 重排 staging、V Entry 分配的页，以及 D 接收 staging。V pool MR 即使一直注册，提前归还 Entry 页也可能导致静默数据覆盖，而不出现 rkey 错误。

原生依据固定为 Mooncake 源码提交 `719735896c86b56fabec6cf3e825fb2ea640597a`（本项目核验的 `0.3.13.post1` 源码），不使用随分支变化的链接。安装包版本检查不能替代对实际二进制构建来源的核验。

| 核验位置 | 结论 |
| --- | --- |
| [Python 绑定 transferSync](https://github.com/kvcache-ai/Mooncake/blob/719735896c86b56fabec6cf3e825fb2ea640597a/mooncake-integration/transfer_engine/transfer_engine_py.cpp#L479) | 等待达到墙钟期限后可直接返回 -1，没有向 Python 返回原生批次句柄或排空证明。 |
| [transferSubmitWrite / transferCheckStatus](https://github.com/kvcache-ai/Mooncake/blob/719735896c86b56fabec6cf3e825fb2ea640597a/mooncake-integration/transfer_engine/transfer_engine_py.cpp#L806) | 已有异步提交与轮询接口。提交失败返回 0；轮询 1/-1 为完成/失败并释放批次，-2 为超时且不释放，0 为未完成。不能重复轮询已释放的句柄。 |
| [RdmaTransport 提交](https://github.com/kvcache-ai/Mooncake/blob/719735896c86b56fabec6cf3e825fb2ea640597a/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp#L822) | 可达到水位后先提交部分 slices，再继续处理；后续失败不等于之前没有提交。 |
| [RdmaTransport 状态](https://github.com/kvcache-ai/Mooncake/blob/719735896c86b56fabec6cf3e825fb2ea640597a/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp#L1023) | 成功与失败 slice 数之和等于总数时才报告任务终态；否则为 WAITING。 |
| [MultiTransport::freeBatchID](https://github.com/kvcache-ai/Mooncake/blob/719735896c86b56fabec6cf3e825fb2ea640597a/mooncake-transfer-engine/src/multi_transport.cpp#L108) | 未结束的任务返回 BatchBusy，不释放批次。Python 提交失败路径不检查该释放结果，仍返回 0，调用者可能失去跟踪句柄。 |

上述结论依据代码控制流。真实驱动、QP 故障、GPU 可见性与内存回收顺序仍需硬件验收，不宣称已完成底层 RDMA 正确性证明。

## 2. 方案选择与范围

推荐：使用现有异步提交/轮询接口，保留原生句柄；把业务结果和传输安全状态分开；对无法证明终态的异常保留资源并熔断。

替代方案一：保留同步接口，负返回一律隔离。改动较少，但丢失句柄后不能自动回收，恢复能力较差。

替代方案二：修改 Mooncake 绑定，为部分提交失败返回可排空句柄及明确的未提交状态。恢复语义更完整，但需要额外构建、分发和核验原生包；本轮不修改外部 Mooncake 仓库，也不自动升级依赖。

本轮保持 Router 选择 P/V/D、完整 Prompt KV、每 M token 刷新、D 生成 KV 常驻、Entry/EntryShard/Delivery 分离，以及现有 TP/rail 配置。不得顺带实现检索、隐藏网络延迟的流水线或扩大硬件支持范围。底层改为异步不意味着允许 D 用未完成的 KV 做 forward；现有 Decode 刷新屏障保持。

## 3. 安全不变量

1. 描述符交给发送方之前，接收方先固定目标分配及注册资源；发送方提交前固定源张量及注册资源。
2. 超时、取消、HTTP 断连、TTL 到期和 consumer lease 消失，都不证明传输已停止。
3. 只有明确从未进入原生提交，或持有句柄并观察到原生终态，才能减少 transport pin；网络失联不是终态。
4. 接收内存还需得到发送方的终态确认，并阻止该传输身份再次提交，才能复用。单有接收端 `cuda.synchronize()` 不证明远端不会继续写。
5. MR unregister、张量引用释放、allocator 归还页、staging 内容重写，都受同一资源生命周期约束。
6. 业务状态终结后，transport 记录必须存活到安全回收；active delivery 业务计数不能充当资源 pin 计数。
7. 已被判定业务失败的请求，之后即使原生 WRITE 成功，也只触发回收，不重新提交 KV 或恢复该请求。

## 4. 状态与单一所有者

TransferHandle 保留业务结果，新增独立的传输状态与原生 ID。状态约定如下：

| 情况 | 传输状态 | 回收 |
| --- | --- | --- |
| 本地校验/额度拒绝，尚未调用原生提交 | NOT_SUBMITTED | 可以回收本次尚未对外授权的资源。已发出的目标授权另需关闭。 |
| 提交取得非零句柄 | IN_FLIGHT | 不可以；继续轮询。 |
| 业务超时、取消或原生轮询 -2 | DRAINING | 不可以；业务可失败，轮询继续。 |
| 原生轮询 1 或 -1 | TERMINAL_SUCCESS / TERMINAL_FAILED | 发送侧可解除本次传输 pin；接收侧经匹配 fence 确认后解除。 |
| 原生提交返回 0、进入原生调用后异常、无法再可靠查询句柄 | UNKNOWN | 不可以；隔离相关资源并熔断新传输。 |

`0` 的含义依接口区分：提交返回 0 是不明提交失败，轮询返回 0 是仍在途。不得以同一分支处理。原生检查会在终态释放批次：每个句柄只允许一个轮询所有者，缓存终态供其他调用读取，不再触碰已释放的原生 ID。

每个共享原生 engine 对应一个生命周期管理器，负责句柄、注册引用、poll 驱动和额度。`from_existing()` 生成的多个 adapter 必须共享该管理器，不能各自维护互不知晓的生命周期。后台 poll 不持有 VectorKVStore/Coordinator 的业务锁等待网络；锁只保护短状态变更。

`release_memory()` 改为回收申请：有 pin 则延迟，无 pin 才 unregister；native unregister 失败时保留注册记录和张量所有权并报错，不能先从 registry 删除。成功 unregister 后才解除最终引用。

## 5. 两段传输协议

### P → V

V 在返回接收描述符前创建上传授权和目标 allocation pin。授权身份包括 P/V 进程 incarnation、Entry key、shard、upload ID 和 region/allocation generation。

P 提交前 pin staging；成功或失败终态后报告上传结束。V 校验身份、关闭该授权后才允许取消/TTL 回收目标页。成功提交 Entry 仍要求全部 shard 完整到达；失败终态只允许回收，不能发布 STORED。

V 取消上传时先阻止新授权，并向原 P 查询/关闭已有授权；P 对同一身份建立禁止再提交的 tombstone，等已有句柄终态后回复。没有匹配答复，V 保留目标页。P 重启或不可达不等于已排空。上传结束通知丢失时允许幂等查询与重试，但不能仅靠租约过期强制释放。

### V → D

D 在提供接收描述符前 pin staging；每次 refresh 使用独立 retrieval/delivery ID 和目标 generation。V 提交前 pin Entry 页，异构 TP 的重排 staging 还需单独 pin。

`fence_retrieval()` 先关闭该 retrieval 的提交入口，再检查所有涉及的 delivery/shard 原生终态。只有全部安全时才回复 `fenced=true`；否则回复 pending 并由后台 poll 推进，UNKNOWN 明确报告隔离原因。HTTP 超时不转为成功。

fence 回复必须匹配协议版本、发送方进程 incarnation、retrieval ID、目标 region/generation，以及预期 shard 集合。D 不能接受仅有 `fenced=true` 的旧版本回复。迟到的 reserve/start/重复请求必须被已关闭的授权拒绝。

Entry cancel、TTL reaper、consumer release 只能标记逻辑删除；allocator 在上传 pin 和全部 delivery pin 都清零后归还页。一个 Delivery 的失败不能释放其他 Delivery 仍使用的 Entry。

## 6. 容量、清理与重启

采用预先额度控制，而不是隔离后才检查上限。每个 GPU worker 的 staging 字节预算与在途传输条数预算分别限制 P/V/D；V Entry 目标页还受现有 pool 容量约束。活动、待排空和 UNKNOWN 隔离资源都计入预算，不能通过换状态绕开计费。

建议新增 `--pvd-transfer-staging-budget-bytes` 和 `--pvd-transfer-max-inflight`，本轮采用显式正整数配置，不猜测适合所有 GPU 的默认显存大小。P/D 参数与 V 启动参数使用相同语义。先预留预算，再分配张量/发布描述符；复用同一 staging 不重复计费，只有真正回收后才归还预算。超出预算时拒绝新的传输准备并返回可识别的容量错误，不建立无界等待队列。

TP rank 必须先汇总额度预留/资源准备结果；任一 rank 失败则整组停止本次提交，释放未发布资源，已发布授权按 fence 流程关闭。不得让部分 rank 开始 forward、其他 rank 等待清理。

UNKNOWN 立即让所属 engine 停止新 PVD 传输。已有可追踪传输继续 drain；隔离字节受之前预留预算限制。后台回收任务按有界资源记录驱动，不为每个超时无限新增重试任务。

停止服务先停止 admission，再尝试 drain；超过关闭等待时间必须明确报告未排空数量/字节，不宣称安全回收。进程重启、旧 incarnation 丢失或通信中断时，不自动确认旧授权安全，不自动复用旧槽位。首版无法恢复跟踪的 UNKNOWN 需要操作者停掉相关发送者并对相关 P/V/D worker 做协调重启；仅重启 D 或缩短 TTL 不是恢复方案。自动跨崩溃恢复不在本轮范围。

## 7. 协议兼容与可观测性

P/V/D 启动和建立上传/下载授权时要求双方具备 `pvd_transfer_lifecycle_v1` 能力，并携带进程 incarnation。缺失能力的旧节点拒绝参与新生命周期协议，不静默退回旧 fence。原有 PD 路径不要求此能力。

保留 fresh metadata 的版本锁定与启动前配置；另检查原生异步 submit/check API 存在，不能通过能力检测就解除版本锁。协议身份只用于 PVD 授权和回收，不声称它能刷新 Mooncake rkey。

日志关联 Entry/upload/retrieval/delivery ID、rank、region generation、native handle、业务结果、transport 状态、pin 变化和 fence 原因。健康检查区分 ready 与 draining/isolated，并暴露 in-flight、draining、unknown 数量、staging 当前/峰值/预算和拒绝计数。不得将 native handle 或地址当作跨重启稳定身份。

## 8. 每一步的验证门槛

以下是后续实现的验收顺序，不代表已经执行完毕。每一步先让故障测试暴露旧行为，再实现并运行相应测试；未通过不进入下一步。

1. **传输状态与资源 pin**：可控延迟 fake；超时/abort 后源和目标仍保留；晚成功/晚失败后恰好释放一次；重复释放、unregister 失败保持所有权。
2. **Mooncake adapter**：模拟原生 0/非零 submit、poll 0/-2/1/-1、抛异常；区分返回语义；终态只检查一次；共享 adapter 不重复 poll/unregister；UNKNOWN 不释放、不继续 admission。
3. **V 生命周期**：同构 Entry 页、异构重排 staging、并发 Delivery、cancel/TTL/close；在途时 allocator 不能重用相关页；故障探针中的提前 fence 必须变成 pending。
4. **P 上传与 D 接收**：上传失败/延迟提交/确认丢失、取消与提交竞态、错误 incarnation/region、旧协议 fence；任何一段未排空都不回收目标，不发布 KV_READY 或执行 forward。
5. **有界故障和 TP 一致性**：连续超时达到额度后内存不再增长，新准备被拒绝；单 rank 准备失败时所有 rank 退出本轮；poll/reaper/fence 并发不死锁。
6. **完整 CPU 回归**：现有 86 项以及新增测试全部通过；检查 diff、格式和语法；不修改普通 PD 行为。
7. **真实机器验收**：先正常单请求，再顺序地址复用、并发 refresh、HTTP 取消、链路异常/超时和恢复。链路故障注入仅在用户指定实验环境获准后执行。保存 P/V/D 对齐日志、内存曲线与生成结果；核验 fence 前没有回收、终态后回收、隔离达到预算后停止增长。CPU 通过不能替代此项。

## 9. 自审结论

安全范围覆盖 P→V 与 V→D 两段，而非只修改 D 的 unregister。传输成功不是 CUDA 后续使用完成，D 仍须遵守现有回填/forward 同步规则。业务释放与 transport pin 分离，也不意味着无限保留：容量在授权之前预留，无法排空时停止新工作。

首版明确接受的可用性代价：原生提交失败丢失句柄时，不能在不修改 Mooncake 绑定的情况下可靠自动回收；采用隔离和协调重启。文档审阅通过后再编写实现计划、修改生产代码。
