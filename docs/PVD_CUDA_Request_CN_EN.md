# CUDA 每请求刷新控制 / CUDA per-request refresh control

`CUDAPrefetchRequest` 将 CUDA query bridge、现有逐 shard HTTP search、GQA
有界并集、CUDA sparse Delivery 和本地 TP1 runtime 接通。调用者须先独立安装
完整 Prompt；初始就绪条件不依赖检索索引。禁止在已配置的 Delivery 路径中
偷偷改用本地 `pack_source`。

The explicit CUDA controller connects query bridging, shard HTTP search, bounded
GQA union, CUDA sparse Delivery and the local TP1 runtime. Complete Prompt must
already be installed independently of retrieval indexes. A local `pack_source`
cannot replace the configured Delivery path.

控制器复用 CPU 的同一请求窗口状态机，但保留不同的类型/设备约束。每请求一轮
在途；边界前启动用预测 Q，边界上首次启动用正式前缀 Q，不运行 draft；超过
未安装边界拒绝。HTTP 返回后重新核对请求、Entry、版本和绝对 deadline，不能
因为检索已完成就推进 token 时钟。安装及全组 RESUMED 才放行，之后 ACK。

CPU and CUDA share the request-window state machine, not device assumptions.
Only one round may be in flight per request. Before-boundary starts use predicted
Q; first starts at the boundary use committed-prefix Q without draft execution;
starts beyond an uninstalled boundary fail. Request, Entry, versions and absolute
deadline are rechecked after HTTP. Search completion does not advance the clock:
installation/all-rank RESUMED gates Decode, followed by Delivery ACK.

同步/异步 close 都不能在 query copy、probe、draft 或 RNG 仍隔离时声称资源
已释放。取消不代替 native WRITE 的成功/排空证明，既有接收器继续持有 destination。

Neither synchronous nor asynchronous close reports success while query copying,
probe, draft or RNG ownership remains quarantined. Cancellation is not a native
WRITE fence; the existing receiver retains its destination until native proof.

验证用例覆盖两轮刷新、边界正式 Q 补查、初始 Prompt 门控、CPU/CUDA 类型拒绝、
检索拒绝及隔离关闭。HTTP 与 exact V store 为真实实现；模型 Q 和 CUDA tensor/
driver 使用 CPU 策略替身、payload 用受控 fake copy。测试的 bootstrap 为预建索引
下的协议 fixture，不能替代生产 index-independent 完整 Prompt 引导。

Tests cover two rounds, committed-Q fallback, initial-Prompt gating, CPU/CUDA type
separation, search refusal and quarantined close. HTTP/exact V store are real;
model Q and CUDA placement/driver use CPU substitutes and payloads use controlled
fake copies. Fixture bootstrap uses a prebuilt index for protocol setup, not the
production index-independent full-Prompt bootstrap.

尚未接入生产 Scheduler 工厂/队列，也未实现真实模型 TP、多请求 GPU batch 的
完整装配。此控制器不自动开启 `--pvd-draft-*` 服务能力，且没有 GPU/RDMA 验收。

Production Scheduler factory/queue integration, actual model TP and complete
multi-request GPU batch assembly remain separate tasks. This controller does not
activate draft flags in serving and carries no GPU/RDMA acceptance claim.
