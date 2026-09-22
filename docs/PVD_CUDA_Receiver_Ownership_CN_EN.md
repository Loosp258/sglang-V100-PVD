# 接收 session 的刷新所有权交接 / Full-receiver refresh ownership

## 接管条件 / Preconditions

`CUDARefreshDriver.claim_received_session(session)` 是单向、逐请求交接。
顺序必须为：完整接收完成凭据 → `install_received` → 创建 controller →
driver.register → CUDARequestRelease → claim。检查同一个 Req、导入凭据、
runtime group、原始池 owner 和 target 锁；初始 Prompt 已安装且 D 正式 token 数
仍为 0。没有设置新请求或旧请求的时钟，也不产生/接受 draft 输出 token。

Claim is a one-way, per-request transfer. Complete the real receive, run
install_received, construct the controller, register it, attach CUDARequestRelease,
then claim. The Req, import receipt, runtime group, original pool owner and target
lock must match. Full Prompt must be installed and D's committed count still zero.
Claim neither resets any request's clock nor produces/accepts draft output tokens.

## 运行与结束 / Running and retirement

- 已接管 session 的 due 为 false，prepare 直接拒绝；原 refresher 也不再处理其
  租约错误。CUDA driver 在读取正式 Req/启动 probe 前检查原 session 的身份、
  租约和关闭状态，错误交给同一个 controller 处理。
- 原 session 保持 V consumer lease 和初始接收 MR；不能为了切换而提前 close。
- 原 cleanup_finished/release_request 只发出 driver cancellation intent，不 pop
  session 或停 keepalive。多个结束通知保持幂等。
- driver 先 await 稀疏 controller.aclose，再把 session.close 提交到其原 control
  loop 并等待 concurrent future。不能跨 event loop 直接 await keepalive Task。
- full session 的完成结果必须为 True，接收 guard/WRITE pin 已释放；然后才允许
  原始请求池回收，最后删除 manager session 和 driver registration。

Claimed sessions are excluded from legacy full refresh, including lease-error
handling. The CUDA driver checks source identity, lease and liveness before
observing Req/probing. The original session keeps its consumer lease and receive
MR. Ordinary finish/abort callbacks cancel the driver without removing the session
or stopping keepalive. The driver drains the sparse controller first, schedules
source close on its original control loop and awaits that concurrent future.
Only a true drained result with no receive guard/WRITE pin permits original pool
retirement, followed by registry removal.

## 失败边界 / Failure boundary

稀疏 close、原 session close 或原池回收失败不返还 admission，不自动重试释放。
保留 controller/session/池 owner；共享 arbiter 保持占用（或保留已存在的 capture
lease），两个真实池 poison，拒绝新的分配和 driver 工作。这是 worker 隔离，
不是网络超时即可取消 WRITE 的假设，也不提供“再试同步即可恢复”的入口。

Failed sparse/source close or original pool retirement does not refund admission
or retry release. Real owners remain retained. The shared arbiter is held (or an
existing capture lease stays retained), both real pools are poisoned and new
driver work is refused. This is worker quarantine, not an assumption that a
timeout cancels a WRITE, and not an automatic recovery-by-resynchronization path.

## 验证与未完成项 / Evidence and remaining work

12 个新增 CPU 策略用例覆盖接管前置条件、重复接管、旧刷新抑制、lease 丢失、
原 finished/abort 入口、两阶段异步关闭、registry 替换及 UNKNOWN 隔离。测试实际
运行独立 control loop 和 PVDDecodeSession.close；模型执行/稀疏关闭完成信号、
MR 和最终 allocator return 是明确的 double，不声称设备验证。
WSL driver/接收/真实池定向 66 passed（3 条已有 CPU 平台警告）。

Twelve CPU policy cases cover claim validation/replay, legacy-refresh exclusion,
lease loss, original finish/abort hooks, two-stage asynchronous close, registry
replacement and quarantine. They run a real control loop and source-session close;
model execution/sparse completion, MR and final allocator return are explicit
doubles. Focused WSL driver/receiver/real-pool tests: 66 passed, three existing
CPU-platform warnings. This is not GPU or RDMA evidence.

本步骤接通了明确绑定请求的所有权切换及原结束入口；**尚未提供生产启动工厂或
waiting-queue 自动装配**。默认服务仍保持完整 Prompt 刷新；不能仅传 draft 参数
便认为此入口已启用。真实模型 TP、原生 CAGRA 和 GPU/RDMA 验收仍未完成。

This connects explicitly bound requests to ownership transfer and original finish
hooks. It does **not** provide a serving startup factory or automatic waiting-queue
assembly. Default serving remains full-Prompt refresh; draft flags alone do not
activate this path. Actual model TP, native CAGRA and GPU/RDMA acceptance remain.
