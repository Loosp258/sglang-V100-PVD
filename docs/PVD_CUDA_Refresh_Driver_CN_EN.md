# CUDA 同线程刷新驱动 / CUDA owner-thread refresh driver

## 行为 / Behavior

`CUDARefreshDriver` 将同步调度器的逐轮 `poll()` 接到现有
`CUDAPrefetchRequest` 异步流程。同步调用者使用私有 asyncio loop，每次 poll
只推进一个不等待网络的 iteration；异步调用者借用原 loop，不创建后台 GPU 线程。
probe、query copy、Delivery stage 和安装都留在 owner thread。probe 自身仍是
同步计算，不能将“不等待网络”说成“不占用计算时间”。

The driver connects synchronous owner polling to the existing asynchronous CUDA
request controller. A private event loop advances one nonblocking iteration per
poll, or an async caller lends its existing loop. No background GPU thread is
created. Probe, query copy, Delivery staging and installation stay on the owner
thread. A synchronous probe still consumes compute time; this is not proof of
latency hiding or compute overlap.

每个 request 的正式时钟来自 `len(Req.output_ids) - 1`，P 的首 token 不计入 D
生成数。驱动从不写入 token，不运行第二个 sampler。保留上次看到的 prefix
仅用于验证 append-only：身份、Prompt、slot、已有输出被修改均停止该请求。
正式 Req 使用的 `array("q")`、以及 list/tuple token 序列均有界读取。

The authoritative clock is `len(Req.output_ids) - 1`, excluding P's first token.
The driver never writes tokens or resamples. Its previous-prefix copy verifies
append-only observation, not a second output ledger. Identity, Prompt, slot or
existing-output changes stop the request. Actual Req `array("q")` fields and
list/tuple inputs are read with explicit prefix bounds.

只在完整初始 Prompt 安装完成且有 P 首 token 后注册；新请求不修改旧请求的
task、deadline 或刷新时钟。每轮最多排队一个 probe，排队前取得共享 arbiter
许可，进入 capture 前重验正式 prefix。捕获后释放许可，HTTP 等待不占用目标
模型。每请求只有一轮在途刷新；提前就绪等边界安装，迟到等原请求，未提前启动
则使用边界处的正式前缀 Q，绝不为该补查调用 draft。

Admission follows complete initial Prompt installation and P's first token. New
requests do not reset existing tasks, deadlines or clocks. At most one capture
is queued per poll, reserving the shared target arbiter before dispatch and
revalidating the authoritative prefix before capture. HTTP waits hold no target
lease. One refresh is in flight per request: early readiness waits for its
boundary, late work waits for the original result, and a missed window uses
committed-prefix Q without running draft.

## 所有权与接入义务 / Ownership and integration obligations

- 所有 CUDA batch execution 必须共用此 arbiter 和 pipeline target RLock。
  poll 只能在同步 forward/结果处理之外调用，不支持 overlap serving。
- request 数和 prefix 长度显式有界。超时/取消停止新操作，但不证明远端 WRITE
  完成；只有 controller 的 `aclose()` 真正成功后才释放 registration 容量。
- 清理失败保留 request/controller/task/异常；不自动重试 UNKNOWN，也不允许
  `close_loop()` 宣称完成。probe/copy/provider UNKNOWN 还保留共享 arbiter 许可：
  单独保留 RLock 不够，因为 owner thread 可以再次进入同一可重入锁。
- 驱动不会归还原 Req/KV allocator rows，不会关闭调用者提供的 HTTP client。
  调用者仍须按各自的完成证明退还这些资源，并保证 loop 使用身份一致。

All target batches must use the same arbiter and pipeline RLock. Poll only
between synchronous forwards and result processing; overlap serving is not
supported. Request count and prefix length are bounded. Timeout/cancellation
does not establish remote WRITE completion. Registration capacity is returned
only after successful controller close. Failed cleanup retains actual owners
and refuses automatic retry/loop closure. Unknown probe/copy/provider completion
also retains the shared arbiter: an RLock alone cannot exclude its own thread.
Original Req/KV allocator rows and caller-owned HTTP clients remain the caller's
responsibility; this driver does not free or close them.

## 验证与尚缺 / Evidence and remaining work

17 个新用例：Windows 16 passed / 1 skipped；WSL 含真实 Req 的组合测试
33 passed（3 条已有 CPU 平台警告）。覆盖两轮真实 HTTP 检索/交付、独立请求时钟、
迟到等待、边界补查、prefix 变更、错误线程、排队取消、超时和 UNKNOWN 所有权。
计算和 payload 使用 CPU/fake 替代，测试不能证明 CUDA/RDMA。

Seventeen new cases: Windows 16 passed / 1 skipped; the focused WSL combination
passes 33 cases, including a real Req, with three existing CPU-platform warnings.
The tests exercise real localhost HTTP with CPU placement and fake payloads,
not CUDA/RDMA. The true serving startup factory, waiting-queue admission hook,
authoritative batch-result binding, original allocator retirement and actual
distributed model TP remain integration tasks. This driver does not silently
activate a new serving mode or replace those obligations.

最终 Windows 全量 / Final Windows full regression: **2168 passed / 25 skipped**.
The real Req import is skipped on Windows and passes in WSL; it is not a
hardware test and is not counted as verified in the Windows run.
