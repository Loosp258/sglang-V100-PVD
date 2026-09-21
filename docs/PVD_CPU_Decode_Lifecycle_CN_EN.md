# CPU Decode 生命周期接点 / CPU Decode lifecycle seam

2026-09-21。上一阶段真实检索消费已提交为 `8f482e631`，未 push。
本文新增生命周期代码、测试及 smoke 接线尚未提交。

## 本轮实现 / Implementation

`cpu_decode_lifecycle.py` 提供 owner-thread-only 的 `CPUDecodeLifecycle` 与
`TargetExecutionArbiter`，目前驱动离线真实 CPU 模型验证，不直接操作服务 Scheduler。

| 接口 / API | 明确契约 / Contract |
|---|---|
| 构造 / construct | 不可变 Prompt、P 首 token、request ID、显式共享 arbiter；初态 waiting |
| `admit(controller)` | 所有 CPU bank 完整 Prompt 已安装到边界 0，身份/长度匹配；同一 controller 不可重复占用 |
| `snapshot()` | 前向之间从已提交输出构造不可变前缀；P 首 token 不算 D tick |
| `launch_refresh` | 提前窗口内或恰在边界发起；排队前占目标执行权，防止 probe 启动前 D 推进计数 |
| capture scope | 覆盖同步 draft/probe；结束即释放执行权，HTTP 等待不占目标模型 |
| `begin_decode()` | 已准入、非边界阻塞、无其他目标执行后发唯一 `DecodePermit` |
| `complete_decode` | 仅实际在途票据可提交一个正式 token；重复、复制、旧实例票据拒绝 |
| `try_install` | rank 计数必须匹配本请求真实 committed count，且恰在原边界；不能伪造未来计数 |
| `terminate` / `fail_decode` | EOS、取消、超时、失败终止实例；失败 token 不提交、不自动重试 |
| `close()` | Decode 完成/报告失败、HTTP task drain 后，才关闭 CPU bank |

`finished=True` 是调用方已判定 EOS/长度限制的通知，不是本模块猜测特殊 token。
EOS 最后一个成功 token 正常提交，然后停止刷新；取消/超时后迟到的输出丢弃。
终止通知不能归还仍执行中的 target lease，也不能证明 RDMA/GPU 已结束。

All requests using the same target execution context must share one arbiter.
The offline probe's process-global ForwardContext forbids simultaneous model
executions. Queued probes hold a lease until synchronous capture finishes;
HTTP search may then overlap Decode. Cancel-before-dispatch retains ownership
until the task is observed done/drained, not merely until cancel is requested.

Refresh timeout is explicit per launch, uses a monotonic clock, covers dispatch
through installation, and is checked by owner-thread `poll()`. It is not a
background timer or preemptive kernel cancellation. Decode eligibility,
completion and installation poll automatically; an idle driver must keep polling.
Task failure becomes terminal status with a reason, still requiring close.

## 验证 / Evidence

真实 CPU 模型脚本现在走 `admit → begin_decode → forward → complete_decode →
snapshot/refresh/install`。9 个 D token，4/8 边界安装；18 次 attention 对照最大误差
`3.5762786865234375e-7`，新增 `lifecycle_dispatch_commit_install: true`。
提前查询迟到等待、正式前缀补查、生成 KV 保留、Prompt 池毒化和失败不提交 token
均通过。25 个新增契约测试覆盖身份、互斥、异线程、超时、EOS/取消/drain、新请求
不影响旧请求；模型数值来自独立严格 smoke，不是 mock 单测。

```bash
PYTHONPATH=python /home/loosp/torch311-env/bin/python \
  test/registered/disaggregation/run_pvd_draft_cpu_smoke.py \
  --probe --search --sparse-decode --controlled-decode
```

Full regression: Windows **1292 passed / 11 skipped**, WSL **1297 passed /
6 skipped**. Skips are not hardware evidence. Changed Python files pass Ruff
check/format check; `git diff --check` passes.

## 已核对的服务接点 / Inspected production touchpoints

- `disaggregation/decode.py::_pvd_enter_waiting_queue` / `process_decode_queue`：
  保留初始完整 Prompt 交付、安装、ACK 后才准入。
- `managers/scheduler.py::update_running_batch`：当前刷新在 `batch.prepare_for_decode`
  前，即下一生成 token 分配之前。
- `managers/scheduler.py::run_batch`：未来绑定整个 batch 的执行权和读者。
- `managers/scheduler_components/batch_result_processor.py::process_batch_result_decode`：
  普通 Decode 在此追加 `req.output_ids` 并调用 `req.update_finish_state`；必须与该
  唯一提交点对齐，不能再向 Req 重复追加。
- `managers/scheduler.py::abort_request` 和 retraction：失效旧实例和结果，同时等待
  执行/远端写入各自的完成证据，不能发 cancel 就释放 MR。

These serving files were inspected, not modified. Existing full-Prompt serving
is unchanged. The CPU output ledger belongs to the isolated driver, not a second
production output authority. No production flag/backend registration was added.

## 下一步 / Next

batch 级 dispatch 契约：一次 batch 共享一个执行 lease，以请求 incarnation/唯一
票据匹配各结果，处理成员变化、EOS、retraction 和部分失败。当前每个 permit 独占
target lease，不能给 batch 每个成员逐一调用 `begin_decode` 冒充 continuous batching。
先实现 batch 契约和测试，再接正式服务路径。

Still absent: real ScheduleBatch wiring, TP collective ordering, GPU sparse
attention, authorized asynchronous sparse Mooncake delivery, GPU/MR fences,
production workspace/host-ledger budgets, CAGRA, real draft/model quality and
latency tests. Tiny random Llama, fixed draft candidates and local copies are
fixtures, not user model choices or proof of network-latency hiding.
