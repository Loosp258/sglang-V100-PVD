# 请求级跨 rank 安装协议 / Request-local rank install contract

2026-09-21，基线 `9c09b7768`。本轮代码未提交、未推送。
这是第三步的 CPU 协议验证，不是已上线的 TP collective。

## 实现 / Implementation

`sparse_install.py` 新增 `RankInstallCoordinator`，每个请求实例独立持有
`PrefetchClock`。参与 rank 与各自 layout 明确传入，没有硬编码 TP2 或网卡名。

| 阶段 / Phase | 条件 / Condition | 能否继续 Decode / May resume |
|---|---|---|
| PREPARING | 提前加载候选工作集，各 rank 上报 prepared | 首轮不可以；周期刷新到边界前可用旧 bank |
| 已到边界 / Parked | 每个 rank 到达同一 committed-token 边界，旧 forward 读者已退出 | 不可以 |
| INSTALLING | 所有 rank 均 prepared + parked，发布安装决定 | 不可以，决定不是完成 |
| 全部 applied / Complete | 每个 rank 确认自己安装了对应候选 bank | 时钟仅推进一次，允许继续 |
| FAILED / CANCELLED | 失败、显式超时或取消 | 本请求实例永久拒绝继续，迟到消息不能重新放行 |

`InstallEpoch` 绑定 request、incarnation、Entry、随机 operation ID、round 与
target_tokens。后者是 D committed-token 计数（不含 P 的首 token），不是绝对序列位置。
`RankInstallReceipt` 还绑定 rank、layout fingerprint 与 staging ID。
重复一致消息幂等；冲突候选、未知 rank、旧 epoch/旧 ACK/旧失败通知被拒绝。
只保留在途轮次与最近完成轮次的记录，不随请求长度无限累积。

The coordinator is owner-thread-only. Worker/network callbacks must enqueue
notifications onto that thread. Its receipts are trusted participant assertions,
not proof of native RDMA completion, GPU visibility, or buffer ownership.
Failure/cancel closes the request clock but does not free or unregister memory.
Timeout policy is caller-driven (`fail`), not a timer that silently frees buffers.

## CPU 实体工作集驱动 / CPU bank driver

`CPUInstallGroup` 在**同一进程**中独占多个 CPU bank，用于验证协议与真实 tensor
生命周期能否组合，而不是模拟 TCP/Gloo/NCCL 的故障语义。

1. `stage` 验证 payload 属于当前 epoch，使用既有 bank 预算复制并生成准备收据。
2. `CPUSparseWorkingSet.install_candidate()` 返回与本次分配绑定的候选标识。
   同一 operation/边界下，释放后重新 stage 也产生不同 staging ID，旧收据不能复用。
3. `try_install` 在改变任何 rank 前检查全部候选仍匹配、计数一致且 CPU 读者退出。
4. 安装期间经过 group 的新读取被门控；仅全部安装 ACK 后提交请求时钟。
5. 某 rank 失败时，已安装的 rank 不回滚，所有 rank 的新读取均拒绝。
   显式 close 才尝试回收 CPU bank；仍有读者的 bank 保留预算，退出后可再 close。

All reads in this driver MUST go through `CPUInstallGroup.read()`. Raw bank
access bypasses the group gate and is forbidden under its exclusive-ownership
contract. The existing CPU model adapter is NOT yet bound through this driver.
There is no claim that physical swaps are simultaneous: visibility is gated
until all ACKs arrive. On partial failure the request must abort; old banks may
already be retired and rollback is neither implemented nor safe to assume.

## 验证 / Evidence

- 新增 **34** 个测试：1/2/3 个显式 rank（包括非连续 rank ID）、乱序/重复通知、
  缺 rank 等待、过早 ACK/错误边界/陈旧身份拒绝、取消及超时后拒绝迟到消息。
- CPU 双 bank 验证：旧 reader 阻止任何安装；释放 reader 后切换成功；rank 1 在
  安装前、安装后但 ACK 前失败，或安装时取消，都不能暴露 rank 0 已换好的新 bank。
- 候选替换检测在任何 rank 安装前拒绝；回收保留活跃读者的预算。
- 新请求的创建、初始安装和取消均不改变旧请求时钟、候选 staging ID 与预算。
- 完整回归：Windows **1220 passed / 11 skipped**；WSL **1225 passed / 6 skipped**。
- Ruff 检查与格式化通过。硬件相关跳过项不是验收证据。
- 再次运行严格 `--probe --search --sparse-decode` 验收通过：真实 CPU 模型 Q、HTTP
  检索和稀疏 attention 的既有证据保持不变；它们尚未与新协调器组成同一条 pipeline。

Reproduce the new tests in the existing environment:

```bash
PYTHONPATH=python /home/loosp/torch311-env/bin/python \
  test/registered/disaggregation/run_pvd_cpu_tests.py \
  test/registered/disaggregation/test_pvd_sparse_install.py -q --tb=short
```

## 下一步的实际接线缺口 / Concrete remaining integration gap

后续已完成[受控 CPU 闭环](PVD_Controlled_Request_Loop_CN_EN.md)：控制端统一身份
在捕获前注入 session，一次 probe 后分配子查询，各 shard 独立钉住 index 版本，
贯穿 union/payload/install。**没有给已返回的独立结果改 ID。**
用户确认到边界才首次启动时用正式前缀目标 Q 补查，不调用 draft；提前启动但迟到
时等待原结果。两条路径均经过真实 CPU Llama + 本地 HTTP 验证。

Next, consume the controlled results in the SAME real CPU Decode sequence;
the real sparse model adapter and the controlled search/install loop still have
separate fixtures. Then wire the serving Scheduler with explicit collective
ordering and trusted local completion/visibility evidence. Boundary misses must
wait, not silently consume old or partially installed KV. The evidence counts
above are historical for the install-contract step; see the newer note for the
combined regression results and limits.

Not done: wire serialization/authentication, actual multi-process TP agreement,
rank/coordinator crash recovery, GPU fences, sparse authorized Mooncake delivery,
source/index leases, online model execution arbitration, CAGRA, output quality,
latency or memory-savings measurement. Membership is fixed per request instance;
there is no automatic failed-rank exclusion or failover. Existing full-Prompt
serving (`decode_refresh.py`) is unchanged.
