# 受控请求刷新闭环 / Controlled request refresh loop

2026-09-21，基线 `9c09b7768`，本轮与前一轮安装协议修改尚未提交。

## 本轮完成 / Implemented

`cpu_prefetch_request.py` 的 `CPUPrefetchRequest` 把以下环节接成一个**离线 CPU**
请求闭环，不是生产 Scheduler 接线：

1. 初始完整 Prompt 已经由 `CPUInstallGroup` 安装；不依赖检索或索引就绪。
2. 一个请求控制端发出 `InstallEpoch`，统一 request/incarnation/Entry/operation/
   round/目标边界；取正式前缀的不可变快照。
3. 一次 draft + target probe 产生所有 shard 的查询。`ProbeSearchSession` 从创建
   window 时就绑定该 epoch，不能在搜索完成后修改身份来拼接结果。
4. `fork_prepared` 在搜索前分割实际捕获的查询行，每行恰好属于一个 shard。
   子 session 共享同一不可变 window，各自钉住对应 shard 的 index/mapping 版本。
5. 并发本地 HTTP 搜索；同层同 KV head 的 Q heads 结果取并集、去重、校验显式上限。
6. `pack_source(rank, specs)` 独立验证源 Entry/index/mapping，作用域内复制 K/V；
   各 CPU bank stage 到同一个 epoch。返回 ready 不代表已经安装或推进时钟。
7. 到边界后，全部 rank prepared + parked + applied，才推进该请求时钟并允许读取。
   失败或取消关闭该请求实例；不存在自动回滚、自动换 rank 或旧结果重新放行。

One controller owns the request identity and epoch **before** query generation.
The prepared query rows are partitioned before HTTP search; replies are never
relabelled into a common epoch. Each shard pins its own index/mapping versions.
The pack callback must independently check those against current source state
and own/pin the source throughout its scope. This callback is not an RDMA grant
or a production source-lease implementation. Initial full-Prompt bootstrap stays
separate from periodic retrieval.

## 已确认的迟到处理 / Confirmed boundary policy

- 已提前发起，但结果迟到：D 在原边界等待原来的查询，不重新预测，不移动边界。
- 错过提前窗口，到边界才首次发起：D 暂停，使用**当前正式前缀的目标模型 Q**补查；
  不调用 draft，不把实际 token 伪装为预测 token。仍等待所有 rank 安装确认。
- 已越过尚未安装的边界：拒绝；不能跳过这一轮或挪到下一边界掩盖错误。

`refresh` 按 committed D-token count 与边界的关系选择来源；调用方必须显式提供
绝对 `query_positions`：提前路径在预测 continuation 内，边界补查在正式前缀内。
计数不含 P 的首 token，不能当作绝对位置。测试选择正式前缀最后一个 token 的 Q；
API 允许显式有界的前缀位置，但不隐式决定多位置合并策略。

新增 `TargetProbe.capture_committed` 和 `PredictionPipeline.committed_query_branch`。
不支持此能力的 probe 明确拒绝，不偷偷降级回 draft。当前真实实现仍为私有 CPU 池中
完整前缀重算的 exact Llama/FP32/TP1/TorchNative 子集；Q 复制数量受原显式预算约束。
这得到实际 token 序列上的目标模型 Q，**不是**复用在线稀疏 attention 已算出的 Q，
也未证明与在线稀疏历史下的 Q 数值相同。全前缀重算有计算成本，不是低延迟承诺。

If prefetch was already launched, wait for that operation at the unchanged
boundary. If first launched at the boundary, use target-model Q recomputed from
the actual committed prefix, with no draft invocation. Positions must explicitly
lie within that prefix. A count past an uninstalled boundary is an error.
This offline implementation recomputes full-context Q; it does not claim to reuse
or numerically equal Q from a live sparse-attention Decode trajectory.

## 生命周期与验证 / Lifecycle and evidence

- 同请求至多一轮在途；新请求初始安装/关闭不改变旧请求的时钟或查询。
- 正常 committed 进展不取消近似预测。前缀或 Entry 替换取消整个 controller，
  重建新实例，旧 HTTP 回复不能安装。不能沿用旧 incarnation 重试。
- 一个 shard 失败时取消并等待兄弟 HTTP task 退出；这不是原生传输的 fence。
- index 重建后的旧查询版本在源打包前拒绝；部分已 stage 不允许继续 Decode。
- 单进程 CPU reader/安装门控已验证；真实稀疏模型 attention gate 仍是另一个 fixture。

严格 smoke 已同时验证提前路径和正式前缀补查：真实 tiny random Llama 的 Prompt K/Q，
两个真实本地 V HTTP shard，四组 layer/KV-head 工作集，同一 operation 安装到边界 4。
补查无 draft 调用；补查 Q 与普通目标前向独立 RoPE hook oracle 最大误差 **0**。
既有稀疏 Decode 的 full-selection logits 误差 **0**，独立 attention 最大误差
`1.1920928955078125e-7`。测试采用固定 draft 候选/正式前缀 fixture token，
不是运行中的 D 输出，也不是任务质量或近似检索召回证据。

```bash
PYTHONPATH=python /home/loosp/torch311-env/bin/python \
  test/registered/disaggregation/run_pvd_draft_cpu_smoke.py \
  --probe --search --sparse-decode
```

## 尚未完成 / Still missing

本轮完整回归：Windows **1251 passed / 11 skipped**；WSL **1256 passed / 6 skipped**。
跳过项不作为 GPU 或原生运行时验收证据。
其余修改 Python 文件 Ruff 检查通过；`prediction.py` 保留基线已有的 21 条
旧式 typing 注解提示（UP006/UP035/UP045），已对照 HEAD，未新增该类问题。
排除这三类基线提示后该文件检查通过；格式检查与 `git diff --check` 通过。

1. 将本闭环选出的工作集交给**同一个持续生成 token 的真实 CPU D**消费，
   用真实输出构造下一次不可变前缀；目前检索/安装和真实 sparse Decode 仍分开验收。
2. Scheduler 接线、目标执行串行化/并发仲裁、真实 TP collective 顺序与安装门控。
3. 生产源 lease、异步稀疏授权传输、GPU 可见性/MR fence 和预算、故障恢复。
4. V100S CAGRA、真实模型/draft 的选择、GPU/RDMA/质量/延迟隐藏实验。

No production flags/backend registration were added. The original full-Prompt
serving path is unchanged. The controller requires a positive configured lead
window, has at most 64 aggregate routes and 64 positions per query, runs CPU
prediction/probing synchronously and uses an in-process bank group. Two V shards
are local HTTP fixtures, not two GPU workers. GPU fences, distributed TP safety,
KV memory savings and network-latency hiding are **not established**.

Next: join the controlled search/install loop to an actual CPU Decode sequence
with group-gated reads and real committed output snapshots. Preserve generated
KV, request-local clocks, initial full Prompt and the boundary fallback above.
Then address online/GPU/RDMA gates; do not turn this reference into production
by merely registering its class or enabling a CLI flag.
