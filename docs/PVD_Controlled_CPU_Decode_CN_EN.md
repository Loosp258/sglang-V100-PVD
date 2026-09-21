# 同一真实 CPU Decode 序列的检索消费 / Retrieval consumed by one real CPU Decode sequence

2026-09-21。上一阶段已提交为 `3c4b0c479`（未推送）；本文所述后续修改尚未提交。

## 完成了什么 / What now runs

新增 `CPUInstalledPromptView`：在同一进程内，把 `CPUInstallGroup` 的各分片
工作集交给 TP1 模型消费。所有读取必须经过 group 的边界门控，整个模型前向期间
持有每个 bank 的 reader；不能在两层之间切换工作集。

- 共同 request/incarnation/Entry/layout、Prompt 长度、head_dim 必须匹配。
- layer/KV-head 归属不能重叠，消费者要求覆盖模型需要的全部 layer/head。
- 每个分片必须属于同一 operation/目标边界，并与协调器已安装边界一致。
- 模型绝对 query position 必须等于 Prompt 长度 + 已提交 D-token 计数。
  P 的首 token 不计入刷新时钟；第一次 Decode 输入这个首 token，计数为 0。
- 边界尚未安装、取消或旧计数均拒绝进入模型前向，不能降级读旧 bank。

This is a scoped **in-process CPU read view**, not AllGather, a new wire protocol,
or real distributed TP. Shards must have compatible metadata and disjoint
layer/KV-head ownership. The current CPU model consumer still requires all heads
locally and TP1. Reads hold every bank until the entire forward exits.

## 真实模型验收 / Real-model acceptance

新增 `pvd_controlled_decode_smoke.py`，通过现有严格验证入口启用：

```bash
PYTHONPATH=python /home/loosp/torch311-env/bin/python \
  test/registered/disaggregation/run_pvd_draft_cpu_smoke.py \
  --probe --search --sparse-decode --controlled-decode
```

同一次请求中：

1. 真实 tiny random Llama 计算 Prompt K/V 和第一个输出 token，初始化完整 Prompt bank。
2. 原 Prompt 池写入 NaN、原 Prompt 映射写为 -1，防止消费者暗中回退到完整池。
3. 每次只将目标模型 greedy logits 的 token 追加到正式输出；snapshot 使用这些实际
   输出，不再使用固定的正式前缀 fixture token。
4. n=3 时从不可变实际前缀运行独立目标 probe，真实本地 HTTP 查询两个 V shard；
   独立 draft 候选仍是固定测试 token，不是正式小模型输出。
5. 故意延迟 shard 1 回复。D 使用旧工作集执行 n=3→4；到 n=4 后，实际消费者
   binding 被拒绝，不能发起下一前向。释放回复、全分片安装确认后继续生成。
6. 第二轮故意漏掉提前预取：n=8 暂停，直接从正式前缀捕获目标 Q 补查，**不调用
   draft**。返回 KV 安装到原边界 8 后，再执行下一 Decode。
7. 逐层显式 softmax oracle 使用选中 Prompt KV + 常驻 D 的生成 KV 验证 attention。
   记录每次消费的 operation/边界，确认 n<4、4≤n<8、n≥8 分别使用正确一代 KV。
8. 另跑一个请求，在 n=5 的实际 attention 写入之后注入失败：本次 token 不提交，
   controller 终止，不重试部分执行；释放 reader、请求行和 KV 页。

Observed strict smoke: **9 committed D tokens**, installation boundaries **[4, 8]**,
**18** attention comparisons, maximum attention absolute error
**3.5762786865234375e-7**. Both real-output snapshots match the actual generated
sequence, only one draft invocation occurs, and at least one refreshed workset
is a strict Prompt subset. Previous generated KV remains unchanged. Failed
forward output is not committed; pool capacity and scoped budgets recover.

本轮还新增 16 个 CPU 契约测试，覆盖所有分片 reader 作用域、边界/过界/取消/旧计数、
混合代际、不同元数据、后分片读取失败清理、位置与计数不一致，以及前向 reader
未退出时所有分片均禁止安装。实际模型测试不放在 mock 单测里冒充通过。

完整回归：Windows **1267 passed / 11 skipped**，WSL **1272 passed / 6 skipped**。
本轮修改 Python 文件 Ruff check/format check 与 `git diff --check` 均通过。
跳过项不是硬件验收证据。

## 尚未证明 / Limits

- 离线 CPU/FP32/TorchNative/TP1 Llama；两个 V shard 是本地 HTTP 服务，非两个 GPU。
- P→V 和 V→D 的 KV 字节是本地复制，没有 RDMA、GPU 可见性或 MR fence 证据。
- 不同于上一阶段，现在**同一个 Decode 序列**确实消费检索结果；但驱动仍在独立
  验证脚本内，没有接入服务 Scheduler 或 continuous batching。
- 固定长度 greedy 测试，不验证用户采样策略、EOS 处理或真实任务输出质量。
- 人工延迟 HTTP 的一小段等待与 CPU Decode 交叠，不等于证明真实网络被隐藏，
  没有任何 TPOT/吞吐/显存节省结论。原完整 Prompt 页仍保留分配以供毒化检查。
- probe 仍全前缀重算、同步且需要目标执行静止；没有和目标计算并发运行。
- 没有加载真实独立 draft，也没有 CAGRA。attention 临时 workspace 尚无生产预算。
- 前向失败没有事务回滚；之前层可能已写入当前 token KV，必须终止并回收请求。

## 下一步 / Next gate

进入 Scheduler 接线前，先定义并测试 Scheduler 线程独占的 dispatch/完成/安装/abort
生命周期接口：真实 committed count、不可变前缀、目标执行互斥、waiting/running
状态、EOS/cancel/超时、batch 加入不影响旧请求。之后再为受支持的模型/后端接实际
服务路径，并用真实硬件验收 TP collective、稀疏授权交付和 GPU fence。

Do not simply register this CPU reference as a production backend. Preserve
initial full-Prompt bootstrap, independent clocks, approximate prediction,
all-rank installation, boundary committed-Q fallback and generated-KV residency.
