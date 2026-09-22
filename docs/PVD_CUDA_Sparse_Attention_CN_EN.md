# CUDA 稀疏 attention 消费基线 / Sparse attention consumption baseline

## 已实现与未实现 / Implementation boundary

`cuda_sparse_attention.py` 提供独立、显式创建的 attention workspace；不注册为
SGLang serving backend，不修改默认完整 Prompt 刷新路径。支持单 token Decode、
TP1 full causal MHA/GQA/MQA、FP16/FP32 存储、FP32 累加。无 sliding window、MLA、
量化 KV、logit cap、dropout 或 native speculative generation。

This is an explicitly constructed workspace, not a registered production backend.
It consumes one-token TP1 full-causal MHA/GQA/MQA, FP16/FP32 storage with FP32
accumulation. Other attention variants are not implemented. The math and CUDA
ownership path are implemented; serving/model-pool integration is still separate.

## 数学与内存 / Math and memory

对每个 Q head，按显式 `QueryHeadMapping` 找到 KV head，分块遍历该 head 的稀疏
Prompt 和完整 generated KV，用在线 softmax 合并每块结果。不拼接整份 K/V，不生成
长度等于完整上下文的 score 数组，也不跨 head/layer 合并检索分数。
每块使用同一组 FP32 scratch，显式 tensor 总元素数为：

```text
2 * chunk_tokens * head_dim + chunk_tokens + 3 * head_dim + 6
```

Each Q head reads its mapped Prompt union and all locally generated KV through
fixed FP32 tiles and online softmax. The explicit scratch formula above is
reserved before allocation and does not grow with total context. The output is
caller-owned and is not allocated by the workspace. No RoPE is recomputed.

**预算边界**：公式只覆盖本类的显式 scratch tensor，不覆盖调用方 Q/K/V/output、
CUDA/cuBLAS context、底层库内部 workspace 或 PyTorch allocator cache；不能据此
声称已给出整个 GPU 峰值显存硬上限。输入与输出必须由调用方独立记账。
This is not a hard bound on total device memory: caller tensors, library/context
workspace and allocator caches are separate. Native workspace and production
attention budgeting still need target-stack integration and measurement.

## 调用与生命周期 / Calling contract

- 必须传入 `CUDARankInstallParticipant`，由其 read scope 确保本地安装已经 RESUME。
  上层整组调度仍须等待全部 RESUMED；局部 read scope 不证明全组可执行。
  Use the group runtime's permit as well as the participant's local read gate.
- `ResourceGuard(AttentionBuffers(...), release)` 保护 Q、generated K/V 和 output。
  release 必须绑定真实 owner 的回收，不能只用一个 no-op 假装已保护 pool 行。
  The guard owner must prevent allocator-row reuse, not merely keep Python tensor
  references. The guard covers both successful and failed execution.
- generated K/V 形状严格为 `[decode_tokens + 1, KV heads, head_dim]`，按原始绝对
  位置 `prompt_tokens ... prompt_tokens + decode_tokens` 排列，包含当前 query 的 KV。
  所有 Q/K 已 post-RoPE；本类不从 shape 推断或证明这些语义。
  The caller supplies post-RoPE Q/K and every generated position in order, including
  the current token. Shape validation does not certify semantic origin or encoding.
- output 必须独立于输入、Prompt 和 scratch 的 storage。全部元数据检查在计算前；
  不修改 Prompt 或 generated KV。失败时 output 可能部分写入，调用方不得提交 token。
  Failed compute can leave partial output: discard it rather than committing it.
- 计算结束或抛异常都同步该设备后才撤销 guard。同步未知保留 scratch 预算和全部
  guard；close 不强制回收。释放回调期间也不能重入 workspace。
  Unknown completion quarantines storage/charges. No force-free or CPU fallback.

## 验证 / Validation

27 个 CPU 数值/生命周期用例，含不允许 `torch.cat` 的对照、跨块最大值变化、
部分计算失败、容量拒绝、源释放失败与回调重入。CPU 策略 fixture 不等于 CUDA。
3 个独立真实 CUDA 用例分别覆盖 FP16/FP32 数值和两轮 bank/participant/workspace
集成；本地无 CUDA 环境跳过。尚无真实模型 forward、V100S、RDMA 或性能证据。

Twenty-seven CPU cases exercise the actual tiled math and explicit ownership
policies. Three distinct real-CUDA cases cover FP16/FP32 math and two-round bank /
participant / workspace composition; these skip locally. No real-model CUDA
forward, V100S, RDMA, production activation or performance claim follows.

本步验证 / Step regression: Windows 全量 **1957 passed / 23 skipped**；
WSL attention/bank/rank 定向 **73 passed / 6 skipped**。GPU 用例跳过不计为通过。
Skipped GPU cases are unverified, not passing GPU acceptance.

Python 按 head/tile 提交很多小 kernel，且当前同步整个设备。本实现用于建立可验证
的正确性基线，不是“网络已被隐藏”或最终高性能 kernel。后续要在验证数值与资源
生命周期的基础上接入模型池、stream/event 与更高效的 kernel。
The Python head/tile loop and whole-device synchronization are conservative and
potentially slow. Kernel fusion/overlap is future work, not an implied speedup.

```bash
PYTHONPATH=python python test/registered/disaggregation/run_pvd_cpu_tests.py \
  test/registered/disaggregation/test_pvd_cuda_sparse_attention.py -q
```
