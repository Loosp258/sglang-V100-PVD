# PVD target-Q probe: offline CPU reference / 离线 CPU 参考实现

Date / 日期：2026-09-21. Previous work was committed and pushed as
`6dba510dba4129c8c94266085366968f7811f37a` on `pvd-disaggregation`.
The changes described here are the subsequent, not-yet-committed step.

## What exists / 已实现

`pvd/target_probe.py` implements `TargetProbe` with **existing target weights**,
not draft Q, a copied target model, or a third model. The initial capability
subset is explicit: exact SGLang `LlamaForCausalLM`, CPU FP32, TP1/PP1/CP1,
non-DP `torch_native`, no quantization or native speculative generation.
This is an offline correctness reference, not a new serving option.

实现路径：

1. `branch()` 先预留私有池、Q 副本及调用者声明的临时空间预算。
2. 校验请求/前缀版本、token 范围、总长度和预测长度。
3. 为 probe 创建独立真实请求池、KV 池和 attention backend；不替换目标 runner 的池。
4. 将完整 committed prefix 与 predicted tokens 拼接，在独立位置 `0..N-1` 重算。
5. `ForwardBatch.pvd_query_capture` 是 batch 局部、默认关闭的显式收集器。
   `LlamaAttention` 在 RoPE 后、attention 前交付 Q，只复制所需预测位置/层/Q heads。
6. 复用目标 backbone 执行，不调用 LM head、sampler、输出提交或 refresh 时钟。
7. 返回带 request、prefix version、唯一捕获版本、layer、global Q head、absolute
   positions 和 `rope_applied` 协议标签的 `QueryVectors`（语义为 post-RoPE）。
   Q-head → KV-head 映射仍由现有桥接负责。推进 Step 1 时修复了旧 `post_rope` 字面标签
   与 V 协议不一致的实际阻断；未放宽 V 的严格校验。
8. 前向结束释放私有行和请求槽；branch 结束清理收集器并退预算。若释放失败，保留
   私有状态及预算、隔离该 probe 并拒绝复用。不要通过重试强行退回隔离预算。

There are no runtime model hooks and no mutable capture state on shared model
layers. Ordinary target forwards have no sink. Draft forwards explicitly refuse
a nonempty sink. Test-only hooks provide an independent numerical oracle.

## Verification / 验证

```bash
PYTHONPATH="$PWD/python" python test/registered/disaggregation/run_pvd_draft_cpu_smoke.py --probe
```

The script remains offline and generates only a deterministic tiny random
Llama fixture. It does not select the user's experiment model. The environment
is the one recorded in [the CPU execution report](PVD_Draft_CPU_Execution_CN_EN.md).
Failures are nonzero exits, not skips or fallback to doubles.

本轮真实模型结果：

- 两层 Q 与普通目标前向的 RoPE 输出逐元素一致：最大绝对误差 `0.0`。
- 明确检查结果不同于 pre-RoPE Q；位置为 `(4, 5)`，选取 global Q heads `[1, 3)`。
- 原目标权重、整个目标 KV 池、请求映射、空闲行/槽和 CPU RNG 保持不变。
- 复用目标权重对象，没有构建第二个目标 ModelRunner。
- 前向中途异常后恢复原 `ForwardContext`、退还预算，后续 probe 可以重用。
- 释放失败时保留预算和私有池引用，并拒绝下一轮。
- 请求身份/版本不匹配、越界 token、预测过长、分支外调用和预算不足均拒绝。
- WSL 完整回归：**1120 passed / 6 skipped**。其中 15 个新增纯 CPU 收集器测试；
  draft sink 拒绝测试也随新字段增加一个用例。真实 forward 另由严格 smoke 执行。
- Windows 最小依赖环境：**1115 passed / 11 skipped**；额外五个 skip 是该环境
  无法导入真实 serving 类的检查，不能当作通过。新增文件 Ruff 检查通过。

Test-oracle correction: `get_rope` can share one module between layers. An
initial oracle registered one hook per layer, overwriting both layer entries on
each invocation. It was fixed to register once per distinct module and collect
actual invocation order, asserting the number of calls. The production capture
never relied on hooks and was not changed to accommodate that wrong oracle.

## Limits that must not be hidden / 不得省略的限制

- **NOT wired into the Scheduler or live Decode.** No new request/batch refresh
  policy, initial KV delivery, output-token ownership or per-request M clock changes.
- `ForwardContext` is process-global, not thread-local. The adapter refuses
  non-main-thread calls and nested branches and requires a quiescent target.
  **This is not proof that another thread cannot concurrently use the target.**
  Serving integration must first establish a common execution boundary; a lock
  used only by the probe is insufficient.
- It refuses an active piecewise compilation context. No CUDA graph, CUDA RNG,
  GPU/TP>1, V100S, RDMA, overlap or production checkpoint evidence exists.
- Full-prefix recomputation is a correctness baseline, not the desired fast
  pipeline. No latency or O(prefix) feasibility claim is made.
- Target identity is an explicit caller/deployment binding, not a weight hash.
  Production must bind it to the same Entry/model epoch used by V and forbid
  concurrent weight replacement; declaring two equal strings proves no such binding.
- The budget accounts for KV/mapping/Q bytes plus caller-declared temporary
  headroom. It is not a PyTorch hard allocation cap or a measured peak bound.
  Query tensor lifetime is branch-scoped: materialize search inputs before scope
  exit; retaining arbitrary tensor references outside the scope is not accounted.
- No CAGRA, query-quality/recall claim, sparse transfer/install, GPU attention
  consumption, or draft/probe/V end-to-end serving path is added.

## Next / 下一步

Keep this numerical reference. Add a real-query → exact V search integration
test with explicit identity, head mapping and logical token/page IDs. Separately
design the D execution boundary/private GPU state before enabling an online
target probe. Do not infer live concurrency safety from these CPU tests.

继续保留可自定义 draft 模型；此 Llama 小模型只是回归夹具。不得开启原生 speculative
generation，也不得让预测 token 变成最终输出。新请求不重置旧请求时钟、不取消旧预取。
