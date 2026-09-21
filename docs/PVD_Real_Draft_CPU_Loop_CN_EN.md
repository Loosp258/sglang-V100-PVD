# 独立真实 Draft 的 CPU 闭环 / Independent real draft CPU loop

后续于 ScheduleBatch 结果接点 `bef8a59c4`，本阶段将闭环内固定候选替换为真实
独立 SGLang ModelRunner：target 是 2 层/hidden=32 的随机 Llama，draft 是
1 层/hidden=16 的随机 Llama。共享 64-token toy tokenizer，只为执行验证，
不是生产模型选择。draft 仍然只预测，正式输出全部由 target 与原 Req 结果处理器提交。

The controlled loop now accepts an injected real DraftProvider. The strict
`--real-draft-loop` fixture loads two independent random tiny Llamas locally,
using a shared toy tokenizer. Actual SGLang draft prefill/continuation produces
the candidates, followed by target post-RoPE Q, local V HTTP search, sparse KV
installation, real batched target Decode and the normal Req result processor.
No download or production-model selection is performed.

## 修复 / Fix

原私有池验证未沿真实 allocator 的 `_kvcache` 找到 K/V：两个不同 allocator
共享同一底层 KV 可能漏检。现在遍历 backing cache，拒绝 K 或 V 的 storage alias，
有循环的未知 wrapper 会终止并报告未验证。4 个新增回归测试。

Private-pool verification now follows allocator backing caches. Distinct
allocator objects over aliased K or V fail; genuine separate storage is verified.
It never treats the absence of inspectable backing storage as proof of privacy.

## 验证 / Evidence

完整 CPU suite：Windows 1364 passed / 11 skipped；WSL 1369 passed / 6 skipped。
新增测试/fixture Ruff 通过；`draft_sglang.py` 保留既有旧 typing lint，不重写无关代码。

```bash
PYTHONPATH=python python test/registered/disaggregation/run_pvd_draft_cpu_smoke.py \
  --real-draft-loop
```

- Draft 真正运行 2 次 forward（prefix + continuation），生成 `(13, 13)`。
- old 请求预取使用这些预测；边界首次补查使用实际前缀 Q，不调用 draft；
  新请求加入不触发自身或旧请求的额外预测。
- 每个 draft branch 验证 target 权重、KV、req map、pool capacity、CPU RNG 不变；
  draft 私有 pool capacity 恢复、scratch 归零、无 quarantine。
- 21 次真实 attention 对照，最大误差约 `3.58e-7`。请求加入/重排/撤回、wait-all、
  实际前向失败、Req 输出上限均通过，D token 计数 old=9/new=2/third=0/length-limit=1。
- Draft retained tensor footprint 为 39,712 bytes；persistent charge 在测试进程寿命内
  保留，非“加载前 admission”或整个运行时峰值内存测量。初始化另一个 ModelRunner
  会设置全局 server args，fixture 显式恢复 target 配置并隔离初始化 RNG。

No quality, GPU/TP/RDMA, CAGRA or latency evidence. The fixture measures retained
tensors after construction; it does not prove loading-time budget enforcement.
The independent worker is assembled explicitly in the standalone test, not yet
automatically from production Scheduler flags. Real-model vocab compatibility,
model/backend compatibility and loading/global-state ownership still need a
production integration gate. Prefix recomputation is a correctness baseline,
not a performance claim. Next: automatic request-local CPU refresh progression.
