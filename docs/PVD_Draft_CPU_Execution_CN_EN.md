# PVD draft: real CPU execution / 真实 CPU 执行记录

Date / 日期: 2026-09-21. Base HEAD: `3b1b4e75b`; changes remain uncommitted.

## Result / 结果

The prediction-only adapter now runs a **real SGLang ModelRunner**, real
`ForwardBatch`, `ReqToTokenPool`, `TokenToKVPoolAllocator` and
`TorchNativeAttnBackend` on CPU. This is not a recording double or a replacement
model forward. TP1 uses real Gloo groups, normally initialized by D's target.

现在已通过真实模型前向，而不只是构造真实类：离线随机初始化的两层 Llama，
hidden=32、Q heads=4、KV heads=2、head_dim=8、vocab=64、FP32、page_size=1。
它只是数值回归夹具，**没有替用户选定实验 target/draft 模型**，也不下载权重。

Strict smoke result / 严格检查结果：

- 33 real forwards; six incremental-decode versus full-prefix logit comparisons.
- Maximum absolute error: `8.344650268554688e-7` (atol `2e-5`, rtol `2e-4`).
- Deliberately corrupting the prefix KV map changes logits; minimum maximum
  logit difference across canaries: `0.9022974967956543`.
- Real `SGLangDraftHandle` greedy continuations agree with full recomputation.
- Successful generation, early release before generation, repeated release and
  subsequent reuse restore pool capacity and clear the request's mapping.
- WSL PVD suite: **1104 passed / 6 skipped**. Hardware skips remain skips.
- Windows minimal environment: **1099 passed / 11 skipped**; five additional
  skips are real-class imports unavailable in that environment.

反向检查会覆盖当前 token 的深层 KV，所以恢复映射后也必须重新计算当前 token，
再做后续对比。默认 dummy loader 权重较小，不足以作为可靠的故障敏感性检查；
此脚本给测试模型设置了固定种子的较大随机权重与单位 norm 权重，并显式断言故障影响。

## Reproduce / 复现

From the repository root in a serving-capable Linux environment:

```bash
PYTHONPATH="$PWD/python" python test/registered/disaggregation/run_pvd_draft_cpu_smoke.py
python -c 'import glob,runpy,sys; sys.argv=["run_pvd_cpu_tests.py",*sorted(glob.glob("test/registered/disaggregation/test_pvd*.py")),"-q","--tb=short","-p","no:cacheprovider"]; runpy.run_path("test/registered/disaggregation/run_pvd_cpu_tests.py",run_name="__main__")'
```

The smoke selects `SGLANG_USE_CPU_ENGINE=1` before SGLang imports and forces
`HF_HUB_OFFLINE=1`. Missing serving dependencies or runtime errors cause a
nonzero exit, not a skip. Run it as a separate process, not inside a live D
server: it initializes distributed/global runtime state and test weights.

本地验证环境为 WSL、Python 3.11、torch `2.14.0+cpu`、transformers `5.8.1`。
沿用已有 serving 依赖环境，本轮补充 `msgspec==0.21.1`。
这份版本记录是复现证据，不是要求正式 draft 模型固定成某个版本。
Registry 报出的其他模型可选依赖缺失警告不代表本次 Llama 路径失败；严格脚本的
退出码与 JSON 结果才是本次检查结果。首次全量测试还出现 pytest cache 写权限警告，
上面的命令关闭 cacheprovider，不改变测试内容。

New smoke/admission files pass the repository's full Ruff check; touched draft
files pass the focused correctness rules `E9,F63,F7,F82` and formatting check.
The broad Ruff invocation still reports pre-existing style issues in the older
draft files (for example typing modernization); this is not a claim of a
repository-wide clean lint run. `git diff --check` passes.

## Defects corrected / 修复

1. ModelRunner's SM70 backend selection queried CUDA even for `device=cpu`.
   Guarded by device and uses the runner's GPU id for actual CUDA queries.
2. Stale native speculative draft path/revision could override the PVD model
   when `is_draft_worker=True`; clear those overrides and draft quantization in
   the private config. The target config remains unchanged. Native speculative
   generation remains prohibited.
3. `scratch_bytes()` counted KV only but claimed logits/workspace coverage.
   The executor must now declare `transient_bytes(prefix_tokens, predict_tokens)`;
   unknown/invalid values reject admission. The adapter has an explicit
   `transient_bytes_bound` with **no guessed default**.
4. Source-reading tests now use UTF-8 explicitly, fixing three Windows GBK
   decoding failures without changing the source contracts being tested.

## Memory contract and limits / 预算约束与限制

Private KV pools are persistent physical allocations; per-branch KV bytes are
**capacity credits**, not additional allocated physical bytes. Non-KV transient
reservations must cover inputs, simultaneous logits, activations and backend
workspace across admitted shapes. Do not sum the KV credits and persistent KV
pool into a claimed physical peak.

非 KV 上限必须来自部署后端与允许形状的可靠分析/测量；本轮没有给生产环境填入
未经验证的数值。预算是准入核算，不是拦截 PyTorch 的硬内存上限。
独立 smoke 直接驱动 adapter/handle，不经过 provider 预算准入，因而**不能证明**
provider 的生产峰值预算已验证。已有 persistent 预算逻辑发生在 worker 创建后，
也不能当成权重加载前的硬配额保护。这仍是正式服务接线前的待办。

## Still missing / 尚未完成

- Actual chosen checkpoint and `TpModelWorker` construction path validation;
  this smoke constructs `ModelRunner` directly and no tokenizer is involved.
- CUDA/V100S kernels, GPU memory safety/peaks, RDMA and Mooncake execution.
- Real target-model post-RoPE Q capture in an isolated probe branch.
- Scheduler-driven prediction/probe/search, CAGRA, sparse transfer/install and
  attention consumption; the existing initial/full-KV delivery is unchanged.
- Measured prefix recompute cost, quality/recall, compute overlap and whether
  retrieval/transfer fits within the lookahead window.
- Sampling/RNG isolation beyond existing limited tests, TP>1, paged draft
  allocation, or concurrent forwards. No such support is inferred here.

Next / 下一步：继续独立 target-Q probe 的真实模型接口和私有状态隔离验证；
同时为用户将来提供的 checkpoint/device 准备严格验证入口。不能把此次小模型
CPU 通过，描述成 PVD 预测检索流水线已端到端上线。
