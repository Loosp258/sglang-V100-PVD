# CUDA 模型池与 sparse backend 接点 / Model-pool adapter

## 范围 / Scope

`cuda_model_attention.py` 将 CUDA bank/tiled attention 接到真实模型使用的
`req_to_token_pool`、`token_to_kv_pool` 和 attention backend 接口。提供显式
`make_cuda_sparse_backend` 工厂；**不自动修改生产 Scheduler 或默认 backend**。
首版支持无量化 Llama、TP1/PP1、torch_native、page size 1、FP16/FP32，禁用
CUDA graph、overlap scheduling、DP attention 和原生 speculative generation。

The explicit factory binds the CUDA bank/workspace to SGLang model-pool and
backend interfaces. It does not activate production Scheduler retrieval. Its
supported subset is unquantized TP1/PP1 Llama, torch_native, page size 1,
FP16/FP32, without graphs/overlap/DP attention/native speculation.

## 所有权与接口约束 / Ownership contract

- 每个 `CUDADecodeBinding` 给出 request slot、正式 D token 计数、participant 和
  exact exchange；只能在完整 Prompt 初始安装及 RESUMED 后执行。
  Each binding names its exact slot/count/participant/exchange. Initial full
  Prompt and all-rank RESUMED are mandatory; multi-rank model execution is refused.
- `consumer.bind(..., pool_owner=...)` 必须覆盖**整个 model forward**。Pool owner
  的值是 `CUDAModelPools`，release 必须连到真实请求槽位/KV allocator 回收；
  Scheduler 不得绕过该 owner 复用行。只保留 tensor 引用不能保护 allocator 行。
  The pool guard must control actual allocator retirement, not merely tensor
  references. This caller obligation is not established by CPU interface doubles.
- 所有正式 target forward 和 probe 使用同一个 execution lock。bind 持锁至所有
  CUDA/Prompt reader 完成及清理后；这是串行正确性基线，不是计算重叠方案。
  Target and probe share one lock; it remains held through completion/cleanup.
- 每层先验证整个 batch 的位置、shape、dtype、slot、generated row 映射及跨请求
  无重叠，再写新 token KV；Prompt 从 bank 读取，不读模型池的 Prompt 映射。
  Whole-batch validation precedes writes. Generated KV remains in the model pool;
  discontiguous rows feed fixed tiles directly, while Prompt comes from the bank.
- output 独立预算在 forward 前预留，覆盖本次所有层输出。返回 attention tensor
  是 forward scope 内借用，调用方不得持有它越过 bind；不包括模型其他 activation、
  native workspace 或 allocator cache。不是总设备显存硬上限。
  Layer outputs are borrowed within the whole-forward scope and must not escape
  it. Output reservations exclude other model activations, native workspace and
  allocator caches; this is not a whole-device memory bound.
- 异常后先排空再退租约/预算；排空或回收失败保留锁、池 lease、Prompt reader 和
  output 状态，进入 quarantine。取消发生在 forward 中也不得接受该次结果。
  Uncertain completion/retirement quarantines owners and lock; cancellation
  invalidates results. There is no force-free or automatic recovery API.

## 验证边界 / Evidence

新增 37 个 CPU interface/math/lifecycle 用例，覆盖模型池非连续行、GQA 数值对照、
整批写入前验证、全组门控、输出预算、取消、排空/回收失败与回调重入。
CUDA placement/同步在这些测试中被替换：**不证明 GPU forward 已执行**。

Thirty-seven CPU policy/math cases use real CPU tensors and explicit placement /
completion doubles. They are not a real CUDA model forward or native RDMA proof.

未来有设备后运行严格实模检查（随机 tiny Llama，不下载或锁定用户 checkpoint）：
Run the strict real-CUDA smoke when a compatible device/runtime is available:

```bash
PYTHONPATH=python python test/registered/disaggregation/run_pvd_cuda_model_smoke.py --dtype float16
PYTHONPATH=python python test/registered/disaggregation/run_pvd_cuda_model_smoke.py --dtype float32
```

它执行 5 次真实 Decode forward，初始 full Prompt 后切换到 sparse Prompt，逐层与
独立 dense SDPA 对照，并通过真实 allocator 回收槽位。无设备返回 blocked/退出码 2；
模型/依赖/断言失败返回 failed/1，不能当作跳过后通过。本地只运行到 blocked。
The smoke requires five model forwards, a full-to-sparse transition, per-layer
independent SDPA checks and actual allocator retirement. No device means blocked
(2), execution/import/assertion failures mean failed (1). Locally it is blocked.

仍未完成：生产工厂/队列装配、实际 TP2 模型、GPU 数值/资源实测、原生 CAGRA、
RDMA、多请求服务压力与性能。新增 backend 工厂不等于这些事项已完成。
Serving assembly, real model TP2, GPU validation, native CAGRA/RDMA, concurrent
service load and performance remain separate implementation/acceptance tasks.
