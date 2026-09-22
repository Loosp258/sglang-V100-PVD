# CUDA 目标 Q probe / CUDA target-Q probe

`CUDALlamaTargetProbe` 复用现有 Llama 权重和 batch-local post-RoPE capture，使用
独立 CUDA Req/KV pool、原始绝对位置和完整正式前缀重算。独立 draft 的 token 只追加
在该 probe 分支；不会进入正式 Req 输出，也不启用原生 speculative generation。
错过预取窗口时，`capture_committed` 只使用真实正式前缀的指定位置，不调用 draft。

The CUDA probe reuses target weights but allocates private request/KV pools and
captures target post-RoPE Q at original positions. Predicted tokens exist only in
that branch; committed-prefix fallback does not append predictions. No sampler,
LM-head output commit or mutation of the target's live request pool is introduced.

## 支持边界 / Supported subset

- exact `LlamaForCausalLM`、TP1/PP1、non-DP/non-CP、非量化、`torch_native`，
  目标权重必须全部位于显式 CUDA device，dtype 为统一 FP16 或 FP32。
- 模型位置、词表和层/head 范围依现有真实配置检查，模型路径不被固定。
- 主线程执行；调用方必须给出和正式目标 forward **同一个** execution lock。
  忙时拒绝执行，不等待死锁；锁本身不是任意外部 target 正在停机的证明。

The caller must wire the same lock into every target forward. Merely constructing
a lock for the probe does not establish quiescence. This is an explicit component,
not a production Scheduler factory or a claim of real model TP support.

## 生命周期 / Lifetime

共享 probe 核心现在在首个 pool 构造前发布 owner，覆盖“构造了一部分才抛异常”和
request slot 分配失败。forward 后先排空设备再 clear/free，清理可能产生 GPU 操作，
因此清理后再排空一次。已完成异常帧中的 tensor 引用也在退预算前清理。

Owners are published before constructors allocate. Both normal and failed forwards
drain before allocator cleanup, then drain cleanup work before dropping state.
Finished exception-frame tensor references are cleared before refund. CPU weakref
tests observe actual tensor release rather than checking counters alone.

任意 CUDA 完成未知或清理失败都会保留私有 state、预算和 target execution lock，
拒绝复用该 probe。没有 force-free，不会因 Python 异常就假设 GPU 已经停止。
Quarantine deliberately blocks target execution until external recovery/restart;
an exception is not a CUDA fence. Cancellation does not return a possibly-live pool.

预算中的 pool/query 上限按 FP32 计算，FP16 不会据此少计；`transient_bytes_bound`
仍是调用方对激活/原生 workspace 的声明，**不是已经测得或强制限制的峰值**。
每次重算完整前缀且同步设备，尚未证明开销能放进预取窗口，也未实现 target/probe
计算重叠。这些性能和完整峰值显存约束不能由 CPU 测试替代。

## 可执行验收 / Executable acceptance

```bash
PYTHONPATH=python python test/registered/disaggregation/run_pvd_cuda_probe_smoke.py --dtype float16
# Also verify the explicitly supported FP32 path if required:
PYTHONPATH=python python test/registered/disaggregation/run_pvd_cuda_probe_smoke.py --dtype float32
```

独立 Linux CUDA 进程，随机 tiny Llama，不下载 checkpoint。脚本使用真实 ModelRunner、
pool 和 NCCL TP1，对照普通目标 forward 的 RoPE hook oracle，并检查目标权重、KV、
映射和 CUDA RNG 不变。hook 仅用于测试 oracle，probe 实现不装 hook。
失败为非零退出，无设备为 blocked，不把它标成 skip/pass。

The strict smoke owns a standalone process. A future pass would validate only this
random-model/device/dtype case, not production checkpoints, sparse Decode, CAGRA,
RDMA or latency. It has **not** run a CUDA model forward locally.

本步 CPU 策略测试使用明确 doubles；真实 CPU 四场景矩阵在共享核心修改后再次
全部通过，完整场景 attention 最大误差约 3.58e-7。该 CPU 模型证据不改名为 GPU。
The real CPU four-case matrix passed after the shared-core refactor; these results
remain CPU/fake-transport evidence. The CUDA smoke currently reports blocked.
