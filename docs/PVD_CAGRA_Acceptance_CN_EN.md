# CAGRA 验收边界 / Acceptance gate

## 2026-09-23 V100S 候选环境实测 / Candidate V100S execution

在 node-1 的独立 `pvd-cagra25-venv` 中安装了 `cuvs-cu12==25.2.0`
（实际导入版本字符串 `25.02.00`）、`libcuvs-cu12==25.2.1`、
`rmm-cu12==25.2.0` 和 `cupy-cuda12x==13.3.0`。原 Conda 环境和运行中的
V worker 未修改。仓库 `check_cagra.py` 在 V100S/SM70、CUDA runtime 12.6
上完成真实 4096×128 index build 和 32-query Top-10 search；合成 recall@10
为 0.953125，分数最大绝对误差约 6.4e-6。

新增的 `run_pvd_cagra_backend_gpu.py` 还实际执行了**本项目**的
`CagraIndexBackend`：原生 RMM 限额/共享注册表探针、IVF-PQ build、16-query
search、分数核对和 dispose 全部成功。512 MiB 每索引原生上限的本次合成
配置通过，64/128/256 MiB 上限均在真实 RMM 限额处拒绝构建；这些数值不是
对不同数据规模的通用上限。仅加载模块或官方平台兼容表均不能替代这些实测。

The isolated cuVS 25.02 candidate on node-1 V100S completed a real 4096×128
CAGRA build and 32-query Top-10 search (synthetic recall@10 0.953125). PVD's
own `CagraIndexBackend` also passed its native RMM bridge/cap probe, IVF-PQ
build, 16-query search, score check and dispose with a 512 MiB per-index cap.
The same synthetic build was refused at 64, 128 and 256 MiB by the native
limiter. This is an artifact-specific execution result, not a general memory
bound, model-query recall, production serving or performance acceptance.

当前实现为每个 `(layer, KV head)` 索引在**整个生命周期**保留完整 native cap。
Qwen2.5-7B TP2 的每个 V rank 有 28×2=56 个此类索引；若都用 512 MiB，
保留预算就是 28 GiB/rank，尚未计入 KV pool、向量副本和其他显存。
因此不能因为单图通过就直接在 32 GiB V100S 上打开全部生产索引；需要
新的跨索引共享/峰值预算设计，或实测更小的安全 cap，并通过真实 Prompt/Q
与并发 Entry 验收。The current whole-lifetime per-index cap would reserve
28 GiB per rank for 56 Qwen2.5-7B TP2 graphs at 512 MiB each, before the
KV pool and other users. Do not enable all graphs from this synthetic result.

## Compatibility is artifact-specific / 兼容性不能凭标签判断

The current [cuVS installation guide](https://docs.nvidia.com/cuvs/installation)
lists Ampere or newer for current source builds. The broader
[RAPIDS platform table](https://docs.nvidia.com/datascience/platform-support/)
still lists Volta for CUDA 12 combinations. These different scopes are not
proof that any particular cuVS wheel or source revision runs on V100S. Record
the actual artifact/revision, CUDA and GPU, then execute build/search on V100S.
Do not silently upgrade the environment, switch GPUs or replace CAGRA.

官方通用平台表与 cuVS 自身安装页的范围不同。不能从 RAPIDS/Volta 标签、导入成功
或版本字符串推导具体 cuVS 包适配 V100S；也不能据此断言全部历史版本都不支持。
具体软件组合必须实测。默认 inventory 仅收集环境；没有 GPU 时仍可推进通用代码。

## Score verification / 分数数值验收

The [cuVS 25.06 CAGRA Python implementation](https://github.com/rapidsai/cuvs/blob/branch-25.06/python/cuvs/cuvs/neighbors/cagra/cagra.pyx)
defines inner-product and squared-Euclidean metrics and returns scores together
with neighbor IDs. The PVD probe now compares each returned score against its
actual selected row under the requested metric, using an independent float64
CPU calculation. Higher dot products and lower squared distances must not be
confused. This inspection is an API reference, not a required-version choice.

Previously finite but incorrect scores could pass if neighbor recall was good.
Now wrong signs, wrong metrics, ID/score mispairing, duplicate/out-of-range IDs,
nonfinite scores and low recall fail. Exact kth-score ties are interchangeable
for recall, without widening the cutoff using numerical tolerance.

原先只检查有限分数与邻居召回，会漏掉正确 ID 搭配错误分数的问题。现在增加独立
CPU 数值验证；分数容差为 rtol/atol 各 1e-3，仅用于数值检查，不扩大 recall 的
Top-K cutoff。新增 10 个 CPU oracle 测试通过；这不代表实际执行了 cuVS kernel。

```bash
# No GPU import/build/search, no dependency changes
python scripts/pvd/check_cagra.py --mode inventory
# Only on the candidate experimental environment
python scripts/pvd/check_cagra.py --mode smoke --expected-gpu V100S
```

The smoke is synthetic, not real-query recall or generation quality. No CAGRA
serving backend has been enabled. Native allocation bounds (retained graph,
build/search scratch), stream completion, installed-package capability and
real-model query evaluation remain necessary before such a backend can satisfy
the existing IndexBackend lifetime/budget contract. CPU doubles cannot establish
those native properties. No GPU/RDMA/performance acceptance is claimed here.
