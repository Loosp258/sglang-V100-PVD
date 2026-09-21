# CAGRA 验收边界 / Acceptance gate

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
