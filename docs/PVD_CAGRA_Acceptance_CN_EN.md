# CAGRA 验收边界 / Acceptance gate

## 2026-09-24 管理器与真实原生图联合门控 / Manager/native integration gate

node-1 的 GPU 0 和 GPU 1 都在 V100S/cuVS 25.02 隔离候选环境运行
`run_pvd_cagra_shared_manager_gpu.py`：真实 Prompt KV packer 将合成
1024-token、2-layer、rank-local 1-KV-head 数据打包；`PromptIndexManager`
提取向量，为两个 Entry 共构建四个原生 CAGRA 图，并通过管理器完成一次检索。
共享 RMM 根上限 671088640 bytes、每图子上限 536870912 bytes。管理器启动时
已计费 671088640；四图存活时原生根占用 524288，预算占用 671612928
（根上限加 524288 向量副本）；关闭两个 Entry 后原生根占用归零，预算仍保留
671088640 的根预留。状态 `passed`。这证明了合成数据组件接线，不代表真实 V
worker、真实目标 Q、56 图容量、并发服务、召回或吞吐验收。

The isolated node-1 V100S/cuVS 25.02 component gate passed separately on
GPU 0 and GPU 1. It packed synthetic
1024-token Prompt KV, extracted rank-local K through `PromptIndexManager`,
built four native CAGRA graphs across two Entries, and searched through the
manager. The root cap was 671088640 bytes and each child cap 536870912.
Initial charge was exactly the root cap; four live graphs used 524288 native
root bytes, and budget charge was root plus 524288 vector-copy bytes. Closing
both Entries returned native root usage to zero while retaining the root
budget charge. Status: `passed`. No real V service, real-model Q, 56-graph
capacity, concurrent serving, recall or throughput claim follows.

```bash
PYTHONPATH=python python test/registered/disaggregation/run_pvd_cagra_shared_manager_gpu.py \
  --device 0 --expected-gpu V100S --expect-cuvs-version 25.02.00
PYTHONPATH=python python test/registered/disaggregation/run_pvd_cagra_shared_manager_gpu.py \
  --device 1 --expected-gpu V100S --expect-cuvs-version 25.02.00
```

## 2026-09-24 共享原生上限能力探针 / Shared native-cap capability gate

`CagraNativeRuntime` 新增可选父级 RMM limiter：每个图仍有
自己的 per-index limiter，但其上游可指向同一个全局 limiter。node-1 V100S、
cuVS 25.02 的隔离探针先同时构建并检索两个 1024×32 原生索引；两个子 limiter
各保留 131072 bytes，父 limiter 精确记录总计 262144 bytes。随后从两个
不同 cuVS Resources 各申请 360 MiB：第一笔通过，第二笔在共享 640 MiB
上限处被拒，尽管它未超过每图 512 MiB 上限。第一笔安全释放、两个索引
逆序销毁后父级计数回到零；门控输出 `passed`。

这只证明 cuVS C API 与嵌套 RMM resource 在这套实测环境中共享计数和限额。
随后增加的 `--prompt-index-cagra-global-native-bytes N` 已把父级 cap 纳入
`PromptIndexManager` 的一次性预算预留：`N` 至少覆盖一个图 cap，不得超过总索引
预算；每图 child limiter 保留，向量和其他内存另计。该接线的 node-2 焦点回归
**219 passed / 3 skipped**，但尚未在真实 V 服务中验证，亦不能据此断言
56 图或多个 Entry 的容量。

`CagraNativeRuntime` now has an optional
parent RMM limiter shared by the per-index child limiters. The isolated
node-1 V100S/cuVS 25.02 gate built and searched two 1024×32 indexes. The
root recorded their combined 262144 retained bytes. A 360 MiB cuVS C-API
allocation from one index succeeded; a second 360 MiB request from the other
was rejected by the shared 640 MiB root even though each child allowed up to
512 MiB. After safe release and reverse disposal, root allocation returned
to zero. The later `--prompt-index-cagra-global-native-bytes N` integration
reserves the parent cap once in `PromptIndexManager`, provided it covers a
child cap and fits the total index budget. Focused node-2 regressions passed
**219 / 3 skipped**. The combined serving path and safe 56-graph admission
remain unverified.

```bash
PYTHONPATH=python python test/registered/disaggregation/run_pvd_cagra_shared_cap_gpu.py \
  --expected-gpu V100S --expect-cuvs-version 25.02.00
```

## 2026-09-24 短 Prompt 自动回退 / Explicit short-Prompt fallback

新增显式 `--prompt-index-backend cagra-auto`，纯 `cagra` 模式保持原行为。
短 Prompt（行数不超过 `intermediate_degree`）采用同一 V GPU 上的有界精确
后端；长 Prompt 仍用原生 cuVS CAGRA。索引管理器按实际选择的后端分别预留
保留内存与 build/search scratch，native UNKNOWN 会阻止整个组合后端继续运行。
node-0 V100S 上索引/管理器相关回归 **213 passed / 3 skipped**，其中 GPU
精确回退真的在 CUDA tensor 上运行；跳过项不是此回退用例。

node-1 隔离 cuVS 25.02 候选环境的原生门控同时构建 16×32 精确 GPU 索引与
1024×32 CAGRA 索引，各检索 4 条 query、Top-4，短/长自向量命中均 4/4，
最大分数误差分别为 0 与约 3.81e-6；两个索引均正常销毁。短索引保留
2048 bytes，长索引构建后保留 131072 bytes；长索引仍需 512 MiB 原生
构建上限，不能把保留量当作峰值。没有真实模型 Q、56 图、多个 Entry、
生产 Scheduler 或端到端性能证据。

The explicit `cagra-auto` mode uses bounded exact search on the same V GPU
for rows at or below `intermediate_degree`, and native cuVS CAGRA above it;
pure `cagra` is unchanged. The manager charges each actual path separately,
and native UNKNOWN poisons the combined mode. Node-0 V100S index regressions
passed **213 / 3 skipped**, including a real CUDA exact-fallback case.
In the isolated node-1 cuVS 25.02 candidate, a 16×32 exact GPU index and a
1024×32 native index coexisted, each searched four Top-4 queries with 4/4
self-hits, then both disposed. Score errors were 0 and about 3.81e-6.
Retained bytes after build (2048 and 131072) are **not** peak-build bounds;
the native test still used a 512 MiB cap. Real-model Q, 56 graphs,
multi-Entry admission, production serving and latency remain unvalidated.

在隔离候选环境从仓库根目录复验 / Reproduce from the repository root in the
isolated candidate environment:

```bash
PYTHONPATH=python python test/registered/disaggregation/run_pvd_cagra_auto_gpu.py \
  --device 0 --expected-gpu V100S --expect-cuvs-version 25.02.00 \
  --native-cap-bytes 536870912
```

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
同一 backend 上再运行 `--index-count 2`，两个索引同时存活，各自检索 16 个
query，32/32 个自向量命中，逆序释放 2/2 个索引；这仍不是完整 56 图服务验收。
两个索引构建后各只保留 524288 bytes 的 RMM 分配，远低于本配置构建阶段
必须允许的 >256 MiB 峰值。这说明当前“每图终生保留完整 512 MiB cap”是
保守而昂贵的**预算策略**，不是该合成图的真实静态占用；不可据此直接把 cap
降到 512 KiB，因为构建会失败。

The isolated cuVS 25.02 candidate on node-1 V100S completed a real 4096×128
CAGRA build and 32-query Top-10 search (synthetic recall@10 0.953125). PVD's
own `CagraIndexBackend` also passed its native RMM bridge/cap probe, IVF-PQ
build, 16-query search, score check and dispose with a 512 MiB per-index cap.
The same synthetic build was refused at 64, 128 and 256 MiB by the native
limiter. This is an artifact-specific execution result, not a general memory
bound, model-query recall, production serving or performance acceptance.
With `--index-count 2`, both native indexes coexisted, searched independently
and disposed in reverse order (32/32 self-neighbor hits, 2/2 disposals).
Each retained 524,288 RMM bytes after build. The peak allowance required by
build is much larger than the retained graph, so a future shared transient
budget could improve capacity only after preserving concurrent build/search
admission and native-completion safety.

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
Top-K cutoff。新增 10 个 CPU oracle 测试通过；后续独立的真实 V100S 探针见本页顶部。

```bash
# No GPU import/build/search, no dependency changes
python scripts/pvd/check_cagra.py --mode inventory
# Only on the candidate experimental environment
python scripts/pvd/check_cagra.py --mode smoke --expected-gpu V100S
```

The GPU smoke is synthetic, not real-query recall or generation quality. No
CAGRA serving backend has been enabled. The bounded native adapter probe
establishes one version/device/configuration, not safe multi-Entry admission,
all 56 graphs, real-model queries, RDMA integration or performance. CPU doubles
remain useful for lifecycle edge cases but cannot establish those GPU properties.
