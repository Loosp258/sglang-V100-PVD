# V-side CAGRA backend / V 侧 CAGRA 后端

## Scope / 范围

`cagra_backend.py` implements the existing `IndexBackend` interface and is now
selectable by both V launch modes. The default remains the exact CPU backend.
This is native-call implementation plus CPU contract evidence. A later isolated
V100S candidate completed a bounded synthetic GPU adapter probe (see
[acceptance](PVD_CAGRA_Acceptance_CN_EN.md)); this is not a claim that
real-target-query retrieval quality is acceptable or that D's production
predictive Scheduler has been assembled.

`cagra_backend.py` 实现现有索引接口，两个 V 启动模式均可显式选择；默认仍为
CPU 精确检索。后续隔离候选环境已完成有界 V100S 合成探针，见
[验收记录](PVD_CAGRA_Acceptance_CN_EN.md)；这不代表真实目标 Q 召回率合格
或生产 D 预测检索 Scheduler 已启用。

## Explicit activation / 显式启用

在已验证的 V100S/cuVS 25.02 隔离环境，原生 CAGRA 必须通过仓库顶层的
`python -m pvd_cagra_server` 启动 V，而不是
`python -m sglang.srt.disaggregation.pvd.server`。前者在导入 SGLang/torch
之前加载 cuVS；后者在该候选环境可因 CUDA 动态库加载顺序报
`libcuvs_c.so` 缺失。只使用精确索引或不开索引时，继续用原 V 启动命令，
不需要安装 cuVS。此启动入口不改变 CLI 参数或传输协议。

In the validated V100S/cuVS 25.02 candidate, launch native CAGRA with
`PYTHONPATH=python python -m pvd_cagra_server` instead of
`python -m sglang.srt.disaggregation.pvd.server`. The standalone entry point
loads cuVS before SGLang/torch; the ordinary module entry point can fail to
resolve `libcuvs_c.so` in this environment because of CUDA library load
order. Exact/index-off V service keeps its original launcher and does not
require cuVS. The flags and transfer protocol are otherwise unchanged.
An installed wheel also exposes `pvd-cagra-server` with the same arguments;
cuVS remains an explicitly installed optional dependency.

Add these to the existing V command; choose budgets from the deployment's real
capacity, not from this document:

在已有 V 命令中添加以下参数；两个预算必须根据实际容量配置，不提供猜测值：

```text
--prompt-index-backend cagra
--prompt-index-vector-space <target-model-and-encoding-identity>
--prompt-index-budget-bytes <total-index-budget-per-V-rank>
--prompt-index-cagra-native-bytes <native-cap-per-layer-and-KV-head-index>
--prompt-index-cagra-global-native-bytes <optional-shared-native-cap-per-V-rank>
--prompt-index-cagra-graph-degree 64
--prompt-index-cagra-intermediate-degree 128
--prompt-index-cagra-itopk-size 512
```

对短 Prompt，可显式将 `--prompt-index-backend cagra` 换成
`--prompt-index-backend cagra-auto`。当实际索引行数 `<= intermediate_degree`
时，它在同一 V GPU 上使用有界精确检索；更长的 Prompt 仍走原生 CAGRA。
`cagra` 模式原来的短 Prompt 拒绝行为不变。自动模式分别向索引预算申报
精确副本或 native cap，短 Prompt 不预留完整 CAGRA cap；但 GPU 精确副本
仍和权威 KV pool 竞争显存，因此必须为它留出明确预算。

For short prompts, explicitly select `--prompt-index-backend cagra-auto`
instead of `cagra`. At `rows <= intermediate_degree`, this uses bounded
exact search on the **same V GPU**; larger indexes still use native CAGRA.
Pure `cagra` retains its prior short-prompt refusal. Each path declares its
actual retained and scratch footprints to the index budget. Short indexes
avoid the full native cap, but their exact GPU copy still competes with the
authoritative KV pool and must be budgeted.

The three numeric graph/search defaults above are configuration defaults, not
tuned V100S results. CAGRA requires an actual indexed CUDA device; the factory
passes each V shard's real local device, not a hard-coded GPU zero. CPU-test
override is refused. No installation/version pin is performed automatically.

上面三个 graph/search 数值是参数默认值，不是 V100S 调优结果。工厂传递各 V
shard 的实际 CUDA device，禁止 CPU override；不自动安装或强制指定 cuVS 版本。

Current positive capability subset: contiguous float32 vectors, IP or squared-L2
native metric, IVF-PQ build, one graph per layer/KV head. Indexed row count must
exceed the configured intermediate graph degree. Shorter prompts are explicitly
refused in pure `cagra` mode (full-Prompt delivery remains available); only
the explicitly selected `cagra-auto` mode uses exact GPU fallback. `top_k`
must not exceed the row count or itopk bound on the native path.
No page representatives, head averaging, cross-head score merge, or out-of-core
CAGRA implementation is implied.

当前支持连续 float32、IP/L2、IVF-PQ build、每个 layer/KV head 独立图。纯
`cagra` 模式要求行数大于 intermediate degree，否则明确拒绝；仅显式
`cagra-auto` 模式会对短 Prompt 使用 GPU 精确回退。完整 Prompt 交付不受索引状态
限制。top-k 受行数和原生 itopk 限制。未实现页代表向量、跨 head
平均/合分或 out-of-core CAGRA。

## Ownership and bounds / 所有权与容量

The manager first reserves the extracted float32 copies, one full native cap per
head, and bounded Torch validation scratch. Each native cap remains reserved
throughout the graph's lifetime, covering retained graph/dataset bytes **and
native build/search workspace**. This intentionally over-reserves compared with
the final graph's size. Torch query/output scratch is separately reserved for
each search. This is application allocation accounting, not a GPU free-memory
guarantee: CUDA contexts, library code, allocator rounding/cache and other process
users still need deployment headroom.

manager 在分配前保留提取向量、每 head 的完整 native cap 和 Torch 校验临时量。
native cap 在整个索引生命周期保留，包含图/数据以及原生 build/search workspace，
因此比最终图大小更保守。每次查询另留 Torch 输出/校验临时量。此预算不等同于
GPU 空闲显存保证，CUDA context、库代码、分配器粒度/cache 和其他使用者仍需余量。

Each index owns an RMM `LimitingResourceAdaptor` over `CudaMemoryResource`.
The adapter serializes PVD CAGRA operations per device, switches the current RMM
resource only inside that scope, and restores the previous resource even on
exceptions. Other components in the V process must not independently swap RMM's
per-device resource concurrently with these operations.

An isolated hardware candidate also verified an **optional runtime-only**
parent limiter under all per-index limiters: it counts native allocations
across two graphs and rejects an aggregate over-cap cuVS C-API allocation.
The serving factory does not create this parent and `PromptIndexManager` does
not reserve its cap once yet. Until that accounting is integrated, production
still charges the full native cap per graph for its entire lifetime.

每个索引独占 RMM 限额资源；同设备 PVD CAGRA 操作串行，在受锁范围设置并恢复
当前 RMM 资源。V 进程的其他组件不能绕开该锁并发更换同设备的 RMM 资源。

隔离硬件探针还验证了可选的**仅运行时**父级 limiter：两个子索引的原生分配
计入同一个总上限，第二笔使总量超限的 cuVS C API 分配会被拒绝。生产工厂尚未
创建此父级 limiter，也未在 `PromptIndexManager` 中对总上限一次性计费。
完成预算接入前，生产仍按每图完整 native cap 终生预留。

A 256-byte allocation/free through the **loaded cuVS C library** must change the
Python limiter's count, and an allocation above the cap must fail. This checks the
actual C++/Python allocator bridge rather than assuming separately loaded libraries
share a registry. Missing symbols/APIs, a mismatched registry, or an ambiguous
allocation result refuse the operation. A pointer with an ambiguous free outcome
stays attached to its owner; it is not blindly freed again.

通过实际加载的 cuVS C 库进行 256-byte 分配/释放并核对 Python limiter 计数，
再验证超 cap 分配被拒绝，以确认实际共享 allocator registry。API/符号缺失、
不共享 registry 或结果不明确都拒绝；释放结果未知的指针保留在 owner，不能盲目重试。

Explicit device fences run before publishing, after search, and on failure cleanup.
Destroying the native index/resources must leave zero bytes in the limiter before
its owner can retire. An unproved fence or remaining allocation produces sticky
`IndexCompletionUnknown`: the backend refuses new work and the manager retains
the vectors, handles and reservations. Cancellation or a Python exception is not
a native completion fence. Limiter count does not prove absence of every possible
host-side leak inside an external library.

发布、search 和异常清理均显式 fence。销毁后 limiter 必须归零方可回收；未证明
完成或仍有分配则进入粘性的 UNKNOWN，停止新操作且保留向量、句柄和预算。Python
异常/取消不等于原生完成。设备计数不能证明外部库完全不存在 host-side leak。

## Interface compatibility / 接口兼容

The runtime passes a `cuvs.common.Resources` bound to the current Torch stream,
and explicit uint32-neighbor/float32-distance output buffers through DLPack. It
returns the existing logical selection contract: IP score is unchanged;
squared-L2 is converted to **negative Euclidean distance**, e.g. 25 becomes -5,
not -25. Negative squared distances are rejected, not clamped. Existing selection
checks still reject invalid IDs, duplicates and nonfinite scores. Q/K vector-space,
post-RoPE and layer/head identity checks remain in the manager.

传入绑定当前 Torch stream 的 cuVS Resources，显式提供 uint32 行号和 float32
距离 buffer。IP 保留原 score，平方 L2 转负欧氏距离（25 → -5）；负平方距离拒绝，
不 clamp。原有 logical ID、重复/有限性、Q/K 空间、post-RoPE、layer/head 校验不变。

Source-reviewed reference (not a required version or execution certificate):
源码核对参考，不是版本锁定或运行认证：

- [cuVS 25.02 CAGRA Python wrapper](https://github.com/rapidsai/cuvs/blob/v25.02.00/python/cuvs/cuvs/neighbors/cagra/cagra.pyx)
- [cuVS Resources and auto-sync wrapper](https://github.com/rapidsai/cuvs/blob/v25.02.00/python/cuvs/cuvs/common/resources.pyx)
- [cuVS RMM C API implementation](https://github.com/rapidsai/cuvs/blob/v25.02.00/cpp/src/core/c_api.cpp)
- [RMM resource adaptor implementation](https://github.com/rapidsai/rmm/blob/v25.02.00/python/rmm/rmm/pylibrmm/memory_resource.pyx)
- [RAFT workspace resource selection](https://github.com/rapidsai/raft/blob/v25.02.00/cpp/include/raft/core/resource/device_memory_resource.hpp)

See also [release-specific compatibility](PVD_CAGRA_Compatibility_CN_EN.md).
The reviewed wrapper's implicit auto-sync is not a finally-block error fence;
the backend does not rely on it for exception safety.

参见版本兼容文档；参考 wrapper 的隐式同步不是 finally 异常 fence，因此不能依赖
它保证异常释放安全。

## Evidence and next hardware check / 证据与设备验收

`test_pvd_cagra_backend.py` has CPU tests for policy, actual runtime call argument
construction with library/GPU doubles, allocator-bridge decisions, failure fences,
native ownership, startup selection, and real VectorKVStore/PromptIndexManager
budget lifetimes. Doubles do not execute cuVS.

测试覆盖策略、用 library/GPU doubles 验证实际 runtime 参数构造、allocator bridge
决策、失败 fence、原生 owner、启动配置和真实 store/manager 预算生命周期。
doubles 不构成 cuVS 执行证据。

On a prepared GPU environment, explicitly request the two real native cases:

在已准备好的 GPU 环境显式执行两项原生测试：

```bash
PYTHONPATH=python PVD_RUN_CAGRA_NATIVE=1 python \
  test/registered/disaggregation/run_pvd_cpu_tests.py \
  test/registered/disaggregation/test_pvd_cagra_backend.py \
  -k native_build_search_scores_and_dispose -q -rs
```

Once requested, missing CUDA/cuVS is a failure, not a skip. The fixture cap is
256 MiB per synthetic index, **not a suggested production cap**. These cases
exercise actual build/search/dispose, selected-row metric correctness, uniqueness,
ordering and limiter retirement. They do not establish actual-model recall,
V100S latency, distributed RDMA or full serving. Use the existing CAGRA acceptance
tools for query recall and record the installed versions/device separately.

显式请求后缺少 CUDA/cuVS 会失败而非 skip。fixture 每索引 cap 为 256 MiB，仅用于
合成验收，不是生产建议。验证实际 build/search/dispose、返回行对应数值、唯一性、
排序和 limiter 清空；真实模型召回、V100S 延迟、分布式 RDMA 和全服务另行验收。
