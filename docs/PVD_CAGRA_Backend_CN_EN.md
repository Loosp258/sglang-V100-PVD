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

Add these to the existing V command; choose budgets from the deployment's real
capacity, not from this document:

在已有 V 命令中添加以下参数；两个预算必须根据实际容量配置，不提供猜测值：

```text
--prompt-index-backend cagra
--prompt-index-vector-space <target-model-and-encoding-identity>
--prompt-index-budget-bytes <total-index-budget-per-V-rank>
--prompt-index-cagra-native-bytes <native-cap-per-layer-and-KV-head-index>
--prompt-index-cagra-graph-degree 64
--prompt-index-cagra-intermediate-degree 128
--prompt-index-cagra-itopk-size 512
```

The three numeric graph/search defaults above are configuration defaults, not
tuned V100S results. CAGRA requires an actual indexed CUDA device; the factory
passes each V shard's real local device, not a hard-coded GPU zero. CPU-test
override is refused. No installation/version pin is performed automatically.

上面三个 graph/search 数值是参数默认值，不是 V100S 调优结果。工厂传递各 V
shard 的实际 CUDA device，禁止 CPU override；不自动安装或强制指定 cuVS 版本。

Current positive capability subset: contiguous float32 vectors, IP or squared-L2
native metric, IVF-PQ build, one graph per layer/KV head. Indexed row count must
exceed the configured intermediate graph degree. Shorter prompts are explicitly
refused for indexing (full-Prompt delivery remains available), not silently
switched to another algorithm. `top_k` must not exceed the row count or itopk bound.
No page representatives, head averaging, cross-head score merge, or out-of-core
CAGRA implementation is implied.

当前支持连续 float32、IP/L2、IVF-PQ build、每个 layer/KV head 独立图。行数必须
大于 intermediate degree，否则明确拒绝建索引，完整 Prompt 交付不受此索引状态
限制；不会静默切算法。top-k 受行数和 itopk 限制。未实现页代表向量、跨 head
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

每个索引独占 RMM 限额资源；同设备 PVD CAGRA 操作串行，在受锁范围设置并恢复
当前 RMM 资源。V 进程的其他组件不能绕开该锁并发更换同设备的 RMM 资源。

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
