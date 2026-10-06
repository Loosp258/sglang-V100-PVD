# 逐层交付固定开销：五步实施

2026-10-06。用户要求按顺序实施，每步验证后本地 commit，不上传 GitHub。
工作树 `sglang-V100-PVD-oasiskv`，分支 `codex/pvd-oasiskv`。
保留快图、V/CAGRA、Q/TopK/驻留预算与 actual-only 写回。
CloudLab 已到期，无 GPU；明确区分 CPU 验证与 CUDA/RDMA/TPOT 实测。
临时输出只放在本工作树 `artifacts/layer_overheads_20261006/`。

1. V 融合搜索直接消费结果对象，最终响应单次编码；保留所有请求与大小校验。
2. 两个 rank 的 owned ACK/close 并行推进，仍尝试并 join 全部清理，失败可见。
3. ACK/fence 复用请求级 channel，绑定完整 WriteIdentity，断线使用原 HTTP fence
   恢复；清理不挤占原两项搜索预算，通道总容量明确有界。
4. 本地 Q D2H、KV D2H、bank H2D 事件完成以异步等待推进，保持原 RDMA ordering。
   owner、临时内存与接收槽只在实际完成证明后退休；GPU 正确性待有 GPU 验证。
5. 精确缓存按版本传增量，绑定请求/Entry/layer/head；ACK 延迟不阻塞查询。
   丢版本/重排不得造成假命中；可重新发送完整快照恢复，不改变选择与 missing 集。

新模式默认关闭。每步记录代码、测试、资源/失败边界及 commit；有 GPU 后逐项
ABBA，只改变一项，检查精确输出、流量、峰值内存、D wait 与 TPOT。

## 本地完成情况

1. `7e109d953`：V typed batch/final encode，155 passed/2 CUDA skipped。
2. `41d4e7d3c`：并行 owned rank cleanup，51 passed。
3. `3b935ed72`：持久 channel ACK/fence，97 passed/1 CUDA skipped。
4. `31c845bd9`：local CUDA event async wait，70 passed/2 CUDA skipped。
5. 精确 cache delta/full resync；最终跨步 567 passed/4 CUDA skipped。

分步测试重叠，不相加。详见 `benchmark/results/pvd_layer_overheads_local_20261006.md`。
五项代码按顺序本地 commit；不 push。CUDA/RDMA/真实模型/TPOT gates 仍待 GPU。

## 可选配置

V 环境变量：`PVD_TYPED_BATCH_RESULTS=1`。

以下 D 字段加入已经通过旧配置校验的 JSON；新字段均默认 false：

```json
{
  "parallel_owned_cleanup": true,
  "channel_cleanup": true,
  "async_cuda_completion": true,
  "cache_delta_snapshots": true,
  "request_scratch_bytes": 33554432
}
```

要求已有 `ready_before_cleanup=true`、`async_layer_jobs=true`、
`fused_search_delivery=true`、`binary_queries=true`、`binary_control_channel=true`
和 `compact_cache_snapshots=true`。原 pinned/event/receive reuse 可继续使用；
`fused_zero_miss_proof=true` 可省掉全命中额外 fence。容量仍为32、workers仍为2。
local async+pinned 在 capacity32 增加32KiB rank分区；delta另按Prompt上限预留。
先逐项GPU对照，再测组合，不把开启全部视作已经验证收益。
