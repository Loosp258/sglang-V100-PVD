# 缓存后处理与 Q 冻结：本地结果

2026-10-08，`codex/pvd-oasiskv`，每步本地 commit，不 push。
无 GPU；以下计时仅是 CPU 函数，包括输入校验与结果构造，不是 native CAGRA、
RDMA、D 等待或 TPOT。保持当前快图、Q、TopK、容量和写回规则。

## 1. 精确缓存直接 membership

- 默认关闭 `PVD_DIRECT_CACHE_MEMBERSHIP=1`，D/V 同时设置。
- bitmap 严格校验后只读 chosen token 的位；大 sorted sparse 做 searchsorted，
  最多64项的小 sparse 使用有界集合。保留 JSON list、原排序、摘要和 missing KV。
- `step1/gate03`：182 passed、1 CUDA skipped，一个既有 asyncio_mode warning。
  84种 Prompt/密度/编码组合，对照原集合；禁止 bitmap 展开，坏编码拒绝；
  两 rank 真实 CPU HTTP 通道验证 miss/mixed/hit 的交付字节与零 miss 行为。
- 同进程 CPU ABBA/BAAB，7轮，每批120次，排除15次暖身。
  两 head、同候选28项、bank32/max_new16、半数已缓存：

| Prompt | 原 choose_wire | 新 choose_wire | CPU 函数降幅 |
|---|---:|---:|---:|
|2159|299.9 us|108.1 us|64.0%|
|8192|744.3 us|166.9 us|77.6%|
|32768|2509.5 us|206.6 us|91.8%|

- 全缓存各形状降幅75.0–97.4%。每隔127项缓存的 sparse 分别12.9%、0.3%、
  **-0.4%**，短路径有噪声，不能声称所有密度都受益。
- 初始 sparse 向量路径出现负收益，加入小集合后复测；保留
  `step1/abba_before_small_sparse.json`，不混合不同源码计时。
- 原始样本、源码 hash、日志与命令：`artifacts/cache_compute_20261008/step1/`。
