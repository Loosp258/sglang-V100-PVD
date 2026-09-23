# PVD CAGRA 兼容性与接入约束 / Compatibility and integration gates

首次核对 / Initially checked: 2026-09-22. This section records the historical
candidate selection before GPU access; the later V100S execution evidence is in
[CAGRA acceptance](PVD_CAGRA_Acceptance_CN_EN.md).

## 版本不能混为一谈 / Requirements are release-specific

- 当前 [cuVS 安装文档](https://docs.nvidia.com/cuvs/installation) 将当前源码
  构建要求写为 CUDA 12.2+、Ampere / SM80+。不能据此选择最新版用于 V100S。
  Current source-build requirements specify CUDA 12.2+ and Ampere / SM80+;
  do not assume the latest source supports V100S.
- 官方 [v25.02.00 构建文档](https://github.com/rapidsai/cuvs/blob/v25.02.00/docs/source/build.rst)
  列出 CUDA 11.4+、Volta / SM70+。它是值得验证的历史候选，不是已验证的软件栈，
  也不是本项目的强制版本。发行包的 CUDA/Python/驱动依赖仍需逐项确认。
  The tagged release lists CUDA 11.4+ and Volta / SM70+. It is a candidate for
  validation, not an accepted stack or a mandatory project pin. Wheel and
  dependency compatibility must be checked separately.
- [该版本 Python 源码](https://github.com/rapidsai/cuvs/blob/v25.02.00/python/cuvs/cuvs/neighbors/cagra/cagra.pyx)
  提供 `IndexParams`, `SearchParams`, `build`, `search`；search 返回
  `(distances, neighbors)`，不是 PVD 的 `(rows, scores)`。原生 L2 是平方距离、
  越小越好；当前 exact backend 的 PVD `l2` 是负欧氏距离，因此 L2 适配须
  先转换为欧氏距离再取负，而非仅对平方距离取负。IP 不应擅自改成
  cosine 或做归一化。仍须以实际结果的独立数值 oracle 验证。
  The tagged Python implementation returns distances before neighbor IDs.
  An adapter must reorder outputs and convert squared-L2 to negative Euclidean
  distance to match the current exact backend's `l2` scores, not merely negate
  squared-L2. Inner product must not silently become cosine. Verify numerically.

## 无设备可做什么 / Hardware-independent checks

```bash
python scripts/pvd/check_cagra.py --mode inventory
```

inventory 不导入 GPU 库，不安装任何依赖，成功表示完成采集而非 GPU 验收。
The default inventory neither imports GPU libraries nor installs dependencies;
exit zero means collection, not hardware acceptance.

设备就绪后，在隔离的候选环境运行 / When hardware is available, in an isolated
candidate environment:

```bash
python scripts/pvd/check_cagra.py --mode smoke --expected-gpu V100S
# Optional exact assertion against the version you deliberately installed:
python scripts/pvd/check_cagra.py --mode smoke --expected-gpu V100S --expect-cuvs-version 25.2.0
```

例中的版本仅为候选示例，不安装、不锁定、不自动降级；参数是对已加载
`cuvs.__version__` 的精确断言。JSON v3 同时记录发行包清单和实际导入的 cuVS /
CAGRA 路径，防止 PYTHONPATH 或 editable checkout 覆盖包时把 inventory 当成
执行证据。版本不匹配、API 缺失时在导入 CuPy/创建 CUDA context 前拒绝。

The example version is not installed, pinned or selected by the probe. The optional
flag asserts the imported version exactly. Schema v3 records imported module paths
alongside distribution inventory, exposing editable/PYTHONPATH shadowing. A version
mismatch or missing API fails before CuPy import/CUDA context creation.

版本一致不等于架构支持；`architecture_support=unverified` 不因版本相符而变为
通过。smoke 成功也只证明所报告的合成配置，不能代替真实目标 Q recall、
模型质量、生产 PVD 集成、RDMA 或性能测试。失败不回退其他算法/metric。
Matching versions do not prove architecture support. A successful synthetic smoke
does not establish model-query recall, generation quality, serving integration,
RDMA or performance. Failure never selects a different algorithm or metric.

## 接入仍需完成 / Remaining backend requirements

1. 将真实 cuVS 调用接到 `IndexBackend`，保留逻辑 token/page 映射及严格 Q 身份。
   Implement the native backend while preserving logical IDs and query identity.
2. 确定索引常驻、构建临时峰值、查询 scratch 的可执行预算约束；不能把 graph
   最终大小当作构建峰值，不能把经验测量当作硬上限。
   Enforce retained/build/search budgets; final graph size is not peak build cost,
   and a measurement is not a hard bound.
3. 明确原生句柄和流的生命周期；build/search 异常不天然证明 GPU 已完成。
   最后一个 reader 退出且原生操作完成后才能销毁/退预算。
   Establish native handle/stream lifetime and completion on errors before release.
4. 用同一批真实目标 post-RoPE Q 与 exact backend 对比，再做多 Entry、并发、
   close/build/search 交错的 GPU 验证。目前没有这些 GPU 证据。
   Compare real target post-RoPE queries to exact search and validate concurrent
   Entry lifecycle on the selected GPU stack. These GPU results do not exist yet.
