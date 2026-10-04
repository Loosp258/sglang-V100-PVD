# 五步交付优化与发布汇总

2026-10-05，分支 `codex/pvd-oasiskv`。按用户要求先上传全部历史未发布修改，
再按顺序实现、commit、push；大文件处理与旧提交映射见
`docs/pvd_github_publication_20261005.md`。未推送本地备份分支，未修改远端主分支。

| 步骤 | 实现提交 | 本地验证 |
|---|---|---|
| V 发送 staging/MR 复用 | `58a147863` | 334 passed/32 CUDA skipped；840 捕获 bank 字节一致，注册 770/772→2 |
| D READY 与 owned cleanup 分离 | `1c07fa293` | 64 passed；慢 ACK 前 bank 可消费，同线程清理及 close join |
| scoped producer event/adapter readiness | `e8d0e0406` | 128 passed/15 CUDA skipped；exact physical slice proof 与 UNKNOWN 保留 |
| 二进制 float32 Q | `ba377d83c` | 129 passed；真实 CPU HTTP 新旧查询结果及 Q bytes 一致 |
| 融合 search/reserve/start | `873fb2c20` | 120 passed/1 Linux CUDA skipped；两逻辑 rank、原选择策略、wire/重放/失败 |

各阶段测试存在重叠，数量不能相加。每步保持默认关闭，按独立选项准备 ABBA，
保留当前 fast V/CAGRA、逐层 Q、actual-only writeback、候选预算及既有 bootstrap。

## 最终提交上的证据

- 最终跨步骤 gate：**279 passed、16 actual CUDA skipped**；另有一项既有
  `asyncio_mode` pytest 配置 warning。
- 171 个源码/测试/benchmark/launcher 文件逐个验证 normalized-LF hash，
  与 `873fb2c209ef0819447b66726f6cc333eb4cac53` 的 Git blob 一致。
- benchmark AST、launcher shell syntax、全部 comparison scope/配置差异映射
  及五个错误 fused counter 拒绝检查通过。counter fixture 为明确合成数据，
  不冒充实际 native counters。
- 完整测试列表、源码 hash 和各步状态见同名 JSON。原始失败修复、日志及
  46782 字节证据包保存在项目内
  `artifacts/ordered_delivery_20261005/final/`。

## 资源恢复后还需完成

当前无 GPU，CloudLab 已到期。Linux CUDA/Mooncake、双物理 V rank、实际输出
质量、峰值预算/取消负载及每步独立 ABBA 的 D wait/TPOT 尚未执行。上述结果
证明本地协议、字节及生命周期，**不能证明线上已经加速**。

重点观察 READY 后 cleanup 是否把后续层排队拉长；fused 无 miss 时新增的
预授权注册/fence 成本；scoped event 在真实并行 workload 的 source readiness。
不把注册、RPC 次数和单阶段收益相加，也不自动组合五个实验。
