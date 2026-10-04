# V 发送 staging/MR 复用：本地验证

2026-10-05，顺序优化第 1 步。实现提交 `58a147863`，默认关闭；配置及公平
全路径对照已加入 `v-source-slots` 实验。保留原 CUDA fences、精确 native
终态与 cleanup 证明、候选和传输预算。

## 已执行结果

- CPU gate：334 passed、32 CUDA skipped。两次早期失败为 CLI 测试 fixture
  参数错误，修正后最终 gate 通过。使用显式 transport/CUDA policy doubles，
  不计作原生 GPU 或 RDMA 验证。
- 两份已捕获真实 KV 回放，共 840 个消费 bank，逐字节结果一致。每份回放
  注册次数由 770/772 降到 2，后续交付无需物理注册。
- 串行回放两 rank 实际持久缓冲占用 64 KiB；配置上限为 128 KiB。
  关闭后 staging charge 归零。UNKNOWN 与未完成 cleanup 保留槽和预算。
- 覆盖并发容量、重复 start、取消、短写、注册/复制/注销失败、物理 MR 身份
  及本地 owner 退休。相关基准脚本 AST 和 launcher shell syntax 通过。

详细输入/source hashes、失败记录及结果见同名 JSON。原始日志、证据包及
源码证明保存在项目内 `artifacts/ordered_delivery_20261005/step1/`。

## 尚未执行

CloudLab 已到期且没有 GPU；实际 CUDA 非默认 stream、Mooncake、双物理 rank
及完整 D wait/TPOT ABBA 对照待资源恢复。注册次数下降不能直接换算为等待
时间收益，因此仍保持默认关闭。
