# 继续优化逐层 Q→KV 交付

2026-10-05。按用户要求依次实现，每步验证后本地 commit，不上传 GitHub。
工作树 `sglang-V100-PVD-oasiskv`，分支 `codex/pvd-oasiskv`。
保留现有快图、V/CAGRA、逐层配对前向、actual-only 写回和原检索预算。
没有 GPU；CPU 字节/协议/生命周期验证不作为 CUDA/RDMA 或 TPOT 证据。
所有临时产物放在本工作树 `artifacts/async_delivery_20261005/`。

1. **融合 READY 与清理分离。** 复用 `ready_before_cleanup`，使融合交付
   在 terminal-success 与本地缓存复制后先安装 bank、发布 READY，再由原
   owner 执行 ACK/close。允许与融合二进制、接收槽、pinned/event/channel
   组合。清理失败必须可见，资源在实际完成前保留。
2. **零 miss 证明随响应返回。** V 用原 WriteIdentity 安装 absent-write
   tombstone，响应携带精确的 fence 证明；D 验证后才能免掉额外 fence。
   丢响应、身份不符与已有 writer 仍使用原恢复机制。默认关闭新协议优化。
3. **冻结 Q 只编码一次。** 原始二进制快照同时用于摘要与发送；不可变字节
   绑定 metadata，保持旧摘要和搜索结果语义。测试 producer mutation、
   字节边界与两端一致性，避免重复数组打包/解析。
4. **请求级有界异步调度。** 分别管理有效查询/交付与退休清理的在途预算，
   网络 await 不独占层 worker。保持两项搜索/交付预算；不通过增加 native
   查询并发取得收益。新模式默认关闭，线程与注册 owner 一致，close 必须
   join 全部 admitted work，UNKNOWN 保留。测阻塞 ACK 时后续任务进展、
   精确层身份、deadline、取消、资源预算与实际网络并发。

每步报告修改、测试与未验证边界。GPU 恢复后逐步 ABBA 对照，只改一个
变量，记录搜索、交付、bank READY、ACK、排队、D wait、TPOT 和精确输出。
